"""Exact weighted PEFT LoRA delta merger via rank concatenation.

For ordinary linear LoRA, delta(W) = scale * B @ A. Concatenating factor
pairs with their coefficients applied to B computes the exact sum of deltas,
without materializing the base LLM or approximating by averaging A/B factors.

Deliberately rejects exotic PEFT extensions (DoRA, LoHa, LoftQ, bias,
modules_to_save, nonuniform rank/alpha patterns, embedding LoRA) rather than
silently producing a misleading or unloadable adapter.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .spec import Spec
from .weighting import is_non_text_tensor, language_layer_count, tensor_coefficients


class LoRAMergeError(RuntimeError):
    pass


def _adapter_source(root: Path) -> tuple[dict, Path]:
    cfg_path = root / "adapter_config.json"
    weights_path = root / "adapter_model.safetensors"
    if not cfg_path.is_file() or not weights_path.is_file() or cfg_path.is_symlink() or weights_path.is_symlink():
        raise LoRAMergeError(f"Expected local PEFT adapter_config.json and adapter_model.safetensors in {root}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("peft_type") != "LORA":
        raise LoRAMergeError(f"Only standard PEFT LORA supported: {root}")
    for unsupported in ("use_dora", "use_qalora", "use_bdlora", "use_arrow", "lora_bias", "alora_invocation_tokens", "use_alora", "use_xlora", "layer_replication", "trainable_token_indices", "enable_lora"):
        if cfg.get(unsupported):
            raise LoRAMergeError(f"Unsupported {unsupported} in {root}")
    if cfg.get("bias", "none") != "none" or cfg.get("modules_to_save"):
        raise LoRAMergeError(f"LoRA bias/modules_to_save are not supported: {root}")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise LoRAMergeError("LoRA rank_pattern/alpha_pattern are not supported in this exact-rank backend")
    if cfg.get("use_xlora") or cfg.get("megatron_config"):
        raise LoRAMergeError("Non-standard LoRA integrations are unsupported")
    rank, alpha = cfg.get("r"), cfg.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise LoRAMergeError("Each adapter requires positive r and lora_alpha")
    if not cfg.get("base_model_name_or_path"):
        raise LoRAMergeError("Missing base_model_name_or_path; ancestry cannot be checked")
    return cfg, weights_path


def _order_free(value):
    """PEFT stores target_modules as a set, so its list order varies between processes."""
    if isinstance(value, (list, tuple)):
        try:
            return sorted(value)
        except TypeError:
            return list(value)
    return value


def verify_adapters(spec: Spec) -> dict:
    """Metadata-only strict preflight; weights checked during build."""
    if spec.mode != "lora":
        raise LoRAMergeError("Expected mode: lora")
    configs = []
    for parent in spec.parents:
        cfg, _ = _adapter_source(Path(parent.model))
        configs.append(cfg)
    for cfg in configs[1:]:
        if cfg["base_model_name_or_path"] != configs[0]["base_model_name_or_path"]:
            raise LoRAMergeError("Adapter base_model_name_or_path values differ")
        for field in ("target_modules", "exclude_modules", "layers_to_transform", "layers_pattern"):
            if _order_free(cfg.get(field)) != _order_free(configs[0].get(field)):
                raise LoRAMergeError(f"Adapter {field} differs")
        if cfg.get("fan_in_fan_out", False) != configs[0].get("fan_in_fan_out", False):
            raise LoRAMergeError("Adapter fan_in_fan_out differs")
        if cfg.get("task_type") != configs[0].get("task_type"):
            raise LoRAMergeError("Adapter task_type differs")
        if cfg.get("revision") != configs[0].get("revision"):
            raise LoRAMergeError("Adapter base revision differs")
    return {"declared_base": configs[0]["base_model_name_or_path"],
            "ranks": [cfg["r"] for cfg in configs],
            "output_rank": sum(cfg["r"] for cfg in configs),
            "parents": len(configs)}


def build_lora(spec: Spec, genes: list[float], destination: Path, *, method: str = "linear") -> dict:
    if method not in ("linear", "task_arithmetic"):
        raise LoRAMergeError("LoRA backend supports only linear and task_arithmetic")
    if len(genes) != spec.genome_size:
        raise LoRAMergeError("Genome size mismatch")
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise LoRAMergeError('Install dependencies: pip install -e ".[lora]"') from exc

    meta = verify_adapters(spec)
    if meta["output_rank"] > spec.max_output_rank:
        raise LoRAMergeError(f"Output rank {meta['output_rank']} exceeds max_output_rank={spec.max_output_rank}; reduce the parents or raise the explicit budget")
    configs, sources = zip(*(_adapter_source(Path(parent.model)) for parent in spec.parents))
    from contextlib import ExitStack
    # The output is a normal PEFT adapter, not a full checkpoint.
    if destination.exists() and any(destination.iterdir()):
        raise LoRAMergeError(f"Destination not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, torch.no_grad():
        handles = [stack.enter_context(safe_open(str(p), framework="pt", device="cpu")) for p in sources]
        keys = set(handles[0].keys())
        if not keys:
            raise LoRAMergeError("Empty adapter tensors")
        if any(set(h.keys()) != keys for h in handles[1:]):
            raise LoRAMergeError("LoRA adapter tensor keys differ (target layer mismatch)")
        modules = {}
        for key in keys:
            if key.endswith(".lora_A.weight"):
                part = "A"
                module = key[:-len(".lora_A.weight")]
            elif key.endswith(".lora_B.weight"):
                part = "B"
                module = key[:-len(".lora_B.weight")]
            else:
                raise LoRAMergeError(f"Unsupported adapter parameter: {key}")
            modules.setdefault(module, {})[part] = key
        if any(set(pair) != {"A", "B"} for pair in modules.values()):
            raise LoRAMergeError("Missing LoRA factor A or B")

        # Header-only resource estimate: reject over-budget outputs before
        # loading any large tensors. Peak RAM will be HIGHER than this bound.
        bytes_per_item = {"float16": 2, "bfloat16": 2, "float32": 4}[spec.out_dtype]
        expected_tensor_bytes = 0
        for module, pair in modules.items():
            a_shape = handles[0].get_slice(pair["A"]).get_shape()
            b_shape = handles[0].get_slice(pair["B"]).get_shape()
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != configs[0]["r"] or b_shape[1] != configs[0]["r"]:
                raise LoRAMergeError(f"Unexpected LoRA factor header shapes: {module}")
            for cfg, h in zip(configs[1:], handles[1:]):
                other_a = h.get_slice(pair["A"]).get_shape()
                other_b = h.get_slice(pair["B"]).get_shape()
                if (len(other_a) != 2 or len(other_b) != 2 or
                        other_a[0] != cfg["r"] or other_b[1] != cfg["r"] or
                        other_a[1] != a_shape[1] or other_b[0] != b_shape[0]):
                    raise LoRAMergeError(f"Incompatible LoRA factor header shapes: {module}")
            expected_tensor_bytes += meta["output_rank"] * (a_shape[1] + b_shape[0]) * bytes_per_item
        if expected_tensor_bytes > spec.max_lora_output_mib * 1024 * 1024:
            raise LoRAMergeError(
                f"Estimated LoRA tensor output {expected_tensor_bytes / 1048576:.1f} MiB exceeds "
                f"max_lora_output_mib={spec.max_lora_output_mib} (peak RAM is higher)")

        # Only module-backed adapter tensors; no code from the model is imported.
        config_path = Path(spec.base_model) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            # A remote base may not exist locally; module indices then reveal the
            # number of layers, with a conservative fallback to max_index+1.
            config = {}
        import re
        layer_indices = [int(m.group(1)) for module in modules
                         if not is_non_text_tensor(module)
                         and (m := re.search(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.", module))]
        if not layer_indices:
            raise LoRAMergeError("Layer indexing not recognized; cannot assign layer-wise genes safely")
        detected_layers = max(layer_indices) + 1
        n_layers = language_layer_count(config) or detected_layers
        if not isinstance(n_layers, int) or n_layers < detected_layers:
            raise LoRAMergeError("base config layers incompatible with LoRA tensors")

        output = {}
        for module, pair in sorted(modules.items()):
            As, Bs = [], []
            coeffs = tensor_coefficients(spec, genes, module, n_layers)
            for cfg, h, coefficient in zip(configs, handles, coeffs):
                a, b = h.get_tensor(pair["A"]), h.get_tensor(pair["B"])
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != cfg["r"] or b.shape[1] != cfg["r"]:
                    raise LoRAMergeError(f"Unexpected rank/shapes for {module}")
                if not a.is_floating_point() or not b.is_floating_point():
                    raise LoRAMergeError(f"LoRA factors must be floating point: {module}")
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise LoRAMergeError(f"Non-finite LoRA factor values: {module}")
                if As and (a.shape[1] != As[0].shape[1] or b.shape[0] != Bs[0].shape[0]):
                    raise LoRAMergeError(f"LoRA dimensions differ: {module}")
                rank = cfg["r"]
                scale = cfg["lora_alpha"] / (math.sqrt(rank) if cfg.get("use_rslora", False) else rank)
                if method == "task_arithmetic":
                    scale *= spec.task_scale
                As.append(a.to(torch.float32))
                Bs.append(b.to(torch.float32) * (float(coefficient) * scale))
            output[pair["A"]] = torch.cat(As, dim=0).contiguous()
            output[pair["B"]] = torch.cat(Bs, dim=1).contiguous()
        if len(output) != len(keys):
            raise LoRAMergeError("Output adapter tensor count mismatch")
        target_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[spec.out_dtype]
        output = {k: v.to(target_dtype).contiguous() for k, v in output.items()}
        if any(not torch.isfinite(v).all() for v in output.values()):
            raise LoRAMergeError("LoRA coefficients overflowed target dtype; use float32 or smaller task_scale")
        save_file(output, str(destination / "adapter_model.safetensors"), metadata={"format": "pt"})

    final = dict(configs[0])
    final.update({"r": meta["output_rank"], "lora_alpha": meta["output_rank"],
                  "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {},
                  "inference_mode": True, "use_dora": False})
    (destination / "adapter_config.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    details = {"engine": "lora", "method": method, "delta_concatenation_in_fp32_before_output_cast": True,
               "input_ranks": meta["ranks"], "output_rank": meta["output_rank"],
               "estimated_output_tensor_bytes": expected_tensor_bytes,
               "declared_base": meta["declared_base"], "modules": len(modules),
               "parameter_tensors": len(output)}
    (destination / "modelbreeder_provenance.json").write_text(
        json.dumps({**details, "parents": [p.model for p in spec.parents],
                    "gene_groups": list(spec.gene_groups), "genes": genes}, indent=2) + "\n", encoding="utf-8")
    return details
