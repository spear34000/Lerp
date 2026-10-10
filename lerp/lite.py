"""Minimal real, torch-CPU safetensors merge backend for linear/task_arithmetic.

This backend reads one tensor from each parent at a time, writes HF shard files,
and avoids importing transformers or downloading weights. It supports only
identically named/shaped safetensors checkpoints, no adapters, no quantization,
and no stochastic TIES / DARE. Those require MergeKit.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .weighting import language_layer_count, tensor_coefficients
from .mergeops import NEW_METHODS, needs_base
from .spec import Spec


class LiteMergeError(RuntimeError):
    pass


_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


def _source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise LiteMergeError(f"Lite engine requires local HF checkpoint folders, got {root}")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise LiteMergeError(f"No .safetensors weight shards found in {root}. LoRA or GGUF is not supported")
    return files


def _tensor_index(root: Path, stack: ExitStack, safe_open: Any) -> dict[str, Any]:
    index = {}
    for file in _source_files(root):
        handle = stack.enter_context(safe_open(str(file), framework="pt", device="cpu"))
        for key in handle.keys():
            if key in index:
                raise LiteMergeError(f"Duplicate tensor key {key!r} in {root}")
            index[key] = handle
    return index


def _merge_weights_for_tensor(genes: list[float], spec: Spec, name: str, layer_count: int) -> list[float]:
    return tensor_coefficients(spec, genes, name, layer_count)


# Tensors above this many elements (e.g. Gemma 4's 2.8B-element per-layer embedding table) are merged in
# row chunks read via safetensors slices, so peak RAM stays near the output size instead of ~3x the fp32 tensor.
LARGE_TENSOR_ELEMS = 128 * 1024 * 1024
CHUNK_ELEMS = 32 * 1024 * 1024
_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


def _merge_large_tensor(key: str, slices: list, weights: list[float], method: str, spec: Spec, cast_to, torch) -> Any:
    shapes = [tuple(s.get_shape()) for s in slices]
    if any(s != shapes[0] for s in shapes):
        raise LiteMergeError(f"Tensor shape mismatch: {key}")
    if any(s.get_dtype() != slices[0].get_dtype() for s in slices):
        raise LiteMergeError(f"Tensor dtype mismatch: {key}")
    shape = shapes[0]
    row_elems = max(1, math.prod(shape[1:]))
    rows = max(1, CHUNK_ELEMS // row_elems)
    output = torch.empty(shape, dtype=cast_to)
    for start in range(0, shape[0], rows):
        end = min(shape[0], start + rows)
        parts = [s[start:end] for s in slices]
        if any(not torch.isfinite(part).all() for part in parts):
            raise LiteMergeError(f"Non-finite parent tensors: {key}")
        if method == "linear":
            acc = torch.zeros_like(parts[0], dtype=torch.float32)
            for coef, part in zip(weights, parts):
                acc.add_(part.float(), alpha=coef)
        else:
            base32 = parts[0].float()
            acc = base32.clone()
            for coef, part in zip(weights, parts[1:]):
                acc.add_(part.float() - base32, alpha=spec.task_scale * coef)
        chunk = acc.to(dtype=cast_to)
        if not torch.isfinite(chunk).all():
            raise LiteMergeError(f"Merged tensor overflowed target dtype: {key}")
        output[start:end] = chunk
    return output


def _merge_new_method(key: str, handles: list, weights: list[float], method: str, spec: Spec, cast_to, torch):
    """SLERP / TIES / DARE for one tensor (see lerp.mergeops); ``handles`` are the safetensors handles of [base] + parents (or parents)."""
    from .mergeops import TensorMerger
    shapes = [tuple(h.get_slice(key).get_shape()) for h in handles]
    dtypes = {h.get_slice(key).get_dtype() for h in handles}
    if any(s != shapes[0] for s in shapes) or len(dtypes) != 1:
        raise LiteMergeError(f"Tensor shape or dtype mismatch: {key}")
    shape = shapes[0]

    def read_cpu(i: int, lo: int, hi: int):
        handle = handles[i]
        part = handle.get_slice(key)[lo:hi] if shape else handle.get_tensor(key)
        if not torch.isfinite(part).all():
            raise LiteMergeError(f"Non-finite parent tensors: {key}")
        return part.float()

    merger = TensorMerger(torch, method, weights, shape=shape, key=key, task_scale=spec.task_scale, density=spec.density,
                          seed=spec.seed, device="cpu")
    output = torch.empty(shape, dtype=cast_to)
    merger.merge(read_cpu, read_cpu, output)
    if not torch.isfinite(output).all():
        raise LiteMergeError(f"Merged tensor overflowed target dtype: {key}")
    return output


def build_lite(spec: Spec, genes: list[float], destination: Path, *, max_shard_mb: int = 128, method: str | None = None) -> dict:
    """Produce actual HF-format safetensors tensors in destination.

    Destination must be empty/nonexistent and must be private to the caller.
    No model code is executed, and output is only committed by caller on success.
    """
    if spec.mode != "full":
        raise LiteMergeError("Use --engine lora for LoRA adapters")
    chosen_method = method or spec.method
    if chosen_method not in {"linear", "task_arithmetic", *NEW_METHODS}:
        raise LiteMergeError(f"Lite supports linear, task_arithmetic, slerp, ties, dare_ties and dare_linear, not {chosen_method}. Use --engine mergekit")
    if chosen_method == "slerp" and len(spec.parents) != 2:
        raise LiteMergeError("slerp merges exactly two parents")
    if max_shard_mb < 1:
        raise LiteMergeError("max_shard_mb must be positive")
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise LiteMergeError('Install the CPU dependencies: pip install -e ".[lite]"') from exc

    source_roots = [Path(p.model) for p in spec.parents]
    if needs_base(chosen_method):
        source_roots = [Path(spec.base_model)] + source_roots
    for root in source_roots:
        _source_files(root)
        if not (root / "config.json").is_file():
            raise LiteMergeError(f"Missing config.json: {root}")
    if destination.exists() and any(destination.iterdir()):
        raise LiteMergeError(f"Destination not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    config_source = Path(spec.base_model) if needs_base(chosen_method) else Path(spec.parents[0].model)
    config = json.loads((config_source / "config.json").read_text(encoding="utf-8"))
    num_layers = language_layer_count(config)
    if type(num_layers) is not int or num_layers < 1:
        raise LiteMergeError("Lite requires num_hidden_layers or n_layer in config.json (or in text_config)")

    cast_to = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[spec.out_dtype]
    tensor_to_shard: dict[str, str] = {}
    total_size = 0
    shard_limit = max_shard_mb * 1024 * 1024
    tensor_buffer = {}
    buffer_size = 0
    shard_idx = 0
    shards: list[str] = []

    def flush() -> None:
        nonlocal tensor_buffer, buffer_size, shard_idx
        if not tensor_buffer:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(tensor_buffer, str(destination / name), metadata={"format": "pt"})
        for tensor_name in tensor_buffer:
            tensor_to_shard[tensor_name] = name
        shards.append(name)
        tensor_buffer = {}
        buffer_size = 0

    with ExitStack() as stack, torch.no_grad():
        sources = [_tensor_index(root, stack, safe_open) for root in source_roots]
        expected_keys = set(sources[0])
        for root, index in zip(source_roots[1:], sources[1:]):
            if set(index) != expected_keys:
                missing = expected_keys - set(index)
                extra = set(index) - expected_keys
                raise LiteMergeError(f"Tensor keys differ for {root}: missing={len(missing)}, extra={len(extra)}")

        for key in sorted(expected_keys):
            slices = [index[key].get_slice(key) for index in sources]
            if chosen_method in NEW_METHODS and slices[0].get_dtype() in _FLOAT_DTYPES:
                weights = _merge_weights_for_tensor(genes, spec, key, num_layers)
                output = _merge_new_method(key, [index[key] for index in sources], weights, chosen_method, spec, cast_to, torch)
                size = output.numel() * output.element_size()
                if tensor_buffer and buffer_size + size > shard_limit:
                    flush()
                tensor_buffer[key] = output
                buffer_size += size
                total_size += size
                del output, slices
                continue
            if (math.prod(slices[0].get_shape()) > LARGE_TENSOR_ELEMS and len(slices[0].get_shape()) >= 1
                    and slices[0].get_dtype() in _FLOAT_DTYPES):
                weights = _merge_weights_for_tensor(genes, spec, key, num_layers)
                output = _merge_large_tensor(key, slices, weights, chosen_method, spec, cast_to, torch)
                size = output.numel() * output.element_size()
                if tensor_buffer and buffer_size + size > shard_limit:
                    flush()
                tensor_buffer[key] = output
                buffer_size += size
                total_size += size
                del output, slices
                continue
            tensors = [index[key].get_tensor(key) for index in sources]
            if any(t.shape != tensors[0].shape for t in tensors):
                raise LiteMergeError(f"Tensor shape mismatch: {key}")
            if any(t.dtype != tensors[0].dtype for t in tensors):
                raise LiteMergeError(f"Tensor dtype mismatch: {key}")
            if any(t.is_floating_point() and not torch.isfinite(t).all() for t in tensors):
                raise LiteMergeError(f"Non-finite parent tensors: {key}")
            if not tensors[0].is_floating_point():
                if any(not torch.equal(tensors[0], t) for t in tensors[1:]):
                    raise LiteMergeError(f"Non-floating tensor differs between parents: {key}")
                output = tensors[0].contiguous()
            else:
                weights = _merge_weights_for_tensor(genes, spec, key, num_layers)
                if chosen_method == "linear":
                    output = torch.zeros_like(tensors[0], dtype=torch.float32)
                    for coef, tensor in zip(weights, tensors):
                        output.add_(tensor.float(), alpha=coef)
                else:
                    base, *parents = tensors
                    base32 = base.float()
                    output = base32.clone()
                    for coef, parent in zip(weights, parents):
                        output.add_(parent.float() - base32, alpha=spec.task_scale * coef)
                output = output.to(dtype=cast_to).contiguous()
                if not torch.isfinite(output).all():
                    raise LiteMergeError(f"Merged tensor overflowed target dtype: {key}")
            size = output.numel() * output.element_size()
            if tensor_buffer and buffer_size + size > shard_limit:
                flush()
            tensor_buffer[key] = output
            buffer_size += size
            total_size += size
            del tensors, output
        flush()

    # HF sharded tensor index is used even for one shard; supported by transformers.
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": tensor_to_shard}, indent=2) + "\n",
        encoding="utf-8",
    )
    # Restrict copies to the metadata/tokenizer allowlist; no remote code.
    metadata_names = {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "tokenizer.model", "chat_template.jinja", "sentencepiece.bpe.model", "vocab.txt",
    }
    for filename in sorted(metadata_names):
        source = config_source / filename
        if source.is_file() and not source.is_symlink():
            shutil.copyfile(source, destination / filename)
    return {"tensors": len(tensor_to_shard), "weight_bytes": total_size,
            "shards": shards, "engine": "lite", "floating_point_accumulator": "float32"}
