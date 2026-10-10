"""Exact combination of LoRA adapters with signed coefficients, and the graft crossover built on it.

``combine_adapters`` writes the adapter whose delta is ``sum_i coef_i * scale_i * B_i A_i`` (scale_i = alpha_i / r_i) by concatenating ranks, the
same exact construction as ``lerp.lora.build_lora`` but without confining the coefficients to a convex blend, so it can express sums and differences.

``graft_parts`` is the crossover that keeps what a parent learned: ``child = A + (B - init_B)``, where ``init_B`` is the adapter B started its own
training from. Adding only B's *learned change* (and not B's whole delta) means shared ancestry is counted once; a plain ``A + B`` of two
siblings would count the founders they both inherited twice, then four times in the next generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


class CombineError(ValueError):
    pass


def _order_free(value):
    return sorted(value) if isinstance(value, (list, tuple, set)) else value


def adapter_delta_scale(cfg: dict) -> float:
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise CombineError("only plain LoRA adapters can be combined")
    return float(cfg["lora_alpha"]) / int(cfg["r"])


def combine_adapters(parts: Sequence[tuple[Path, float]], dest: Path, *, out_scale: float = 2.0) -> dict:
    """Adapter at ``dest`` with delta ``sum coef_i * scale_i * B_i A_i``, stored with scale ``out_scale`` (alpha = out_scale * r, B divided by out_scale).

    ``out_scale=2`` matches freshly trained adapters (alpha = 2r), so inherited adapters learn at the same effective rate."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if not parts:
        raise CombineError("nothing to combine")
    if out_scale <= 0:
        raise CombineError("out_scale must be positive")
    dirs = [Path(p) for p, _ in parts]
    configs = [json.loads((d / "adapter_config.json").read_text(encoding="utf-8")) for d in dirs]
    for cfg in configs[1:]:
        if cfg.get("base_model_name_or_path") != configs[0].get("base_model_name_or_path"):
            raise CombineError("adapters were trained on different base models")
        if _order_free(cfg.get("target_modules")) != _order_free(configs[0].get("target_modules")):
            raise CombineError("adapters target different modules")
    scales = [adapter_delta_scale(c) for c in configs]
    handles = [safe_open(str(d / "adapter_model.safetensors"), framework="pt") for d in dirs]
    try:
        keys = set(handles[0].keys())
        if any(set(h.keys()) != keys for h in handles[1:]):
            raise CombineError("adapter tensor keys differ")
        out: dict = {}
        with torch.no_grad():
            for key in sorted(k for k in keys if k.endswith(".lora_A.weight")):
                b_key = key.replace(".lora_A.", ".lora_B.")
                if b_key not in keys:
                    raise CombineError(f"missing {b_key}")
                As = [h.get_tensor(key).float() for h in handles]
                Bs = [h.get_tensor(b_key).float() * (coef * s / out_scale) for h, (_, coef), s in zip(handles, parts, scales)]
                if any(a.shape[1] != As[0].shape[1] for a in As) or any(b.shape[0] != Bs[0].shape[0] for b in Bs):
                    raise CombineError(f"module dimensions differ: {key}")
                out[key] = torch.cat(As, 0).contiguous()
                out[b_key] = torch.cat(Bs, 1).contiguous()
    finally:
        for h in handles:
            del h
    if not out:
        raise CombineError("no LoRA modules found")
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise CombineError(f"destination not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest / "adapter_model.safetensors"), metadata={"format": "pt"})
    rank = sum(int(c["r"]) for c in configs)
    final = dict(configs[0])
    final.update({"r": rank, "lora_alpha": out_scale * rank, "rank_pattern": {}, "alpha_pattern": {}})
    (dest / "adapter_config.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return {"rank": rank, "parts": len(parts), "out_scale": out_scale}


def graft_parts(primary: Path, secondary: Path, secondary_init: Path | None) -> list[tuple[Path, float]]:
    """Parts of ``primary + (secondary - secondary_init)``; a founder has no ``secondary_init`` (it started from the base model, i.e. zero)."""
    parts = [(Path(primary), 1.0), (Path(secondary), 1.0)]
    if secondary_init is not None:
        parts.append((Path(secondary_init), -1.0))
    return parts
