"""LoRA learning that continues from an inherited adapter, and rank compression so inherited adapters do not keep growing."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence


class LearningError(RuntimeError):
    pass


def pairs_sha256(pairs: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for prompt, completion in pairs:
        h.update(json.dumps([prompt, completion], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_adapter(base: str, pairs: Sequence[tuple[str, str]], out: Path, *, init: Path | None = None, steps: int = 100, lr: float = 2e-4,
                  rank: int = 16, seed: int = 0, device: str = "cpu", dtype: str = "bfloat16", batch: int = 4, accum: int = 2,
                  max_len: int = 64, ordered: bool = False, data_offset_steps: int = 0, schedule_total: int | None = None,
                  schedule_offset: int = 0, warmup: int | None = None, state_in: Path | None = None, state_out: Path | None = None) -> dict:
    """Supervised fine-tuning of a LoRA adapter on (prompt, completion) pairs; the loss is on the completion tokens only.

    ``init`` continues training from an existing adapter (an inherited, possibly merged one) instead of starting from a fresh one.

    Segmented training that is identical to one continuous run needs three things, each its own switch:

    * ``ordered=True`` consumes ``pairs`` in the given order (no shuffling) starting at ``data_offset_steps`` optimizer steps into the stream, so
      consecutive segments read consecutive slices of one prepared stream;
    * ``schedule_total`` / ``schedule_offset`` place this segment inside a global learning-rate schedule (default: a schedule of its own);
    * ``state_in`` / ``state_out`` carry the optimizer state (Adam moments and step count) from one segment to the next.

    Without them every call is an independent run with its own warm-up, cosine decay and fresh optimizer."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not pairs:
        raise LearningError("no training pairs")
    if steps < 1:
        raise LearningError("steps must be positive")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype))
    if init is not None:
        model = PeftModel.from_pretrained(model, str(init), is_trainable=True)
    else:
        from ..targets import discover
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                 target_modules=discover(model)))
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise LearningError("the adapter has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)
    encoded = []
    for prompt, completion in pairs:
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(prompt + completion, add_special_tokens=False)["input_ids"][len(p):]
        if len(p) + len(c) <= max_len and c:  # a cut-off answer would teach the wrong thing
            encoded.append((p, c))
    if not encoded:
        raise LearningError("every training pair is longer than max_len")
    order = list(range(len(encoded)))
    if not ordered:
        rng.shuffle(order)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    sched_total = schedule_total or steps
    warm = warmup if warmup is not None else max(1, min(20, sched_total // 5))
    if state_in is not None:
        opt.load_state_dict(torch.load(str(state_in), map_location="cpu"))
    model.train()
    cursor, last = data_offset_steps * accum * batch, 0.0
    for local in range(steps):
        step = schedule_offset + local
        factor = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, sched_total - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * factor
        running = 0.0
        for _ in range(accum):
            items = [encoded[order[(cursor + i) % len(order)]] for i in range(batch)]
            cursor += batch
            width = max(len(p) + len(c) for p, c in items)
            ids = torch.tensor([p + c + [pad] * (width - len(p) - len(c)) for p, c in items])
            lab = torch.tensor([[-100] * len(p) + c + [-100] * (width - len(p) - len(c)) for p, c in items])
            mask = torch.tensor([[1] * (len(p) + len(c)) + [0] * (width - len(p) - len(c)) for p, c in items])
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=lab.to(device)).loss / accum
            loss.backward()
            running += float(loss.detach())
        total = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))  # clip_grad_norm_ crashes Intel Arc XPU
        scale = torch.clamp(1.0 / (total + 1e-6), max=1.0)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = running
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    if state_out is not None:
        Path(state_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(opt.state_dict(), str(state_out))
    return {"steps": steps, "lr": lr, "seed": seed, "final_loss": last, "pairs": len(encoded), "pairs_sha256": pairs_sha256(pairs),
            "init": None if init is None else str(init)}


def compress_adapter(src: Path, dst: Path, rank: int, out_scale: float = 1.0) -> dict:
    """Best rank-``rank`` approximation (truncated SVD) of every module delta ``(alpha/r) B A``, written as a standard adapter with scale ``out_scale``
    (alpha = out_scale * r, B divided by out_scale, so the delta is unchanged; 2 matches freshly trained adapters).

    Works on the small factors only (QR of B and A^T, SVD of an r x r matrix), so it is exact for the retained directions and cheap."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise LearningError("compress_adapter supports plain LoRA only")
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / r
    out: dict = {}
    kept = total = 0.0
    with safe_open(str(src / "adapter_model.safetensors"), framework="pt") as f:
        keys = set(f.keys())
        for key in sorted(keys):
            if not key.endswith(".lora_A.weight"):
                continue
            b_key = key.replace(".lora_A.", ".lora_B.")
            if b_key not in keys:
                raise LearningError(f"missing {b_key}")
            A, B = f.get_tensor(key).float(), f.get_tensor(b_key).float()
            Qb, Rb = torch.linalg.qr(B)
            Qa, Ra = torch.linalg.qr(A.T)
            U, S, Vh = torch.linalg.svd(scale * Rb @ Ra.T)
            k = min(rank, S.numel())
            root = S[:k].sqrt()
            out[b_key] = (Qb @ (U[:, :k] * root) / out_scale).contiguous()
            out[key] = ((root[:, None] * Vh[:k]) @ Qa.T).contiguous()
            kept += float((S[:k] ** 2).sum())
            total += float((S ** 2).sum())
        if not out:
            raise LearningError("no LoRA modules in the adapter")
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
    new = dict(cfg)
    new.update({"r": min(rank, r), "lora_alpha": out_scale * min(rank, r), "rank_pattern": {}, "alpha_pattern": {}})
    (dst / "adapter_config.json").write_text(json.dumps(new, indent=2), encoding="utf-8")
    return {"rank_in": r, "rank_out": min(rank, r), "energy_kept": kept / total if total else 1.0, "out_scale": out_scale}
