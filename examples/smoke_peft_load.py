"""P0 smoke verification of actual child LoRA loading and short generation.

This is a separate, OPTIONAL, real-world gate and was NOT run by the release
maintainers without real PEFT/Transformers models. Install lerp[lora]
and use the same local frozen base as the parents.

Usage:
  python examples/smoke_peft_load.py \
    --base checkpoints/base \
    --adapter runs/lora/generations/gen-000/cand-000/model \
    --prompt 'Answer briefly: what is 2 + 2?' \
    --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Load actual PEFT child and generate tokens (P0 gate)")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello! Answer briefly: what is 2 + 2?")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("--max-new-tokens must be between 1 and 64")
    if not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error("Adapter does not contain adapter_model.safetensors")
    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("Adapter does not contain adapter_config.json")

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; Intel Arc XPU is not CUDA")
    start = time.perf_counter()
    configuration = PeftConfig.from_pretrained(str(args.adapter))
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=False)
    model = PeftModel.from_pretrained(base_model, str(args.adapter), is_trainable=False)
    model = model.to(args.device).eval()
    prepared = tokenizer(args.prompt, return_tensors="pt")
    prepared = {key: value.to(args.device) for key, value in prepared.items()}
    with torch.inference_mode():
        logits = model(**prepared).logits
        if logits.dim() != 3 or not torch.isfinite(logits.float()).all().item():
            raise RuntimeError("Non-finite or malformed logits from loaded PEFT adapter")
        new_ids = model.generate(
            **prepared, do_sample=False, max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = new_ids[0, prepared["input_ids"].shape[1]:]
    weight_file = args.adapter / "adapter_model.safetensors"
    digest = hashlib.sha256()
    with weight_file.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    print(json.dumps({
        "result": "PEFT_LOADED_AND_GENERATED",
        "note": "This is a loadability smoke test, NOT a quality benchmark",
        "adapter_config_declared_base": configuration.base_model_name_or_path,
        "loaded_base": args.base,
        "adapter_sha256": digest.hexdigest(),
        "device": args.device,
        "finite_logits": True,
        "logits_shape": list(logits.shape),
        "new_tokens": int(generated.numel()),
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "elapsed_seconds": round(time.perf_counter()-start, 3),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
