"""Verify Lerp LoRA linear merge on Qwen2.5-0.5B.

1) For every module: child B'@A' (scale = alpha/r = 1) vs  sum_i c_i * (alpha_i/r_i) * B_i @ A_i
2) PEFT-load the child adapter and compare logits with a dense base where the same
   weighted deltas are added directly into the base weights.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

base_dir = Path(sys.argv[1])
child_dir = Path(sys.argv[2])
parent_dirs = [Path(sys.argv[3]), Path(sys.argv[4])]
coeffs = [float(sys.argv[5]), float(sys.argv[6])]  # per-parent weights from genome
print("coefficients:", coeffs, "sum =", sum(coeffs))


def load_lora(adapter: Path):
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    tensors = {}
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).float()
    # map module prefix -> (A, B)
    mods = {}
    for k, v in tensors.items():
        if ".lora_A." in k:
            mods.setdefault(k.split(".lora_A.")[0], {})["A"] = v
        elif ".lora_B." in k:
            mods.setdefault(k.split(".lora_B.")[0], {})["B"] = v
    return mods, scale


child_mods, child_scale = load_lora(child_dir)
print("child modules:", len(child_mods), "child scale (alpha/r):", child_scale)
ccfg = json.loads((child_dir / "adapter_config.json").read_text(encoding="utf-8"))
print("child r =", ccfg["r"], "lora_alpha =", ccfg["lora_alpha"])

parents = [load_lora(p) for p in parent_dirs]
worst_rel = 0.0
for mod, cd in child_mods.items():
    delta_child = cd["B"] @ cd["A"] * child_scale
    expected = 0.0
    for c, (pm, ps) in zip(coeffs, parents):
        expected = expected + c * ps * (pm[mod]["B"] @ pm[mod]["A"])
    rel = ((delta_child - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    worst_rel = max(worst_rel, rel)
print(f"[1] delta check over {len(child_mods)} modules: worst relative error = {worst_rel:.3e} (bf16 output cast)")

# ---- 2) runtime equivalence ----
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(base_dir))
ids = tok("The capital of France is Paris. Python is a programming language.", return_tensors="pt")["input_ids"]

base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cpu")
base.eval()
dense = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cpu")
dense.eval()
named = dict(dense.named_modules())
with torch.no_grad():
    for mod, _ in child_mods.items():
        # mod like base_model.model.model.layers.0.self_attn.q_proj -> strip PEFT prefix
        target = mod.replace("base_model.model.", "", 1)
        lin = named[target]
        total = torch.zeros_like(lin.weight)
        for c, (pm, ps) in zip(coeffs, parents):
            total += c * ps * (pm[mod]["B"] @ pm[mod]["A"])
        lin.weight.add_(total)

peft_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cpu"), str(child_dir))
peft_model.eval()
with torch.no_grad():
    lp = peft_model(ids).logits
    ld = dense(ids).logits
    lb = base(ids).logits
print("[2] PEFT child loaded; logits finite:", bool(torch.isfinite(lp).all()))
print("    max|PEFT child - dense merged| =", (lp - ld).abs().max().item(), " (scale of logits ~", lp.abs().max().item(), ")")
print("    max|PEFT child - base|         =", (lp - lb).abs().max().item(), "(non-zero => adapter actually changes the model)")

with torch.no_grad():
    enc = tok("Q: What is 2 + 3?\nA:", return_tensors="pt")
    out = peft_model.generate(**enc, max_new_tokens=24, do_sample=False)
print("[3] generation:", repr(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
