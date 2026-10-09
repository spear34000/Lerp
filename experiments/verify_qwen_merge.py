"""Verify a Lerp full-checkpoint linear merge of Qwen2.5 parents.

Checks (1) tensor-level arithmetic against the recorded genome and (2) that the
child loads with transformers and produces finite logits and text.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

child = Path(sys.argv[1])
parent_a = Path(sys.argv[2])  # weight on genome gene (parent 1, e.g. instruct)
parent_b = Path(sys.argv[3])  # weight 1 - gene (parent 2, e.g. coder)
alpha = float(sys.argv[4])


def weight_map(root: Path) -> dict:
    """Tensor name -> shard file. Single-file checkpoints have no index."""
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
    return None


def tensor_from(root: Path, mapping, name: str):
    shard = mapping[name] if mapping else "model.safetensors"
    with safe_open(str(root / shard), framework="pt") as f:
        return f.get_tensor(name)


child_map = weight_map(child)
a_map = weight_map(parent_a)
b_map = weight_map(parent_b)

print("== 1) tensor arithmetic (fp32 expected vs child)")
probe = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.12.self_attn.o_proj.weight",
    "model.layers.23.mlp.up_proj.weight",
    "model.layers.23.input_layernorm.weight",
]
worst = 0.0
for name in probe:
    c = tensor_from(child, child_map, name).float()
    pa = tensor_from(parent_a, a_map, name).float()
    pb = tensor_from(parent_b, b_map, name).float()
    expected = alpha * pa + (1 - alpha) * pb
    diff = (c - expected).abs().max().item()
    worst = max(worst, diff)
    print(f"  {name:55s} max|child-expected|={diff:.3e}  shape={tuple(c.shape)}")
print(f"  worst deviation (bf16 rounding expected ~1e-2 relative): {worst:.3e}")

print("== 2) transformers load + generation")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(child))
model = AutoModelForCausalLM.from_pretrained(str(child), dtype=torch.float32, device_map="cpu")
model.eval()
prompts = ["The capital of France is", "Write a Python function that adds two numbers:\n"]
for p in prompts:
    enc = tok(p, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        finite = bool(torch.isfinite(logits).all())
        out = model.generate(**enc, max_new_tokens=32, do_sample=False)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  prompt={p!r}\n  finite_logits={finite}\n  completion={text!r}")
