import json, sys
import torch
from safetensors import safe_open

child, a_dir, b_dir, alpha = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
wm = json.load(open(child + "/model.safetensors.index.json"))["weight_map"]
fa = safe_open(a_dir + "/model.safetensors", "pt"); fb = safe_open(b_dir + "/model.safetensors", "pt")
print("child tensors:", len(wm), "| parent tensors:", len(list(fa.keys())), "| same key set:", set(wm) == set(fa.keys()))
handles = {}
def child_slice(name):
    sh = wm[name]
    if sh not in handles: handles[sh] = safe_open(child + "/" + sh, "pt")
    return handles[sh].get_slice(name)
probe = [
    "model.language_model.embed_tokens_per_layer.weight",    # 2.8B elements: chunked path
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.41.self_attn.q_proj.weight",
    "model.language_model.layers.20.post_attention_layernorm.weight",
    "model.language_model.norm.weight",
    "model.audio_tower.layers.5.self_attn.q_proj.linear.weight" ,
    "model.vision_tower.encoder.layers.3.mlp.down_proj.linear.weight",
]
keys = set(wm)
probe = [k for k in probe if k in keys] + [k for k in sorted(keys) if "audio_tower" in k and k.endswith("weight")][:1] + [k for k in sorted(keys) if "vision_tower" in k and k.endswith("weight")][:1]
for name in dict.fromkeys(probe):
    cs, sa, sb = child_slice(name), fa.get_slice(name), fb.get_slice(name)
    rows = cs.get_shape()[0]
    idx = sorted({0, rows // 3, rows // 2, rows - 1}) if len(cs.get_shape()) > 1 else [None]
    worst = 0.0
    for r in idx:
        sl = (slice(r, r + 1) if r is not None else slice(None))
        c = cs[sl].float(); e = alpha * sa[sl].float() + (1 - alpha) * sb[sl].float()
        worst = max(worst, ((c - e).norm() / e.norm().clamp_min(1e-9)).item())
    print(f"{name[:70]:70s} dtype={cs.get_dtype()} shape={tuple(cs.get_shape())} rel_err={worst:.2e}")
