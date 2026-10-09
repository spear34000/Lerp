"""Merge two MoE checkpoints with a different weight per module group and check every group against the arithmetic.

genes are flat over depth, so the expected tensor is simply  w_group * parent1 + (1 - w_group) * parent2.
"""
import collections
import json
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from lerp.lite import build_lite
from lerp.spec import parse_spec
from lerp.weighting import tensor_group

config, out = sys.argv[1], Path(sys.argv[2])
weights = {"attention": 0.9, "mlp": 0.1, "router": 0.6, "other": 0.3}
spec = parse_spec(yaml.safe_load(open(config, encoding="utf-8")))
genes = []
for group in spec.gene_groups:
    genes += [weights[group]] * spec.block_size
t0 = time.time()
info = build_lite(spec, genes, out)
print(f"built {info['tensors']} tensors, {info['weight_bytes'] / 2**30:.1f} GiB in {time.time() - t0:.0f}s", flush=True)

index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]


def parent_index(root):
    root = Path(root)
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    return {k: root / v for k, v in idx.items()}


p1, p2 = (parent_index(p.model) for p in spec.parents)
groups = collections.defaultdict(list)
for name in index:
    groups[tensor_group(name, spec.tensor_rules) if tensor_group(name, spec.tensor_rules) in weights else "other"].append(name)
print({g: len(v) for g, v in groups.items()}, flush=True)
rng = random.Random(0)
worst = {}
for group, names in groups.items():
    sample = rng.sample(names, min(40, len(names)))
    for name in sample:
        with safe_open(str(out / index[name]), "pt") as f: c = f.get_tensor(name).float()
        with safe_open(str(p1[name]), "pt") as f: a = f.get_tensor(name).float()
        with safe_open(str(p2[name]), "pt") as f: b = f.get_tensor(name).float()
        w = weights[group]
        e = w * a + (1 - w) * b
        denom = e.norm().clamp_min(1e-9)
        worst[group] = max(worst.get(group, 0.0), ((c - e).norm() / denom).item())
for group, err in worst.items():
    print(f"  group {group:10s} w_parent1={weights[group]}  tensors checked {min(40, len(groups[group])):3d}  worst relative error {err:.2e}")
routers = [n for n in groups["router"]][:3]
print("router tensors:", routers)
print("experts sample:", [n for n in groups["mlp"] if ".experts." in n][:2])
