"""Real GP + expected-improvement search over 3 module-group merge weights, using Lerp's LoRA builder.

Search space: weight of parent 1 (ARC) for [attention, mlp, other], each repeated over the 3 depth control points
(9 genes -> 3 parameters). Each evaluation = build merged adapter + lm-eval on the same first-N items as the
Lerp runs, scored with the same fitness (mean - penalty * gap).
"""
import argparse
import glob
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from lerp.lora import build_lora
from lerp.spec import parse_spec

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--lm-eval", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=8)
ap.add_argument("--init", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--device", default="xpu")
args = ap.parse_args()

raw = yaml.safe_load(open(args.config, encoding="utf-8"))
spec = parse_spec(raw)
tasks = {name: opts["metric"] for name, opts in raw["evaluation"]["tasks"].items()}
penalty = float(raw["evaluation"].get("penalty", 0.0))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log_path = out / "bo_log.json"
history = []


def evaluate(w):
    genes = [float(w[0])] * 3 + [float(w[1])] * 3 + [float(w[2])] * 3
    idx = len(history)
    model_dir = out / f"cand-{idx:02d}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    t0 = time.time()
    build_lora(spec, genes, model_dir / "model")
    res_dir = model_dir / "eval"
    cmd = [args.lm_eval, "run", "--model", "hf", "--model_args", f"pretrained={spec.base_model},peft={model_dir / 'model'}",
           "--tasks", ",".join(tasks), "--device", args.device, "--batch_size", "8", "--limit", str(args.limit),
           "--num_fewshot", "0", "--output_path", str(res_dir)]
    with open(model_dir / "eval.log", "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"lm-eval failed for candidate {idx} (see {model_dir / 'eval.log'})")
    results = json.load(open(sorted(glob.glob(str(res_dir / "**" / "results_*.json"), recursive=True))[-1], encoding="utf-8"))["results"]
    metrics = {t: results[t][m] for t, m in tasks.items()}
    vals = list(metrics.values())
    fit = sum(vals) / len(vals) - penalty * (max(vals) - min(vals))
    history.append({"idx": idx, "w": [float(x) for x in w], "metrics": metrics, "fitness": fit, "seconds": round(time.time() - t0)})
    json.dump(history, open(log_path, "w", encoding="utf-8"), indent=1)
    print(f"[{idx + 1}/{args.budget}] w(attn,mlp,other)={np.round(w, 2).tolist()} metrics={ {k: round(v, 3) for k, v in metrics.items()} } fitness={fit:.4f} ({history[-1]['seconds']}s)", flush=True)
    return fit


def rbf(a, b, ls):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d / ls ** 2)


def propose(rng, ls=0.3, noise=2e-4):
    X = np.array([h["w"] for h in history]); y = np.array([h["fitness"] for h in history])
    cand = rng.uniform(0.05, 0.95, size=(4000, 3))
    mu0, scale = y.mean(), max(y.std(), 5e-3)
    K = rbf(X, X, ls) + noise / scale ** 2 * np.eye(len(X)) * scale ** 2 / scale ** 2
    Ks = rbf(cand, X, ls)
    alpha = np.linalg.solve(K, (y - mu0) / scale)
    mu = mu0 + scale * (Ks @ alpha)
    v = np.linalg.solve(K, Ks.T)
    sd = scale * np.sqrt(np.maximum(1.0 - (Ks * v.T).sum(1), 1e-9))
    z = (mu - y.max()) / sd
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    ei = (mu - y.max()) * cdf + sd * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return cand[int(np.argmax(ei))]


rng = np.random.default_rng(args.seed)
for i in range(args.init):  # space-filling start: stratified random points
    evaluate(rng.uniform(0.1, 0.9, size=3) if i else np.array([0.5, 0.5, 0.5]))
while len(history) < args.budget:
    evaluate(propose(rng))
best = max(history, key=lambda h: h["fitness"])
print("BEST", json.dumps(best), flush=True)
