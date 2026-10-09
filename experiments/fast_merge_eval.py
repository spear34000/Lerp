"""Merge-and-score a LoRA pair in one resident process: seconds per candidate instead of minutes.

The base model stays on the accelerator. A candidate is  W = W0 + sum_i c_i * scale_i * B_i @ A_i  written in place into the
target weights (identical coefficients to Lerp's `build_lora`, via `tensor_coefficients`), then the items are scored
with the same prompts, continuation tokenization and metrics as lm-eval (loglikelihood; acc_norm = argmax ll / len(choice)).
Only the hidden states at continuation positions go through the LM head, which keeps memory small.

    python fast_merge_eval.py --config CFG.yaml --verify 0.5,0.5,0.5 0.25,0.25,0.25
    python fast_merge_eval.py --config CFG.yaml --budget 12 --batch 1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from lerp.gp import features, genes_from_features, propose_batch, _sample_features
from lerp.spec import parse_spec
from lerp.weighting import language_layer_count, tensor_coefficients

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--start", type=int, default=0, help="first item index (use >= limit for fresh items)")
ap.add_argument("--device", default="xpu")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--verify", nargs="*", help="group weights 'attn,mlp,other' of parent 1 to score, then exit")
ap.add_argument("--budget", type=int, default=0, help="GP evaluations (including --init random-design points)")
ap.add_argument("--init", type=int, default=3)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--strategy", choices=["gp", "random"], default="gp")
ap.add_argument("--validate-limit", type=int, default=0, help="re-score the top picks and parents on this many FRESH items after the search")
ap.add_argument("--top", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

raw = yaml.safe_load(open(args.config, encoding="utf-8"))
raw["search"] = "gp"
spec = parse_spec(raw)
penalty = float(raw["evaluation"].get("penalty", 0.0))
dev = torch.device(args.device)
dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

# ---------------------------------------------------------------- data (lm-eval prompt formats)
tok = AutoTokenizer.from_pretrained(spec.base_model)


def arc_docs(lo, hi):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    for row in list(ds)[lo:hi]:
        yield f"Question: {row['question']}\nAnswer:", list(row["choices"]["text"]), row["choices"]["label"].index(row["answerKey"])


def boolq_docs(lo, hi):
    ds = load_dataset("aps/super_glue", "boolq", split="validation")
    for row in list(ds)[lo:hi]:
        yield f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:", ["no", "yes"], int(row["label"])


def _hs_pre(text):
    import re
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def hellaswag_docs(lo, hi):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    for row in list(ds)[lo:hi]:
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        yield _hs_pre(row["activity_label"] + ": " + ctx), [_hs_pre(e) for e in row["endings"]], int(row["label"])


DOCS = {"arc_easy": arc_docs, "boolq": boolq_docs, "hellaswag": hellaswag_docs}
METRIC = {name: opts["metric"].split(",")[0] for name, opts in raw["evaluation"]["tasks"].items()}  # acc or acc_norm

class Items:
    def __init__(self, lo, hi):
        self.requests, self.gold, self.char_len = [], {}, {}  # request = (task, doc, choice, context ids, continuation ids)
        for task in METRIC:
            for d, (ctx, choices, label) in enumerate(DOCS[task](lo, hi)):
                self.gold[(task, d)] = label
                for c, choice in enumerate(choices):
                    whole = tok(ctx + " " + choice, add_special_tokens=False)["input_ids"]
                    cenc = tok(ctx, add_special_tokens=False)["input_ids"]
                    self.requests.append((task, d, c, whole[:len(cenc)], whole[len(cenc):]))
                    self.char_len[(task, d, c)] = float(len(choice))
        self.order = sorted(range(len(self.requests)), key=lambda i: len(self.requests[i][3]) + len(self.requests[i][4]))
        self.by_doc = {}
        for i, r in enumerate(self.requests):
            self.by_doc.setdefault((r[0], r[1]), []).append(i)
        print(f"items [{lo},{hi}): {len(self.requests)} requests, {sum(len(r[3]) + len(r[4]) for r in self.requests)} tokens", flush=True)


search_items = Items(args.start, args.start + args.limit)

# ---------------------------------------------------------------- model + adapters
model = AutoModelForCausalLM.from_pretrained(spec.base_model, dtype=dtype).to(dev).eval()
config = json.loads((Path(spec.base_model) / "config.json").read_text(encoding="utf-8"))
n_layers = language_layer_count(config)
modules = {}  # module path -> dict(name, W0 fp32, Bs[list], As[list], scales[list])
for parent in spec.parents:
    adapter = Path(parent.model)
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if ".lora_A." not in key:
                continue
            prefix = key.split(".lora_A.")[0]
            entry = modules.setdefault(prefix, {"A": [], "B": [], "scale": []})
            entry["A"].append(f.get_tensor(key).to(dev, torch.float32))
            entry["B"].append(f.get_tensor(key.replace(".lora_A.", ".lora_B.")).to(dev, torch.float32))
            entry["scale"].append(scale)
for prefix, entry in modules.items():
    layer = model.get_submodule(prefix.replace("base_model.model.", "", 1))
    entry["layer"] = layer
    entry["W0"] = layer.weight.detach().cpu().clone()  # original (bf16) weights live in host RAM; the GPU holds only the working copy
    entry["A_cat"] = torch.cat(entry["A"], 0)
    entry["ranks"] = [a.shape[0] for a in entry["A"]]
print(f"{len(modules)} adapter modules, device memory {torch.xpu.memory_allocated() / 2**30:.1f} GiB" if dev.type == "xpu" else f"{len(modules)} modules", flush=True)


@torch.no_grad()
def apply(genes):
    for prefix, e in modules.items():
        coeffs = tensor_coefficients(spec, genes, prefix, n_layers)
        cols = torch.cat([torch.full((r,), c * s, device=dev) for r, c, s in zip(e["ranks"], coeffs, e["scale"])])
        delta = (torch.cat(e["B"], 1) * cols) @ e["A_cat"]
        e["layer"].weight.copy_((e["W0"].to(dev).float() + delta).to(e["layer"].weight.dtype))


@torch.no_grad()
def score(items=None, max_tokens=9000):
    items = items or search_items
    requests, order = items.requests, items.order
    ll = {}
    pos = 0
    while pos < len(order):
        width = len(requests[order[pos]][3]) + len(requests[order[pos]][4])
        batch = []
        while pos < len(order):
            r = requests[order[pos]]
            w = len(r[3]) + len(r[4])
            if batch and (len(batch) + 1) * max(w, width) > max_tokens:
                break
            batch.append(order[pos]); width = max(width, w); pos += 1
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, i in enumerate(batch):
            seq = requests[i][3] + requests[i][4]
            ids[row, :len(seq)] = torch.tensor(seq); mask[row, :len(seq)] = 1
        hidden = model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).last_hidden_state
        sel, targets, owner = [], [], []
        for row, i in enumerate(batch):
            c0, n = len(requests[i][3]), len(requests[i][4])
            sel.append(hidden[row, c0 - 1:c0 - 1 + n]); targets.extend(requests[i][4]); owner.extend([i] * n)
        sel = torch.cat(sel)
        tgt = torch.tensor(targets, device=dev)
        lps = torch.cat([torch.log_softmax(model.lm_head(sel[s:s + 1024]).float(), -1).gather(1, tgt[s:s + 1024, None])[:, 0]
                         for s in range(0, len(sel), 1024)]).cpu().tolist()
        for i, lp in zip(owner, lps):
            ll[i] = ll.get(i, 0.0) + lp
    metrics = {}
    for task in METRIC:
        docs = [d for (tk, d) in items.gold if tk == task]
        hits = 0
        for d in docs:
            idx = items.by_doc[(task, d)]
            vals = [ll[i] / (items.char_len[(task, d, requests[i][2])] if METRIC[task] == "acc_norm" else 1.0) for i in idx]
            hits += int(max(range(len(vals)), key=vals.__getitem__) == items.gold[(task, d)])
        metrics[task] = hits / len(docs)
    return metrics


def fitness(metrics):
    v = list(metrics.values())
    return sum(v) / len(v) - penalty * (max(v) - min(v))


def evaluate(w):
    t0 = time.time()
    genes = genes_from_features(list(w), spec)
    apply(genes)
    if dev.type == "xpu":
        torch.xpu.synchronize()
    t1 = time.time()
    m = score()
    return genes, m, fitness(m), t1 - t0, time.time() - t1


if args.verify:
    t0 = time.time(); score(); print(f"warm-up scoring pass {time.time() - t0:.1f}s (kernel compilation)", flush=True)
    for item in args.verify:
        w = [float(x) for x in item.split(",")]
        _, m, fit, t_apply, t_score = evaluate(w)
        print(f"w={w} -> {m} fitness={fit:.4f}   apply {t_apply:.2f}s  score {t_score:.1f}s", flush=True)
    raise SystemExit

# ---------------------------------------------------------------- search
@torch.no_grad()
def apply_base():
    for e in modules.values():
        e["layer"].weight.copy_(e["W0"].to(dev))


history = []
rng = random.Random(args.seed)
queue = [[0.5] * 3] + _sample_features(rng, spec, max(0, args.init - 1))
t_start = time.time()
while len(history) < args.budget:
    if not queue:
        if args.strategy == "gp":
            obs = [(genes_from_features(h["w"], spec), h["fitness"]) for h in history]
            n = min(args.batch, args.budget - len(history))
            queue = [features(g, spec) for g in propose_batch(obs, spec, n, args.seed * 1000 + len(history))]
        else:
            queue = _sample_features(rng, spec, 1)
    w = queue.pop(0)
    genes, m, fit, t_apply, t_score = evaluate(w)
    history.append({"idx": len(history), "w": w, "metrics": m, "fitness": fit, "seconds": round(t_apply + t_score, 1)})
    print(f"[{len(history)}/{args.budget}] w={[round(x, 2) for x in w]} {m} fitness={fit:.4f} best={max(h['fitness'] for h in history):.4f} "
          f"({t_apply + t_score:.1f}s, total {time.time() - t_start:.0f}s)", flush=True)
ranked = sorted(history, key=lambda h: -h["fitness"])
print("TOP", [(round(h["fitness"], 4), [round(x, 2) for x in h["w"]]) for h in ranked[:args.top]], flush=True)

report = {"strategy": args.strategy, "budget": args.budget, "limit": args.limit, "seconds": round(time.time() - t_start),
          "history": history, "validation": None}
if args.validate_limit:
    fresh = Items(args.start + args.limit, args.start + args.limit + args.validate_limit)
    rows = {}
    apply_base(); rows["base"] = score(fresh)
    for name, w in (("parent1_only", [1.0] * 3), ("parent2_only", [0.0] * 3)):
        apply(genes_from_features(w, spec)); rows[name] = score(fresh)
    for k, h in enumerate(ranked[:args.top]):
        apply(genes_from_features(h["w"], spec)); rows[f"top{k + 1}"] = score(fresh); rows[f"top{k + 1}_w"] = [round(x, 2) for x in h["w"]]
    for name, m in rows.items():
        if name.endswith("_w"):
            continue
        print(f"fresh {name:13s} {m} fitness={fitness(m):.4f}", (rows.get(name + "_w") or ""), flush=True)
    report["validation"] = {k: (v if k.endswith("_w") else {"metrics": v, "fitness": fitness(v)}) for k, v in rows.items()}
if args.out:
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
print("BEST", json.dumps(ranked[0]), flush=True)
