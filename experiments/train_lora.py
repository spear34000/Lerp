"""Train a small PEFT LoRA on Qwen2.5-0.5B for one benchmark's TRAIN split.

Skills:
  arc       -> allenai/ai2_arc ARC-Easy train      (evaluated on arc_easy test)
  hellaswag -> Rowan/hellaswag train                (evaluated on hellaswag validation)
  piqa      -> baber/piqa train                     (evaluated on piqa validation)
  boolq     -> aps/super_glue boolq train           (evaluated on boolq validation)

Prompt formats mirror lm-eval's so loglikelihood scoring sees the same layout.
Loss is applied to the answer tokens only. Train/eval splits are disjoint.
"""
import argparse
import math
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--skill", choices=["arc", "hellaswag", "piqa", "boolq"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="xpu")
ap.add_argument("--debug", action="store_true")
ap.add_argument("--targets", default="auto", help="auto = attention + dense MLP of the language model (family rules shared with the merger), or a comma list of module names")
ap.add_argument("--attn", default="eager")
ap.add_argument("--pad-to", type=int, default=32, help="pad batch width to a multiple of this (stable XPU kernel shapes)")
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)


def hs_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def arc_examples():
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    for row in ds:
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue
        gold = row["choices"]["text"][labels.index(row["answerKey"])]
        yield f"Question: {row['question']}\nAnswer:", " " + gold


def hellaswag_examples():
    ds = load_dataset("Rowan/hellaswag", split="train")
    for row in ds:
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        query = hs_preprocess(row["activity_label"] + ": " + ctx)
        yield query, " " + hs_preprocess(row["endings"][int(row["label"])])


def piqa_examples():
    for row in load_dataset("baber/piqa", split="train"):
        yield f"Question: {row['goal']}\nAnswer:", " " +[row["sol1"], row["sol2"]][int(row["label"])]


def boolq_examples():
    for row in load_dataset("aps/super_glue", "boolq", split="train"):
        yield f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:", " " + ["no", "yes"][int(row["label"])]


examples = list({"arc": arc_examples, "hellaswag": hellaswag_examples, "piqa": piqa_examples,
                 "boolq": boolq_examples}[args.skill]())
# Truncation would cut off the answer tokens, so drop examples that do not fit.
_tok = AutoTokenizer.from_pretrained(args.base)
_before = len(examples)
examples = [(p, c) for p, c in examples if len(_tok(p + c, add_special_tokens=False)["input_ids"]) <= args.max_len]
if len(examples) != _before:
    print(f"dropped {_before - len(examples)} examples longer than {args.max_len} tokens", flush=True)
random.shuffle(examples)
print(f"{args.skill}: {len(examples)} training examples", flush=True)

tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation=args.attn)
if args.targets == "auto":
    from lerp.targets import discover
    targets = discover(model)
    print(f"auto LoRA targets: {len(targets)} Linear layers (e.g. {targets[0]})", flush=True)
else:
    targets = args.targets.split(",")
cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=targets)
model = get_peft_model(model, cfg)
model.to(args.device)
params = [p for p in model.parameters() if p.requires_grad]
print("trainable params:", sum(p.numel() for p in params), "dtype:", params[0].dtype, flush=True)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, foreach=False)


def lr_at(step: int) -> float:
    warm = 20
    if step < warm:
        return args.lr * (step + 1) / warm
    return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))


def make_batch(items):
    ids, labels = [], []
    for prompt, cont in items:
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        c = tok(prompt + cont, add_special_tokens=False)["input_ids"][len(p):]
        seq = (p + c)[: args.max_len]
        lab = ([-100] * len(p) + c)[: args.max_len]
        ids.append(seq)
        labels.append(lab)
    width = max(map(len, ids))
    width = min(args.max_len, -(-width // args.pad_to) * args.pad_to)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    x = torch.tensor([s + [pad] * (width - len(s)) for s in ids])
    y = torch.tensor([l + [-100] * (width - len(l)) for l in labels])
    mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in ids])
    return x, y, mask


model.train()
t0 = time.time()
cursor = 0
running = 0.0
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    for _ in range(args.accum):
        batch = [examples[(cursor + i) % len(examples)] for i in range(args.batch)]
        cursor += args.batch
        x, y, mask = make_batch(batch)
        x, y, mask = x.to(args.device), y.to(args.device), mask.to(args.device)
        loss = model(input_ids=x, attention_mask=mask, labels=y).loss / args.accum
        loss.backward()
        running += loss.item()
    # torch.nn.utils.clip_grad_norm_ kills the Intel Arc XPU device (DEVICE_LOST), so clip manually.
    total = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
    scale = torch.clamp(1.0 / (total + 1e-6), max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if args.debug:
        torch.xpu.synchronize()
        print(f"  dbg step {step} shape {tuple(x.shape)} mem {torch.xpu.memory_allocated() // 2**20}MiB "
              f"peak {torch.xpu.max_memory_allocated() // 2**20}MiB", flush=True)
    if (step + 1) % 20 == 0:
        print(f"step {step + 1}/{args.steps} loss {running / 20:.4f} elapsed {time.time() - t0:.0f}s", flush=True)
        running = 0.0

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(out))
print("saved", out, f"total {time.time() - t0:.0f}s", flush=True)
