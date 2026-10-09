"""Real-model check of generative scoring: GSM8K exact match of two parents and a blend through the resident session,
plus an independent one-prompt-at-a-time decode of the same prompts to confirm the batched path agrees.

    python experiments/gen_check.py --device xpu --items 30
"""
import argparse
import time

from lerp.resident import ResidentSession
from lerp.spec import parse_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--parent", action="append", help="name=path, twice; LoRA adapters or full checkpoints")
    ap.add_argument("--mode", default="lora", choices=["lora", "full"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    task = {"dataset": "openai/gsm8k", "config": "main", "split": "test", "prompt": "Question: {question}\nAnswer:",
            "answer": {"field": "answer", "regex": "#### (-?[0-9.,]+)"},
            "generate": {"max_new_tokens": args.tokens, "stop": ["Question:", "</s>", "<|im_end|>", "<|endoftext|>"]},
            "extract": {"regex": "(-?[0-9][0-9,]*\\.?[0-9]*)", "pick": "last"},
            "normalize": ["remove_commas", "strip_period", "number"]}
    spec = parse_spec({"name": "gen", "base_model": args.base, "genes": 3, "out_dtype": "bfloat16",
                       "gene_groups": ["attention", "mlp", "other"], "population": 3,
                       "mode": args.mode, "parents": [{"name": n, "model": m} for n, m in (x.split("=", 1) for x in args.parent)],
                       "evaluation": {"limit": args.items, "tasks": {"gsm8k": {"metric": "exact_match,none", "task": task}}}})
    session = ResidentSession(spec, args.device, args.dtype)
    window = (0, args.items)
    for name in [p.name for p in spec.parents]:
        t0 = time.time()
        print(name, session.evaluate_reference(name, window), f"{time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    print("blend 0.5", session.evaluate([0.5] * 9, window), f"{time.time() - t0:.0f}s", flush=True)

    # independent decode of the first prompts, one at a time, against the batched completions
    import torch
    gen = session._generative(window)
    first = gen.order[:4]
    batched = session._generate_batch(gen, first)
    same = 0
    for i, text in zip(first, batched):
        ids = torch.tensor([gen.items[i][2]], device=session.device)
        out = session.model.generate(input_ids=ids, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     max_new_tokens=args.tokens, pad_token_id=gen.pad_id())
        single = session.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        task_def = gen.tasks[gen.items[i][0]]
        same += int(task_def.extract_answer(single) == task_def.extract_answer(text))
    print(f"batched vs single extracted answers equal: {same}/{len(first)}")
    session.close()


if __name__ == "__main__":
    main()
