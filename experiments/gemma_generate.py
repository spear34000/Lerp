import gc, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
prompts = ["The capital of France is", "Q: What is 12 + 30?\nA:"]
for label, path in [a.split("=", 1) for a in sys.argv[1:]]:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    print(f"[{label}] loaded {type(model).__name__} in {time.time()-t0:.0f}s", flush=True)
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            out = model.generate(**enc, max_new_tokens=16, do_sample=False)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[{label}] bos_first={enc['input_ids'][0][0].item()==2} finite={bool(torch.isfinite(logits).all())} prompt={p!r} -> {text!r}", flush=True)
    del model; gc.collect()
