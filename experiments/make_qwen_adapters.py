"""Create two PEFT LoRA adapters on Qwen2.5-0.5B for merge-mechanics testing.

Adapters are NOT trained: lora_B is filled with seeded small random values so the
delta is non-zero (PEFT initializes lora_B to zero). Results therefore test merge
mechanics and PEFT loading, not model quality.
"""
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

for name, seed, rank in [("adapter_a", 1, 8), ("adapter_b", 2, 8)]:
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cpu")
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=TARGETS, lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.01)
    model.save_pretrained(str(out_dir / name))
    print("saved", name, "trainable:", sum(p.numel() for n, p in model.named_parameters() if "lora_" in n))
