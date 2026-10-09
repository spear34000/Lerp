# Toward a model-family-independent train / merge / search ecosystem

Goal: any Hugging Face causal LM (dense, multimodal, mixture-of-experts, any size) can be trained with LoRA, merged, searched and evaluated by the same
pipeline, and supporting a *new* family means writing configuration, not code. This file records what is general today, what is verified, and what is still missing.

## Layers and how family-specific each one is

| Layer | Mechanism | Family-specific input | Status |
|---|---|---|---|
| Compatibility check | config keys (nested `text_config` flattened), tensor-name/shape set, tokenizer core hash | none | verified: Qwen2.5/3, Gemma 4 (E4B, 26B-A4B config), OLMoE |
| Module families | `weighting.DEFAULT_RULES` (attention / mlp / router / norm / embedding / other) | `tensor_rules: [{match: regex, group: ...}]` in the experiment YAML | dense, multimodal and MoE names covered by tests |
| Depth profile | layer index from `layers.N.`, count from `text_config.num_hidden_layers` | non-text towers (`audio`, `vision`, ...) excluded from depth | verified on Gemma 4 E4B |
| Full-checkpoint merge | streaming per tensor, row-chunked above 128M elements | none | Gemma 4 E4B (16 GB, 5 min), OLMoE-1B-7B (12.9 GB, 55 s) |
| LoRA merge | exact rank concatenation | PEFT adapters with the standard linear layout | Qwen2.5-0.5B, Qwen3-4B |
| LoRA training targets | `targets.discover`: attention + dense MLP of the language model; skips experts, routers, towers, head | same rules as the merger | probed on 6 architectures without weights |
| Search | evolution or GP + expected improvement over module-group weights | none | evolution = 24 evals, GP = 8 evals for the same quality |
| Evaluation | lm-eval through `lerp`, or the resident loop (`examples/fast_merge_eval.py`) | task format function per benchmark | 3 tasks in the resident loop |

## Adding a new family

1. `lerp check -c config.yaml` - fix reported config/tokenizer/tensor mismatches (soft warnings are fine).
2. Print the tensor names and see which group each lands in (`python -c "from lerp.weighting import tensor_group; ..."`). If a family uses unusual names, add
   `tensor_rules` to the YAML, for example `- {match: '\.ffn\.', group: mlp}`; no source change.
3. Choose `gene_groups` (add `router` for mixture-of-experts so routers get their own weight).
4. Train adapters with `qwen_merge_test/train_lora.py --targets auto` (the same family rules pick the Linear layers).

## Known gaps (honest list)

- Fused expert parameters (Gemma 4 26B-A4B stores experts as one 3-D parameter, not `nn.Linear`) are merged but cannot receive LoRA through PEFT's module targeting.
- Merged MoE checkpoints above ~14 GB cannot be loaded for evaluation on a 16 GB GPU / 31 GB RAM machine without quantization (a GGUF + llama.cpp backend is the missing piece).
- Router interpolation has no quality guarantee; give it its own gene group and let the search decide.
- The resident evaluation loop supports two LoRA parents and three tasks; a declarative task definition (dataset, split, prompt template, choices, label) would remove that limit.
- Quantized sources (GPTQ/AWQ/GGUF) must be dequantized first; there is no direct support.
- Training beyond LoRA (full fine-tuning, DPO, MoE expert adapters) is not wired in.
