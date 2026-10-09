<div align="center">

# Lerp

**Find the right blend of language models — seconds per candidate, every claim auditable.**

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20alpha-orange)

[English](README.md) · [한국어](README.ko.md) · [Reference manual](docs/REFERENCE.md) · [Measured results](experiments/RESULTS.md) · [Ecosystem map](ECOSYSTEM.md)

</div>

Lerp blends fine-tuned models that share a base — **LoRA adapters or full checkpoints, dense, multimodal or mixture-of-experts** — and searches for the blend weights
per module group (attention, MLP, router, ...). `W = W0 + w · Δ` is a linear interpolation, hence the name. It does **not** train new models from scratch,
and a successful merge does not imply a better model: Lerp's job is to find good blends cheaply and to prove (or disprove) that they are better.

## Why Lerp

- **Seconds per candidate.** A resident loop keeps the base model on the accelerator, writes the blended weights in place and scores the items itself:
  5 s per candidate for a 0.5B model, 29 s for Qwen3-4B (the same step took ~285 s and ~11-14 min through a fresh `lm-eval` run per candidate).
- **Sample-efficient search.** Gaussian-process Bayesian search (`search: gp`) reached the same blend quality as evolution with a third of the evaluations; evolution and Pareto selection remain available when you want specialists.
- **One pipeline across model families.** A single rule table maps tensors to module groups; mixture-of-experts routers and experts, multimodal towers and unfamiliar architectures are handled by configuration (`tensor_rules`), not code.
- **Auditable.** Inputs and outputs are SHA-256 pinned, winners are re-scored on items the search never saw, simulated scores cannot leak into real results, and the docs list what is *not* verified.

## How it works

```mermaid
flowchart LR
    A["Parents<br/>LoRA adapters or full checkpoints"] --> B["check<br/>family rules, tokenizer, tensor names"]
    B --> C["freeze<br/>SHA-256 pin of every input"]
    C --> D{"search"}
    D -->|"evolution / GP"| E["merge candidate<br/>lite / lora / mergekit"]
    D -->|"resident loop"| F["blend weights in place<br/>score in seconds"]
    E --> G["lm-eval"]
    G --> H["winner"]
    F --> H
    H --> I["validate<br/>held-out tasks and fresh items"]
```

## Quick start

```bash
pip install "lerp[lora,gp,eval] @ git+https://github.com/spear34000/Lerp"
lerp --version

# No-download demo with fake scores (never reported as real results)
lerp init -c examples/multi_parent_demo.yaml -o runs/toy
lerp simulate -r runs/toy && lerp advance -r runs/toy --allow-simulated && lerp report -r runs/toy
```

Two LoRA adapters trained on the same base, searched with GP and validated:

```bash
cp examples/gp_demo.yaml my.yaml            # point base_model and parents at your adapters
lerp check  -c my.yaml                      # family rules, tokenizer, tensor names
lerp init   -c my.yaml -o runs/my && lerp freeze -r runs/my --strict
lerp cycle  -r runs/my --rounds 3 --engine lora      # search: gp  ->  3 + 3 + 3 evaluations
lerp validate -r runs/my -c examples/holdout_evaluation.yaml --baseline all

# Or the resident loop: seconds per candidate, 300 items per task, fresh-item check at the end
python examples/fast_merge_eval.py --config my.yaml --limit 300 --budget 30 --strategy gp --validate-limit 300
```

Windows: set `PYTHONUTF8=1`. More in the [reference manual](docs/REFERENCE.md).

## What has been measured

Small models on one 16 GB Intel Arc machine; sample sizes are 100-300 items per task, so differences below ~0.05 are noise. Full tables, seeds and caveats are in [`experiments/RESULTS.md`](experiments/RESULTS.md).

| Question | Result |
|---|---|
| How fast is the resident loop? | Qwen2.5-0.5B: 285 s → 5 s per candidate. Qwen3-4B: 11-14 min → 29 s (47 s with 150 items per task). Scores match `lm-eval` within one item per 100 |
| Does merging help? | **Only when the skills are complementary.** An ARC LoRA + a BoolQ LoRA (Qwen2.5-0.5B) blend to 0.78 on fresh items against 0.71 / 0.69 for the parents (+0.07, about 3 standard errors). Weak or redundant pairs gave no detectable gain over the better parent |
| GP vs evolution? | 8 GP evaluations found blends as good as 24 evolution evaluations (fresh-item fitness 0.739 vs 0.741). Evolution did **not** beat plain random search |
| GP vs random? | Indistinguishable when the weight landscape is a wide plateau (30 evaluations, 300 items). The speed-up comes from the evaluation loop, not from a smarter optimizer |
| Large and unusual checkpoints | Gemma 4 E4B (multimodal, 16 GB): merged in 5 min, arithmetic verified, generates. OLMoE-1B-7B (64 experts per layer): 3,219 tensors, 12.9 GiB in 55 s, each module group matches its weight to 2e-3, scored between its parents |
| Different families (Qwen3 + Gemma 4) | Cannot be weight-merged; `check` refuses with architecture, vocabulary, tokenizer and tensor-name mismatches |

## Models verified so far

| Family | Kind | What was run |
|---|---|---|
| Qwen2.5-0.5B | dense | LoRA training and merging, full-checkpoint merging, evolution, GP |
| Qwen3-4B (Base, post-trained) | dense | LoRA training and merging, full-checkpoint merging, GP, resident loop |
| Gemma 4 E4B (pt, it) | multimodal | full-checkpoint merge and generation |
| OLMoE-1B-7B (base, Instruct) | mixture-of-experts | full-checkpoint merge, generation, evaluation |
| Gemma-4-26B-A4B, Qwen3-30B-A3B | MoE | LoRA targets and tensor groups probed from the config only; **not merged** |

## Limits worth knowing

- Ancestry cannot be proven from files: matching configs and shapes do not show that two checkpoints share a base revision.
- Merged checkpoints larger than ~14 GB cannot be evaluated on a 16 GB GPU without quantization (no GGUF backend yet). Merging itself streams tensor by tensor.
- The resident loop supports two LoRA parents and the tasks `arc_easy`, `boolq` and `hellaswag`.
- No MergeKit (TIES/DARE) run has been exercised here; quantized sources (GPTQ/AWQ/GGUF) are not supported directly.
- Read the [critical review](docs/CRITICAL_REVIEW_KO.md) and the [technical audit](docs/V04_TECHNICAL_AUDIT.md) before quoting any result.

## Repository

```
lerp/          package: spec, family rules, lite / lora engines, GP search, integrity, CLI
examples/      demo configs, PEFT smoke tests, the resident fast_merge_eval loop
experiments/   training, verification and search scripts + RESULTS.md with every table
docs/          reference manual, architecture, audits
tests/         124 tests (numerical merge checks, crash recovery, GP, model families)
```

## Roadmap

1. Declarative evaluation tasks (dataset, prompt template, choices in YAML) to lift the three-task limit.
2. The resident loop for full checkpoints (frozen base on the accelerator, parent difference in host RAM).
3. A GGUF / llama.cpp evaluation backend so large MoE merges can be scored locally.
4. Expert-level adapters for fused-expert MoE models.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Lerp grew out of ModelBreeder v0.4 (MIT); its notice is kept in [NOTICE](NOTICE). The old `modelbreeder` command still works as an alias.
