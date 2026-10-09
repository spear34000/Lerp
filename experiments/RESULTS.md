# Trained-LoRA breeding on Qwen2.5-0.5B (2026-10-08)

Setup: base Qwen2.5-0.5B; two LoRAs (r=16, 600 steps) trained on the **train** splits of ARC-Easy and HellaSwag
(`train_lora.py`). Lerp `lora` engine, population 4, 2 generations, screening 30 items -> top 2 promoted at 200 items.
Evaluation: lm-eval 0.4.13, Intel Arc XPU, 0-shot, first 200 items per task (standard error ~ +-0.035 per score).

## Search tasks (acc_norm)
| model | arc_easy | hellaswag | fitness |
|---|---:|---:|---:|
| base | 0.560 | 0.520 | 0.536 |
| arc LoRA (parent) | 0.710 | 0.540 | 0.608 |
| hellaswag LoRA (parent) | 0.575 | 0.550 | 0.560 |
| **merged g000-c001** (0.5/0.5) | 0.705 | 0.570 | **0.624** (best) |
| merged g000-c000 (0.25/0.75) | 0.700 | 0.560 | 0.616 |
| merged g001-c002 (evolved) | 0.695 | 0.565 | 0.617 |
| merged g001-c000 (evolved) | 0.705 | 0.550 | 0.612 |

## Holdout (tasks not used in search): piqa acc_norm / winogrande acc
| model | piqa | winogrande | fitness |
|---|---:|---:|---:|
| base | 0.710 | 0.580 | 0.645 |
| arc LoRA | 0.710 | 0.595 | 0.653 |
| hellaswag LoRA | 0.690 | 0.555 | 0.623 |
| merged g000-c001 | 0.715 | 0.590 | 0.653 |

Caveats: 200 items only; differences below ~0.05 are within noise; piqa/winogrande independence from pretraining data is unverified;
the best candidate is a seeded uniform 0.5/0.5 blend, i.e. evolution did not improve on its starting points.

---

# Qwen3-4B-Base (2026-10-09)

Two LoRAs (r=16, 300 steps, effective batch 4) on the train splits of ARC-Easy / HellaSwag; 1 generation of 4 candidates
(screening 20 items -> top 2 promoted at 100 items). 100 items per task: standard error ~ +-0.04, so gaps below ~0.08 are noise.

| model | arc_easy | hellaswag | fitness |
|---|---:|---:|---:|
| base | 0.77 | 0.70 | 0.728 |
| arc LoRA | 0.86 | 0.66 | 0.740 |
| hellaswag LoRA | 0.83 | 0.67 | 0.734 |
| merged g000-c002 (0.75 arc / 0.25 hellaswag) | 0.87 | 0.69 | **0.762** |
| merged g000-c000 (0.25 / 0.75) | 0.89 | 0.67 | 0.758 |

Holdout (piqa acc_norm / winogrande acc): base 0.79/0.81 (0.800), arc LoRA 0.79/0.78 (0.785), hellaswag LoRA 0.82/0.77 (0.795), merged c002 0.77/0.79 (0.780).
Conclusion: the merge keeps both parents' ARC gain; no measurable gain on the holdout.

# Gemma 4 E4B (full checkpoint, `lite` engine)

Parents: google/gemma-4-E4B (pt) and google/gemma-4-E4B-it (it); candidate = 0.25 pt + 0.75 it (bf16). 2130 tensors, 16 GB, built in 5 min.
Text tensors match 0.25*pt + 0.75*it to ~1.7e-3 relative error (bf16 rounding); audio/vision towers are identical in both parents.
lm-eval directly (CPU, bf16, `add_bos_token=True`, 100 items, acc_norm):

| model | arc_easy | hellaswag |
|---|---:|---:|
| pt | 0.88 | 0.72 |
| it | 0.78 | 0.64 |
| merged | 0.84 | 0.65 |

The merge lies between its parents on both tasks (monotone interpolation). Without `<bos>` the `it` checkpoint degenerates into repeated text,
so every Gemma comparison must use the same tokenizer behaviour.
Cross-family (Qwen3-4B base + Gemma 4 parents) is rejected by `check`: architecture, layer count, vocab, tokenizer and tensor keys all differ.

---

# Qwen3-4B-Base, 3 training seeds (2026-10-09)

Seeds 0, 1, 2 change the LoRA init and data order (adapters retrained each time); everything else identical. 100 items per task (standard error ~ +-0.04).
For each seed, candidates are the uniform blends with weight 0.25 / 0.50 / 0.75 on the ARC adapter (`aggregate_seeds.py`). Base is shared (0.770 arc_easy / 0.700 hellaswag).

| seed | arc LoRA (arc/hs) | hellaswag LoRA (arc/hs) | merge 0.25 | merge 0.50 | merge 0.75 |
|---:|---|---|---|---|---|
| 0 | 0.86 / 0.66 | 0.83 / 0.67 | 0.89 / 0.67 | 0.89 / 0.68 | 0.87 / 0.69 |
| 1 | 0.89 / 0.68 | 0.80 / 0.63 | 0.89 / 0.67 | 0.90 / 0.68 | 0.89 / 0.67 |
| 2 | 0.88 / 0.67 | 0.83 / 0.64 | 0.88 / 0.66 | 0.89 / 0.65 | 0.88 / 0.65 |

Mean +- sd over seeds (arc_easy | hellaswag | mean of both):
- base: 0.770 | 0.700 | 0.735
- arc LoRA: 0.877 +- 0.015 | 0.670 +- 0.010 | 0.773
- hellaswag LoRA: 0.820 +- 0.017 | 0.647 +- 0.021 | 0.733
- merge 0.50: 0.893 +- 0.006 | 0.670 +- 0.017 | 0.782

Merge 0.50 minus best parent on arc_easy: +0.03, +0.01, +0.01. On hellaswag: +0.01, 0.00, -0.02.

Findings: the ARC adapter gains +0.09..+0.12 on arc_easy in every seed (clearly above noise). The hellaswag adapter never improves hellaswag
(0.63-0.67, below the 0.70 base in all seeds), so the experiment is effectively one working skill plus one that does not transfer. The merge keeps the ARC gain
and is >= the ARC parent on arc_easy in 3/3 seeds, but each gap is within one standard error; on hellaswag it matches the parents and stays below the base.
Not run: per-item paired statistics, more than 3 seeds, larger samples.

---

# Evolution vs random search (Qwen2.5-0.5B, ARC-Easy LoRA + PIQA LoRA) (2026-10-09)

Setup: both runs use 30 candidate evaluations scored on the FIRST 100 items of arc_easy and piqa (fitness = mean of the two acc_norm minus 0.1 * gap).
Evolution: population 6 x 5 generations (`q05_evo.yaml`). Random search: one generation of 30 (`q05_rand.yaml`; the first 6 are identical to evolution's generation 0).
Top-3 of each run (by search fitness) were re-evaluated on FRESH items 100-599 of both tasks (500 items, standard error ~0.02).

Search trajectory (evolution): best fitness per generation 0.710, 0.714, 0.724, 0.720, 0.724; mean 0.698, 0.704, 0.713, 0.705, 0.711 (all-candidate mean 0.706).
Random search: best 0.720, mean 0.705.

Fresh-set results (acc_norm; fresh fit = mean - 0.1 * gap):
| model | search fit | arc-weight | arc_easy | piqa | fresh fit |
|---|---:|---:|---:|---:|---:|
| base | | | 0.596 | 0.706 | 0.640 |
| ARC LoRA | | 1.0 | 0.760 | 0.722 | 0.737 |
| PIQA LoRA | | 0.0 | 0.614 | 0.724 | 0.658 |
| evo g002-c005 | 0.724 | 0.64 | 0.742 | 0.738 | 0.740 |
| evo g004-c002 | 0.724 | 0.77 | 0.746 | 0.742 | 0.744 |
| evo g002-c000 | 0.720 | 0.71 | 0.736 | 0.740 | 0.738 |
| rand g000-c026 | 0.720 | 0.72 | 0.742 | 0.738 | 0.740 |
| rand g000-c016 | 0.716 | 0.52 | 0.722 | 0.728 | 0.724 |
| rand g000-c008 | 0.714 | 0.66 | 0.746 | 0.740 | 0.742 |

Top-3 mean fresh fitness: evolution 0.740, random 0.735 (difference far below the standard error).

Why neither search can show an advantage: over the 60 evaluated candidates (arc weights 0.23-0.77) fitness spans only 0.684-0.724, its sd across candidates is 0.010,
smaller than the evaluation noise (~0.03 on 100 items). Fitness vs arc weight: corr +0.24, slope +0.019 per unit weight (arc_easy slope +0.07, piqa slope -0.035).
Merged models are ~equal to the ARC LoRA on fitness, slightly below it on arc_easy (-0.015) and slightly above both parents on piqa (+0.015..+0.02, ~1 standard error).

---

# Complementary skills: ARC-Easy LoRA + BoolQ LoRA (Qwen2.5-0.5B) and fast GP search (2026-10-09)

Search score = fitness on the first 100 items (arc_easy acc_norm, boolq acc; fitness = mean - 0.1 * gap). Fresh = items 100-249 (150 items, standard error ~0.04).
Single adapters: ARC LoRA arc_easy 0.72 / boolq 0.63 on fresh data; BoolQ LoRA 0.57 / 0.79 (each helps only its own task).

Evolution (6 x 4 generations = 24 evals) vs random search (24 evals), fixed seeds (3 shared candidates) excluded:
- mean fitness 0.732 (evo) vs 0.718 (rand), difference +0.014, t = 4.4; 9/21 vs 1/21 candidates with fitness >= 0.735; best 0.742 vs 0.738.
- Evolution concentrates samples where fitness is high (mean ARC weight 0.31 vs 0.50), but its best point is not better than random's best.

Fresh-set check (fitness / arc_easy / boolq): base 0.573; ARC LoRA 0.664; BoolQ LoRA 0.659; evo best 0.741 / 0.693 / 0.813; fixed seed w=0.25 0.735 / 0.687 / 0.807;
random best 0.705 / 0.647 / 0.793; GP best 0.739 / 0.693 / 0.807. Merges beat both parents by ~0.07 (about 1.7 standard errors) because the two skills are complementary.

GP + expected improvement over 3 module-group weights (`bo_search.py`, 8 evaluations, ~25 min on idle hardware vs ~70 min for evolution):
best search fitness 0.746 at weights (attention 0.76, mlp 0.07, other 0.05); 0.740 already after 3 evaluations. Fresh fitness 0.739 (evolution 24 evals: 0.741).
Offline replay on the random pool (`surrogate_analysis.py`): evaluations needed to reach fitness >= 0.74: random 12.4, GP+EI 6.6.
Group analysis: boolq depends on attention and mlp weights (-0.064 / -0.086 per +1 ARC weight), "other" has no effect; ARC skill barely depends on the weights.

Caveats: one seed, 100/150-item samples, small pool for the replay; GP-vs-evolution fresh difference (0.739 vs 0.741) is far below the standard error.

Research notes (sources): SIP-BMM / AP-BMM / BAMBO (https://arxiv.org/abs/2512.09972), multi-fidelity merging (https://arxiv.org/abs/2502.04030), surrogate benchmarks (https://arxiv.org/abs/2509.02555),
data-free covariance merging ACTMat (https://arxiv.org/abs/2604.01329), LARV (https://arxiv.org/abs/2602.09413), FroM (https://arxiv.org/abs/2506.02478): TIES/DARE can lower LoRA-merge accuracy.

---

# Qwen3-4B full-checkpoint merge with the new `search: gp` (2026-10-09)

Parents: Qwen3-4B (post-trained, weight w) and Qwen3-4B-Base (weight 1 - w); lite engine (CPU merge), scored on the Intel Arc XPU with lm-eval,
arc_easy acc_norm and hellaswag acc_norm on the first 40 items (standard error ~0.07 per task, so only large gaps mean anything). Generation 0 = seeded design (3), generation 1 = one GP batch (3).
Preflight needed two fixes: BPE merges stored as strings vs pairs / `ignore_merges` null vs false are now treated as equal, extra added tokens (`<think>`) and a different `max_position_embeddings` are warnings.

| candidate (w_post: attention / mlp / other) | arc_easy | hellaswag | fitness |
|---|---:|---:|---:|
| Qwen3-4B-Base | | | 0.710 |
| Qwen3-4B post-trained | | | 0.660 |
| g0 0.25 / 0.25 / 0.25 | 0.750 | 0.675 | 0.705 |
| g0 0.50 / 0.50 / 0.50 | 0.775 | 0.650 | 0.700 |
| g0 0.75 / 0.75 / 0.75 | 0.750 | 0.650 | 0.690 |
| GP 0.22 / 0.55 / 0.03 | 0.700 | 0.725 | 0.710 |
| GP 0.55 / 0.17 / 0.08 | 0.750 | 0.700 | **0.720** |
| GP 0.26 / 0.02 / 0.60 | 0.675 | 0.575 | 0.615 |

Reading: interpolating base and post-trained weights is smooth at 4B (no collapse) except when 60% of the post-trained "other" tensors (embeddings, norms) are mixed in (0.615).
Merged candidates sit at 0.69-0.72, i.e. around the better parent (base 0.710) - no gain is detectable at this sample size; log-likelihood tasks do not reward chat post-training.
Not measured: generative benchmarks (GSM8K, IFEval), which is where a base/post-trained blend could differ. Evaluation, not merging, is the bottleneck on this 16 GB GPU:
8B+ models do not fit the GPU in bf16, and a 30B+ checkpoint could be merged (tensor streaming) but not evaluated here.

---

# Resident merge-and-score loop: seconds per candidate (2026-10-10)

`fast_merge_eval.py` keeps the base model on the XPU, overwrites the LoRA-targeted weights with the weighted delta and scores the items in-process (same prompts and metrics as lm-eval).
Agreement with lm-eval on the same candidates: within one item per 100 (e.g. w=(0.25,0.25,0.25) on Qwen2.5-0.5B: lm-eval 0.73/0.78, resident 0.73/0.77).
Time per candidate: 0.5B 5 s (100 items/task) or 14 s (300 items) vs ~285 s; Qwen3-4B 29 s (100 items) or 47 s (150 items) vs ~11-14 min.

Qwen2.5-0.5B, ARC LoRA + BoolQ LoRA, 30 evaluations on 300 items per task (7 min each), top picks re-scored on 300 FRESH items (standard error ~0.025):
| | base | ARC LoRA only | BoolQ LoRA only | GP top 1-3 | random top 1-3 |
|---|---:|---:|---:|---:|---:|
| fresh fitness | 0.611 | 0.713 | 0.692 | 0.780 / 0.789 / 0.785 | 0.787 / 0.777 / 0.777 |
Merges beat both parents by about +0.07 (about 3 standard errors). GP and random are indistinguishable with this budget: the weight landscape is a wide plateau
(search best 0.7533 vs 0.7487; GP had 2 evaluations >= 0.75, random none; mean fitness 0.7378 vs 0.7355).

Qwen3-4B-Base, ARC LoRA + HellaSwag LoRA, GP, 10 evaluations on 150 items per task (475 s total), re-scored on 150 fresh items (standard error ~0.037):
base 0.641, ARC only 0.713, HellaSwag only 0.684, GP top1 0.723 (w = 0.07/0.93/0.89), GP top2 0.728 (0.10/0.82/0.73).
Merges are about +0.08 over the base and +0.01 over the better parent, which is inside the noise. The best weights give attention mostly to the HellaSwag adapter and the MLP to the ARC adapter.

Conclusion: the speedup comes from removing fixed per-candidate overhead, not from a cleverer search; once evaluation costs seconds, 300+ items per candidate become affordable and the earlier noise problem largely disappears.

---

# Model-family support and a real MoE merge: OLMoE-1B-7B (2026-10-10)

Rule table for module families (attention / mlp / router / norm / embedding / other), user `tensor_rules`, MoE-aware LoRA target discovery (`lerp/targets.py`).
LoRA targets discovered from meta-device models, no weights downloaded: Qwen2.5-0.5B 168, Qwen3-4B 252, Gemma 4 E4B 258 (towers skipped), Gemma 4 26B-A4B 205, Qwen3-30B-A3B 192 (6,144 experts skipped), OLMoE-1B-7B 64.

OLMoE-1B-7B base + Instruct (64 experts/layer, top-8), `lite` engine, per-group weights of the Instruct parent attention 0.9 / experts 0.1 / router 0.6 / other 0.3:
3,219 tensors, 12.9 GiB, 55 s; groups: attention 112, mlp 3,072, router 16, other 19; worst relative error per group 1.6e-3..2.1e-3.
Child loads on the 16 GB XPU and generates. lm-eval (100 items, acc_norm; fitness = mean - 0.1 * gap): base 0.730 / 0.680 (0.700), Instruct 0.770 / 0.750 (0.758), merged 0.740 / 0.710 (0.722).
The merge sits between its parents. The 100-item standard error is ~0.045, so the order inside that band is not established.
