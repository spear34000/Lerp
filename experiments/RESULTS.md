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


---

# `lerp search`: the resident evaluator inside the product (2026-10-10)

Declarative tasks (`lerp/tasks.py`): arc_easy, arc_challenge, boolq, hellaswag and piqa built in; openbookqa and sciq written only as YAML `task:` blocks. All seven compared with lm-eval's own
document conversion (context, choices, correct index) on 40 real documents each: identical.

Resident engine (`lerp/resident.py`, `lerp search`): LoRA blending and full-checkpoint blending (linear and task arithmetic), per-expert -> fused-expert name mapping, tied weights, startup self-check,
out-of-memory backoff. Verified with tiny offline models against the merge engines' own output (dense, task arithmetic, LoRA, MoE; logits equal to 1e-6 / 1e-4) and against a brute-force log-likelihood scorer.

Real models on the Intel Arc 140V:
- Qwen2.5-0.5B, ARC LoRA + BoolQ LoRA, GP, 3 generations of 3 + 3 baselines, 100 items per task: 12 scorings in 90 s including model load, 4.3 s per candidate.
  Candidate (0.25 everywhere): lerp 0.73 / 0.77 vs lm-eval 0.73 / 0.78; (0.5): 0.72 / 0.74 vs 0.71 / 0.73. `board`, `compare` (vs resident baselines), `report`, `build`, `status` work on the run.
- OLMoE-1B-7B base + Instruct (64 experts per layer, 12.9 GiB), weights attention 0.9 / experts 0.1 / router 0.6 / other 0.3: session start 64 s; per candidate 44 s (30 s to write the weights, 14 s to score) vs about 10 min
  through merge + lm-eval. Scores 0.73 / 0.70 vs lm-eval 0.74 / 0.71 (another batch composition gave 0.75 / 0.71: bf16 rounding moves 100-item accuracies by 1-3 items). Base 0.74 / 0.69 (lm-eval 0.73 / 0.68), Instruct 0.74 / 0.74 (lm-eval 0.77 / 0.75).

Bugs found by running it on real models: the first self-check compared different batch shapes and produced a false alarm (bf16 logits are quantized to 0.125 at magnitude 16-32); it now compares
`lm_head(base_model(x))` with `model(x).logits` on the identical batch. Stacking experts on the host cut the weight-writing time from 78 s to 30 s; fp32 temporaries for a whole layer ran out of memory on the 16 GB device, so experts are processed in groups.

## Generative scoring in the resident evaluator (GSM8K, Qwen2.5-0.5B + ARC LoRA)

`lerp/generative.py`: greedy decoding of a fixed window, stop strings, answer extraction by regex, exact match. Zero-shot GSM8K, first 30 test items, 256 new tokens, bf16 on the Arc 140V
(`experiments/gen_check.py`).

| scorer | exact match | note |
|---|---:|---|
| `lerp` resident, ARC LoRA | 0.100 (3/30) | |
| `lm-eval` gsm8k, flexible-extract, same adapter, `max_gen_toks=256` | 0.100 +/- 0.056 | strict-match 0.000 (a zero-shot base model does not write `####`) |
| `lerp` resident, PIQA LoRA / 0.5 blend | 0.067 / 0.000 | |

Agreement with lm-eval is exact on this one comparison, but 3 correct answers out of 30 cannot tell scorers (or models) apart; this is a plumbing check, not a measurement of merge quality.
Batched left-padded decoding against one-prompt-at-a-time decoding: 4/4 identical extracted answers in float32, 2/4 in bf16. The difference is numerical (a different batch shape changes bf16 rounding and a
greedy decode of 128+ tokens diverges after one flipped token), so generative scores carry batch-composition noise on top of the sampling error. The first candidate of a session takes several minutes (warm-up
kernels), later ones 7-30 s for 8-30 prompts. The tiny-model tests pin the padding logic (batched = single in float32, any batch size).

## Paired item-level comparison (`lerp pairs`, McNemar)

`lerp search` now stores the 0/1 outcome of every scored item (`items.json` next to each score); `lerp pairs -r RUN -g G -i I` compares a candidate with the base and each parent
item by item: exact McNemar test on the discordant items and a 95% interval for the accuracy difference (Wald with +0.5 per cell, Agresti-Min). `compare-samples` adds the same block for
binary scores. Exactness is tested against the binomial tail (8 vs 2 discordant -> p = 0.1094).

Real run: Qwen2.5-0.5B, ARC-Easy LoRA + BoolQ LoRA, 150 items per task, one generation of 4 candidates (4 fixed weights), bf16 on the Arc 140V (7 s per candidate, 7 s per baseline).
Best candidate (g0-c3, fitness 0.739) against the parents:

| vs | task | cand | baseline | diff [95% CI] | only cand / only base | p |
|---|---|---:|---:|---|---:|---:|
| arc LoRA | arc_easy | 0.707 | 0.707 | +0.000 [-0.056, +0.056] | 9 / 9 | 1.000 |
| arc LoRA | boolq | 0.787 | 0.720 | +0.067 [-0.003, +0.134] | 19 / 9 | 0.087 |
| arc LoRA | pooled | 0.747 | 0.713 | +0.033 [-0.011, +0.077] | 28 / 18 | 0.184 |
| boolq LoRA | arc_easy | 0.707 | 0.620 | +0.087 [+0.019, +0.152] | 20 / 7 | 0.019 |
| boolq LoRA | boolq | 0.787 | 0.773 | +0.013 [-0.021, +0.047] | 4 / 2 | 0.688 |
| boolq LoRA | pooled | 0.747 | 0.697 | +0.050 [+0.012, +0.087] | 24 / 9 | 0.014 |

Reading: the blend keeps each parent's strength (equal on its own task) and gains on the other parent's task, but against the *better parent per task* no difference is significant at 150 items
(pooled against the ARC parent +0.033, p = 0.18). Caveats: the winner was chosen on these same items (selection bias favours it; `lerp validate` on fresh items is the real test), four candidates and nine comparisons
were looked at without multiplicity correction, and the earlier +0.07 result on fresh items (see above) is the better evidence for this pair.

## New merge methods on a real checkpoint pair (OLMoE-1B-7B, `experiments/merge_methods_check.py`)

SLERP, TIES, DARE-TIES and DARE-linear (`lerp/mergeops.py`) applied in place by the resident evaluator to OLMoE-1B-7B Instruct and pretrained (the pretrained model is also the base, so its task vector is zero and the
task-vector methods effectively move toward the Instruct model only), weight 0.5 for every group, density 0.5, task_scale 1, seed 5, 100 items per task, bf16 on the Arc 140V. One run, one seed.

| model | arc_easy | boolq | apply | score |
|---|---:|---:|---:|---:|
| instruct | 0.74 | 0.75 | - | 83 s (first, includes warm-up) |
| pretrained | 0.74 | 0.72 | - | 34 s |
| linear | 0.75 | 0.78 | 30 s | 18 s |
| slerp | 0.75 | 0.79 | 62 s | 19 s |
| ties | 0.77 | 0.73 | 214 s | 19 s |
| dare_ties | 0.76 | 0.77 | 88 s | 19 s |
| dare_linear | 0.76 | 0.74 | 76 s | 20 s |

Every difference is within the noise of 100 items (standard error about 0.04): **no method is shown to be better than linear** here. What the run establishes is that the methods execute on a 6.9B-parameter mixture-of-experts
checkpoint with sane scores. Their equality with the `lite` engine is established on tiny models in the tests, not at this scale. Cost: because fused experts are merged expert by expert and TIES needs an order
statistic per tensor on the CPU, applying a candidate takes 2-7x longer than linear blending (30 s -> 62-214 s); the apply step is the target of the planned speed work.

## Closed evolutionary learning loop (`lerp evolve`), first run: **not shown to beat plain training**

Loop (`lerp/evolution/`): founders learn skills (LoRA SFT), survivors are crossed (LoRA rank concatenation), each child **learns a new skill** on verifiable problems (continuing from the inherited merged adapter, with 50% replay of the founders'
skills), is compressed back to rank 16 by truncated SVD (energy kept 0.85-0.99), scored on a dev split, and the best survive as the next parents. Problems are arithmetic with exact answers (`add`, `mul`; the new skill `chain`: `a + b * c`,
`a * b - c`); train, dev and test questions are disjoint (asserted), the final comparison uses a test split no step of the loop ever saw. Criteria were fixed in `orchestrator.CRITERIA` before the run: the evolved best must beat the best founder on
the new skill by >= 0.05 (McNemar p < 0.01, CI above 0), lose at most 0.05 on the old skills, and beat the **compute-matched plain-training control** (same total training steps, one adapter, same data mix, started from the 0.5 merge of the founders) with p < 0.05.

Setup: Qwen2.5-0.5B, 2 founders x 150 steps, 3 generations, <= 4 children per generation x 100 steps (11 children, 1100 steps total; the control trains 1100 steps), seed 1, 300 test items per family, bf16 on the Arc 140V, 40 min wall clock.

| model | add | mul | chain (new) |
|---|---:|---:|---:|
| base | 0.597 | 0.227 | 0.007 |
| founder add | 0.927 | 0.260 | 0.010 |
| founder mul | 0.820 | 0.313 | 0.027 |
| merge-only (0.5, no learning) | 0.933 | 0.303 | 0.017 |
| **evolved best** (g3-c0, lineage g0-add, g0-mul, g1-c0, g1-c1, g2-c2, g2-c3, g3-c0) | 0.907 | 0.307 | 0.080 |
| **plain training control** (1100 steps) | 0.930 | 0.343 | 0.130 |

* New skill learned: evolved vs best founder on `chain` +0.053, 95% CI [+0.020, +0.086], p = 0.0025 (21 vs 5 items) - the gate is passed, narrowly. Old skills retained (-0.020 add, -0.007 mul).
* **Evolution did not beat plain training: on `chain` the control is better, evolved - control = -0.050 [-0.090, -0.009], p = 0.024 (12 vs 27 items); on `mul` -0.037 [-0.068, -0.005], p = 0.035.** Verdict from the pre-set criteria: *learned and retained, but NOT shown to beat plain training*.
* The absolute numbers are low: a 0.5B model learns `a + b * c` only to 13% with 1100 steps. The loop spreads the same compute over 11 children of 100 steps each, so every lineage gets little learning; selection on a 100-item dev split picks among near-equal candidates (dev chain 0.06-0.10) mostly by noise.

Limits: one seed, one configuration, a model too small for the new skill, no tuning of the loop. This does not show that an evolutionary outer loop can never help (more steps per child, a skill the founders' recombination actually helps with, larger populations); it shows that
at this scale it is not better than training one adapter for the same number of steps, and the loop is not "evolution that accumulates abilities" until that comparison is won.

## Does merging beat both parents? Pre-registered test (`experiments/merge_proof.py`): **not proven**

Question asked: in every one of 3 training seeds, does a blend of two skill LoRAs beat BOTH parents on the pooled accuracy of the two tasks by >= +0.03, on 1000 fresh items per task (never used in the search), with exact McNemar p < 0.01 and a 95%
interval above 0? Criteria fixed before any run. Qwen2.5-0.5B, LoRA rank 16, 200 SFT steps per skill (400 for the multi-task LoRA, the strongest control: one adapter trained on both skills), search = 12 GP evaluations on
200 items (window 0-200), final scoring on items 500-1500. Per-seed raw results and the verdict output: `experiments/results/merge_proof/`.

| pair | pooled accuracy, mean of 3 seeds: base / parent A / parent B / 0.5 blend / searched blend / multi-task LoRA |
|---|---|
| ARC-Easy + BoolQ | 0.602 / 0.675 / 0.706 / 0.732 / 0.734 / 0.737 |
| PIQA + HellaSwag | 0.596 / 0.602 / 0.596 / 0.601 / 0.600 / 0.607 |

Findings (searched blend; the 0.5 blend is within 0.005 of it everywhere):

* **ARC + BoolQ: the blend beats both parents in all 3 seeds, significantly.** Against the ARC parent +0.054 to +0.062 (p < 0.0001). Against the better parent (BoolQ) +0.025 / +0.042 / +0.018 (p = 0.0014 / < 0.0001 / 0.021),
  every interval above 0. The pre-set bar (+0.03 over *both* parents with p < 0.01 in *every* seed) is met in 1 of 3 seeds, so the claim as pre-registered is **not proven**; the honest reading is a real but modest gain over the better parent, about +0.03.
* **The blend is not better than one adapter trained on both skills.** Blend minus multi-task: -0.005 / +0.004 / -0.006 (p = 0.45-0.64). Merging reaches what joint training reaches without a joint training run, but does not exceed it.
* **The search added nothing over the fixed 0.5 blend.** Searched versus 0.5: pooled within 0.004 in every seed; 12 evaluations bought no measurable gain on this plateau.
* **PIQA + HellaSwag: nothing to prove.** The LoRAs barely moved the base model (0.602 and 0.596 against base 0.596), so there are no skills to combine; the blend equals everything within +-0.01 (all p > 0.05). Training did not create a skill here,
  which is a failed premise, not evidence about merging.

What this shows and does not show: for two clearly complementary skills, merging two small LoRAs recovers essentially all of the benefit of joint training and beats each parent by roughly +0.02 to +0.06 on fresh items, robustly across seeds. It does not show
"noticeably better than existing models" nor better than joint training; only one task pair at 0.5B shows a gain and the requested bar was not met.

## Evolution loop after the fixes: crossover ablation (`experiments/evolution_ablation.py`): **no arm beats plain training**

Fixes applied after the diagnosis of the first run, all switchable in the config: `graft` crossover (`child = A + (B - init_B)`, so shared ancestry is counted once, with the start adapter of every organism kept), adapters stored at the scale of
freshly trained ones (alpha = 2r; the first run's inherited adapters learned at half speed), a single ranking key for survivor selection and the final pick, fitness that weights the new skill 2x, the forgetting gate anchored to the founders' dev scores, 300 dev and 600 test items per
family. Arms differ **only** in the crossover (`graft`, the original `blend`, and `none` = each child keeps training a survivor); they share the same founders, the same control, the same total training steps (1200 = 3 generations x 2 children x 200) and the same test items.
Qwen2.5-0.5B, same arithmetic families as before, seed 1; 96 min of wall clock in total (graft 35 min including the shared founders and the shared control, blend 27, none 34).

| model | add | mul | chain (new) | steps in final lineage / discarded |
|---|---:|---:|---:|---:|
| base | 0.585 | 0.233 | 0.005 | |
| founder add / founder mul | 0.915 / 0.783 | 0.278 / 0.342 | 0.008 / 0.020 | |
| merge-only (0.5, no learning) | 0.927 | 0.327 | 0.012 | |
| **plain training control** (1200 steps) | 0.930 | 0.355 | **0.115** | |
| evolved, `blend` (g2-c0) | 0.890 | 0.318 | 0.097 | 400 / 800 |
| evolved, `graft` (g1-c0) | 0.892 | 0.325 | 0.085 | 200 / 1000 |
| evolved, `none` (g2-c0) | 0.892 | 0.345 | 0.093 | 400 / 800 |

Paired comparisons on the new skill (600 items, same items for every model; difference [95% CI], exact McNemar p):

* every arm beats the best founder clearly (+0.065 to +0.077, p < 0.0001): the loop does learn the new skill;
* arm minus control: blend -0.018 [-0.045, +0.009] p = 0.22; graft -0.030 [-0.058, -0.001] p = 0.051; none -0.022 [-0.047, +0.004] p = 0.12. **No arm beats the control; all point below it.**
* arm against arm: blend - graft +0.012 (p = 0.47), blend - none +0.003 (p = 0.89), graft - none -0.008 (p = 0.62): **the crossover mode makes no measurable difference**, including no crossover at all.
* old skills: every arm ends 0.017-0.025 below the best founder on `add` and `mul` (inside the 0.05 tolerance, but nonzero) except `none` on `mul` (+0.003).

What the logs show (facts): `graft` got stuck - from generation 2 every child was rejected by the founder-anchored forgetting gate (their `add`/`mul` dev scores were 0.05-0.1 below the founders'),
so the survivor set never changed and 1000 of its 1200 training steps were discarded; its final organism comes from generation 1 (200 lineage steps), so **this ablation never exercised the accumulation hypothesis that graft was built for**.
`blend` and `none` kept a lineage of two generations (400 steps) and discarded 800. Every arm spends two thirds of its steps on children that never reach the final organism, while the control spends all 1200 on one adapter.
The `control-plain` and `merge-only` rows are bit-identical across the three arms' result files (same adapter, same items), so the cross-arm comparisons carry no scoring noise from them.

Resolution: with one seed and 600 items the arm-versus-arm intervals are about +-0.027, so differences smaller than roughly 0.03 cannot be seen; "no measurable difference between crossover modes" means exactly that, not that they are equal.
Graft against the control is borderline (CI excludes 0, p = 0.0505), not significant.

Candidate explanations for the remaining gap to the control, **all untested**:

1. The structural waste above (two thirds of the steps never reach the final organism).
2. Re-compression every generation (rank 32 -> 16 keeps only 83-99% of the energy; the control is compressed once, at the end).
3. Every child restarts the optimizer and a 20-step warm-up inside its 200 steps; the control does this once.
4. A flaw in the graft itself (a hypothesis): the stored `B` is the compressed version of the trained adapter but `init_B` is not, so `B - init_B` is not exactly "what B learned"; it also contains minus B's truncation residual, which lies mostly in the founders' subspace. Each graft would then strip some
   founder content, consistent with the `add`/`mul` drops that tripped the gate - but the summing of two skills' full deltas could equally explain it, and neither was checked.

Compared with the first run (blend, unfixed): the gap to the control shrank from -0.050 (p = 0.024) to -0.018 (p = 0.22) and the evolved model is no longer significantly worse, but several things changed at once (scale, fitness, dev size, fewer and longer children), so this cannot be attributed to any single fix. One seed, one model size, one new skill:
the result does not say evolution can never help, it says that here none of the variants shows a benefit over simply training one adapter for the same number of steps.

## Evolution loop diagnostics: where the gap to plain training does *not* come from (`experiments/evolution_followup.py`)

Cheap follow-ups on the crossover ablation above, reusing its founders, its control and their test scores (only the new organisms are trained and scored; same 1200 training steps, same 600 test items, seed 1, Qwen2.5-0.5B). Raw outputs:
`experiments/results/evolution_followup/`.

| variant | add | mul | chain | lineage steps / discarded | chain vs control-plain (0.115) |
|---|---:|---:|---:|---:|---|
| `none-seq`: no crossover, 1 child per generation, 6 generations of 200 steps | 0.928 | 0.340 | 0.092 | 600 / 600 | -0.023 [-0.049, +0.003] p = 0.098 |
| `graft-r32`: graft, children stored at rank 32 | 0.900 | 0.312 | 0.083 | 200 / 1000 | -0.032 [-0.061, -0.003] p = 0.042 |
| `blend-r32`: blend, children stored at rank 32 | 0.890 | 0.308 | 0.080 | 200 / 1000 | -0.035 [-0.063, -0.007] p = 0.019 |

None of the variants differs from the earlier arms (all arm-versus-arm p > 0.3, resolution about +-0.03).

What this rules out, for this configuration:

* **Compression loss is not the cause.** At rank 32 the energy kept per child is 0.995-0.999 (it was 0.83-0.93 at rank 16) and the results did not move. `graft` still stalled: every child from generation 2 was rejected by the founder-anchored gate (their `add` fell to 0.79-0.85, `mul` to 0.22-0.26).
  With truncation gone, the graft's loss of the old skills must have another cause (not found; the earlier "truncation residual" hypothesis for the graft is not supported).
* **Waste from parallel children is not the whole story.** `none-seq` has no parallel children (one lineage of six 200-step segments) and still sits 0.023 below the control (p = 0.098, not significant at 600 items but pointing the same way as every other arm).
* **Dev-set selection noise does not explain it.** The dev-selected organism of `none-seq` was g4-c0 (600 steps); scoring the last segment g6-c0 (1200 steps) on the test items gives the same `chain` accuracy (0.092 both), so choosing the latest segment instead would not have closed the gap.

What remains consistent with the data, **untested**: each segment restarts the optimizer state and a 20-step warm-up and runs its own cosine decay to zero, while the control uses one schedule over 1200 steps. `chain` accuracy of the lineage did not improve between 600 and 1200 steps (0.092 -> 0.092)
while the control reached 0.115; a restart-per-segment schedule is a candidate cause. A direct test is to carry the optimizer state and a single learning-rate schedule across a lineage (or to measure the control at 600 steps to see whether the plateau is real).
One seed and 600 items cannot resolve differences below about 0.03, so even the direction of the small gaps is uncertain.
