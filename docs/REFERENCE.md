# Lerp reference manual

Full command reference, mathematics and compatibility boundaries. For an overview start with the [README](../README.md).

## Implemented and tested

| Feature | Reality / boundary |
|---|---|
| 2–6 compatible parents | Genetic coefficients per layer, interpolated between control points; same base/architecture required |
| Full model merges | `--engine lite` for CPU Torch `safetensors` linear and task arithmetic, verified on tiny real tensor files. `--engine mergekit` supports five recipe types; its heavyweight run was **not** exercised here |
| LoRA adapter merge | `--engine lora` concatenates ranks and applies scaling to produce a sum of LoRA update matrices, **numerically verified on synthetic adapters**. Supports ordinary PEFT linear A/B factors only |
| Module genetics | Separate Attention, MLP and fallback `other` weight groups; optional norm/embedding for full checkpoints (heuristic tensor-name classification) |
| Merge-method evolution | `method: auto` with explicit supported `search_methods`; method inheritance and seeded exploration |
| Pareto + weighted search | Nondominated fronts and crowding-distance, stochastic parent selection. **Inspired by**, not a complete implementation of NSGA-II |
| Staged eval | `screening_limit` on all candidates then full evaluation only of `promote_top` candidates; screen scores never count as final scores |
| Held-out verification | `validate` uses **different task names**, evaluates child and optionally all baselines with one separate protocol, and never changes search scores |
| Resume / integrity | Content-hash frozen local inputs, output artifact SHA-256 manifests, advisory OS-backed CLI write lock, crash-recoverable generation publish and model/eval drafts |
| Real-world audit helpers | `compare-samples` paired bootstrap and exploratory sign-flip test from aligned JSONL; `audit-splits` exact/normalized text overlap. Neither proves generalization or uncontaminated training |
| Output constraints | LoRA rank ceiling and estimated output-tensor size ceiling (NOT peak RAM) |
| Offline HTML dashboard | Per-generation fitness, Pareto plot, status and preliminary screen score display, baseline comparison, lineage export |
| Tests | **124 passing, 1 platform-dependent skip** (symlink test, skipped when the OS denies symlink creation, e.g. Windows without Developer Mode) on Windows 11 / Python 3.12 / torch 2.13 / peft 0.20 / transformers 5.16. Includes numerical LoRA update checks, crash recovery and a mocked staged/holdout workflow |

**Verified on Windows (2026-10-08):** `examples/offline_peft_equivalence.py` passes (max logits error 6.7e-8); a full-checkpoint `lite` merge and a LoRA merge of real Qwen2.5-0.5B models build, load in Transformers/PEFT and generate; `lm-eval` 0.4.13 ran on the merged outputs (30-item smoke run; differences between models were within noise).

**Still unverified:** quality improvement over parent models, trained (non-random) LoRA parents, the MergeKit CLI (TIES/DARE), models above 1.5B, GPU/XPU execution.

## Install

Python 3.10 or later, in PowerShell or bash:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

For checkpoint merging:

```bash
python -m pip install -e '.[lite]'    # CPU HF safetensors
python -m pip install -e '.[lora]'    # CPU LoRA merger + PEFT/Transformers integration dependencies
python -m pip install -e '.[merge]'   # MergeKit CLI
python -m pip install -e '.[eval]'    # lm-evaluation-harness
```

Install only what you need. The LoRA tensor merger itself requires Torch and safetensors; PEFT and Transformers are needed to load/use the resulting adapters. `--engine lite` and `--engine lora` run on CPU; `--cuda` is only passed to MergeKit on supported CUDA hardware. Intel Arc is not CUDA.

### Windows notes

- Set `PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8 = "1"`). Without it some torch builds fail at import with a cp949 `UnicodeDecodeError`.
- Use Windows paths in YAML (`C:/models/base`). Git-Bash style `/c/models/base` is now mapped to `C:/models/base` automatically, but forward-slash drive paths are safest.
- `freeze`'s symlink rejection is tested only where the OS allows creating symlinks.

## GP search (`search: gp`): fewer evaluations

Evolution breeds the next generation from the best candidates and needs dozens of model evaluations. With `search: gp` the same `advance` / `cycle` machinery asks a Gaussian-process surrogate
(fitted to every verified score so far) for the next `population` candidates, chosen by batch expected improvement (kriging believer). Install with `pip install -e ".[gp]"` (needs numpy).

```yaml
selection: weighted      # GP optimizes the scalar fitness (weighted mean minus imbalance penalty)
search: gp
population: 3           # generation 0 is the usual seeded design; later generations are GP batches of 3
gene_groups: [attention, mlp, other]
```

```bash
lerp init -c examples/gp_demo.yaml -o runs/gp
lerp freeze -r runs/gp --strict
lerp cycle -r runs/gp --rounds 3 --engine lora   # 3 + 3 + 3 = 9 evaluations
```

- Search space: one weight per module group and parent (the last parent is implied), flat over depth. This keeps the dimension at `groups x (parents - 1)`; a GP cannot be fitted reliably in the full `groups x genes` space with ten-odd observations.
- Not supported: `method: auto`, and Pareto selection (the scalar fitness is optimized; `selection` still controls the leaderboard).
- Measured (Qwen2.5-0.5B, ARC and BoolQ LoRAs, 100 items per task): 8 GP evaluations found fitness 0.746 on the search items and 0.739 on fresh items; evolution needed 24 evaluations for 0.742 / 0.741.
  The difference between the two final scores is far below the evaluation noise: the gain is cost, not a higher optimum.
- Synthetic landscape test (`tests/test_gp.py`): with 12 evaluations the GP reaches regret 0.013 against 0.067 for random sampling (wins 12/12 seeds).

## Fast resident loop: seconds per candidate (`examples/fast_merge_eval.py`)

Most of the wall-clock time of a normal cycle is not merging or scoring but fixed cost per candidate: writing the adapter, starting `lm-eval`, loading the model and datasets again. For a
LoRA pair the merged weights are linear, so a single resident process can keep the base model on the accelerator, overwrite the target weights with
`W0 + sum_i c_i * scale_i * B_i @ A_i` (the same coefficients as `build_lora`, via `tensor_coefficients`) and score the items itself (lm-eval prompts, continuation tokenization and acc / acc_norm).

```bash
python examples/fast_merge_eval.py --config my-lora.yaml --verify 0.5,0.5,0.5         # compare with lm-eval first
python examples/fast_merge_eval.py --config my-lora.yaml --limit 300 --budget 30 --strategy gp \n       --validate-limit 300 --out search.json     # GP search + fresh-item validation
```

Measured on an Intel Arc 140V (16 GB), 168-252 adapter modules:

| model | items per task | Lerp + lm-eval | resident loop |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 100 | ~285 s per candidate | 5 s (0.1-0.3 s to merge) |
| Qwen2.5-0.5B | 300 | - | 14 s |
| Qwen3-4B | 100 | ~11-14 min | 29 s (2-3 s to merge) |
| Qwen3-4B | 150 | - | 47 s |

Scores agree with lm-eval within one item per 100 (bf16 numerics). Limits: supports `arc_easy`, `boolq` and `hellaswag` (add a `*_docs` function for other loglikelihood tasks),
two LoRA parents with the standard layout, and original weights are kept in host RAM (about 2 bytes per adapted weight). It is a search tool: confirm the winner with `lerp validate`.

## Model families and mixture-of-experts

Module families come from one rule table (`weighting.DEFAULT_RULES`: attention, mlp, router, norm, embedding, other) shared by the merger and by LoRA target discovery (`lerp.targets`).
Mixture-of-experts checkpoints are covered for per-expert tensors (OLMoE, Qwen3-MoE, Mixtral) and fused 3-D expert tensors (Gemma 4); add `router` to `gene_groups` to give routers an independent weight.
Unfamiliar families need no code: add `tensor_rules: [{match: '<regex>', group: mlp}]` to the experiment YAML. See [ECOSYSTEM.md](../ECOSYSTEM.md) for the support matrix and the known gaps.

Verified on OLMoE-1B-7B (64 experts per layer, 6.9B parameters): base and Instruct merged with weights attention 0.9 / experts 0.1 / router 0.6 / other 0.3 in 55 s (3,219 tensors, 12.9 GiB);
every group matches the expected arithmetic to 2e-3 (bf16 rounding); the child loads and generates on a 16 GB GPU and scores between its parents on arc_easy / hellaswag.

## Multimodal and very large checkpoints (Gemma 4, Qwen-VL, ...)

- `num_hidden_layers` is read from `text_config` when the top-level config has none, and compatibility checks compare the nested language-model settings too.
- Tensors under audio/vision towers (`audio`, `vision`, `visual`, `image`, `video`, `multi_modal` in the name) are **not** interpolated along text-layer depth; they use the mean of the gene control points (a no-op when both parents share that tower).
- `--engine lite` merges tensors above 128M elements in row chunks through safetensors slices, so peak RAM is about the size of the output tensor. Gemma 4 E4B's 2.8B-element per-layer embedding merges on a 32 GB machine in about 5 minutes for the whole 16 GB checkpoint.
- Tokenizer check: differences in vocabulary, merges, added tokens, normalizer, pre-tokenizer or decoder are **errors**; a differing `post_processor` (for example automatic `<bos>` insertion) is only a **warning**, because token ids and embeddings stay aligned. The child copies the first parent's tokenizer files, so evaluate all compared models with the same tokenizer behaviour.
- Different model families (for example Qwen3 with Gemma 4) cannot be weight-merged: `check` reports mismatching architecture, layer count, vocabulary, tokenizer and tensor names.

## How genomes map to merge weights

A genome is a flat list of numbers, split into one block per entry of `gene_groups` (`attention`, `mlp`, `other`, ...). Each block holds `genes` control points that are linearly interpolated across layer depth.

- **2 parents:** each number is the weight of **parent 1**; parent 2 gets `1 - value`. `genes: 3` with three groups gives 9 numbers. `0.25` everywhere means 25% parent 1, 75% parent 2.
- **3-6 parents:** each block holds `parents x genes` numbers (one control-point row per parent), normalized so the weights at every control point sum to 1.
- For `method: linear` the child tensor is `sum(weight_i * parent_i)`. For LoRA the weights scale each parent's update (`B @ A`) before rank concatenation.
- Tensors outside numbered layers (embeddings, final norm) use the mean of the control points.

The first population always contains fixed seeds (2 parents: uniform 0.25, 0.50 and 0.75), so generation 0, candidate 0 is not random.

## No-download demo (fake data)

```bash
python -m lerp init -c examples/multi_parent_demo.yaml -o runs/toy
python -m lerp simulate -r runs/toy
python -m lerp advance -r runs/toy --allow-simulated
python -m lerp simulate -r runs/toy
python -m lerp report -r runs/toy
```

This **never** evaluates or merges a real language model. Toy fitness is intentionally labeled `SIMULATED_TOY` and is barred from automatic champion export and live auto-cycling.

## Real LoRA workflow

Supply 2–6 **local PEFT adapters trained on the same exact base and ideally immutable revision**. All parents must have compatible target modules and A/B tensor keys. Edit `examples/lora_experiment.yaml` with paths and a suitable evaluation task suite. This example expects real local files that are **not included**.

```bash
python -m lerp doctor -c examples/lora_experiment.yaml
python -m lerp check -c examples/lora_experiment.yaml
python -m lerp init -c examples/lora_experiment.yaml -o runs/lora
python -m lerp freeze -r runs/lora --strict  # strict requires every checkpoint locally
python -m lerp verify-inputs -r runs/lora
python -m lerp build -r runs/lora -g 0 -i 0 --engine lora
python -m lerp baseline -r runs/lora --name all
python -m lerp cycle -r runs/lora --rounds 3 --engine lora
python -m lerp board -r runs/lora --pareto --real-only
python -m lerp report -r runs/lora
```

For an **actual PEFT loading/inference smoke test** (requires a real frozen base, a built adapter, and optional dependencies):

```bash
python examples/smoke_peft_load.py --base ./checkpoints/base --adapter ./runs/lora/generations/gen-000/cand-000/model --device cpu
```

This runnable check was *added* but **not executed against real PEFT models** in this release. It checks finite logits and actual generated tokens; it does not show improved quality.

The baseline evaluation for a parent adapter passes `pretrained=BASE,peft=PARENT_ADAPTER` to the harness; generated child uses `pretrained=BASE,peft=CHILD_ADAPTER`. A full base baseline loads `pretrained=BASE` alone. These commands **require** an actual compatible PEFT and evaluation stack to be installed; external evaluator integration was tested with mocks, not real model inference.

### Budgeted evaluation

`screening_limit: 15`, `promote_top: 3` and `limit: 250` means all candidates are evaluated on 15 samples per task, **only 3** are evaluated again using 250 samples per task. This saves *evaluation* compute, not necessarily merge/storage cost, because all candidate artifacts must still be built. `limit: 250` is still a smoke test. Using first N examples repeatedly may introduce sample-selection bias.

Manual per-candidate screening and promotion:

```bash
python -m lerp screen -r runs/lora -g 0 -i 0
python -m lerp promote -r runs/lora -g 0
python -m lerp status -r runs/lora
```

### Independent holdout protocol

After a candidate was evaluated on the search tasks, run a **different**, frozen protocol with distinct task names. Never use these holdout results to repeatedly tune the same model unless you create another untouched final test set.

```bash
python -m lerp validate -r runs/lora -c examples/holdout_evaluation.yaml --baseline all
```

`validation/<protocol SHA prefix>/summary.json` reports child and base/parent scores without affecting the evolutionary leaderboard. Its task-name separation cannot prove absence of train/test contamination, and the `validate` command itself computes **no confidence interval**. `compare-samples` can separately analyze aligned item-level JSONL scores.

### Export

```bash
python -m lerp export -r runs/lora -o runs/best-recipe
python -m lerp lineage -r runs/lora -o runs/lineage.dot
```

Automatic export requires an `lm_eval`-scored candidate; manual scores are unverified, and **the tool cannot guarantee the external evaluator was truthful**. Only recipes and metrics, not full LLM weights, are exported. Verify licenses independently before distributing model artifacts.

## Mathematical notes

For ordinary LoRA layers, `delta W_i = (alpha_i / r_i) B_i @ A_i`, or `alpha_i / sqrt(r_i)` with rsLoRA. The LoRA `cat` construction produces `B' = concat(w_i * scale_i * B_i, axis=1)` and `A' = concat(A_i, axis=0)` such that `B'@A'` equals `sum(w_i * delta W_i)` **before rounding the result to the requested output dtype**. Output rank is the sum of input ranks; RAM/inference cost may grow.

For normalized convex blends, full-checkpoint `task_arithmetic` with `task_scale = 1` is algebraically the same as linear averaging: `BASE + sum(w_i*(PARENT_i-BASE)) = sum(w_i*PARENT_i)`. Search `linear` vs `task_arithmetic` only adds a real degree of freedom when `task_scale != 1`. TIES/DARE are different, but rely on externally tested MergeKit.

## Compatibility boundaries and risks

- Model ancestry must be confirmed by the user. Matching `config.json`, adapter-declared base strings and tensor shapes **do not prove** checkpoints originated from the same frozen revision.
- LoRA: ordinary 2D dense A/B factors only; intentionally rejects DoRA, aLoRA, `modules_to_save`, bias alterations, mixed adapter targets, per-layer rank patterns, embedding-specific LoRA and other extensions. This is fail-closed, not universal PEFT compatibility.
- The rank-concatenation technique does not preserve train-time dropout behavior; output adapters are inference-only.
- No training, recursive checkpoint ancestry, adapter rank compression, dynamic serving, GPU kernels, license checking, semantic de-duplication, or proof of training-data independence. Optional SHA-256 pinning and descriptive paired statistics do not replace authentic model provenance or a controlled study.
- Runs can be very expensive: `population × generations × (full checkpoint or adapter size)`, plus CPU RAM and tokenizer/model-loading costs.
- Re-running statistical selection on the same tiny sample repeatedly overfits the test set. Benchmark configuration and metric direction/scale must be validated by the experimenter.

Full audit: [CRITICAL_REVIEW_KO.md](CRITICAL_REVIEW_KO.md). Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).


## New in v0.4: pin weights, verify outputs and audit results

Before merging and evaluating, freeze all local weights with SHA-256. A hash mismatch now blocks build/evaluate/advance. This is I/O intensive because every source is re-hashed; remote references can be frozen only as **unverified references** without `--strict`. Prefer local immutable model snapshots. Freeze **before scoring**.

```bash
python -m lerp freeze -r runs/lora --strict
python -m lerp verify-inputs -r runs/lora
```

An output `model/modelbreeder_artifact_integrity.json` checks the merged artifact before evaluation. CLI mutating commands use an OS-backed run lock, but direct Python API calls do not. Neither manifest provides cryptographic authentication against an attacker with write access.

Set these optional resources in experiment YAML (`mode: lora`):

```yaml
max_output_rank: 256
max_lora_output_mib: 1024
```

Actual PEFT integration is now scripted with a tiny randomly initialized local Llama: two genuine PEFT adapters, merge, child PEFT load, dense-delta/logits comparison, and a few generated tokens. **Not run in the release environment** (the optional `peft` and `transformers` packages were unavailable).

```bash
python -m pip install -e '.[lora]'
python examples/offline_peft_equivalence.py
```

Optional item-level comparison (scores in [0,1], matching IDs) and development/holdout contamination checks:

```bash
python -m lerp compare-samples --candidate examples/paired_candidate.jsonl --baseline examples/paired_baseline.jsonl --out comparison.json --replicates 5000 --seed 42
python -m lerp audit-splits --development examples/development.jsonl --holdout examples/heldout.jsonl --out leakage_report.json
```

Both are **illustrative small datasets included as demonstration only**, not LLM benchmark results. The CI uses itemwise paired bootstrap, assumes meaningful independent sampling, and is NOT corrected for multiple comparisons. Exact text checks miss paraphrases and content present in model pretraining.

Evaluation outputs include `modelbreeder_protocol.json` with the executed command, device and task settings. Candidate vs baseline comparison refuses protocol mismatches. This recording does **not** certify the external evaluator itself. See [V04_TECHNICAL_AUDIT.md](V04_TECHNICAL_AUDIT.md) and [CRITICAL_REVIEW_V03_KO.md](CRITICAL_REVIEW_V03_KO.md) for limitations.
