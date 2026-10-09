# Lerp v0.5 | Critical technical audit

Release status: **research alpha; LLM quality unverified**. Audited on 2026-10-08.

**Interpretation of test results:** synthetic tensor numerical tests and mocked evaluator integration are valuable engineering checks, but cannot establish genuine PEFT runtime compatibility or improvements to model intelligence. Optional PEFT/Transformers dependencies were unavailable at build time; the real PEFT offline integration test was skipped, not passed.

## Evidence classes

- **Measured locally:** PyTorch/safetensors matrix operations on synthetic input files; source/output SHA-256 tamper detection; staged genetic workflow; JSONL paired statistics; split overlap auditing; ZIP unpack/CLI tests.
- **Mocked:** `lm-eval run` command invocation and synthetic result JSON extraction; candidate screening and baseline/holdout evaluation flow. No external evaluator results were generated.
- **Provided but not executed:** `examples/offline_peft_equivalence.py`, which constructs a randomly initialized tiny local HF Llama and actual PEFT adapters, loads the merged child and compares logits with an explicit dense update.
- **Not verified:** any trained 0.5B-27B LLM; GPU/XPU/real CUDA execution; actual MergeKit and lm-eval installations and runs; dataset independence from real pretraining corpora; statistically reliable performance gain.

## Detailed adversarial review

| Ref | Priority | Attack, missing assumption, or failure mode | v0.4 action | Residual status |
|---|---|---|---|---|
| M01 | P0 | Parent LoRA A/B factors are averaged separately and introduce cross terms | Concatenate rank factors; check result of B@A against sum of parent deltas | Synthetic math tested, PEFT forward unverified |
| M02 | P0 | Different parent ranks and rsLoRA scale are confused | Independent scale per parent, output alpha=rank; 3-parent numerical regression | Synthetic tested |
| M03 | P0 | Parent tensor A/B shape mismatch allocates large RAM before failure | Verify *all* parent tensor headers before reading tensors | Tested on oversized mismatched input |
| M04 | P0 | Output rank explosion and resource exhaustion | Configurable `max_output_rank` (256 default) hard fail | Tested; no peak RAM guarantee |
| M05 | P1 | Large tensor shapes exceed configured output capacity | Header-level `max_lora_output_mib` estimate (1024 MiB default) | Tested; peak RAM can be much higher |
| M06 | P0 | Unsupported DoRA/aLoRA/token extensions silently merged | Reject known nonstandard flags, unexpected key names | Many paths tested; exhaustive PEFT versions not covered |
| M07 | P0 | Two adapters declare different base models/revisions | Compare `base_model_name_or_path`, `revision`, modules and task type | Metadata checks cannot prove ancestry |
| M08 | P0 | Real PEFT rejects merged adapter config or keys | Added tiny no-download PEFT/Transformers roundtrip and logits equality test | **NOT RUN**, release blocker |
| M09 | P1 | Low-precision quantization violates exact math | Compute factors in FP32 and check post-cast finite values | Numerical tolerances required; not bit-exact |
| M10 | P1 | Different architecture names map module groups incorrectly | Recognized-layer-name parsing, fallback groups, strict unknown LoRA layer handling | Untested on non-Llama/Qwen families |
| I01 | P0 | Base/parent weights change after the experiment starts | New opt-in SHA-256 `freeze`, `verify-inputs` and verification before operations | Local file alteration detected; freeze is not automatic |
| I02 | P1 | Mutable Hugging Face reference is treated as immutable revision | Strict freeze refuses remote source; nonstrict explicitly marks unverified | Need trusted offline snapshot and commit pin |
| I03 | P1 | An attacker modifies frozen manifest as well as weights | Hashes are integrity checks, not signed attestations | Requires trusted filesystem/access control |
| I04 | P0 | Generated child changes after build, before evaluation | Output file hash manifest checked before eval/holdout | New builds covered, legacy unpinned artifacts less protected |
| I05 | P1 | Filename symlink redirects SHA to mutable external file | Reject symlink root/inputs/artifacts and nonregular input files | No full adversarial race testing |
| I06 | P1 | Multiple concurrent CLI writers corrupt state | OS-backed per-run mutation lock | Unit contention tested on Linux; not multi-host/API proof |
| I07 | P1 | Crash during generation leaves inconsistent state | Pending directory, atomic rename, generation commit recovery | Fault-injection unit tests, no power-loss filesystem testing |
| I08 | P1 | Repeated full hashing of large base checkpoints is too slow | Preserve full SHA-256 correctness rather than false caching | Performance cost may dominate large experiments |
| I09 | P1 | An unexpected nested checkpoint directory is silently ignored | Output artifact hash scanner rejects unexpected directories | Could reject legitimate specialized checkpoint layout |
| E01 | P0 | A screen score from 3 examples is counted as a full result | Stage-separated screening JSON, promotion policy | `--limit` is still biased by sample ordering |
| E02 | P0 | Candidate score from CUDA is compared with CPU baseline | Store runtime device/batch size and enforce matching protocol | Hardware/driver and software version details still not pinned |
| E03 | P0 | Evaluator ran different tasks or prompt config than recorded | Persist exact CLI command, model args and task/fewshot/settings | Only mocked lm-eval end-to-end; evaluator authenticity unknown |
| E04 | P0 | A manual `1.0` score is accepted as a measured result | Manual/fake sources clearly labeled and blocked by default evolution | User-controlled files can still be forged |
| E05 | P0 | Final holdout equals search set | Separate task-name protocol and output directory; added exact text split audit | No semantic overlap or pretraining contamination proof |
| E06 | P1 | Independent per-task aggregate means are treated as paired outcomes | Separate aligned item-ID JSONL comparison utility | Caller must prepare correct item-level scores |
| E07 | P1 | One point estimate is declared superior | Paired bootstrap CI, win/loss counts, exploratory sign-flip p | IID/exchangeability assumptions and multiple-selection bias remain |
| E08 | P1 | Repeating 100 trials on same validation pool picks a lucky model | Holdout workflow separated from optimizer | Winner's curse / multiple comparison not corrected |
| E09 | P1 | Lower-is-better perplexity is maximized as a score | Fitness constrains metrics to [0,1] and larger-is-better | Raw perplexity/latency require user-managed normalization |
| E10 | P1 | Validation scripts accidentally train/evaluate on the same examples | New `audit-splits` hashes Unicode-normalized, punctuation-folded text | Does not catch paraphrases or data seen during pretraining |
| E11 | P1 | An inaccurate lmeval JSON result is accepted | Require configured task + metric keys and scores in [0,1] | Actual lmeval schema versions not tested |
| A01 | P1 | Genetic search is advertised as original frontier research | Document MergeKit CMA-ES and algorithm overlap | No quality/speed superiority claim |
| A02 | P1 | A merge recipe is described as a newly trained foundation model | Documentation explicitly says no gradient optimization or new pretraining | Appropriate attribution/license review remains |
| A03 | P1 | LoRA exact concatenation is assumed free to serve | Rank ceiling and capacity check | Additional rank still increases latency/VRAM |
| A04 | P1 | Local Intel Arc supports MergeKit CUDA | CPU-lite/LoRA separation and explicit CUDA-only MergeKit flag | Arc XPU/GPU optimization absent |
| A05 | P1 | MIT code license implies permission to distribute child weights | Docs require review of parent and base licenses | No automatic license analysis |
| A06 | P2 | Static HTML report content injections | Existing text escaping tests | Browser security audit not completed |
| A07 | P2 | Linux test success implies Windows support | Cross-platform lock with explicit msvcrt branch | Actual Windows installation/test still outstanding |

## Required real-data Go / No-Go experiment

1. **Immutable ancestry:** download an exact HF commit of a small base and 2 independently trained PEFT LoRA adapters that all share the same frozen base. Inspect tokenizer, target modules and licenses. Record commits outside the codebase.
2. **Input hashing:** initialize the run and perform `lerp freeze -r <run> --strict`. The test should fail when a weight file is edited and should pass when untouched.
3. **Runtime equivalence:** install Transformers/PEFT, run `examples/offline_peft_equivalence.py`; then test an actually trained adapter pair using `examples/smoke_peft_load.py` and compare logits to direct combined deltas.
4. **Real benchmark:** install/run `lm-eval`, verify tasks and metrics against real output, pin package versions, run exact base/parent/child prompts under equivalent settings.
5. **Held-out evidence:** reserve untouched prompts and deduplicate by ID, normalized text, near-duplicate semantics and knowledge of pretraining sources. Do not use it to tune the evolution hyperparameters.
6. **Statistical analysis:** evaluate aligned item outputs, calculate paired CI, repeat on independent seeds, account for repeated experiments/model selection before reporting generalization.
7. **Operational stress:** run large rank and tensor edge cases, interrupt during build and score commit, observe peak CPU RAM, storage and I/O and independently test Windows and XPU constraints.
8. **Publish only with provenance:** keep base/parent revisions, task definitions, generated recipe, measured confidence intervals, limitations and redistribution licenses.

## Release verdict

- **Engineering alpha with working synthetic input, sha-pinning, provenance and measurement utilities.**
- **No trained model quality improvement verified**; no actual PEFT model loaded during this build.
- **80 local passing tests, optional runtime integration skipped** (run the packaged suite for the authoritative count).
- Use for offline experimentation and research, not as proof of superior LLMs or a production-safe automated model breeder.

Reference implementations: [MergeKit evolution](https://github.com/arcee-ai/mergekit/blob/main/docs/evolve.md), [MergeKit methods](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md), [EleutherAI harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), [Hugging Face PEFT](https://huggingface.co/docs/peft/main/developer_guides/model_merging).
