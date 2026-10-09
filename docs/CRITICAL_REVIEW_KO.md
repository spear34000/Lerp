# Lerp v0.5 - Critical Review

The previous Korean-language v0.3 audit has been preserved as `CRITICAL_REVIEW_V03_KO.md`.

**Current release verdict:** Research Alpha. Passing numerical checks does not establish improved real LLM quality.

**Current v0.4 assessment (all 37+ identified risks and mitigations):**

- [V04_TECHNICAL_AUDIT.md](V04_TECHNICAL_AUDIT.md) - current detailed adversarial review, risk register and Go / No-Go gates
- [CRITICAL_REVIEW_V03_KO.md](CRITICAL_REVIEW_V03_KO.md) - previous in-depth Korean critical analysis and baseline deficiencies
- [QUICKSTART_KO.md](../QUICKSTART_KO.md) - Korean setup and latest v0.4 commands

New protections include opt-in content-hash model freezing, output integrity validation, per-run CLI writer lock, adapter rank/size ceilings, paired sample uncertainty analysis and exact/normalized split contamination screening.

**Release blocker:** genuine `peft` + `transformers` model forward and `lm-eval`/MergeKit full execution could not be run in the packaging environment. A runnable offline integration gate is included but was skipped.
