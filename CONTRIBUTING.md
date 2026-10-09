# Contributing to Lerp

## Branches

`main` is always releasable: CI is green and the docs match the code. Nothing is committed to it directly; every change arrives through a pull request from a short-lived branch.

| Prefix | Use | Example |
|---|---|---|
| `feat/` | new capability | `feat/generative-tasks` |
| `fix/` | bug fix | `fix/tied-weights-order` |
| `exp/` | experiment scripts and measured results (no package changes) | `exp/code-math-korean-lora` |
| `docs/` | documentation only | `docs/reference-search` |
| `chore/` | CI, tooling, repo housekeeping | `chore/branch-workflow` |

Rules of thumb:

- One branch, one purpose; keep it under about a week and rebase on `main` before opening the PR (`git fetch && git rebase origin/main`).
- Merge with **squash** so `main` reads as one commit per change. Delete the branch after merging.
- Releases are tags on `main` (`v0.5.0`); fixes to an old release go on a `release/0.x` branch only if one is needed.
- Experiment results go in `experiments/RESULTS.md` in the same PR as the script that produced them, with seeds, sample sizes and the noise caveat.

## Before opening a PR

```bash
PYTHONUTF8=1 python -m pytest -q
```

New behaviour needs a test; anything that changes numerical merge results needs a check against the `lite` / `lora` engines (see `tests/test_resident.py`).

## Commit messages

Imperative, one line under 72 characters, a body only when the *why* is not obvious.
