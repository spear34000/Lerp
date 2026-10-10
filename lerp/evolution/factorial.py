"""Effects of a 2x2 comparison on per-item 0/1 outcomes of four models scored on the same items, with an item bootstrap.

Cells (names follow the restart experiment): ``c1`` optimizer state kept + global schedule, ``a`` state kept + schedule restarted every segment,
``b`` fresh optimizer every segment + global schedule, ``c2`` fresh optimizer + restarted schedule. Effects are differences of accuracies:

* state effect    = ((a - c2) + (c1 - b)) / 2   (what keeping the optimizer state is worth, averaged over both schedule settings)
* schedule effect = ((b - c2) + (c1 - a)) / 2   (what one global schedule is worth, averaged over both state settings)
* interaction     = c1 - a - b + c2             (positive: the two together are worth more than the sum of their separate effects)

The bootstrap resamples ITEMS (the same item indices for all four models) and therefore measures item sampling uncertainty only, not the variation
between training seeds.
"""
from __future__ import annotations

import random
from typing import Sequence


def effects(c1: float, a: float, b: float, c2: float) -> dict[str, float]:
    return {"state": ((a - c2) + (c1 - b)) / 2, "schedule": ((b - c2) + (c1 - a)) / 2, "interaction": c1 - a - b + c2,
            "restart_cost": c1 - c2}


def _acc(outcomes: Sequence[int], idx: Sequence[int]) -> float:
    return sum(outcomes[i] for i in idx) / len(idx)


def bootstrap(cells_by_seed: Sequence[dict[str, Sequence[int]]], *, resamples: int = 2000, seed: int = 0, confidence: float = 0.95) -> dict:
    """``cells_by_seed``: per training seed, {"c1","a","b","c2"} -> 0/1 outcomes over the same items (and the same items for every seed).

    Returns per-seed point effects and, for the mean over seeds, the percentile interval of each effect over item resamples."""
    n = len(cells_by_seed[0]["c1"])
    for cells in cells_by_seed:
        if any(len(v) != n for v in cells.values()):
            raise ValueError("all models must be scored on the same items")
    full = list(range(n))
    per_seed = [effects(*(_acc(cells[k], full) for k in ("c1", "a", "b", "c2"))) for cells in cells_by_seed]
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {k: [] for k in per_seed[0]}
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        seed_effects = [effects(*(_acc(cells[k], idx) for k in ("c1", "a", "b", "c2"))) for cells in cells_by_seed]
        for name in draws:
            draws[name].append(sum(e[name] for e in seed_effects) / len(seed_effects))
    lo, hi = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    mean = {name: sum(e[name] for e in per_seed) / len(per_seed) for name in per_seed[0]}
    interval = {}
    for name, values in draws.items():
        values.sort()
        interval[name] = (values[int(lo * (resamples - 1))], values[int(hi * (resamples - 1))])
    return {"per_seed": per_seed, "mean": mean, "interval": interval, "resamples": resamples}
