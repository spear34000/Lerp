"""Paired per-item comparison of a candidate with its baselines, from the item outcomes `lerp search` records.

``lerp search`` writes ``items.json`` next to every score (candidates and baselines): for each task the 0/1 outcome of every item in
the scoring window. Two models scored on the same window can then be compared item by item with McNemar's exact test and an
interval for the accuracy difference (see ``lerp.statistics.mcnemar``) instead of eyeballing two accuracies.
"""
from __future__ import annotations

import json
from pathlib import Path

from .experiment import BreederError, candidate_dir, load_run
from .statistics import StatisticsError, mcnemar

ITEMS_FILE = "items.json"


def write_items(folder: Path, items: dict[str, list[int]], window: tuple[int, int], protocol_file: str) -> None:
    doc = {"window": list(window), "protocol": protocol_file, "tasks": {k: list(v) for k, v in items.items()}}
    (folder / ITEMS_FILE).write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")


def _load(folder: Path) -> dict:
    path = folder / ITEMS_FILE
    if not path.is_file():
        raise BreederError(f"{path} not found: item outcomes are recorded by `lerp search` (scores from other backends have none)")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_pairs(run: Path, gen: int, idx: int, against: list[str] | None = None, *, confidence: float = .95) -> list[dict]:
    """Candidate versus each baseline (default: base and every parent), per task and pooled over tasks."""
    spec, _ = load_run(run)
    child = _load(candidate_dir(run, gen, idx))
    names = against or ["base"] + [p.name for p in spec.parents]
    rows = []
    for name in names:
        other = _load(run / "baselines" / name)
        if child["window"] != other["window"] or child["protocol"] != other["protocol"]:
            raise BreederError(f"baseline {name} was scored on a different window or protocol; outcomes are not paired")
        pooled_a: list[int] = []
        pooled_b: list[int] = []
        for task in sorted(child["tasks"]):
            if task not in other["tasks"]:
                raise BreederError(f"baseline {name} has no outcomes for task {task}")
            try:
                rows.append({"baseline": name, "task": task, **mcnemar(child["tasks"][task], other["tasks"][task], confidence=confidence)})
            except StatisticsError as exc:
                raise BreederError(str(exc)) from exc
            pooled_a += child["tasks"][task]
            pooled_b += other["tasks"][task]
        if len(child["tasks"]) > 1:
            rows.append({"baseline": name, "task": "(all tasks pooled)", **mcnemar(pooled_a, pooled_b, confidence=confidence)})
    return rows


def format_rows(rows: list[dict]) -> str:
    lines = [f"{'vs':<12}{'task':<20}{'n':>5}{'cand':>7}{'base':>7}{'diff':>8}  {'95% CI':<16}{'+only':>6}{'-only':>6}{'p':>8}"]
    for r in rows:
        ci = f"[{r['ci']['lower']:+.3f},{r['ci']['upper']:+.3f}]"
        lines.append(f"{r['baseline']:<12}{r['task']:<20}{r['items']:>5}{r['a_accuracy']:>7.3f}{r['b_accuracy']:>7.3f}"
                     f"{r['difference']:>+8.3f}  {ci:<16}{r['a_only']:>6}{r['b_only']:>6}{r['p_exact_two_sided']:>8.3f}")
    lines.append("")
    lines.append("+only = items only the candidate gets right, -only = only the baseline; p = exact McNemar test on those. "
                 "One fixed item set, several comparisons: treat p as exploratory.")
    return "\n".join(lines)
