"""Paired, sample-level evaluation statistics. Never treats task means as samples.

Inputs are independent JSONL files containing the same set of IDs and per-item
scores in [0,1]. A paired bootstrap estimates uncertainty across those items.
No claim of independent/identically distributed items or multiple-test control.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


class StatisticsError(ValueError):
    pass


def _sha_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def _load_samples(path: Path, id_field: str, score_field: str) -> dict[str, float]:
    data: dict[str, float] = {}
    with path.open('r', encoding='utf-8') as f:
        for line_number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise StatisticsError(f'{path}:{line_number}: invalid JSON: {e}') from e
            if not isinstance(row, dict) or id_field not in row or score_field not in row:
                raise StatisticsError(f'{path}:{line_number}: expected {id_field!r} and {score_field!r}')
            identifier = row[id_field]
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or str(identifier) == '':
                raise StatisticsError(f'{path}:{line_number}: invalid sample id')
            key = str(identifier)
            if key in data:
                raise StatisticsError(f'{path}:{line_number}: duplicate sample id {key!r}')
            raw_score = row[score_field]
            if not isinstance(raw_score, (float, int, bool)):
                raise StatisticsError(f'{path}:{line_number}: score must be numeric')
            value = float(raw_score)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise StatisticsError(f'{path}:{line_number}: score must be finite in [0,1]')
            data[key] = value
    if len(data) < 2:
        raise StatisticsError(f'{path}: need at least two aligned samples')
    return data


def _quantile(values: list[float], probability: float) -> float:
    pos = (len(values) - 1) * probability
    bottom = int(math.floor(pos))
    top = int(math.ceil(pos))
    return values[bottom] + (values[top] - values[bottom]) * (pos - bottom)


def compare_samples(candidate: Path, baseline: Path, *, id_field: str = 'id',
                    score_field: str = 'score', seed: int = 42,
                    replicates: int = 5000, confidence: float = .95) -> dict:
    if replicates < 200 or replicates > 100_000:
        raise StatisticsError('replicates must be within [200,100000]')
    if not .5 < confidence < 1:
        raise StatisticsError('confidence must be between 0.5 and 1')
    child = _load_samples(candidate, id_field, score_field)
    base = _load_samples(baseline, id_field, score_field)
    if child.keys() != base.keys():
        only_a = sorted(child.keys() - base.keys())
        only_b = sorted(base.keys() - child.keys())
        raise StatisticsError(f'Unpaired sample IDs: candidate-only={len(only_a)}, baseline-only={len(only_b)}; '
                              f'examples={only_a[:3]}/{only_b[:3]}')
    identifiers = sorted(child)
    n = len(identifiers)
    diffs = [child[i] - base[i] for i in identifiers]
    improvement = sum(diffs) / n
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    alpha = (1 - confidence) / 2
    lo = _quantile(draws, alpha)
    hi = _quantile(draws, 1 - alpha)
    # Exploratory random sign-flip test under an exchangeable symmetric null.
    # It is NOT an unqualified significance test for independent benchmarks.
    count = 0
    for _ in range(replicates):
        permuted = sum(v * (1 if rng.randrange(2) else -1) for v in diffs) / n
        if abs(permuted) >= abs(improvement) - 1e-12:
            count += 1
    wins = sum(v > 1e-12 for v in diffs)
    losses = sum(v < -1e-12 for v in diffs)
    binary = all(v in (0.0, 1.0) for v in list(child.values()) + list(base.values()))
    extra = {'mcnemar': mcnemar([int(child[i]) for i in identifiers], [int(base[i]) for i in identifiers], confidence=confidence)} if binary else {}
    return {
        **extra,
        'type': 'paired_sample_exploratory',
        'candidate': str(candidate.resolve()), 'baseline': str(baseline.resolve()),
        'candidate_sha256': _sha_file(candidate), 'baseline_sha256': _sha_file(baseline),
        'samples': n, 'candidate_mean': sum(child.values()) / n,
        'baseline_mean': sum(base.values()) / n, 'paired_mean_difference': improvement,
        'wins': wins, 'ties': n-wins-losses, 'losses': losses,
        'win_rate_excluding_ties': wins / (wins+losses) if wins+losses else None,
        'paired_bootstrap_ci': {'confidence': confidence, 'lower': lo, 'upper': hi,
                                'resamples': replicates, 'seed': seed},
        'signflip_exploratory_p_two_sided': (count+1)/(replicates+1),
        'warnings': ([f'Only {n} paired items: interval and test are unstable'] if n < 30 else []) + [
            'Items may be correlated, contaminated, or selected on this metric.',
            'Do not claim superiority solely from this exploratory CI/p-value.',
            'No multiple-comparison, dataset-leakage, or task-selection correction.',
        ],
    }


# ---------------------------------------------------------------------------------------------- binary outcomes
def mcnemar(a: list[int], b: list[int], *, confidence: float = .95) -> dict:
    """Paired comparison of two systems on the same items, each scored right (1) or wrong (0).

    Only the discordant items carry information: ``a_only`` (A right, B wrong) and ``b_only``. The p-value is the exact two-sided
    binomial test of ``a_only`` against ``a_only + b_only`` at probability 1/2 (McNemar's exact test). The interval for the accuracy
    difference A - B is the Wald interval after adding 0.5 to each cell of the 2x2 table (Agresti & Min), which stays sensible when
    one count is zero. Items are assumed independent; both systems are scored on one fixed set, so the interval says nothing about
    other items.
    """
    if len(a) != len(b):
        raise StatisticsError(f'unpaired outcomes: {len(a)} vs {len(b)} items')
    n = len(a)
    if n < 2:
        raise StatisticsError('need at least two paired items')
    if any(v not in (0, 1) for v in a) or any(v not in (0, 1) for v in b):
        raise StatisticsError('McNemar needs 0/1 outcomes')
    if not .5 < confidence < 1:
        raise StatisticsError('confidence must be between 0.5 and 1')
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    both = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    discordant = a_only + b_only
    if discordant == 0:
        p = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)) / 2 ** discordant
        p = min(1.0, 2 * tail)
    n2, x, y = n + 2.0, a_only + .5, b_only + .5
    se = math.sqrt(max(x + y - (x - y) ** 2 / n2, 0.0)) / n2
    z = _z(confidence)
    diff = (a_only - b_only) / n
    centre = (x - y) / n2
    return {
        'items': n, 'a_accuracy': sum(a) / n, 'b_accuracy': sum(b) / n, 'difference': diff,
        'both_right': both, 'a_only': a_only, 'b_only': b_only, 'both_wrong': n - both - discordant,
        'p_exact_two_sided': p,
        'ci': {'confidence': confidence, 'lower': max(-1.0, centre - z * se), 'upper': min(1.0, centre + z * se),
               'method': 'Wald with +0.5 per cell (Agresti-Min)'},
    }


def _z(confidence: float) -> float:
    """Two-sided normal quantile by bisection on erf (no scipy)."""
    target = confidence
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
