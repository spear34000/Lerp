"""Deterministic genetic search, multi-parent mixtures, and Pareto tournament selection.

With two parents the genome retains v0.1's compact [alpha] format. For 3+
parents the genome is parent-major (each parent has `points` weights). Every
control point is normalized across parents to form a convex combination.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalize_genome(genes: Sequence[float], parents: int, points: int) -> list[float]:
    if len(genes) != (points if parents == 2 else parents * points):
        raise ValueError("Unexpected genome size")
    if any(not math.isfinite(x) for x in genes):
        raise ValueError("Genome must contain only finite values")
    if parents == 2:
        return [round(clamp(v), 6) for v in genes]
    result = [0.0] * (parents * points)
    for i in range(points):
        row = [max(0.0, genes[p * points + i]) for p in range(parents)]
        total = sum(row)
        if total < 1e-12:
            row = [1.0 / parents] * parents
        else:
            row = [v / total for v in row]
        # Avoid zero-division in downstream tools: force sum to exactly 1.
        for p in range(parents - 1):
            result[p * points + i] = round(row[p], 8)
        result[(parents - 1) * points + i] = max(0.0, round(1 - sum(result[p * points + i] for p in range(parents - 1)), 8))
    return result


def weights_by_parent(genes: Sequence[float], parents: int, points: int) -> list[list[float]]:
    dna = normalize_genome(genes, parents, points)
    if parents == 2:
        return [list(dna), [round(1 - v, 8) for v in dna]]
    return [dna[p * points:(p + 1) * points] for p in range(parents)]


def initial_genomes(population: int, points: int, seed: int, parents: int = 2, *, groups: int = 1) -> list[list[float]]:
    if groups < 1:
        raise ValueError("groups must be positive")
    if groups > 1:
        blocks = [initial_genomes(population, points, seed + 1009 * group, parents) for group in range(groups)]
        return [sum((blocks[group][i] for group in range(groups)), []) for i in range(population)]
    rng = random.Random(seed)
    if parents == 2:
        samples = [[v] * points for v in (0.25, 0.50, 0.75)]
        while len(samples) < population:
            pivot = rng.uniform(0.15, 0.85)
            samples.append([clamp(pivot + rng.uniform(-0.18, 0.18)) for _ in range(points)])
        return [normalize_genome(g, parents, points) for g in samples[:population]]
    # Include near-pure parent mixtures and a uniform baseline first.
    samples = []
    for dominant in range(parents):
        samples.append([0.85 if p == dominant else 0.15 / (parents - 1)
                        for p in range(parents) for _ in range(points)])
    samples.append([1.0 / parents] * (parents * points))
    while len(samples) < population:
        pivots = [rng.gammavariate(1.0, 1.0) for _ in range(parents)]
        samples.append([max(0.001, pivots[p] + rng.uniform(-0.08, 0.08))
                        for p in range(parents) for _ in range(points)])
    return [normalize_genome(g, parents, points) for g in samples[:population]]


def crossover(a: Sequence[float], b: Sequence[float], rng: random.Random) -> list[float]:
    if len(a) != len(b):
        raise ValueError("Genome lengths differ")
    if rng.random() < 0.25:
        t = rng.uniform(0.15, 0.85)
        return [clamp(t * x + (1 - t) * y) for x, y in zip(a, b)]
    return [x if rng.random() < 0.5 else y for x, y in zip(a, b)]


def mutate(genes: Sequence[float], sigma: float, rng: random.Random) -> list[float]:
    result = []
    for value in genes:
        if rng.random() < 0.65:
            value += rng.gauss(0.0, sigma)
        result.append(round(clamp(value), 8))
    return result


def dominates(a: dict, b: dict, tasks: Sequence[str]) -> bool:
    """Strict Pareto dominance for maximization of all task metrics."""
    left = a["score"]["metrics"]
    right = b["score"]["metrics"]
    return all(left[t] >= right[t] for t in tasks) and any(left[t] > right[t] for t in tasks)


def pareto_fronts(entries: list[dict], tasks: Sequence[str]) -> list[list[dict]]:
    if not entries:
        return []
    # Using integer indices avoids collisions with lineage IDs.
    n = len(entries)
    domination_counts = [0] * n
    beaten: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(entries[i], entries[j], tasks):
                beaten[i].append(j)
                domination_counts[j] += 1
            elif dominates(entries[j], entries[i], tasks):
                beaten[j].append(i)
                domination_counts[i] += 1
    current = [i for i in range(n) if domination_counts[i] == 0]
    fronts: list[list[dict]] = []
    while current:
        fronts.append([entries[i] for i in current])
        next_ids = []
        for i in current:
            for j in beaten[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_ids.append(j)
        current = next_ids
    return fronts


def crowding_distance(front: list[dict], tasks: Sequence[str]) -> dict[str, float]:
    """NSGA-II crowding; extremes on every objective are preferred."""
    distances = {e["id"]: 0.0 for e in front}
    if len(front) < 3:
        return {key: math.inf for key in distances}
    for task in tasks:
        ordered = sorted(front, key=lambda e: (e["score"]["metrics"][task], e["id"]))
        low = ordered[0]["score"]["metrics"][task]
        high = ordered[-1]["score"]["metrics"][task]
        if high <= low:
            continue
        distances[ordered[0]["id"]] = math.inf
        distances[ordered[-1]["id"]] = math.inf
        for i in range(1, len(ordered) - 1):
            if math.isfinite(distances[ordered[i]["id"]]):
                left = ordered[i - 1]["score"]["metrics"][task]
                right = ordered[i + 1]["score"]["metrics"][task]
                distances[ordered[i]["id"]] += (right - left) / (high - low)
    return distances


def rank_entries(entries: list[dict], strategy: str, tasks: Sequence[str] = ()) -> list[dict]:
    if strategy == "weighted":
        return sorted(entries, key=lambda e: (-e["score"]["fitness"], e["id"]))
    if strategy != "pareto" or len(tasks) < 2:
        raise ValueError("Pareto ranking needs at least two tasks")
    ranked = []
    for front in pareto_fronts(entries, tasks):
        crowding = crowding_distance(front, tasks)
        ranked.extend(sorted(front, key=lambda e: (-crowding[e["id"]], -e["score"]["fitness"], e["id"])))
    return ranked


def select_parents(entries: list[dict], rng: random.Random, *, strategy: str = "weighted", tasks: Sequence[str] = ()) -> tuple[dict, dict]:
    if not entries:
        raise ValueError("No scored parents")
    ranked = rank_entries(entries, strategy, tasks)
    weights = [1.0 / (i + 1) ** 1.5 for i in range(len(ranked))]
    x, y = rng.choices(ranked, weights=weights, k=2)
    if len(ranked) > 1 and x["id"] == y["id"]:
        y = ranked[(ranked.index(x) + 1) % len(ranked)]
    return x, y


def breed_generation(
    entries: list[dict], population: int, sigma: float, seed: int, *,
    parents: int = 2, points: int | None = None,
    strategy: str = "weighted", tasks: Sequence[str] = (), groups: int = 1,
) -> list[tuple[list[float], list[str]]]:
    rng = random.Random(seed)
    if points is None:
        points = len(entries[0]["genes"]) // (groups * (1 if parents == 2 else parents))
    block_size = points if parents == 2 else parents * points
    if groups < 1 or any(len(e["genes"]) != block_size * groups for e in entries):
        raise ValueError("Incompatible grouped genomes")
    children = []
    existing = {tuple(e["genes"]) for e in entries}
    seen: set[tuple[float, ...]] = set()
    for _ in range(population):
        for attempt in range(30):
            x, y = select_parents(entries, rng, strategy=strategy, tasks=tasks)
            raw = mutate(crossover(x["genes"], y["genes"], rng), sigma, rng)
            genes = sum((normalize_genome(raw[g * block_size:(g + 1) * block_size], parents, points)
                         for g in range(groups)), [])
            fingerprint = tuple(genes)
            # Don't generate checkpoints already tested in current/previous generations.
            if fingerprint not in existing and fingerprint not in seen or attempt == 29:
                seen.add(fingerprint)
                children.append((genes, [x["id"], y["id"]]))
                break
    return children


def demo_objective(genes: Sequence[float], *, parents: int = 2, points: int | None = None, groups: int = 1) -> dict[str, float]:
    """FAKE toy objective, deliberately not a real-model benchmark."""
    if points is None:
        points = len(genes) // (groups * (1 if parents == 2 else parents))
    group_scores = []
    block_size = points if parents == 2 else parents * points
    for group in range(groups):
        group_scores.append(weights_by_parent(genes[group * block_size:(group + 1) * block_size], parents, points)[0])
    # Toy scores depend on every group, so grouped genomes exercise selection.
    weights = [[sum(values[i] for values in group_scores) / groups for i in range(points)]]
    a = weights[0]
    # Two conflicting metrics create a genuinely nontrivial Pareto frontier.
    code_target = [0.80 - 0.25 * i / max(1, points - 1) for i in range(points)]
    reason_target = [0.20 + 0.25 * i / max(1, points - 1) for i in range(points)]
    score_a = max(0.0, 1.0 - sum((v - t) ** 2 for v, t in zip(a, code_target)) / points)
    score_b = max(0.0, 1.0 - sum((v - t) ** 2 for v, t in zip(a, reason_target)) / points)
    return {"toy_code": round(score_a, 6), "toy_reason": round(score_b, 6)}
