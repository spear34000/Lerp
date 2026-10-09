"""Gaussian-process Bayesian search over merge weights (``search: gp``).

Evolution needs dozens of expensive evaluations. A GP surrogate fitted to the scores that already exist proposes the next
batch where expected improvement is highest, which reaches good recipes with far fewer model evaluations.

Search space: one weight per module group and parent (the last parent is implied), i.e. the depth profile is flat.
Parameterising by the mean over the depth control points keeps the dimension at ``groups x (parents - 1)`` (3 for the
usual attention/mlp/other split of two parents) instead of ``groups x genes``, which is what makes a GP usable with a
handful of observations. Proposals are expanded back into the normal genome layout, so building, evaluating,
integrity checks, resume, leaderboard and export work unchanged.

Objective: the scalar ``fitness`` of each scored candidate (weighted mean minus the imbalance penalty).
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from .genetics import normalize_genome, weights_by_parent
from .spec import Spec

_LENGTHSCALES = (0.15, 0.25, 0.4, 0.6, 1.0)
_NOISES = (1e-3, 1e-2, 1e-1)
_POOL = 3000


class GPSearchError(RuntimeError):
    pass


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise GPSearchError('search: gp needs numpy: pip install -e ".[gp]"') from exc
    return np


def features(genes: Sequence[float], spec: Spec) -> list[float]:
    """Mean weight (over depth) of every parent except the last, for each gene group."""
    parents = len(spec.parents)
    out: list[float] = []
    for group in range(len(spec.gene_groups)):
        block = genes[group * spec.block_size:(group + 1) * spec.block_size]
        per_parent = weights_by_parent(block, parents, spec.genes)
        for p in range(parents - 1):
            out.append(sum(per_parent[p]) / len(per_parent[p]))
    return out


def genes_from_features(feat: Sequence[float], spec: Spec) -> list[float]:
    """Inverse of :func:`features` with a flat depth profile; always a valid normalized genome."""
    parents = len(spec.parents)
    k = parents - 1
    genes: list[float] = []
    for group in range(len(spec.gene_groups)):
        w = list(feat[group * k:(group + 1) * k])
        if parents == 2:
            block = [min(1.0, max(0.0, w[0]))] * spec.genes
        else:
            row = [max(0.0, x) for x in w] + [max(0.0, 1.0 - sum(max(0.0, x) for x in w))]
            block = [row[p] for p in range(parents) for _ in range(spec.genes)]
        genes.extend(normalize_genome(block, parents, spec.genes))
    return genes


def _sample_features(rng: random.Random, spec: Spec, count: int) -> list[list[float]]:
    parents = len(spec.parents)
    groups = len(spec.gene_groups)
    rows = []
    for _ in range(count):
        row: list[float] = []
        for _g in range(groups):
            if parents == 2:
                row.append(rng.uniform(0.02, 0.98))
            else:
                draws = [rng.gammavariate(1.0, 1.0) for _ in range(parents)]
                total = sum(draws)
                row.extend(d / total for d in draws[:-1])
        rows.append(row)
    return rows


def _rbf(np, a, b, lengthscale):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / lengthscale ** 2)


def _fit(np, X, z):
    """Pick lengthscale and noise by log marginal likelihood (z is standardized fitness)."""
    scale = math.sqrt(X.shape[1])
    best = None
    for ls in _LENGTHSCALES:
        for noise in _NOISES:
            K = _rbf(np, X, X, ls * scale) + noise * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
            lml = -0.5 * float(z @ alpha) - float(np.log(np.diag(L)).sum())
            if best is None or lml > best[0]:
                best = (lml, ls * scale, noise)
    if best is None:
        raise GPSearchError("GP fit failed (degenerate observations)")
    return best[1], best[2]


def _expected_improvement(np, X, z, pool, lengthscale, noise, xi=0.01):
    K = _rbf(np, X, X, lengthscale) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
    Ks = _rbf(np, pool, X, lengthscale)
    mu = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    sd = np.sqrt(np.maximum(1.0 - (v * v).sum(0), 1e-12))
    gain = mu - z.max() - xi
    u = gain / sd
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(u / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * u ** 2) / math.sqrt(2.0 * math.pi)
    return gain * cdf + sd * pdf, mu


def propose_batch(observations: Sequence[tuple[Sequence[float], float]], spec: Spec, count: int, seed: int) -> list[list[float]]:
    """Return ``count`` genomes (kriging-believer batch expected improvement).

    ``observations`` is a list of (genome, fitness) for every scored candidate. With fewer than two distinct
    observations the surrogate is undefined, so space-filling random proposals are returned instead.
    """
    np = _numpy()
    rng = random.Random(seed)
    if count < 1:
        raise GPSearchError("count must be positive")
    feats = [features(g, spec) for g, _ in observations]
    ys = [float(f) for _, f in observations]
    if len({tuple(round(x, 4) for x in f) for f in feats}) < 2:
        return [genes_from_features(f, spec) for f in _sample_features(rng, spec, count)]

    X = np.array(feats, dtype=float)
    y = np.array(ys, dtype=float)
    spread = y.std()
    z = (y - y.mean()) / (spread if spread > 1e-12 else 1.0)
    lengthscale, noise = _fit(np, X, z)
    pool = np.array(_sample_features(rng, spec, _POOL), dtype=float)

    proposals: list[list[float]] = []
    Xc, zc = X.copy(), z.copy()
    min_dist = 0.03 * math.sqrt(X.shape[1])
    for _ in range(count):
        ei, mu = _expected_improvement(np, Xc, zc, pool, lengthscale, noise)
        dist = np.sqrt(((pool[:, None, :] - Xc[None, :, :]) ** 2).sum(-1)).min(1)
        ei = np.where(dist < min_dist, -1.0, ei)
        pick = int(np.argmax(ei))
        proposals.append(genes_from_features(pool[pick].tolist(), spec))
        Xc = np.vstack([Xc, pool[pick]])
        zc = np.append(zc, mu[pick])  # believe the surrogate's mean so the next pick spreads out
    return proposals
