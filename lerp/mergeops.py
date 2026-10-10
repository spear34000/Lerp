"""SLERP, TIES, DARE-TIES and DARE-linear for one tensor, shared by the ``lite`` engine and the resident evaluator.

Both engines call :class:`TensorMerger`, so a candidate scored in place and the checkpoint written by ``lerp build`` are the same
numbers. Everything is chunk-independent: a tensor is processed in row blocks (``chunk_rows``), but anything that needs the whole
tensor is computed first and does not depend on how the rows are blocked:

* SLERP: the dot product and norms of the two parents, accumulated in float64 over the blocks.
* TIES: per parent, the magnitude threshold that keeps the largest ``density`` fraction of the task vector. Exact for tensors up to
  ``EXACT_ELEMS`` elements; above that the threshold is estimated from every ``stride``-th row (a fixed function of the shape).
* DARE: the keep/drop decision of an element is a hash of (seed, tensor name, parent, element index), not a random stream, so it is
  identical on CPU and accelerator, for any block size, and for the fused-expert layout.

Definitions (task vector d_i = P_i - base, weights w_i from the genome):

* ``slerp`` (two parents): ``a * P1 + b * P2`` with ``a = sin((1-t) w) / sin w``, ``b = sin(t w) / sin w``, ``w`` the angle between the
  flattened parents and ``t`` the weight of the second parent; plain linear interpolation when the parents are almost parallel.
* ``ties``: trim each d_i to its largest-magnitude ``density`` fraction, elect the sign of the weighted sum, average the entries that
  agree with it (weights normalized over the agreeing parents): ``base + task_scale * merged``.
* ``dare_ties``: DARE (drop each entry with probability ``1 - density``, rescale the rest by ``1 / density``) instead of trimming, then the TIES election.
* ``dare_linear``: DARE, then ``base + task_scale * sum_i w_i d_i``.

These follow the papers; they are not bit-identical to MergeKit's implementations.
"""
from __future__ import annotations

import math
import zlib
from typing import Callable, Sequence

BASE_FREE = ("linear", "slerp")
NEW_METHODS = ("slerp", "ties", "dare_ties", "dare_linear")
CHUNK_ELEMS = 32 * 1024 * 1024
EXACT_ELEMS = 64 * 1024 * 1024
SAMPLE_ELEMS = 16 * 1024 * 1024
_PARALLEL = 0.9995


def needs_base(method: str) -> bool:
    return method not in BASE_FREE


def chunk_rows(shape: Sequence[int]) -> int:
    per_row = max(1, math.prod(shape[1:]))
    return max(1, CHUNK_ELEMS // per_row)


def _hash_seed(seed: int, key: str, parent: int) -> int:
    return (zlib.crc32(f"{seed}|{key}|{parent}".encode("utf-8")) * 2654435761 + parent) & 0xFFFFFFFF


def uniform(torch, start: int, count: int, mix: int, device):
    """Deterministic U[0,1) values for element indexes start .. start+count-1 (32-bit integer hash held in int64, never overflowing)."""
    if start + count >= 2 ** 32:
        raise ValueError("tensors above 2^32 elements are not supported by the DARE hash")
    x = torch.arange(start, start + count, dtype=torch.int64, device=device) ^ mix
    for _ in range(2):
        x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return x.to(torch.float64) / 4294967296.0


class TensorMerger:
    """Merges one tensor. ``read(i, lo, hi)`` returns rows lo:hi of source ``i`` as float32 on the compute device; ``read_cpu`` the same on
    the CPU (used for order statistics). Sources are ``[base, parent_1 ... parent_n]`` for base-relative methods, ``[parent_1 ... parent_n]``
    for ``slerp``."""

    def __init__(self, torch, method: str, coefs: Sequence[float], *, shape: Sequence[int], key: str, task_scale: float,
                 density: float, seed: int, device):
        if method not in NEW_METHODS:
            raise ValueError(f"unsupported method {method}")
        if method == "slerp" and len(coefs) != 2:
            raise ValueError("slerp merges exactly two parents")
        self.torch, self.method, self.coefs = torch, method, list(coefs)
        self.shape, self.key, self.task_scale, self.density, self.seed, self.device = tuple(shape), key, task_scale, density, seed, device
        self.rows = self.shape[0] if self.shape else 1
        self.per_row = max(1, math.prod(self.shape[1:])) if self.shape else 1
        self.step = chunk_rows(self.shape) if self.shape else 1
        self.slerp_ab: tuple[float, float] | None = None
        self.thresholds: list[float] = []

    # -- statistics over the whole tensor -------------------------------------------------------------
    def prepare(self, read_cpu: Callable[[int, int, int], object]) -> None:
        torch = self.torch
        if self.method == "slerp":
            dot = n1 = n2 = 0.0
            for lo in range(0, self.rows, self.step):
                hi = min(self.rows, lo + self.step)
                a, b = read_cpu(0, lo, hi).double(), read_cpu(1, lo, hi).double()
                dot += float((a * b).sum())
                n1 += float((a * a).sum())
                n2 += float((b * b).sum())
            self.slerp_ab = _slerp_coefficients(dot, n1, n2, self.coefs[1])
        elif self.method == "ties":
            numel = self.rows * self.per_row
            stride = 1 if numel <= EXACT_ELEMS else max(1, math.ceil(numel / SAMPLE_ELEMS))
            for parent in range(1, len(self.coefs) + 1):
                pieces = []
                for lo in range(0, self.rows, self.step):
                    hi = min(self.rows, lo + self.step)
                    first = (-lo) % stride  # sampled rows are the global rows divisible by the stride, whatever the blocking
                    delta = (read_cpu(parent, lo, hi) - read_cpu(0, lo, hi)).abs()
                    pieces.append(delta[first::stride].reshape(-1) if self.shape else delta.reshape(-1))
                sample = torch.cat(pieces)
                keep = max(1, int(round(self.density * sample.numel())))
                self.thresholds.append(float(sample.kthvalue(sample.numel() - keep + 1).values))

    # -- one block of rows ----------------------------------------------------------------------------
    def combine(self, read: Callable[[int, int, int], object], lo: int, hi: int):
        torch = self.torch
        if self.method == "slerp":
            a, b = self.slerp_ab
            return read(0, lo, hi) * a + read(1, lo, hi) * b
        base = read(0, lo, hi)
        count = (hi - lo) * self.per_row
        deltas = []
        for parent in range(1, len(self.coefs) + 1):
            delta = read(parent, lo, hi) - base
            if self.method == "ties":
                delta = delta * (delta.abs() >= self.thresholds[parent - 1])
            else:
                u = uniform(torch, lo * self.per_row, count, _hash_seed(self.seed, self.key, parent), self.device)
                keep = (u < self.density).reshape(delta.shape)
                delta = delta * keep / self.density
            deltas.append(delta)
        if self.method == "dare_linear":
            merged = torch.zeros_like(base)
            for w, d in zip(self.coefs, deltas):
                merged.add_(d, alpha=w)
        else:
            mass = torch.zeros_like(base)
            for w, d in zip(self.coefs, deltas):
                mass.add_(d, alpha=w)
            elected = torch.sign(mass)
            merged = torch.zeros_like(base)
            total = torch.zeros_like(base)
            for w, d in zip(self.coefs, deltas):
                agree = ((torch.sign(d) == elected) & (d != 0)).to(base.dtype)
                merged.add_(d * agree, alpha=w)
                total.add_(agree, alpha=w)
            merged = merged / total.clamp_min(1e-12)
        return base + merged * self.task_scale

    def merge(self, read, read_cpu, out) -> None:
        """Whole tensor into ``out`` (a tensor of the target dtype with this shape), block by block."""
        self.prepare(read_cpu)
        if not self.shape:
            out.copy_(self.combine(read, 0, 1).reshape(()).to(out.dtype))
            return
        for lo in range(0, self.rows, self.step):
            hi = min(self.rows, lo + self.step)
            out[lo:hi].copy_(self.combine(read, lo, hi).to(out.dtype))


def _slerp_coefficients(dot: float, n1: float, n2: float, t: float) -> tuple[float, float]:
    denom = math.sqrt(n1) * math.sqrt(n2)
    if denom < 1e-30:
        return 1.0 - t, t
    cos = max(-1.0, min(1.0, dot / denom))
    if abs(cos) > _PARALLEL:
        return 1.0 - t, t
    omega = math.acos(cos)
    sin = math.sin(omega)
    return math.sin((1.0 - t) * omega) / sin, math.sin(t * omega) / sin
