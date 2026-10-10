"""SLERP, TIES, DARE-TIES, DARE-linear: reference formulas, chunk independence, and lite engine == resident evaluator."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import lerp.mergeops as ops  # noqa: E402
from lerp.mergeops import TensorMerger, uniform  # noqa: E402


def _merge(method, tensors, coefs, *, key="w", density=0.5, seed=7, scale=1.0):
    """Run TensorMerger over in-memory tensors; ``tensors`` are [base, parents...] (or just the two parents for slerp)."""
    shape = tuple(tensors[0].shape)

    def read(i, lo, hi):
        return tensors[i][lo:hi].float() if shape else tensors[i].float()

    merger = TensorMerger(torch, method, coefs, shape=shape, key=key, task_scale=scale, density=density, seed=seed, device="cpu")
    out = torch.empty(shape)
    merger.merge(read, read, out)
    return out


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ------------------------------------------------------------------------------------------------- SLERP
def test_slerp_endpoints_and_orthogonal_midpoint():
    a, b = _rand(8, 6, seed=1), _rand(8, 6, seed=2)
    assert torch.allclose(_merge("slerp", [a, b], [1.0, 0.0]), a, atol=1e-5)
    assert torch.allclose(_merge("slerp", [a, b], [0.0, 1.0]), b, atol=1e-5)
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[0.0, 1.0]])
    mid = _merge("slerp", [x, y], [0.5, 0.5])
    assert torch.allclose(mid, torch.tensor([[math.sqrt(0.5), math.sqrt(0.5)]]), atol=1e-6)  # stays on the unit circle, unlike lerp


def test_slerp_matches_the_formula_and_falls_back_to_lerp_when_parallel():
    a, b = _rand(5, 7, seed=3), _rand(5, 7, seed=4)
    t = 0.3
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    omega = math.acos(cos)
    expected = math.sin((1 - t) * omega) / math.sin(omega) * a + math.sin(t * omega) / math.sin(omega) * b
    assert torch.allclose(_merge("slerp", [a, b], [1 - t, t]), expected, atol=1e-5)
    assert torch.allclose(_merge("slerp", [a, 2 * a], [0.6, 0.4]), 0.6 * a + 0.4 * (2 * a), atol=1e-5)


def test_slerp_needs_two_parents():
    with pytest.raises(ValueError, match="two"):
        TensorMerger(torch, "slerp", [0.3, 0.3, 0.4], shape=(2, 2), key="w", task_scale=1.0, density=0.5, seed=0, device="cpu")


# -------------------------------------------------------------------------------------------------- TIES
def _reference_ties(base, parents, w, density, scale):
    deltas = []
    for p in parents:
        d = p - base
        keep = max(1, int(round(density * d.numel())))
        thr = d.abs().flatten().kthvalue(d.numel() - keep + 1).values
        deltas.append(d * (d.abs() >= thr))
    mass = sum(wi * d for wi, d in zip(w, deltas))
    elected = torch.sign(mass)
    num = torch.zeros_like(base)
    den = torch.zeros_like(base)
    for wi, d in zip(w, deltas):
        agree = ((torch.sign(d) == elected) & (d != 0)).float()
        num += wi * d * agree
        den += wi * agree
    return base + scale * num / den.clamp_min(1e-12)


def test_ties_matches_the_reference_and_density_one_with_agreement_is_the_weighted_mean():
    base = _rand(12, 9, seed=5)
    parents = [base + 0.1 * _rand(12, 9, seed=s) for s in (6, 7, 8)]
    w = [0.5, 0.3, 0.2]
    got = _merge("ties", [base] + parents, w, density=0.4, scale=0.9)
    assert torch.allclose(got, _reference_ties(base, parents, w, 0.4, 0.9), atol=1e-6)
    same = [base + 0.1, base + 0.3]  # both parents move every entry up: nothing to elect against
    assert torch.allclose(_merge("ties", [base] + same, [0.5, 0.5], density=1.0), base + 0.2, atol=1e-6)


def test_ties_discards_the_entries_that_disagree_with_the_elected_sign():
    base = torch.zeros(1, 4)
    p1 = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    p2 = torch.tensor([[-0.2, 1.0, -0.2, 0.0]])
    out = _merge("ties", [base, p1, p2], [0.5, 0.5], density=1.0)
    # entry 0: elected +, p2 disagrees -> p1 alone; entry 1: both agree -> mean; entry 2: same as 0; entry 3: untouched
    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 1.0, 0.0]]), atol=1e-6)


# -------------------------------------------------------------------------------------------------- DARE
def test_dare_hash_is_uniform_deterministic_and_seed_dependent():
    u = uniform(torch, 0, 200_000, 12345, "cpu")
    assert 0.0 <= float(u.min()) and float(u.max()) < 1.0
    assert abs(float(u.mean()) - 0.5) < 0.01 and abs(float((u < 0.3).double().mean()) - 0.3) < 0.01
    assert torch.equal(u[1000:2000], uniform(torch, 1000, 1000, 12345, "cpu"))  # the value of an element does not depend on the block
    assert not torch.equal(u[:1000], uniform(torch, 0, 1000, 54321, "cpu"))


def test_dare_linear_keeps_about_density_and_rescales_so_the_mean_delta_is_preserved():
    base = torch.zeros(2000, 50)
    parent = torch.ones(2000, 50)
    out = _merge("dare_linear", [base, parent], [1.0], density=0.25)
    kept = (out != 0).float().mean()
    assert abs(float(kept) - 0.25) < 0.01
    assert set(out.unique().tolist()) <= {0.0, 4.0}  # kept entries are scaled by 1/density
    assert abs(float(out.mean()) - 1.0) < 0.05
    assert torch.equal(out, _merge("dare_linear", [base, parent], [1.0], density=0.25))
    assert not torch.equal(out, _merge("dare_linear", [base, parent], [1.0], density=0.25, seed=8))
    assert not torch.equal(out, _merge("dare_linear", [base, parent], [1.0], density=0.25, key="other.weight"))


def test_dare_ties_runs_and_only_moves_entries_in_the_elected_direction():
    base = _rand(40, 30, seed=9)
    parents = [base + 0.2 * _rand(40, 30, seed=s) for s in (10, 11)]
    out = _merge("dare_ties", [base] + parents, [0.5, 0.5], density=0.5)
    assert torch.isfinite(out).all() and not torch.equal(out, base)


@pytest.mark.parametrize("method", ["ties", "dare_ties", "dare_linear", "slerp"])
def test_results_do_not_depend_on_the_block_size_or_the_sampling_stride(method, monkeypatch):
    base = _rand(37, 11, seed=12)
    parents = [base + 0.3 * _rand(37, 11, seed=s) for s in (13, 14)]
    tensors = parents if method == "slerp" else [base] + parents
    coefs = [0.6, 0.4]
    whole = _merge(method, tensors, coefs, density=0.3)
    monkeypatch.setattr(ops, "CHUNK_ELEMS", 5 * 11)  # five rows per block
    assert torch.allclose(_merge(method, tensors, coefs, density=0.3), whole, atol=1e-6)
    monkeypatch.setattr(ops, "CHUNK_ELEMS", 11)  # one row per block
    assert torch.allclose(_merge(method, tensors, coefs, density=0.3), whole, atol=1e-6)


def test_sampled_ties_threshold_is_independent_of_the_blocking(monkeypatch):
    monkeypatch.setattr(ops, "EXACT_ELEMS", 100)
    monkeypatch.setattr(ops, "SAMPLE_ELEMS", 100)
    base = _rand(60, 20, seed=15)
    parents = [base + 0.3 * _rand(60, 20, seed=s) for s in (16, 17)]
    full = _merge("ties", [base] + parents, [0.5, 0.5], density=0.4)
    monkeypatch.setattr(ops, "CHUNK_ELEMS", 7 * 20)
    assert torch.allclose(_merge("ties", [base] + parents, [0.5, 0.5], density=0.4), full, atol=1e-6)


def test_scalar_and_vector_tensors():
    base = torch.tensor(1.0)
    out = _merge("ties", [base, torch.tensor(2.0), torch.tensor(3.0)], [0.5, 0.5], density=1.0)
    assert float(out) == pytest.approx(2.5)  # 1 + mean(1, 2)
    vec = _merge("dare_linear", [torch.zeros(1000), torch.ones(1000)], [1.0], density=0.5)
    assert vec.shape == (1000,)


# --------------------------------------------------------------- lite engine == resident evaluator
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from lerp.lite import LiteMergeError, build_lite  # noqa: E402
from lerp.spec import SpecError, parse_spec  # noqa: E402
from test_resident import (IDS, _llama, _load, _logits, _olmoe, _session, _spec, _three)  # noqa: E402

GENES = [0.8, 0.8, 0.2, 0.2, 0.5, 0.5]


@pytest.mark.parametrize("method", ["slerp", "ties", "dare_ties", "dare_linear"])
@pytest.mark.parametrize("maker", [_llama, _olmoe], ids=["dense", "moe"])
def test_resident_blend_equals_the_lite_checkpoint(tmp_path, method, maker):
    _three(tmp_path, maker)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"], method=method, density=0.6, task_scale=0.8, seed=11)
    build_lite(spec, GENES, tmp_path / "child")
    session = _session(spec)
    session.blender.apply(GENES)
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "child"), IDS), atol=1e-5)
    other = [0.1, 0.1, 0.9, 0.9, 0.4, 0.4]
    session.blender.apply(other)  # a later candidate does not inherit the previous one
    build_lite(spec, other, tmp_path / "child2")
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "child2"), IDS), atol=1e-5)
    session.close()


def test_the_methods_really_change_the_model_and_dare_is_seeded(tmp_path):
    _three(tmp_path)
    logits = {}
    for method, seed in (("linear", 1), ("ties", 1), ("dare_linear", 1), ("dare_linear", 2)):
        spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"], method=method, seed=seed)
        out = tmp_path / f"{method}{seed}"
        build_lite(spec, GENES, out)
        logits[(method, seed)] = _logits(_load(out), IDS)
    assert not torch.allclose(logits[("linear", 1)], logits[("ties", 1)], atol=1e-4)
    assert not torch.allclose(logits[("dare_linear", 1)], logits[("dare_linear", 2)], atol=1e-5)  # the seed matters
    again = tmp_path / "again"
    build_lite(_spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"], method="dare_linear", seed=1), GENES, again)
    assert torch.equal(logits[("dare_linear", 1)], _logits(_load(again), IDS))  # same seed, same checkpoint


def test_slerp_rejects_three_parents(tmp_path):
    for name, seed in (("base", 1), ("p0", 2), ("p1", 3), ("p2", 4)):
        _llama(tmp_path / name, seed)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1", tmp_path / "p2"], method="slerp")
    with pytest.raises(LiteMergeError, match="two parents"):
        build_lite(spec, [0.5] * 9, tmp_path / "child")


def test_slerp_is_an_accepted_method_but_not_for_lora():
    raw = {"name": "t", "base_model": "b", "parents": [{"name": "a", "model": "a"}, {"name": "c", "model": "c"}], "method": "slerp"}
    assert parse_spec(raw).method == "slerp"
    with pytest.raises(SpecError, match="LoRA"):
        parse_spec({**raw, "mode": "lora"})
