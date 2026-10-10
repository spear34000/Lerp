"""Segmented training must equal continuous training when (and only when) data position, learning-rate schedule and optimizer state are carried;
the prepared training stream must keep its family shares exactly."""
from __future__ import annotations

import pytest

from lerp.evolution import problems


def _rows():
    return {f: problems.pool(f, 50, 3) for f in ("add", "mul", "chain")}


MIX = {"add": 0.25, "mul": 0.25, "chain": 0.5}


# ------------------------------------------------------------------------------------------------ stream
def test_stream_keeps_the_family_shares_exactly_at_any_length():
    rows = _rows()
    for count in (16, 400, 9600):   # 9600 is far beyond the 50 rows per family: rows are repeated, the mix is not
        pairs = problems.stream(rows, MIX, count, seed=1)
        assert len(pairs) == count
        texts = {f: {problems.sft_pair(r)[0] for r in rows[f]} for f in rows}
        share = {f: sum(1 for p, _ in pairs if p in texts[f]) / count for f in rows}
        assert share == pytest.approx(MIX, abs=1e-9)


def test_stream_is_deterministic_and_depends_on_the_seed():
    rows = _rows()
    assert problems.stream(rows, MIX, 200, 1) == problems.stream(rows, MIX, 200, 1)
    assert problems.stream(rows, MIX, 200, 1) != problems.stream(rows, MIX, 200, 2)


# --------------------------------------------------------------------------------------- resumable training
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from safetensors.torch import load_file  # noqa: E402

from lerp.evolution.learning import train_adapter  # noqa: E402

COMMON = dict(lr=3e-3, seed=1, dtype="float32", batch=2, accum=1, max_len=48, ordered=True)


def _weights(path):
    return load_file(str(path / "adapter_model.safetensors"))


def _same(a, b, atol=1e-6):
    return a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=atol) for k in a)


def _setup(tmp_path):
    from test_evolution import _tiny_base
    base = tmp_path / "base"
    _tiny_base(base)
    pairs = problems.stream(_rows(), MIX, 8 * 2, seed=5)   # 8 optimizer steps of batch 2
    return str(base), pairs


def test_segments_with_carried_state_equal_one_continuous_run(tmp_path):
    base, pairs = _setup(tmp_path)
    train_adapter(base, pairs, tmp_path / "cont", rank=4, steps=8, state_out=tmp_path / "cont.opt", **COMMON)
    train_adapter(base, pairs, tmp_path / "s1", rank=4, steps=3, schedule_total=8, state_out=tmp_path / "s1.opt", **COMMON)
    train_adapter(base, pairs, tmp_path / "s2", init=tmp_path / "s1", steps=5, data_offset_steps=3, schedule_total=8, schedule_offset=3,
                  state_in=tmp_path / "s1.opt", state_out=tmp_path / "s2.opt", **COMMON)
    assert _same(_weights(tmp_path / "cont"), _weights(tmp_path / "s2"))
    a, b = torch.load(str(tmp_path / "cont.opt")), torch.load(str(tmp_path / "s2.opt"))   # Adam moments and step count carried over too
    assert all(torch.allclose(a["state"][i]["exp_avg"], b["state"][i]["exp_avg"], atol=1e-6) and float(a["state"][i]["step"]) == float(b["state"][i]["step"])
               for i in a["state"])


def test_each_missing_ingredient_changes_the_result(tmp_path):
    base, pairs = _setup(tmp_path)
    train_adapter(base, pairs, tmp_path / "cont", rank=4, steps=8, **COMMON)
    ref = _weights(tmp_path / "cont")
    train_adapter(base, pairs, tmp_path / "s1", rank=4, steps=3, schedule_total=8, state_out=tmp_path / "s1.opt", **COMMON)
    full = dict(init=tmp_path / "s1", steps=5, data_offset_steps=3, schedule_total=8, schedule_offset=3, state_in=tmp_path / "s1.opt")
    variants = {
        "wrong data offset": {**full, "data_offset_steps": 0},
        "own schedule": {k: v for k, v in full.items() if k not in ("schedule_total", "schedule_offset")},
        "fresh optimizer": {k: v for k, v in full.items() if k != "state_in"},
    }
    for name, kwargs in variants.items():
        train_adapter(base, pairs, tmp_path / name.replace(" ", "_"), rank=4, **kwargs, **COMMON)
        assert not _same(ref, _weights(tmp_path / name.replace(" ", "_")), atol=1e-5), name   # the check can fail: it is sensitive to each piece
    train_adapter(base, pairs, tmp_path / "ok", rank=4, **full, **COMMON)
    assert _same(ref, _weights(tmp_path / "ok"))


def test_restart_segments_read_consecutive_slices_of_the_stream(tmp_path):
    """Plain restart segments (own schedule, fresh optimizer) must still advance through the stream instead of rereading its start."""
    base, pairs = _setup(tmp_path)
    train_adapter(base, pairs, tmp_path / "a1", rank=4, steps=4, **COMMON)
    train_adapter(base, pairs, tmp_path / "a2", init=tmp_path / "a1", steps=4, data_offset_steps=4, **COMMON)
    train_adapter(base, pairs, tmp_path / "b2", init=tmp_path / "a1", steps=4, data_offset_steps=0, **COMMON)
    assert not _same(_weights(tmp_path / "a2"), _weights(tmp_path / "b2"), atol=1e-5)
