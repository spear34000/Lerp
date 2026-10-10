"""Matched-conditions evolution arm: constant training rank, exact data mix, one global schedule across the lineage, cost accounting."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

import lerp.evolution.matched as matched  # noqa: E402
from lerp.evolution import problems  # noqa: E402
from lerp.evolution.archive import Archive  # noqa: E402
from lerp.evolution.combine import combine_adapters  # noqa: E402
from lerp.evolution.learning import train_adapter  # noqa: E402
from lerp.evolution.matched import MatchedConfig, run_matched  # noqa: E402

FAMS = ["add", "mul", "chain"]


def _workdir(tmp_path):
    from test_evolution import _tiny_base
    base = tmp_path / "base"
    _tiny_base(base)
    work = tmp_path / "work"
    for fam in FAMS:
        problems.write_jsonl(problems.pool(fam, 40, 1), work / "data" / f"{fam}_train.jsonl")
        problems.write_jsonl(problems.pool(fam, 60, 2), work / "data" / f"{fam}_eval.jsonl")
        problems.write_jsonl(problems.pool(fam, 10, 3), work / "data_fresh" / f"{fam}_eval.jsonl")
    pairs = {f: [problems.sft_pair(r) for r in problems.pool(f, 16, 1)] for f in FAMS}
    kw = dict(lr=3e-3, seed=1, dtype="float32", batch=2, accum=1, max_len=48)
    for f in ("add", "mul"):
        train_adapter(str(base), pairs[f], work / "founders" / f"g0-{f}", rank=4, steps=3, **kw)
    combine_adapters([(work / "founders" / "g0-add", 0.5), (work / "founders" / "g0-mul", 0.5)], work / "start", out_scale=2.0)
    train_adapter(str(base), pairs["chain"], work / "controls" / "s1", init=work / "start", steps=3, **kw)
    return str(base), str(work)


def _cfg(base, work, **kw):
    args = dict(base_model=base, workdir=work, device="cpu", dtype="float32", n_dev=6, n_fresh=8, generations=3, children=2, survivors=2,
                child_steps=3, rank=8, lr=3e-3, batch=2, accum=1, max_new_tokens=4, seed=1)
    args.update(kw)
    return MatchedConfig(**args)


def test_matched_arm_keeps_the_conditions_of_the_control(tmp_path, monkeypatch):
    base, work = _workdir(tmp_path)
    calls = []
    real = matched.train_adapter

    def spy(*args, **kwargs):
        calls.append({k: kwargs.get(k) for k in ("steps", "ordered", "data_offset_steps", "schedule_total", "schedule_offset", "state_in", "state_out")})
        return real(*args, **kwargs)

    monkeypatch.setattr(matched, "train_adapter", spy)
    result = run_matched(_cfg(base, work), tmp_path / "out", log=lambda m: None)
    # ONE global schedule across the lineage; each generation reads the next slice of the stream; no optimizer state moves between organisms
    assert len(calls) == 6
    for i, c in enumerate(calls):
        g = i // 2 + 1
        assert c == {"steps": 3, "ordered": True, "data_offset_steps": (g - 1) * 3, "schedule_total": 9, "schedule_offset": (g - 1) * 3,
                     "state_in": None, "state_out": None}
    arc = Archive(tmp_path / "out" / "archive")
    children = [o for o in arc.organisms.values() if o.op == "cross+learn"]
    assert len(children) == 6
    for o in children:   # the stored (= trained) rank is the control's rank for every child, whatever the parents' ranks were
        assert json.loads((arc.root / o.adapter / "adapter_config.json").read_text())["r"] == 8
        assert json.loads((arc.root / o.adapter / "adapter_config.json").read_text())["lora_alpha"] == 16
    first = [o for o in children if o.generation == 1]
    assert all(o.training["pre_training_compression"]["rank_in"] == 8 and o.training["pre_training_compression"]["energy_kept"] == 1.0 for o in first)
    later = [o for o in children if o.generation >= 2]
    for o in later:   # survivors may include a rank-4 founder, so the crossed rank is 8 + 8, 4 + 8 or 4 + 4: compressed only when it exceeds the training rank
        comp = o.training["pre_training_compression"]
        assert comp["rank_out"] == min(comp["rank_in"], 8) and 0.0 < comp["energy_kept"] <= 1.0
    # cost: all children's steps equal the control's, selection evaluations are counted
    cost = result["cost"]
    assert cost["training_steps"] == cost["control_training_steps"] == 18
    assert cost["dev_evaluations"] == 2 + 6 and cost["dev_items_scored"] == (2 + 6) * 6 * 3
    assert cost["steps_in_final_lineage"] + cost["steps_discarded"] == 18
    assert set(result["fresh_accuracy"]) >= {"control", "start", "g0-add", "g0-mul", f"evolved:{result['best']}"}
    assert set(result["vs_control"]) == set(FAMS)
    assert (tmp_path / "out" / "items.json").is_file() and not (tmp_path / "out" / "tmp").exists()


def test_children_of_one_generation_use_different_streams_and_the_first_generation_crosses_the_founders_evenly(tmp_path):
    base, work = _workdir(tmp_path)
    out = tmp_path / "out"
    run_matched(_cfg(base, work, generations=1), out, log=lambda m: None)
    arc = Archive(out / "archive")
    c0, c1 = arc.organisms["g1-c0"], arc.organisms["g1-c1"]
    assert c0.parents == c1.parents and set(c0.parents) == {"g0-add", "g0-mul"} and c0.weights == [0.5, 0.5]
    assert c0.adapter_sha256 != c1.adapter_sha256   # same blend, different data streams


def test_a_cross_wider_than_the_training_rank_is_compressed_before_training_and_the_loss_is_recorded(tmp_path):
    base, work = _workdir(tmp_path)   # founders have rank 4: crossing two of them gives rank 8, above the training rank 4
    result = run_matched(_cfg(base, work, rank=4, generations=2), tmp_path / "out", log=lambda m: None)
    arc = Archive(tmp_path / "out" / "archive")
    for o in (x for x in arc.organisms.values() if x.op == "cross+learn"):
        comp = o.training["pre_training_compression"]
        assert comp["rank_in"] == 8 and comp["rank_out"] == 4 and 0.0 < comp["energy_kept"] < 1.0
        assert json.loads((arc.root / o.adapter / "adapter_config.json").read_text())["r"] == 4
    assert [c["rank_out"] for c in result["cost"]["compressions"]] == [4] * 4
