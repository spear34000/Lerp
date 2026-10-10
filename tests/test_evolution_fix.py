"""Evolution-loop fixes: exact adapter combination and the graft crossover, scale-preserving rewrites, one ranking key, a founder-anchored gate."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from safetensors.torch import load_file, save_file  # noqa: E402

from lerp.evolution.archive import Organism  # noqa: E402
from lerp.evolution.combine import CombineError, combine_adapters, graft_parts  # noqa: E402
from lerp.evolution.learning import compress_adapter  # noqa: E402
from lerp.evolution.orchestrator import (EvolutionConfig, fitness, forgotten, rank_key,  # noqa: E402
                                         run_evolution)

MODULES = ["base_model.model.model.layers.0.self_attn.q_proj", "base_model.model.model.layers.1.mlp.up_proj"]
SHAPES = {MODULES[0]: (12, 10), MODULES[1]: (14, 10)}


def _write(path, tensors, r, alpha):
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha, "base_model_name_or_path": "base",
                                                          "target_modules": ["q_proj", "up_proj"], "peft_type": "LORA"}))


def _random_adapter(path, seed, r=4, alpha=8):
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for m, (out_dim, in_dim) in SHAPES.items():
        tensors[f"{m}.lora_A.weight"] = torch.randn(r, in_dim, generator=g)
        tensors[f"{m}.lora_B.weight"] = torch.randn(out_dim, r, generator=g)
    _write(path, tensors, r, alpha)
    return path


def _delta(path):
    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    t = load_file(str(path / "adapter_model.safetensors"))
    return {m: scale * t[f"{m}.lora_B.weight"] @ t[f"{m}.lora_A.weight"] for m in SHAPES}


def _same(a, b, atol=1e-5):
    return all(torch.allclose(a[m], b[m], atol=atol) for m in a)


def _add(*deltas):
    return {m: sum(d[m] for d in deltas) for m in deltas[0]}


def _scaled(d, c):
    return {m: c * v for m, v in d.items()}


# ----------------------------------------------------------------------------------------------- combine
def test_combine_is_exact_signed_and_handles_different_ranks_and_scales(tmp_path):
    a, b = _random_adapter(tmp_path / "a", 1, r=4, alpha=8), _random_adapter(tmp_path / "b", 2, r=6, alpha=6)
    out = tmp_path / "ab"
    info = combine_adapters([(a, 0.7), (b, -0.4)], out, out_scale=2.0)
    assert info["rank"] == 10
    assert _same(_delta(out), _add(_scaled(_delta(a), 0.7), _scaled(_delta(b), -0.4)))
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["lora_alpha"] / cfg["r"] == 2.0  # stored with the scale of freshly trained adapters


def test_combine_a_a_minus_a_returns_a(tmp_path):
    a = _random_adapter(tmp_path / "a", 3)
    out = tmp_path / "out"
    combine_adapters([(a, 1.0), (a, 1.0), (a, -1.0)], out)
    assert _same(_delta(out), _delta(a))


def test_blend_mode_matches_build_lora(tmp_path):
    from lerp.lora import build_lora
    from lerp.spec import parse_spec
    a, b = _random_adapter(tmp_path / "a", 4), _random_adapter(tmp_path / "b", 5)
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    for p in (a, b):  # build_lora checks the declared base against the others only; make the declared bases agree with a real path
        cfg = json.loads((p / "adapter_config.json").read_text())
        cfg["base_model_name_or_path"] = str(tmp_path / "base")
        (p / "adapter_config.json").write_text(json.dumps(cfg))
    spec = parse_spec({"name": "t", "base_model": str(tmp_path / "base"), "mode": "lora", "method": "linear", "genes": 2, "out_dtype": "float32",
                       "parents": [{"name": "a", "model": str(a)}, {"name": "b", "model": str(b)}]})
    build_lora(spec, [0.3] * spec.genome_size, tmp_path / "ref")
    combine_adapters([(a, 0.3), (b, 0.7)], tmp_path / "mine")
    assert _same(_delta(tmp_path / "ref"), _delta(tmp_path / "mine"), atol=1e-4)


def test_graft_counts_shared_ancestry_once(tmp_path):
    base_a = _random_adapter(tmp_path / "A", 6)
    # B started from A (its init) and then learned a change L; stored B = A + L
    learned = _random_adapter(tmp_path / "L", 7)
    combine_adapters([(base_a, 1.0), (learned, 1.0)], tmp_path / "B", out_scale=2.0)
    combine_adapters([(base_a, 1.0)], tmp_path / "B_init", out_scale=2.0)
    # grafting B onto its own parent A must give B, not A + B (which would count A twice)
    combine_adapters(graft_parts(base_a, tmp_path / "B", tmp_path / "B_init"), tmp_path / "graft")
    assert _same(_delta(tmp_path / "graft"), _delta(tmp_path / "B"))
    assert not _same(_delta(tmp_path / "graft"), _add(_delta(base_a), _delta(tmp_path / "B")))
    # a founder has no init: the graft reduces to A + B
    f1, f2 = _random_adapter(tmp_path / "f1", 8), _random_adapter(tmp_path / "f2", 9)
    combine_adapters(graft_parts(f1, f2, None), tmp_path / "fg")
    assert _same(_delta(tmp_path / "fg"), _add(_delta(f1), _delta(f2)))


def test_two_generations_of_grafts_do_not_grow_the_founder_part(tmp_path):
    f1, f2 = _random_adapter(tmp_path / "f1", 10), _random_adapter(tmp_path / "f2", 11)
    d1, d2 = _delta(f1), _delta(f2)
    # generation 1: two siblings graft the same founders in different order, each then learns something of its own
    for name, (p, s), seed in (("c0", (f1, f2), 12), ("c1", (f2, f1), 13)):
        combine_adapters(graft_parts(p, s, None), tmp_path / f"{name}_init")
        combine_adapters([(tmp_path / f"{name}_init", 1.0), (_random_adapter(tmp_path / f"{name}_L", seed), 1.0)], tmp_path / name)
    # generation 2 graft: c0 + (c1 - c1_init) = c0 + L1; the founders (d1 + d2) appear exactly once
    combine_adapters(graft_parts(tmp_path / "c0", tmp_path / "c1", tmp_path / "c1_init"), tmp_path / "g2")
    expected = _add(d1, d2, _delta(tmp_path / "c0_L"), _delta(tmp_path / "c1_L"))
    assert _same(_delta(tmp_path / "g2"), expected, atol=1e-4)


def test_combine_rejects_mismatches(tmp_path):
    a = _random_adapter(tmp_path / "a", 1)
    b = _random_adapter(tmp_path / "b", 2)
    cfg = json.loads((b / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = "other"
    (b / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(CombineError, match="different base"):
        combine_adapters([(a, 1.0), (b, 1.0)], tmp_path / "x")
    with pytest.raises(CombineError, match="nothing"):
        combine_adapters([], tmp_path / "y")


def test_rescaling_leaves_delta_unchanged_and_compress_honours_out_scale(tmp_path):
    a = _random_adapter(tmp_path / "a", 14, r=6, alpha=6)
    for scale in (1.0, 2.0, 4.0):
        combine_adapters([(a, 1.0)], tmp_path / f"s{scale}", out_scale=scale)
        assert _same(_delta(tmp_path / f"s{scale}"), _delta(a))
        info = compress_adapter(tmp_path / f"s{scale}", tmp_path / f"c{scale}", 6, out_scale=scale)
        cfg = json.loads((tmp_path / f"c{scale}" / "adapter_config.json").read_text())
        assert cfg["lora_alpha"] / cfg["r"] == scale and info["energy_kept"] == pytest.approx(1.0)
        assert _same(_delta(tmp_path / f"c{scale}"), _delta(a), atol=1e-4)


# ----------------------------------------------------------------------------------------------- selection
CFG = EvolutionConfig(base_model="x", founders=["add", "mul"], new_family="chain", new_skill_weight=2.0)


def _org(i, **dev):
    return Organism(id=i, generation=1, op="cross+learn", adapter=f"adapters/{i}", dev=dev)


def test_fitness_weights_the_new_skill_and_one_key_serves_selection_and_the_final_pick():
    assert fitness({"add": .9, "mul": .3, "chain": .1}, CFG) == pytest.approx((.9 + .3 + 2 * .1) / 4)
    tied = [_org("g3-c0", add=.9, mul=.3, chain=.1), _org("g2-c2", add=.9, mul=.3, chain=.1), _org("g2-c1", add=.5, mul=.5, chain=.5)]
    assert sorted(tied, key=lambda o: rank_key(o, CFG))[0].id == min(tied, key=lambda o: rank_key(o, CFG)).id == "g2-c1"
    tied = tied[:2]
    # equal fitness: the same organism wins whether the list is sorted for selection or minimised for the final pick
    assert sorted(tied, key=lambda o: rank_key(o, CFG))[0].id == min(tied, key=lambda o: rank_key(o, CFG)).id == "g2-c2"
    new_skill = _org("n", add=.5, mul=.3, chain=.3)
    old_skill = _org("o", add=.9, mul=.3, chain=.1)
    assert rank_key(new_skill, CFG) < rank_key(old_skill, CFG)  # the weight on the new skill outweighs a noisy add gain


def test_the_forgetting_gate_is_anchored_to_the_founders():
    anchor = {"add": .90, "mul": .40}
    assert forgotten({"add": .86, "mul": .37}, anchor, ["add", "mul"], 0.05) == []
    # a lineage that drifted down 0.03 per generation is still compared with the founders, so the third step is caught
    drifted = {"add": .90 - 0.03 * 3, "mul": .40}
    assert forgotten(drifted, anchor, ["add", "mul"], 0.05) == ["add"]


# ------------------------------------------------------------------------------------------- the loop itself
def _base(tmp_path):
    from test_evolution import _tiny_base
    base = tmp_path / "base"
    _tiny_base(base)
    return base


def _cfg(base, **kw):
    args = dict(base_model=str(base), device="cpu", dtype="float32", n_train=60, n_dev=8, n_test=16, founder_steps=3, child_steps=3, generations=2,
                children=2, survivors=2, rank=4, lr=3e-3, batch=2, accum=1, seed=1, max_new_tokens=4)
    args.update(kw)
    return EvolutionConfig(**args)


@pytest.mark.parametrize("mode", ["graft", "blend", "none"])
def test_every_crossover_mode_runs_keeps_start_adapters_and_stores_the_adapter_scale(tmp_path, mode):
    from lerp.evolution.archive import Archive
    base = _base(tmp_path)
    result = run_evolution(_cfg(base, crossover=mode), tmp_path / "out", log=lambda m: None)
    arc = Archive(tmp_path / "out" / "archive")
    children = [o for o in arc.organisms.values() if o.op == "cross+learn"]
    assert len(children) == 4 and result["training_steps"] == {"evolution": 12, "control": 12}
    for o in children:
        assert (arc.root / o.training["start_adapter"] / "adapter_model.safetensors").is_file()  # kept for later grafts
        cfg = json.loads((arc.root / o.adapter / "adapter_config.json").read_text())
        assert cfg["lora_alpha"] / cfg["r"] == 2.0 and cfg["r"] == 4
    survivors_and_final = [o for o in arc.organisms.values() if o.status in ("survivor", "final")]
    final = [o for o in arc.organisms.values() if o.status == "final"]
    assert len(final) == 1 and final[0].id == result["best"]
    assert final[0].id == min(survivors_and_final, key=lambda o: rank_key(o, _cfg(base))).id  # the selection key, not a second one
    assert result["steps_in_final_lineage"] + result["steps_discarded"] == 12
    assert (tmp_path / "out" / "test_items.json").is_file()
    if mode == "graft":
        assert all(len(o.parents) == 2 for o in children)
    if mode == "none":
        assert all(len(o.parents) == 1 for o in children)


def test_ablation_arms_can_share_founders_and_the_control(tmp_path):
    base = _base(tmp_path)
    first = run_evolution(_cfg(base, crossover="graft"), tmp_path / "a", log=lambda m: None)
    second = run_evolution(_cfg(base, crossover="none", founders_from=str(tmp_path / "a"), control_from=str(tmp_path / "a")),
                           tmp_path / "b", log=lambda m: None)
    from lerp.evolution.learning import file_sha256
    for name in ("g0-add", "g0-mul", "control-plain"):
        assert file_sha256(tmp_path / "a" / "archive" / "adapters" / name / "adapter_model.safetensors") == \
               file_sha256(tmp_path / "b" / "archive" / "adapters" / name / "adapter_model.safetensors")
    assert first["test_accuracy"]["control-plain"] == second["test_accuracy"]["control-plain"]
    assert first["test_accuracy"]["g0-add"] == second["test_accuracy"]["g0-add"]
    with pytest.raises(ValueError, match="cannot reuse"):
        run_evolution(_cfg(base, founder_steps=4, founders_from=str(tmp_path / "a")), tmp_path / "c", log=lambda m: None)
    with pytest.raises(ValueError, match="cannot reuse"):
        run_evolution(_cfg(base, child_steps=4, control_from=str(tmp_path / "a")), tmp_path / "d", log=lambda m: None)


def test_one_child_per_generation_a_higher_store_rank_and_reuse_with_the_same_total_steps(tmp_path):
    from lerp.evolution.archive import Archive
    base = _base(tmp_path)
    first = run_evolution(_cfg(base, crossover="graft"), tmp_path / "a", log=lambda m: None)   # 2 generations x 2 children x 3 steps = 12
    second = run_evolution(_cfg(base, crossover="graft", children=1, generations=4, store_rank=8, founders_from=str(tmp_path / "a"),
                                control_from=str(tmp_path / "a")), tmp_path / "b", log=lambda m: None)  # 4 x 1 x 3 = 12: same compute
    arc = Archive(tmp_path / "b" / "archive")
    children = [o for o in arc.organisms.values() if o.op == "cross+learn"]
    assert len(children) == 4 and all(len(o.parents) == 2 for o in children)   # one child per generation, scored on its own
    assert all(json.loads((arc.root / o.adapter / "adapter_config.json").read_text())["r"] == 8 for o in children)   # store_rank honoured
    # shared contenders are the source run's numbers, not recomputed ones
    for name in ("base", "merge-only", "control-plain", "g0-add", "g0-mul"):
        assert second["test_accuracy"][name] == first["test_accuracy"][name]
    items_a = json.loads((tmp_path / "a" / "test_items.json").read_text())["items"]
    items_b = json.loads((tmp_path / "b" / "test_items.json").read_text())["items"]
    assert items_a["control-plain"] == items_b["control-plain"] and f"evolved:{second['best']}" in items_b
    assert second["training_steps"] == {"evolution": 12, "control": 12}
