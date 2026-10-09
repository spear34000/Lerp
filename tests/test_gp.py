"""search: gp - Gaussian-process batch expected improvement over merge weights."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

pytest.importorskip("numpy")

import lerp.experiment as exp
from lerp.gp import features, genes_from_features, propose_batch, _sample_features
from lerp.spec import SpecError, parse_spec


def _spec(parents=2, groups=("attention", "mlp", "other"), genes=3, **extra):
    raw = {"name": "gp", "base_model": "./base",
           "parents": [{"name": f"p{i}", "model": f"./p{i}"} for i in range(parents)],
           "genes": genes, "gene_groups": list(groups), "search": "gp",
           "evaluation": {"tasks": {"a": {"metric": "acc,none"}, "b": {"metric": "acc,none"}}}, **extra}
    return parse_spec(raw)


@pytest.mark.parametrize("parents", [2, 3, 4])
def test_features_roundtrip_through_a_flat_depth_genome(parents):
    spec = _spec(parents=parents)
    rng = random.Random(1)
    for feat in _sample_features(rng, spec, 20):
        genes = genes_from_features(feat, spec)
        assert len(genes) == spec.genome_size
        assert all(0.0 <= g <= 1.0 for g in genes)
        back = features(genes, spec)
        assert back == pytest.approx(feat, abs=1e-6)


def test_gp_rejects_method_auto_and_unknown_search():
    with pytest.raises(SpecError, match="search: gp"):
        _spec(method="auto", search_methods=["linear", "task_arithmetic"])
    with pytest.raises(SpecError, match="Unsupported search"):
        _spec(search="annealing")


def _best_after(spec, objective, strategy, seed, init=3, batches=3, batch=3):
    rng = random.Random(seed)
    starts = _sample_features(rng, spec, init)
    obs = [(genes_from_features(f, spec), objective(f, rng)) for f in starts]
    if strategy == "gp":
        for b in range(batches):
            for genes in propose_batch(obs, spec, batch, seed * 31 + b):
                obs.append((genes, objective(features(genes, spec), rng)))
    else:
        for f in _sample_features(rng, spec, batches * batch):
            obs.append((genes_from_features(f, spec), objective(f, rng)))
    return max(f for _, f in obs)


def test_gp_search_finds_better_merge_weights_than_random_with_the_same_budget():
    spec = _spec()
    target = [0.7, 0.2, 0.4]

    def objective(feat, rng):  # smooth landscape + evaluation noise, like a small benchmark sample
        return -sum((a - b) ** 2 for a, b in zip(feat, target)) + rng.gauss(0, 0.002)

    gp = [_best_after(spec, objective, "gp", s) for s in range(12)]
    rd = [_best_after(spec, objective, "random", s) for s in range(12)]
    assert sum(gp) / len(gp) > sum(rd) / len(rd)
    assert sum(g > r for g, r in zip(gp, rd)) >= 8  # not a lucky average
    assert sum(gp) / len(gp) > -0.02  # 12 evaluations land close to the optimum (random is ~ -0.05)


def test_batch_proposals_are_distinct_and_valid():
    spec = _spec(parents=3)
    rng = random.Random(3)
    obs = []
    for f in _sample_features(rng, spec, 5):
        obs.append((genes_from_features(f, spec), -sum((x - 0.33) ** 2 for x in f)))
    batch = propose_batch(obs, spec, 4, seed=9)
    assert len(batch) == 4 and len({tuple(g) for g in batch}) == 4
    assert all(len(g) == spec.genome_size for g in batch)


def test_too_few_observations_fall_back_to_random_proposals():
    spec = _spec()
    one = [(genes_from_features([0.5, 0.5, 0.5], spec), 0.7)]
    assert len(propose_batch(one, spec, 3, seed=1)) == 3


def _toy_run(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 1}))
    parents = []
    for n in range(2):
        p = tmp_path / f"parent{n}"
        p.mkdir()
        (p / "adapter_config.json").write_text(json.dumps(dict(
            peft_type="LORA", r=2, lora_alpha=4, base_model_name_or_path=str(base), target_modules=["q_proj"],
            task_type="CAUSAL_LM", bias="none", lora_dropout=0.)))
        name = "base_model.model.model.layers.0.self_attn.q_proj"
        save_file({name + ".lora_A.weight": torch.full((2, 4), .1 + n), name + ".lora_B.weight": torch.full((4, 2), .2 + n)},
                  str(p / "adapter_model.safetensors"))
        parents.append(p)
    raw = {"name": "toy-gp", "base_model": str(base), "mode": "lora", "method": "linear", "search": "gp",
           "parents": [{"name": "one", "model": str(parents[0])}, {"name": "two", "model": str(parents[1])}],
           "genes": 3, "population": 3, "out_dtype": "float32", "gene_groups": ["attention", "mlp", "other"],
           "evaluation": {"tasks": {"a": {"metric": "acc,none"}, "b": {"metric": "acc,none"}}}}
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(raw))
    run = tmp_path / "run"
    exp.init_run(config, run)
    return run


def test_advance_uses_gp_proposals_and_writes_a_normal_generation(tmp_path):
    run = _toy_run(tmp_path)
    assert yaml.safe_load((run / "experiment.yaml").read_text())["search"] == "gp"
    exp.simulate_generation(run, 0)
    assert exp.advance(run, allow_simulated=True) == 1
    for idx in range(3):
        genome = json.loads((run / "generations" / "gen-001" / f"cand-{idx:03d}" / "genome.json").read_text())
        genes = genome["genes"]
        assert len(genes) == 9 and genome["lineage"] == []
        for group in range(3):  # flat depth profile: every control point of a group is identical
            assert len(set(genes[group * 3:(group + 1) * 3])) == 1
