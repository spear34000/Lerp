from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import lerp.experiment as exp
from lerp.compat import check_compatibility, resolve_spec_paths
from lerp.genetics import (
    breed_generation, crowding_distance, initial_genomes, normalize_genome,
    pareto_fronts, rank_entries, weights_by_parent,
)
from lerp.merge import mergekit_config
from lerp.report import create_report
from lerp.spec import SpecError, parse_spec

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/multi_parent_demo.yaml"


def new_run(tmp_path):
    run = tmp_path / "run"
    exp.init_run(DEMO, run)
    return run


def make_entry(id, a, b):
    return {"id": id, "genes": [a, b], "score": {"fitness": (a + b) / 2, "metrics": {"coding": a, "reasoning": b}}}


def test_multi_parent_genome_simplex():
    parents, points = 3, 8
    genomes = initial_genomes(12, points, 2026, parents)
    assert len(genomes) == 12
    for g in genomes:
        assert len(g) == parents * points
        weights = weights_by_parent(g, parents, points)
        assert all(abs(sum(weights[p][i] for p in range(parents)) - 1) < 1e-6 for i in range(points))
        assert all(0 <= v <= 1 for v in g)
    assert genomes == initial_genomes(12, points, 2026, parents)


def test_normalization_zero_genome():
    assert normalize_genome([0] * 6, 3, 2) == pytest.approx([1/3]*6, abs=1e-7)
    with pytest.raises(ValueError):
        normalize_genome([math.nan] * 6, 3, 2)


@pytest.mark.parametrize("parents", [3, 4, 6])
def test_mergekit_multi_parent_recipes(parents):
    raw = {
        "name": "many", "base_model": "foo/base", "method": "dare_ties", "genes": 4,
        "parents": [{"name": f"p{i}", "model": f"foo/model{i}"} for i in range(parents)],
    }
    spec = parse_spec(raw)
    genome = initial_genomes(4, spec.genes, 1, parents)[-1]
    cfg = mergekit_config(spec, genome)
    assert len(cfg["models"]) == parents
    assert cfg["base_model"] == spec.base_model
    assert all(model["parameters"]["density"] == .5 for model in cfg["models"])
    for pos in range(spec.genes):
        assert sum(model["parameters"]["weight"][pos] for model in cfg["models"]) == pytest.approx(1)


def test_7_parents_rejected():
    with pytest.raises(SpecError, match="two and six"):
        parse_spec({"name": "no", "base_model": "base", "parents": [
            {"name": f"p{i}", "model": f"model{i}"} for i in range(7)]})


def test_pareto_fronts_tradeoffs():
    entries = [
        make_entry("a", .90, .10), make_entry("b", .30, .90),
        make_entry("c", .75, .75), make_entry("d", .60, .60),
        make_entry("e", .10, .10),
    ]
    fronts = pareto_fronts(entries, ["coding", "reasoning"])
    assert {x["id"] for x in fronts[0]} == {"a", "b", "c"}
    assert {x["id"] for x in fronts[1]} == {"d"}
    assert {x["id"] for x in fronts[2]} == {"e"}
    ordered = rank_entries(entries, "pareto", ["coding", "reasoning"])
    assert {x["id"] for x in ordered[:3]} == {"a", "b", "c"}
    distance = crowding_distance(fronts[0], ["coding", "reasoning"])
    assert math.isinf(distance["a"]) and math.isinf(distance["b"])


def test_pareto_breeding_is_deterministic_and_new():
    originals = [make_entry(f"x{i}", i / 10, (10-i)/10) for i in range(1, 8)]
    children = breed_generation(originals, 8, .12, 19, strategy="pareto", tasks=["coding", "reasoning"], parents=2, points=2)
    assert children == breed_generation(originals, 8, .12, 19, strategy="pareto", tasks=["coding", "reasoning"], parents=2, points=2)
    assert len({tuple(genes) for genes, _ in children}) == 8
    assert all(len(lineage) == 2 for _, lineage in children)


def test_report_and_demo_pareto_flow(tmp_path):
    run = new_run(tmp_path)
    assert exp.load_run(run)[0].selection == "pareto"
    assert exp.load_run(run)[0].genome_size == 24
    exp.simulate_generation(run, 0)
    assert len(exp.leaderboard(run, pareto=True)) == 12
    with pytest.raises(exp.BreederError, match="Simulated"):
        exp.advance(run)
    assert exp.advance(run, allow_simulated=True) == 1
    exp.simulate_generation(run, 1)
    report = create_report(run, tmp_path / "output.html")
    html = report.read_text()
    assert "Pareto" in html and "SIMULATED DATA" in html
    assert "g001-c011" in html and "Gen 1" not in html or "Generation 1" in html
    assert len(exp.leaderboard(run)) == 24
    assert exp.leaderboard(run, include_simulated=False) == []


def test_recipe_and_config_tampering_detected(tmp_path, monkeypatch):
    run = new_run(tmp_path)
    (run / "experiment.yaml").write_text((run / "experiment.yaml").read_text() + "\n# mutation\n")
    with pytest.raises(exp.BreederError, match="changed"):
        exp.load_run(run)
    run2 = tmp_path / "run2"
    exp.init_run(DEMO, run2)
    merge = exp.candidate_dir(run2, 0, 0) / "merge.yaml"
    merge.write_text(merge.read_text() + "\n# mutated\n")
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)
    with pytest.raises(exp.BreederError, match="modified"):
        exp.build_candidate(run2, 0, 0)


def _mock_merger(cmd, log):
    if cmd[0] == "mergekit-yaml":
        path = Path(cmd[2]); path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"fake")
    else:
        path = Path(cmd[cmd.index("--output_path") + 1]); path.mkdir(parents=True, exist_ok=True)
        (path / "result.json").write_text(json.dumps({"results": {
            "coding": {"acc_norm,none": .82}, "reasoning": {"acc_norm,none": .61},
        }}))


def test_full_mocked_model_eval_and_baselines(tmp_path, monkeypatch):
    run = new_run(tmp_path)
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)
    monkeypatch.setattr(exp, "_execute_logged", _mock_merger)
    model = exp.build_candidate(run, 0, 0)
    assert (model / "model.safetensors").is_file()
    assert not (model.parent / ".model.partial").exists()
    value = exp.evaluate_candidate(run, 0, 0)
    assert value["status"] == "smoke_test"
    assert value["evidence"] and (model.parent / value["evidence"]).is_file()
    parent = exp.evaluate_baseline(run, "math")
    assert parent["fitness"] == value["fitness"]
    results = exp.comparisons(run)
    assert results[0]["delta"] == 0
    with pytest.raises(exp.BreederError, match="already scored"):
        exp.evaluate_baseline(run, "math")
    destination = exp.export_recipe(run, tmp_path / "exported")
    assert (destination / "export.json").exists()
    assert not (destination / "model.safetensors").exists()
    assert "No trained/merged weights" in (destination / "MODEL_CARD_DRAFT.md").read_text()


def test_evaluation_fails_preserving_partial_then_retry(tmp_path, monkeypatch):
    run = new_run(tmp_path)
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)
    monkeypatch.setattr(exp, "_execute_logged", _mock_merger)
    exp.build_candidate(run, 0, 0)
    calls = []

    def failing(cmd, log):
        calls.append(cmd[0]); raise exp.BreederError("fake tool failure")

    monkeypatch.setattr(exp, "_execute_logged", failing)
    with pytest.raises(exp.BreederError, match="fake tool failure"):
        exp.evaluate_candidate(run, 0, 0)
    assert (exp.candidate_dir(run, 0, 0) / ".evaluation.partial").exists()
    with pytest.raises(exp.BreederError, match="Interrupted"):
        exp.evaluate_candidate(run, 0, 0)
    monkeypatch.setattr(exp, "_execute_logged", _mock_merger)
    score = exp.evaluate_candidate(run, 0, 0, retry_partial=True)
    assert score["source"] == "lm_eval"
    assert not (exp.candidate_dir(run, 0, 0) / ".evaluation.partial").exists()


def test_build_fails_safely_then_retry(tmp_path, monkeypatch):
    run = new_run(tmp_path)
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)

    def incomplete(cmd, log):
        Path(cmd[2]).mkdir(parents=True)
        (Path(cmd[2]) / "config.json").write_text("{}")

    monkeypatch.setattr(exp, "_execute_logged", incomplete)
    with pytest.raises(exp.BreederError, match="weight tensors"):
        exp.build_candidate(run, 0, 0)
    with pytest.raises(exp.BreederError, match="Interrupted"):
        exp.build_candidate(run, 0, 0)
    monkeypatch.setattr(exp, "_execute_logged", _mock_merger)
    assert exp.build_candidate(run, 0, 0, retry_partial=True).exists()
    with pytest.raises(exp.BreederError, match="already exists"):
        exp.build_candidate(run, 0, 0)


def test_manual_score_unverified_and_reject_nan(tmp_path):
    run = new_run(tmp_path)
    with pytest.raises(exp.BreederError, match="finite"):
        exp.record_score(run, 0, 0, {"coding": float("nan"), "reasoning": .7}, source="manual")
    result = exp.record_score(run, 0, 0, {"coding": .8, "reasoning": .7}, source="manual")
    assert result["status"] == "unverified_manual"
    assert len(exp.leaderboard(run, include_simulated=False)) == 1


def test_pareto_needs_2_objectives():
    with pytest.raises(SpecError, match="at least two"):
        parse_spec({"name": "single", "base_model": "b", "parents": [
            {"name": "a", "model": "ma"}, {"name": "b", "model": "mb"}],
            "selection": "pareto", "evaluation": {"tasks": {"one": {}}}})


def test_num_fewshot_and_chat_template_passed(tmp_path, monkeypatch):
    raw = {
        "name": "test", "base_model": "x/base", "parents": [
            {"name": "a", "model": "x/a"}, {"name": "b", "model": "x/b"}],
        "evaluation": {"num_fewshot": 3, "apply_chat_template": True,
                        "tasks": {"coding": {"metric": "acc_norm,none"}, "reasoning": {"metric": "acc_norm,none"}}},
    }
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    run = tmp_path / "run"; exp.init_run(path, run)
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)
    commands = []

    def fake_exec(cmd, log):
        commands.append(cmd)
        _mock_merger(cmd, log)

    monkeypatch.setattr(exp, "_execute_logged", fake_exec)
    exp.build_candidate(run, 0, 0)
    exp.evaluate_candidate(run, 0, 0)
    assert "--num_fewshot" in commands[-1]
    assert "--apply_chat_template" in commands[-1]
    assert exp.load_candidate(run, 0, 0)["score"]["status"] == "measured"


def test_safetensors_shape_preflight(tmp_path):
    np = pytest.importorskip("numpy")
    safe = pytest.importorskip("safetensors.numpy")
    for name, width in (("base", 4), ("a", 4), ("b", 5)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 4, "num_hidden_layers": 1}))
        (folder / "tokenizer.json").write_text("{}")
        safe.save_file({"model.layers.0.weight": np.zeros((width, 4), dtype=np.float32)}, str(folder / "model.safetensors"))
    cfg = parse_spec({"name": "t", "base_model": "./base", "parents": [
        {"name": "a", "model": "./a"}, {"name": "b", "model": "./b"}]})
    report = check_compatibility(resolve_spec_paths(cfg, tmp_path))
    assert not report.ok
    assert "shape mismatch" in " ".join(report.errors)


def test_cli_end_to_end_demo(tmp_path):
    run = tmp_path / "demo"
    output = tmp_path / "dashboard.html"
    def cli(*args):
        return subprocess.run([sys.executable, "-m", "lerp", *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)
    assert cli("init", "-c", DEMO, "-o", run).returncode == 0
    assert cli("simulate", "-r", run).returncode == 0
    assert cli("advance", "-r", run, "--allow-simulated").returncode == 0
    assert cli("simulate", "-r", run).returncode == 0
    assert cli("board", "-r", run, "--pareto").returncode == 0
    assert cli("report", "-r", run, "-o", output).returncode == 0
    assert output.exists()


def _make_torch_checkpoint(root: Path, layer0: float, layer1: float, *, add_nonfloat=True) -> None:
    import torch
    from safetensors.torch import save_file
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 2,
        "hidden_size": 2, "num_attention_heads": 1, "vocab_size": 10}))
    (root / "tokenizer.json").write_text("{}")
    state = {
        "model.layers.0.mlp.weight": torch.full((2, 2), layer0, dtype=torch.float32),
        "model.layers.1.mlp.weight": torch.full((2, 2), layer1, dtype=torch.float32),
        "model.embed_tokens.weight": torch.full((2, 2), .5 + layer0, dtype=torch.float32),
    }
    if add_nonfloat:
        state["model.layers.0.counter"] = torch.tensor([0, 1], dtype=torch.int64)
    save_file(state, str(root / "model.safetensors"))


def _read_merged_tensor(model: Path, tensor_name: str):
    from safetensors import safe_open
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][tensor_name]
    with safe_open(str(model / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def test_real_lite_tensor_merge_linear_multilayer(tmp_path):
    import torch
    for name, a, b in (("base", 0, 0), ("a", 1, 2), ("b", 9, 8)):
        _make_torch_checkpoint(tmp_path / name, a, b)
    cfg = {"name": "real-linear", "base_model": str(tmp_path / "base"),
           "parents": [{"name": "a", "model": str(tmp_path / "a")},
                       {"name": "b", "model": str(tmp_path / "b")}],
           "method": "linear", "genes": 3, "out_dtype": "float32"}
    import yaml
    path = tmp_path / "experiment.yaml"; path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"; exp.init_run(path, run)
    # Candidate 0 starts [0.25, 0.25, 0.25].
    model = exp.build_candidate(run, 0, 0, engine="lite")
    assert torch.allclose(_read_merged_tensor(model, "model.layers.0.mlp.weight"), torch.full((2, 2), 7.0))
    assert torch.allclose(_read_merged_tensor(model, "model.layers.1.mlp.weight"), torch.full((2, 2), 6.5))
    assert _read_merged_tensor(model, "model.layers.0.counter").tolist() == [0, 1]
    build_meta = json.loads((model.parent / "build.json").read_text())
    assert build_meta["engine"] == "lite"
    assert build_meta["engine_details"]["tensors"] == 4
    assert (model / "config.json").is_file()
    assert (model / "tokenizer.json").is_file()


def test_real_lite_task_vector_scale(tmp_path):
    import torch
    for name, a, b in (("base", 3, 3), ("a", 7, 7), ("b", 11, 11)):
        _make_torch_checkpoint(tmp_path / name, a, b)
    cfg = {"name": "scale", "base_model": str(tmp_path / "base"),
           "parents": [{"name": "a", "model": str(tmp_path / "a")},
                       {"name": "b", "model": str(tmp_path / "b")}],
           "method": "task_arithmetic", "genes": 3, "out_dtype": "bfloat16", "task_scale": .5}
    import yaml
    path = tmp_path / "exp.yaml"; path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"; exp.init_run(path, run)
    model = exp.build_candidate(run, 0, 0, engine="lite")
    # base=3, 0.5 * (0.25*(7-3) + 0.75*(11-3)) = 6.5
    assert torch.allclose(_read_merged_tensor(model, "model.layers.0.mlp.weight"), torch.full((2, 2), 6.5, dtype=torch.bfloat16))


def test_real_lite_3_parent_merge(tmp_path):
    import torch
    for name, a in (("base", 0), ("a", 1), ("b", 5), ("c", 9)):
        _make_torch_checkpoint(tmp_path / name, a, a)
    cfg = {"name": "three", "base_model": str(tmp_path / "base"),
           "parents": [{"name": x, "model": str(tmp_path / x)} for x in ("a", "b", "c")],
           "method": "linear", "genes": 3, "out_dtype": "float32"}
    import yaml
    path = tmp_path / "exp.yaml"; path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"; exp.init_run(path, run)
    genes = exp.load_candidate(run, 0, 0)["genes"]
    assert genes[0] == pytest.approx(.85)
    model = exp.build_candidate(run, 0, 0, engine="lite")
    # generation zero first candidate is near-pure parent A.
    expected = .85 * 1 + .075 * 5 + .075 * 9
    assert torch.allclose(_read_merged_tensor(model, "model.layers.0.mlp.weight"), torch.full((2, 2), expected))


def test_lite_rejects_unknown_methods(tmp_path):
    from lerp.lite import LiteMergeError, build_lite
    run = new_run(tmp_path)
    spec, _ = exp.load_run(run)
    with pytest.raises(LiteMergeError, match="Lite supports"):
        build_lite(spec, exp.load_candidate(run, 0, 0)["genes"], tmp_path / "out", method="magic")


def test_recover_completed_evaluation_score(tmp_path, monkeypatch):
    run = new_run(tmp_path)
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)
    monkeypatch.setattr(exp, "_execute_logged", _mock_merger)
    exp.build_candidate(run, 0, 0)
    exp.evaluate_candidate(run, 0, 0)
    folder = exp.candidate_dir(run, 0, 0)
    (folder / "score.json").unlink()  # Simulated power loss before atomic score commit.
    result = exp.recover_candidate(run, 0, 0)
    assert result["source"] == "lm_eval"
    assert (folder / "score.json").is_file()
    with pytest.raises(exp.BreederError, match="already scored"):
        exp.recover_candidate(run, 0, 0)
