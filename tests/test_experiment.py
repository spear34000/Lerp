import json
from pathlib import Path

import pytest

import lerp.experiment as exp
from lerp.spec import SpecError

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo.yaml"


def test_init_and_demo_advance(tmp_path):
    run = tmp_path / "run"
    state, _ = exp.init_run(DEMO, run)
    assert state["generation"] == 0
    assert (exp.candidate_dir(run, 0, 0) / "merge.yaml").exists()
    first = exp.simulate_generation(run, 0)
    assert len(first) == 6
    assert all(c["source"] == "SIMULATED_TOY" for c in first)
    assert len(exp.leaderboard(run)) == 6
    assert exp.leaderboard(run, include_simulated=False) == []
    with pytest.raises(exp.BreederError, match="Simulated"):
        exp.advance(run)
    gen = exp.advance(run, allow_simulated=True)
    assert gen == 1
    new = exp.load_candidate(run, 1, 0)
    assert len(new["lineage"]) == 2
    assert len(new["genes"]) == 6
    assert all(0 <= v <= 1 for v in new["genes"])
    exp.simulate_generation(run, 1)
    assert len(exp.leaderboard(run)) == 12
    assert "g000-c000" in exp.lineage_dot(run)


def test_init_refuses_overwrite(tmp_path):
    run = tmp_path / "run"
    exp.init_run(DEMO, run)
    with pytest.raises(exp.BreederError, match="already contains"):
        exp.init_run(DEMO, run)


def test_record_manual_requires_exact_metrics(tmp_path):
    run = tmp_path / "run"
    exp.init_run(DEMO, run)
    with pytest.raises(exp.BreederError, match="exactly"):
        exp.record_score(run, 0, 0, {"arc_easy": .4}, source="manual")
    result = exp.record_score(run, 0, 0, {"arc_easy": .4, "hellaswag": .7}, source="manual")
    assert result["fitness"] == .55
    with pytest.raises(exp.BreederError, match="Already scored"):
        exp.record_score(run, 0, 0, {"arc_easy": .4, "hellaswag": .7}, source="manual")
    with pytest.raises(exp.BreederError, match="between 0 and 1"):
        exp.record_score(run, 0, 1, {"arc_easy": 0.4, "hellaswag": float("nan")}, source="manual")
    with pytest.raises(exp.BreederError, match="Every candidate"):
        exp.advance(run)


def test_build_evaluate_mock_tools(tmp_path, monkeypatch):
    run = tmp_path / "run"
    exp.init_run(DEMO, run)
    seen = []
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)

    def fake_exec(command, log):
        seen.append(command)
        if command[0] == "mergekit-yaml":
            root = Path(command[2])
            root.mkdir(parents=True)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fake-mock-tensor")
        elif command[0] == "lm-eval":
            out = Path(command[command.index("--output_path") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "results_mock.json").write_text(json.dumps({"results": {
                "arc_easy": {"acc_norm,none": 0.66},
                "hellaswag": {"acc_norm,none": 0.44},
            }}))
    monkeypatch.setattr(exp, "_execute_logged", fake_exec)
    out = exp.build_candidate(run, 0, 0)
    assert (out / "config.json").exists()
    assert seen[0][0] == "mergekit-yaml"
    score = exp.evaluate_candidate(run, 0, 0)
    assert score["source"] == "lm_eval"
    assert score["fitness"] == 0.55
    assert "run" in seen[1]
    assert seen[1][0] == "lm-eval"


def test_deterministic_genomes(tmp_path):
    x = tmp_path / "x"
    y = tmp_path / "y"
    exp.init_run(DEMO, x)
    exp.init_run(DEMO, y)
    for i in range(6):
        assert exp.load_candidate(x, 0, i)["genes"] == exp.load_candidate(y, 0, i)["genes"]
    exp.simulate_generation(x, 0)
    exp.simulate_generation(y, 0)
    exp.advance(x, allow_simulated=True)
    exp.advance(y, allow_simulated=True)
    for i in range(6):
        assert exp.load_candidate(x, 1, i)["genes"] == exp.load_candidate(y, 1, i)["genes"]


def test_automatic_cycle_resumes_and_advances(tmp_path, monkeypatch):
    run = tmp_path / "run"
    exp.init_run(DEMO, run)
    monkeypatch.setattr(exp, "_require_command", lambda *args: None)
    calls = []

    def fake_exec(cmd, log):
        calls.append(cmd[0])
        if cmd[0] == "mergekit-yaml":
            path = Path(cmd[2])
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"fake-mock-tensor")
        elif cmd[0] == "lm-eval":
            out = Path(cmd[cmd.index("--output_path") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval.json").write_text(json.dumps({"results": {
                "arc_easy": {"acc_norm,none": 0.61},
                "hellaswag": {"acc_norm,none": 0.52},
            }}))
    monkeypatch.setattr(exp, "_execute_logged", fake_exec)
    assert exp.cycle(run, 2) == [0, 1]
    assert calls.count("mergekit-yaml") == 12
    assert calls.count("lm-eval") == 12
    assert len(exp.leaderboard(run, include_simulated=False)) == 12
    assert exp.cycle(run, 1) == [1]
    assert calls.count("mergekit-yaml") == 12  # no duplicate builds
