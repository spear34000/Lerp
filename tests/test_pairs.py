"""Paired item-level statistics: McNemar numbers, item outcomes recorded by the resident search, the `pairs` command."""
from __future__ import annotations

import json
import math

import pytest

from lerp.statistics import StatisticsError, compare_samples, mcnemar


def test_mcnemar_exact_p_value_matches_the_binomial_tail():
    a = [1] * 8 + [0] * 2 + [1] * 5
    b = [0] * 8 + [1] * 2 + [1] * 5
    r = mcnemar(a, b)
    assert (r["a_only"], r["b_only"], r["both_right"], r["both_wrong"]) == (8, 2, 5, 0)
    assert math.isclose(r["p_exact_two_sided"], 2 * (1 + 10 + 45) / 1024)  # P(X <= 2 | n=10, p=1/2), doubled
    assert r["difference"] == pytest.approx(0.4)
    assert r["ci"]["lower"] < 0.4 < r["ci"]["upper"]


def test_identical_systems_have_p_one_and_an_interval_around_zero():
    r = mcnemar([1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1])
    assert r["p_exact_two_sided"] == 1.0 and r["difference"] == 0.0
    assert r["ci"]["lower"] < 0 < r["ci"]["upper"]  # the +0.5 correction keeps the interval from collapsing to a point


def test_all_discordant_one_way_is_significant_and_symmetric():
    a, b = [1] * 12, [0] * 12
    r = mcnemar(a, b)
    assert r["p_exact_two_sided"] == pytest.approx(2 / 2 ** 12)
    flipped = mcnemar(b, a)
    assert flipped["p_exact_two_sided"] == r["p_exact_two_sided"] and flipped["difference"] == -r["difference"]
    assert flipped["ci"]["lower"] == pytest.approx(-r["ci"]["upper"])


def test_mcnemar_rejects_bad_input():
    with pytest.raises(StatisticsError, match="unpaired"):
        mcnemar([1, 0], [1])
    with pytest.raises(StatisticsError, match="0/1"):
        mcnemar([1, 2], [1, 0])


def test_compare_samples_adds_mcnemar_for_binary_scores(tmp_path):
    for name, values in (("a", [1, 1, 1, 0, 1, 1, 0, 1]), ("b", [1, 0, 0, 0, 1, 0, 0, 1])):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps({"id": i, "score": v}) for i, v in enumerate(values)), encoding="utf-8")
    out = compare_samples(tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    assert out["mcnemar"]["a_only"] == 3 and out["mcnemar"]["b_only"] == 0
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps({"id": i, "score": 0.5}) for i in range(8)), encoding="utf-8")
    assert "mcnemar" not in compare_samples(tmp_path / "a.jsonl", tmp_path / "c.jsonl")


# ------------------------------------------------------------------------------------------ resident search
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
yaml = pytest.importorskip("yaml")

import lerp.experiment as exp  # noqa: E402
from lerp.cli import _run, create_parser  # noqa: E402
from lerp.pairs import compare_pairs, format_rows  # noqa: E402
from lerp.search import search_run  # noqa: E402
from lerp.spec import spec_to_dict  # noqa: E402
from test_resident import _session, _spec, _three  # noqa: E402


def _searched_run(tmp_path):
    _three(tmp_path)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"])
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(spec_to_dict(spec)))
    run = tmp_path / "run"
    exp.init_run(config, run)
    loaded, _ = exp.load_run(run)
    session = _session(loaded)
    search_run(run, 1, session=session, baselines=True, log=lambda _m: None)
    return run, session


def test_search_records_item_outcomes_that_match_the_accuracies(tmp_path):
    run, _ = _searched_run(tmp_path)
    cand = run / "generations" / "gen-000" / "cand-000"
    items = json.loads((cand / "items.json").read_text())
    score = json.loads((cand / "score.json").read_text())
    assert items["window"] == [0, 8] and set(items["tasks"]) == {"toy", "toy2"}
    for task, outcomes in items["tasks"].items():
        assert len(outcomes) == 8 and set(outcomes) <= {0, 1}
        assert sum(outcomes) / 8 == pytest.approx(score["metrics"][task])
    assert (run / "baselines" / "base" / "items.json").is_file() and (run / "baselines" / "p0" / "items.json").is_file()


def test_compare_pairs_reports_each_task_and_the_pool(tmp_path, capsys):
    run, _ = _searched_run(tmp_path)
    rows = compare_pairs(run, 0, 0)
    assert {(r["baseline"], r["task"]) for r in rows} == {(b, t) for b in ("base", "p0", "p1") for t in ("toy", "toy2", "(all tasks pooled)")}
    pooled = next(r for r in rows if r["baseline"] == "p0" and r["task"].startswith("(all"))
    assert pooled["items"] == 16
    assert "McNemar" in format_rows(rows) or "McNemar" in format_rows(rows).replace("\n", " ")
    out = tmp_path / "pairs.json"
    assert _run(create_parser().parse_args(["pairs", "-r", str(run), "-g", "0", "-i", "0", "--against", "p1", "-o", str(out)])) == 0
    assert {r["baseline"] for r in json.loads(out.read_text())} == {"p1"}
    assert "vs" in capsys.readouterr().out


def test_compare_pairs_refuses_scores_without_item_outcomes(tmp_path):
    run, _ = _searched_run(tmp_path)
    (run / "baselines" / "p0" / "items.json").unlink()
    with pytest.raises(exp.BreederError, match="items.json"):
        compare_pairs(run, 0, 0, ["p0"])
