"""Generative tasks: definition parsing, answer extraction, and batched left-padded greedy decoding in the resident session."""
from __future__ import annotations

import pytest

from lerp.spec import SpecError, parse_spec
from lerp.tasks import TaskError, metric_kind, resolve_task, validate_definition

GSM = {"dataset": "local/math", "split": "test", "prompt": "Q: {q}\nA:",
       "answer": {"field": "a", "regex": "#### (.*)$"}, "generate": {"max_new_tokens": 32, "stop": ["Q:"]},
       "extract": {"regex": "(-?[0-9][0-9,]*\\.?[0-9]*)", "pick": "last"}, "normalize": ["remove_commas", "strip_period", "number"]}


def test_gold_and_prediction_are_normalized_the_same_way():
    task = resolve_task("m", GSM)
    assert task.convert_generative({"q": "x", "a": "so\n#### 1,018"}) == ("Q: x\nA:", "1018")
    assert task.extract_answer(" 5+3=8, total $1,018.00.\nQ: next 7") == "1018"
    assert task.extract_answer("no digits here") == ""


def test_stop_string_cuts_the_completion_before_extraction():
    task = resolve_task("m", GSM)
    assert task.extract_answer("answer 4 Q: and then 99") == "4"


def test_pick_first_and_missing_gold_regex():
    first = resolve_task("m", {**GSM, "extract": {"regex": "([0-9]+)", "pick": "first"}})
    assert first.extract_answer("7 then 9") == "7"
    with pytest.raises(TaskError, match="does not match"):
        resolve_task("m", GSM).convert_generative({"q": "x", "a": "no marker"})


def test_builtin_gsm8k_extracts_the_last_number():
    task = resolve_task("gsm8k")
    assert task.generative
    assert task.convert_generative({"question": "q", "answer": "work\n#### 72"})[1] == "72"
    assert task.extract_answer("Natalia sold 48 + 24 = 72 clips.") == "72"


@pytest.mark.parametrize("patch, message", [
    ({"choices": ["a", "b"]}, "multiple-choice"),
    ({"generate": {"max_new_tokens": 0}}, "max_new_tokens"),
    ({"generate": {"temperature": 1}}, "generate accepts"),
    ({"normalize": ["rot13"]}, "normalize"),
    ({"extract": {"regex": "("}}, "bad regex"),
    ({"answer": {"field": "a", "template": "x"}}, "answer must be"),
])
def test_invalid_generative_definitions_are_rejected(patch, message):
    with pytest.raises(TaskError, match=message):
        validate_definition("m", {**GSM, **patch})


def test_exact_match_metric_and_spec_round_trip():
    assert metric_kind("exact_match,none") == "exact_match"
    spec = parse_spec({"name": "t", "base_model": "b", "parents": [{"name": "a", "model": "a"}, {"name": "b", "model": "b"}],
                       "evaluation": {"limit": 4, "tasks": {"m": {"metric": "exact_match,none", "task": GSM}}}})
    assert resolve_task("m", spec.evaluation.tasks[0].definition).generative
    with pytest.raises(SpecError):
        parse_spec({"name": "t", "base_model": "b", "parents": [{"name": "a", "model": "a"}, {"name": "b", "model": "b"}],
                    "evaluation": {"tasks": {"m": {"task": {**GSM, "generate": {"max_new_tokens": 0}}}}}})


# ---------------------------------------------------------------------------------------- resident session
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from test_resident import TASK, _llama, _rows  # noqa: E402
from lerp.resident import ResidentSession  # noqa: E402

GEN = {"dataset": "local/gen", "split": "test", "prompt": "{q}", "answer": {"field": "a"},
       "generate": {"max_new_tokens": 4}, "normalize": ["strip"]}


def _prompts(n: int = 6) -> list[str]:
    return [" ".join(f"w{(i * 7 + k * 3) % 50 + 1}" for k in range(2 + i % 4)) for i in range(n)]  # different lengths -> padding


def _single_greedy(model, tokenizer, prompt: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model.generate(input_ids=ids, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=4,
                             pad_token_id=0)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def _gen_spec(tmp_path, tasks):
    from lerp.spec import parse_spec
    return parse_spec({"name": "tiny", "base_model": str(tmp_path / "base"),
                       "parents": [{"name": "p0", "model": str(tmp_path / "p0")}, {"name": "p1", "model": str(tmp_path / "p1")}],
                       "genes": 2, "out_dtype": "float32", "gene_groups": ["attention", "mlp", "other"], "population": 3,
                       "evaluation": {"limit": 6, "tasks": tasks}})


def test_batched_left_padded_generation_matches_one_by_one(tmp_path):
    for name, seed in (("base", 1), ("p0", 2), ("p1", 3)):
        _llama(tmp_path / name, seed)
    spec = _gen_spec(tmp_path, {"gen": {"metric": "exact_match,none", "task": GEN}})
    prompts = _prompts()
    session = ResidentSession(spec, "cpu", "float32", rows={"gen": [{"q": p, "a": ""} for p in prompts]})
    session.blender.apply([0.5] * 6)
    gold = [_single_greedy(session.model, session.tokenizer, p) for p in prompts]
    session.rows = {"gen": [{"q": p, "a": g} for p, g in zip(prompts, gold)]}
    session._gen_scorers.clear()
    assert len(set(gold)) > 1
    assert session.evaluate([0.5] * 6, (0, 6)) == {"gen": 1.0}
    session._gen_batch = 2  # smaller batches regroup the prompts and must not change a single completion
    assert session.evaluate([0.5] * 6, (0, 6)) == {"gen": 1.0}
    session.close()


def test_generative_and_choice_tasks_share_one_window(tmp_path):
    for name, seed in (("base", 1), ("p0", 2), ("p1", 3)):
        _llama(tmp_path / name, seed)
    spec = _gen_spec(tmp_path, {"choice": {"metric": "acc,none", "task": TASK}, "gen": {"metric": "exact_match,none", "task": GEN}})
    rows = {"choice": _rows(6), "gen": [{"q": p, "a": "w1"} for p in _prompts()]}
    session = ResidentSession(spec, "cpu", "float32", rows=rows)
    metrics = session.evaluate([0.5] * 6, (0, 6))
    assert set(metrics) == {"choice", "gen"} and all(0.0 <= v <= 1.0 for v in metrics.values())
    protocol = session.protocol((0, 6))
    assert protocol["generation"] == {"decoding": "greedy", "prompts": 6}
    session.close()


def test_generative_only_run_skips_the_logit_self_check(tmp_path):
    for name, seed in (("base", 1), ("p0", 2), ("p1", 3)):
        _llama(tmp_path / name, seed)
    spec = _gen_spec(tmp_path, {"gen": {"metric": "exact_match,none", "task": GEN}})
    ResidentSession(spec, "cpu", "float32", rows={"gen": [{"q": p, "a": "x"} for p in _prompts()]}).close()
