"""Declarative log-likelihood tasks: built-in definitions, templates, validation, spec round trip."""
from __future__ import annotations

import pytest
import yaml

from lerp.spec import SpecError, parse_spec, spec_to_dict
from lerp.tasks import TaskError, metric_kind, render, resolve_task


def test_arc_row_matches_the_lm_eval_layout():
    row = {"question": "Why is the sky blue?", "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]}, "answerKey": "C"}
    assert resolve_task("arc_easy").convert(row) == ("Question: Why is the sky blue?\nAnswer:", ["a", "b", "c", "d"], 2)


def test_boolq_row_has_static_choices():
    row = {"passage": "P.", "question": "is it", "label": 1}
    assert resolve_task("boolq").convert(row) == ("P.\nQuestion: is it?\nAnswer:", ["no", "yes"], 1)


def test_hellaswag_row_applies_capitalize_then_the_lm_eval_cleanup():
    row = {"activity_label": "Baking", "ctx_a": "A man stands.", "ctx_b": "he opens [title] the oven",
           "endings": ["he [step] bakes", "x  y", "z", "w"], "label": "1"}
    context, choices, label = resolve_task("hellaswag").convert(row)
    assert context == "Baking: A man stands. He opens. the oven"
    assert choices == ["he bakes", "x y", "z", "w"] and label == 1


def test_piqa_row_reads_two_fields():
    row = {"goal": "Open a jar", "sol1": "twist", "sol2": "shake", "label": 0}
    assert resolve_task("piqa").convert(row) == ("Question: Open a jar\nAnswer:", ["twist", "shake"], 0)


def test_render_supports_paths_indexes_filters_and_escaped_braces():
    doc = {"a": {"b": ["x", "y"]}, "n": 3, "name": "hello world"}
    assert render("{a.b[1]}-{n}-{name|capitalize}-{name|upper}", doc) == "y-3-Hello world-HELLO WORLD"
    assert render("{{literal}} {n}", doc) == "{literal} 3"


def test_custom_task_with_template_choices_and_field_label():
    definition = {"dataset": "x/y", "split": "test", "prompt": "{q}\nA:", "choices": {"template": ["{q} {a1}", "{q} {a2}"]},
                  "label": {"field": "gold"}}
    task = resolve_task("mine", definition)
    assert task.convert({"q": "Q", "a1": "yes", "a2": "no", "gold": 1}) == ("Q\nA:", ["Q yes", "Q no"], 1)


@pytest.mark.parametrize("definition,message", [
    ({"dataset": "x", "split": "t", "prompt": "p", "choices": ["a", "b"]}, "missing required key 'label'"),
    ({"dataset": "x", "split": "t", "prompt": "p", "choices": ["a", "b"], "label": {"field": "l"}, "banana": 1}, "unknown keys"),
    ({"dataset": "x", "split": "t", "prompt": "{open", "choices": ["a", "b"], "label": {"field": "l"}}, "malformed placeholder"),
    ({"dataset": "x", "split": "t", "prompt": "{q|shout}", "choices": ["a", "b"], "label": {"field": "l"}}, "unknown template filter"),
    ({"dataset": "x", "split": "t", "prompt": "p", "choices": {"nope": 1}, "label": {"field": "l"}}, "choices must be"),
    ({"dataset": "x", "split": "t", "prompt": "p", "choices": ["a", "b"], "label": {"index_of": {"value": "k"}}}, "index_of needs"),
    ({"dataset": "x", "split": "t", "prompt": "p", "choices": ["a", "b"], "label": {"field": "l"}, "clean": "nope"}, "unknown clean"),
])
def test_bad_definitions_are_rejected_early(definition, message):
    with pytest.raises(TaskError, match=message):
        resolve_task("bad", definition)


def test_unknown_name_without_definition_is_an_error():
    with pytest.raises(TaskError, match="not built in"):
        resolve_task("no_such_benchmark")


def test_label_outside_the_choices_is_an_error():
    task = resolve_task("piqa")
    with pytest.raises(TaskError, match="outside"):
        task.convert({"goal": "g", "sol1": "a", "sol2": "b", "label": 5})


def test_documents_slice_supplied_rows():
    rows = [{"goal": f"g{i}", "sol1": "a", "sol2": "b", "label": i % 2} for i in range(10)]
    docs = resolve_task("piqa").documents(2, 5, rows=rows)
    assert [d[0] for d in docs] == ["Question: g2\nAnswer:", "Question: g3\nAnswer:", "Question: g4\nAnswer:"]


def test_metric_kind():
    assert metric_kind("acc_norm,none") == "acc_norm" and metric_kind("acc") == "acc"
    with pytest.raises(TaskError):
        metric_kind("exact_match,strict")


def test_spec_carries_and_round_trips_a_declarative_task():
    block = {"dataset": "x/y", "split": "test", "prompt": "{q}", "choices": ["a", "b"], "label": {"field": "l"}}
    raw = {"name": "t", "base_model": "./b", "parents": [{"name": "a", "model": "./a"}, {"name": "b", "model": "./b2"}],
           "evaluation": {"tasks": {"mine": {"metric": "acc,none", "task": block}, "arc_easy": {}}}}
    spec = parse_spec(raw)
    by_name = {task.name: task for task in spec.evaluation.tasks}
    assert by_name["mine"].definition is not None and by_name["arc_easy"].definition is None
    again = parse_spec(yaml.safe_load(yaml.safe_dump(spec_to_dict(spec))))
    assert {task.name: task.definition for task in again.evaluation.tasks} == {task.name: task.definition for task in spec.evaluation.tasks}
    bad = {**raw, "evaluation": {"tasks": {"mine": {"task": {"dataset": "x"}}}}}
    with pytest.raises(SpecError, match="missing required key"):
        parse_spec(bad)


def test_constant_label_and_lstrip_filter_for_sciq_style_rows():
    definition = {"dataset": "allenai/sciq", "split": "test", "prompt": "{support|lstrip}\nQuestion: {question}\nAnswer:",
                  "choices": {"fields": ["distractor1", "distractor2", "distractor3", "correct_answer"]}, "label": {"const": 3}}
    row = {"support": "  Plants make sugar.", "question": "What do plants make?", "distractor1": "a", "distractor2": "b",
           "distractor3": "c", "correct_answer": "sugar"}
    assert resolve_task("sciq_like", definition).convert(row) == (
        "Plants make sugar.\nQuestion: What do plants make?\nAnswer:", ["a", "b", "c", "sugar"], 3)
