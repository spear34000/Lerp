"""Declarative log-likelihood multiple-choice tasks.

A task is data, not code: which Hugging Face dataset and split, how to render the prompt, where the choices are and which one is
correct. Any benchmark in that shape can be added from the experiment YAML without touching Lerp::

    evaluation:
      tasks:
        my_task:
          metric: 'acc_norm,none'
          weight: 1.0
          task:
            dataset: org/name           # Hugging Face datasets id
            config: null                # optional dataset configuration
            split: validation
            prompt: "Question: {question}\\nAnswer:"
            choices: {field: options}   # or {fields: [a, b]} / {template: ["{x} yes", "{x} no"]} / ["no", "yes"]
            label: {field: answer}      # or {index_of: {value: answer_key, in: choices.label}}

Templates use ``{path}`` with dotted paths and ``[i]`` indexes, plus filters ``{path|capitalize}`` (``strip``, ``lstrip``,
``lower``, ``upper``). A label is ``{field: ...}``, ``{index_of: {value: ..., in: ...}}`` or ``{const: N}``. Use ``{{`` and ``}}`` for literal braces. Nothing is evaluated as code.

The built-in definitions below reproduce the prompts, choices and normalization of the corresponding lm-eval tasks, so scores
from the resident evaluator line up with ``lm-eval``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_ALLOWED_KEYS = {"dataset", "config", "split", "prompt", "choices", "label", "clean"}
_TOKEN = re.compile(r"\{([A-Za-z_][\w.\[\]]*)(?:\|(\w+))?\}")
_STEP = re.compile(r"([A-Za-z_]\w*)((?:\[\d+\])*)")
_FILTERS = {"capitalize": str.capitalize, "strip": str.strip, "lstrip": str.lstrip, "lower": str.lower, "upper": str.upper}


class TaskError(ValueError):
    pass


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


CLEANERS = {"hellaswag": _hellaswag_clean}

BUILTIN_TASKS: dict[str, dict[str, Any]] = {
    "arc_easy": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
        "prompt": "Question: {question}\nAnswer:",
        "choices": {"field": "choices.text"},
        "label": {"index_of": {"value": "answerKey", "in": "choices.label"}},
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test",
        "prompt": "Question: {question}\nAnswer:",
        "choices": {"field": "choices.text"},
        "label": {"index_of": {"value": "answerKey", "in": "choices.label"}},
    },
    "boolq": {
        "dataset": "aps/super_glue", "config": "boolq", "split": "validation",
        "prompt": "{passage}\nQuestion: {question}?\nAnswer:",
        "choices": ["no", "yes"],
        "label": {"field": "label"},
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag", "config": None, "split": "validation",
        "prompt": "{activity_label}: {ctx_a} {ctx_b|capitalize}",
        "choices": {"field": "endings"},
        "label": {"field": "label"},
        "clean": "hellaswag",
    },
    "piqa": {
        "dataset": "baber/piqa", "config": None, "split": "validation",
        "prompt": "Question: {goal}\nAnswer:",
        "choices": {"fields": ["sol1", "sol2"]},
        "label": {"field": "label"},
    },
}


def _lookup(doc: Any, path: str) -> Any:
    value = doc
    for part in path.split("."):
        match = _STEP.fullmatch(part)
        if not match:
            raise TaskError(f"bad path segment {part!r} in {path!r}")
        name, indexes = match.groups()
        try:
            value = value[name]
        except (KeyError, TypeError) as exc:
            raise TaskError(f"field {path!r} not found in the dataset row (missing {name!r})") from exc
        for index in re.findall(r"\[(\d+)\]", indexes):
            try:
                value = value[int(index)]
            except (IndexError, TypeError) as exc:
                raise TaskError(f"index [{index}] out of range in {path!r}") from exc
    return value


def render(template: str, doc: Any) -> str:
    protected = template.replace("{{", "\x00").replace("}}", "\x01")

    def replace(match: re.Match) -> str:
        value = _lookup(doc, match.group(1))
        text = value if isinstance(value, str) else str(value)
        flt = match.group(2)
        if flt:
            if flt not in _FILTERS:
                raise TaskError(f"unknown template filter {flt!r}; use one of {sorted(_FILTERS)}")
            text = _FILTERS[flt](text)
        return text

    return _TOKEN.sub(replace, protected).replace("\x00", "{").replace("\x01", "}")


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    dataset: str
    split: str
    prompt: str
    choices: Any
    label: Any
    config: str | None = None
    clean: str | None = None

    def convert(self, doc: Any) -> tuple[str, list[str], int]:
        """One dataset row -> (context, choice strings, index of the correct choice)."""
        clean = CLEANERS[self.clean] if self.clean else (lambda s: s)
        context = clean(render(self.prompt, doc))
        spec = self.choices
        if isinstance(spec, list):
            choices = [clean(render(c, doc)) for c in spec]
        elif "field" in spec:
            choices = [clean(str(c)) for c in _lookup(doc, spec["field"])]
        elif "fields" in spec:
            choices = [clean(str(_lookup(doc, f))) for f in spec["fields"]]
        else:
            choices = [clean(render(t, doc)) for t in spec["template"]]
        if len(choices) < 2:
            raise TaskError(f"task {self.name}: a row has fewer than two choices")
        lab = self.label
        if "field" in lab:
            label = int(_lookup(doc, lab["field"]))
        elif "const" in lab:
            label = int(lab["const"])
        else:
            ref = lab["index_of"]
            pool = list(_lookup(doc, ref["in"]))
            value = _lookup(doc, ref["value"])
            if value not in pool:
                raise TaskError(f"task {self.name}: label {value!r} not in {pool!r}")
            label = pool.index(value)
        if not 0 <= label < len(choices):
            raise TaskError(f"task {self.name}: label {label} outside {len(choices)} choices")
        return context, choices, label

    def documents(self, lo: int, hi: int, rows: Sequence[Any] | None = None) -> list[tuple[str, list[str], int]]:
        """Converted rows ``lo <= i < hi``. ``rows`` lets tests (and offline users) supply the dataset directly."""
        if rows is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise TaskError('Install datasets for dataset-backed tasks: pip install datasets') from exc
            data = load_dataset(self.dataset, self.config, split=self.split)
            rows = data.select(range(min(lo, len(data)), min(hi, len(data))))
            return [self.convert(row) for row in rows]
        return [self.convert(row) for row in list(rows)[lo:hi]]


def validate_definition(name: str, definition: dict) -> None:
    if not isinstance(definition, dict):
        raise TaskError(f"task {name}: definition must be a mapping")
    unknown = set(definition) - _ALLOWED_KEYS
    if unknown:
        raise TaskError(f"task {name}: unknown keys {sorted(unknown)}")
    for key in ("dataset", "split", "prompt", "choices", "label"):
        if key not in definition:
            raise TaskError(f"task {name}: missing required key {key!r}")
    choices = definition["choices"]
    if not (isinstance(choices, list) and choices or isinstance(choices, dict) and len(choices) == 1
            and next(iter(choices)) in ("field", "fields", "template")):
        raise TaskError(f"task {name}: choices must be a list or one of field / fields / template")
    label = definition["label"]
    if not (isinstance(label, dict) and any(set(label) == {key} for key in ("field", "index_of", "const"))):
        raise TaskError(f"task {name}: label must be {{field: ...}}, {{index_of: {{value: ..., in: ...}}}} or {{const: N}}")
    if "index_of" in label and set(label["index_of"]) != {"value", "in"}:
        raise TaskError(f"task {name}: index_of needs exactly value and in")
    if definition.get("clean") is not None and definition["clean"] not in CLEANERS:
        raise TaskError(f"task {name}: unknown clean {definition['clean']!r}; use one of {sorted(CLEANERS)}")
    templates = [definition["prompt"]]
    if isinstance(choices, list):
        templates += choices
    elif "template" in choices:
        templates += list(choices["template"])
    for text in templates:
        _check_template(name, text)


def _check_template(name: str, text: Any) -> None:
    """Catch unbalanced braces and bad placeholders when the experiment is parsed, not mid-search."""
    if not isinstance(text, str):
        raise TaskError(f"task {name}: templates must be strings, got {text!r}")
    leftover = _TOKEN.sub("", text.replace("{{", "").replace("}}", ""))
    if "{" in leftover or "}" in leftover:
        raise TaskError(f"task {name}: malformed placeholder in template {text!r}")
    for match in _TOKEN.finditer(text.replace("{{", "").replace("}}", "")):
        if match.group(2) and match.group(2) not in _FILTERS:
            raise TaskError(f"task {name}: unknown template filter {match.group(2)!r}; use one of {sorted(_FILTERS)}")


def resolve_task(name: str, definition: dict | str | None = None) -> TaskDefinition:
    """Definition from the experiment (dict or its JSON string), else the built-in with that name."""
    if isinstance(definition, str):
        definition = json.loads(definition)
    if definition is None:
        if name not in BUILTIN_TASKS:
            raise TaskError(f"task {name!r} has no definition and is not built in ({sorted(BUILTIN_TASKS)}); add a `task:` block")
        definition = BUILTIN_TASKS[name]
    validate_definition(name, definition)
    return TaskDefinition(
        name=name, dataset=definition["dataset"], split=definition["split"], prompt=definition["prompt"],
        choices=definition["choices"], label=definition["label"], config=definition.get("config"),
        clean=definition.get("clean"),
    )


def metric_kind(metric: str) -> str:
    """'acc_norm,none' -> 'acc_norm'. Only acc and acc_norm are computed by the resident evaluator."""
    kind = metric.split(",")[0]
    if kind not in ("acc", "acc_norm"):
        raise TaskError(f"metric {metric!r} is not supported by the resident evaluator (use acc or acc_norm)")
    return kind
