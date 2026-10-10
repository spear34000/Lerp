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

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_ALLOWED_KEYS = {"dataset", "config", "split", "prompt", "choices", "label", "clean", "answer", "generate", "extract", "normalize"}
_GENERATE_KEYS = {"max_new_tokens", "stop"}
_NORMALIZERS = {
    "strip": str.strip,
    "lower": str.lower,
    "remove_commas": lambda t: t.replace(",", ""),
    "remove_dollar": lambda t: t.replace("$", ""),
    "strip_period": lambda t: t.strip().rstrip("."),
    "collapse_spaces": lambda t: " ".join(t.split()),
    "number": lambda t: _canonical_number(t),
}
_TOKEN = re.compile(r"\{([A-Za-z_][\w.\[\]]*)(?:\|(\w+))?\}")
_STEP = re.compile(r"([A-Za-z_]\w*)((?:\[\d+\])*)")
_FILTERS = {"capitalize": str.capitalize, "strip": str.strip, "lstrip": str.lstrip, "lower": str.lower, "upper": str.upper}


class TaskError(ValueError):
    pass


def _canonical_number(text: str) -> str:
    """'18.00' -> '18', '1,000' -> '1000'; anything that is not a plain number is returned unchanged."""
    cleaned = text.strip().replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return text
    return str(int(value)) if value == int(value) else repr(value)


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
    "gsm8k": {
        "dataset": "openai/gsm8k", "config": "main", "split": "test",
        "prompt": "Question: {question}\nAnswer:",
        "answer": {"field": "answer", "regex": "#### (-?[0-9.,]+)"},
        "generate": {"max_new_tokens": 256, "stop": ["Question:", "</s>", "<|im_end|>"]},
        "extract": {"regex": "(-?[0-9][0-9,]*\\.?[0-9]*)", "pick": "last"},
        "normalize": ["remove_commas", "strip_period", "number"],
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
    answer: Any = None
    generate: Any = None
    extract: Any = None
    normalize: tuple = ()

    @property
    def generative(self) -> bool:
        return self.generate is not None

    def convert_generative(self, doc: Any) -> tuple[str, str]:
        """One dataset row -> (prompt, normalized gold answer)."""
        prompt = render(self.prompt, doc)
        spec = self.answer
        raw = str(_lookup(doc, spec["field"])) if "field" in spec else render(spec["template"], doc)
        if spec.get("regex"):
            found = re.search(spec["regex"], raw)
            if not found:
                raise TaskError(f"task {self.name}: answer regex {spec['regex']!r} does not match {raw[-60:]!r}")
            raw = found.group(1) if found.groups() else found.group(0)
        return prompt, self.normalized(raw)

    def normalized(self, text: str) -> str:
        for name in self.normalize:
            text = _NORMALIZERS[name](text)
        return text.strip()

    def extract_answer(self, generated: str) -> str:
        """Cut at the first stop string, pull the answer out with the extract regex, normalize it."""
        for stop in (self.generate or {}).get("stop", []):
            at = generated.find(stop)
            if at >= 0:
                generated = generated[:at]
        spec = self.extract or {}
        if spec.get("regex"):
            found = list(re.finditer(spec["regex"], generated))
            if not found:
                return ""
            match = found[-1] if spec.get("pick", "last") == "last" else found[0]
            generated = match.group(1) if match.groups() else match.group(0)
        return self.normalized(generated)

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

    def local_rows(self) -> list[dict]:
        """Rows of a local JSONL dataset (``dataset: file:PATH``); the experiment pins it with a SHA-256 in the resident protocol."""
        path = Path(self.dataset[len("file:"):])
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            raise TaskError(f"cannot read local dataset {path}: {exc}") from exc

    def data_sha256(self) -> str | None:
        if not self.dataset.startswith("file:"):
            return None
        return hashlib.sha256(Path(self.dataset[len("file:"):]).read_bytes()).hexdigest()

    def documents(self, lo: int, hi: int, rows: Sequence[Any] | None = None) -> list[tuple]:
        """Converted rows ``lo <= i < hi``. ``rows`` lets tests (and offline users) supply the dataset directly."""
        convert = self.convert_generative if self.generative else self.convert
        if rows is None and self.dataset.startswith("file:"):
            rows = self.local_rows()
        if rows is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise TaskError('Install datasets for dataset-backed tasks: pip install datasets') from exc
            data = load_dataset(self.dataset, self.config, split=self.split)
            rows = data.select(range(min(lo, len(data)), min(hi, len(data))))
            return [convert(row) for row in rows]
        return [convert(row) for row in list(rows)[lo:hi]]


def validate_definition(name: str, definition: dict) -> None:
    if not isinstance(definition, dict):
        raise TaskError(f"task {name}: definition must be a mapping")
    unknown = set(definition) - _ALLOWED_KEYS
    if unknown:
        raise TaskError(f"task {name}: unknown keys {sorted(unknown)}")
    generative = "generate" in definition or "answer" in definition
    required = ("dataset", "split", "prompt", "generate", "answer") if generative else ("dataset", "split", "prompt", "choices", "label")
    for key in required:
        if key not in definition:
            raise TaskError(f"task {name}: missing required key {key!r}")
    if generative:
        _validate_generative(name, definition)
        _check_template(name, definition["prompt"])
        return
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


def _validate_generative(name: str, definition: dict) -> None:
    for key in ("choices", "label", "clean"):
        if key in definition:
            raise TaskError(f"task {name}: {key!r} belongs to multiple-choice tasks, not generative ones")
    gen = definition["generate"]
    if not isinstance(gen, dict) or set(gen) - _GENERATE_KEYS:
        raise TaskError(f"task {name}: generate accepts only {sorted(_GENERATE_KEYS)}")
    tokens = gen.get("max_new_tokens", 256)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= 2048:
        raise TaskError(f"task {name}: generate.max_new_tokens must be an integer between 1 and 2048")
    stop = gen.get("stop", [])
    if not isinstance(stop, list) or not all(isinstance(x, str) and x for x in stop):
        raise TaskError(f"task {name}: generate.stop must be a list of non-empty strings")
    answer = definition["answer"]
    if not isinstance(answer, dict) or set(answer) - {"field", "template", "regex"} or not (("field" in answer) ^ ("template" in answer)):
        raise TaskError(f"task {name}: answer must be {{field|template: ..., regex: optional}}")
    extract = definition.get("extract") or {}
    if not isinstance(extract, dict) or set(extract) - {"regex", "pick"} or extract.get("pick", "last") not in ("first", "last"):
        raise TaskError(f"task {name}: extract must be {{regex: ..., pick: first|last}}")
    for pattern in (answer.get("regex"), extract.get("regex")):
        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise TaskError(f"task {name}: bad regex {pattern!r}: {exc}") from exc
    norm = definition.get("normalize", [])
    if not isinstance(norm, list) or any(n not in _NORMALIZERS for n in norm):
        raise TaskError(f"task {name}: normalize must be a list drawn from {sorted(_NORMALIZERS)}")
    if "template" in answer:
        _check_template(name, answer["template"])


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
        choices=definition.get("choices"), label=definition.get("label"), config=definition.get("config"),
        clean=definition.get("clean"), answer=definition.get("answer"), generate=definition.get("generate"),
        extract=definition.get("extract"), normalize=tuple(definition.get("normalize", ())),
    )


def metric_kind(metric: str) -> str:
    """'acc_norm,none' -> 'acc_norm'. The resident evaluator computes acc / acc_norm (choice tasks) and exact_match (generative)."""
    kind = metric.split(",")[0]
    if kind not in ("acc", "acc_norm", "exact_match"):
        raise TaskError(f"metric {metric!r} is not supported by the resident evaluator (use acc, acc_norm or exact_match)")
    return kind
