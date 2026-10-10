"""Versioned, conservative configuration for reproducible merge experiments."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_METHODS = {"linear", "task_arithmetic", "slerp", "ties", "dare_ties", "dare_linear"}
SUPPORTED_DTYPES = {"float16", "bfloat16", "float32"}
SUPPORTED_SELECTION = {"weighted", "pareto"}
SUPPORTED_MODES = {"full", "lora"}
GENE_GROUPS = {"attention", "mlp", "router", "norm", "embedding", "other"}
LORA_METHODS = {"linear", "task_arithmetic"}


@dataclass(frozen=True)
class Parent:
    name: str
    model: str


@dataclass(frozen=True)
class EvalTask:
    name: str
    metric: str = "acc_norm,none"
    weight: float = 1.0
    definition: str | None = None  # canonical JSON of a declarative `task:` block (see lerp.tasks); None = lm-eval name / built-in


@dataclass(frozen=True)
class EvalConfig:
    tasks: tuple[EvalTask, ...] = ()
    device: str = "cpu"
    batch_size: int = 1
    limit: int | None = None
    penalty: float = 0.0
    num_fewshot: int | None = None
    apply_chat_template: bool = False
    screening_limit: int | None = None
    promote_top: int | None = None


@dataclass(frozen=True)
class Spec:
    name: str
    base_model: str
    parents: tuple[Parent, ...]
    method: str = "linear"
    population: int = 6
    genes: int = 6  # number of control points PER parent
    mutation_sigma: float = 0.12
    seed: int = 42
    out_dtype: str = "bfloat16"
    density: float = 0.5
    task_scale: float = 1.0  # lambda for base-relative merge methods
    selection: str = "weighted"  # or 'pareto' (multi-objective NSGA-II style)
    search: str = "evolution"  # how advance() proposes the next generation: genetic breeding or GP + expected improvement
    mode: str = "full"  # full checkpoints or PEFT LoRA adapter checkpoints
    gene_groups: tuple[str, ...] = ("all",)
    search_methods: tuple[str, ...] = ()
    tensor_rules: tuple[tuple[str, str], ...] = ()  # (regex, group) overrides for unfamiliar model families
    max_output_rank: int = 256
    max_lora_output_mib: int = 1024
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @property
    def genome_size(self) -> int:
        return self.block_size * len(self.gene_groups)

    @property
    def block_size(self) -> int:
        return self.genes if len(self.parents) == 2 else self.genes * len(self.parents)

    @property
    def methods(self) -> tuple[str, ...]:
        return self.search_methods if self.method == "auto" else (self.method,)


class SpecError(ValueError):
    pass


def _task_definition(name: str, block: Any) -> str | None:
    if block is None:
        return None
    from .tasks import TaskError, validate_definition
    try:
        validate_definition(name, block)
    except TaskError as exc:
        raise SpecError(str(exc)) from exc
    return json.dumps(block, sort_keys=True, ensure_ascii=False)


def parse_spec(raw: dict[str, Any]) -> Spec:
    if not isinstance(raw, dict):
        raise SpecError("Experiment YAML must contain a mapping")
    try:
        parents_raw = raw["parents"]
        if not isinstance(parents_raw, list) or not (2 <= len(parents_raw) <= 6):
            raise SpecError("Choose between two and six parent models")
        parents = tuple(Parent(str(p["name"]), str(p["model"])) for p in parents_raw)
        eval_raw = raw.get("evaluation") or {}
        task_defs = eval_raw.get("tasks") or {}
        if not isinstance(task_defs, dict):
            raise SpecError("evaluation.tasks must map task names to settings")
        tasks = tuple(
            EvalTask(
                name=str(name),
                metric=str((opts or {}).get("metric", "acc_norm,none")),
                weight=float((opts or {}).get("weight", 1.0)),
                definition=_task_definition(str(name), (opts or {}).get("task")),
            )
            for name, opts in task_defs.items()
        )
        limit = eval_raw.get("limit")
        fewshot = eval_raw.get("num_fewshot")
        if not isinstance(eval_raw.get("apply_chat_template", False), bool):
            raise SpecError("evaluation.apply_chat_template must be true or false")
        evaluation = EvalConfig(
            tasks=tasks,
            device=str(eval_raw.get("device", "cpu")),
            batch_size=int(eval_raw.get("batch_size", 1)),
            limit=None if limit is None else int(limit),
            penalty=float(eval_raw.get("penalty", 0.0)),
            num_fewshot=None if fewshot is None else int(fewshot),
            apply_chat_template=eval_raw.get("apply_chat_template", False),
            screening_limit=(None if eval_raw.get("screening_limit") is None else int(eval_raw["screening_limit"])),
            promote_top=(None if eval_raw.get("promote_top") is None else int(eval_raw["promote_top"])),
        )
        spec = Spec(
            name=str(raw["name"]),
            base_model=str(raw["base_model"]),
            parents=parents,
            method=str(raw.get("method", "linear")),
            population=int(raw.get("population", 6)),
            genes=int(raw.get("genes", 6)),
            mutation_sigma=float(raw.get("mutation_sigma", 0.12)),
            seed=int(raw.get("seed", 42)),
            out_dtype=str(raw.get("out_dtype", "bfloat16")),
            density=float(raw.get("density", 0.5)),
            task_scale=float(raw.get("task_scale", 1.0)),
            selection=str(raw.get("selection", "weighted")),
            search=str(raw.get("search", "evolution")),
            mode=str(raw.get("mode", "full")),
            gene_groups=tuple(raw.get("gene_groups", ["all"])),
            search_methods=tuple(raw.get("search_methods") or ()),
            tensor_rules=tuple((str(r["match"]), str(r["group"])) for r in (raw.get("tensor_rules") or ())),
            max_output_rank=int(raw.get("max_output_rank", 256)),
            max_lora_output_mib=int(raw.get("max_lora_output_mib", 1024)),
            evaluation=evaluation,
        )
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        if isinstance(exc, SpecError):
            raise
        raise SpecError(f"Invalid experiment config: {exc}") from exc

    if not spec.name.strip() or any(c in spec.name for c in "/\\") or spec.name in {".", ".."}:
        raise SpecError("name must be a non-empty simple identifier")
    if spec.method not in SUPPORTED_METHODS | {"auto"}:
        raise SpecError(f"Unsupported method: {spec.method}")
    if spec.out_dtype not in SUPPORTED_DTYPES:
        raise SpecError(f"Unsupported out_dtype: {spec.out_dtype}")
    if spec.mode not in SUPPORTED_MODES:
        raise SpecError("mode must be full or lora")
    if not isinstance(raw.get("gene_groups", ["all"]), (list, tuple)) or not all(isinstance(g, str) for g in spec.gene_groups):
        raise SpecError("gene_groups must be a list of group names")
    if spec.method == "auto" and not isinstance(raw.get("search_methods"), (tuple, list)):
        raise SpecError("search_methods must be a list")
    if not spec.gene_groups or len(set(spec.gene_groups)) != len(spec.gene_groups):
        raise SpecError("gene_groups must be a nonempty list without duplicates")
    if spec.gene_groups != ("all",) and ("all" in spec.gene_groups or "other" not in spec.gene_groups or
                                              not set(spec.gene_groups) <= GENE_GROUPS):
        raise SpecError("gene_groups must be ['all'] or a subset of attention/mlp/router/norm/embedding PLUS other")
    if spec.method == "auto":
        if not spec.search_methods:
            raise SpecError("method: auto requires explicit search_methods")
        if len(set(spec.search_methods)) != len(spec.search_methods):
            raise SpecError("search_methods cannot contain duplicates")
        if not set(spec.search_methods) <= SUPPORTED_METHODS:
            raise SpecError("search_methods contain an unsupported merge method")
    elif spec.search_methods:
        raise SpecError("search_methods can only be used with method: auto")
    if spec.mode == "lora" and not set(spec.methods) <= LORA_METHODS:
        raise SpecError("LoRA mode supports linear and task_arithmetic only")
    if not 1 <= spec.max_output_rank <= 4096:
        raise SpecError("max_output_rank must be between 1 and 4096")
    if not 1 <= spec.max_lora_output_mib <= 131072:
        raise SpecError("max_lora_output_mib must be between 1 and 131072")
    if spec.selection not in SUPPORTED_SELECTION:
        raise SpecError(f"Unsupported selection: {spec.selection}")
    for pattern, group in spec.tensor_rules:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SpecError(f"tensor_rules: invalid regex {pattern!r}: {exc}") from exc
        if group not in GENE_GROUPS:
            raise SpecError(f"tensor_rules: unknown group {group!r}; use one of {sorted(GENE_GROUPS)}")
    if spec.search not in ("evolution", "gp"):
        raise SpecError(f"Unsupported search: {spec.search} (use evolution or gp)")
    if spec.search == "gp" and spec.method == "auto":
        raise SpecError("search: gp optimizes merge weights for one fixed method; method: auto is not supported")
    if not 2 <= spec.population <= 256:
        raise SpecError("population must be between 2 and 256")
    if not 2 <= spec.genes <= 128:
        raise SpecError("genes must be between 2 and 128")
    if not math.isfinite(spec.mutation_sigma) or not 0 <= spec.mutation_sigma <= 1:
        raise SpecError("mutation_sigma must be within [0,1]")
    if not math.isfinite(spec.density) or not 0 < spec.density <= 1:
        raise SpecError("density must be within (0,1]")
    if not math.isfinite(spec.task_scale) or not 0 <= spec.task_scale <= 3:
        raise SpecError("task_scale must be finite and between 0 and 3")
    if len({p.name for p in spec.parents}) != len(spec.parents):
        raise SpecError("Parent names must be unique")
    if len({p.model for p in spec.parents}) != len(spec.parents):
        raise SpecError("Parent references must be unique")
    if any(not p.name.strip() or not p.model.strip() for p in spec.parents) or not spec.base_model.strip():
        raise SpecError("Model references and parent names cannot be empty")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", p.name) or p.name in {"base", "all"} for p in spec.parents):
        raise SpecError("Parent names must be folder-safe (letters/numbers/._-) and cannot be 'base' or 'all'")
    if spec.evaluation.batch_size < 1 or spec.evaluation.limit is not None and spec.evaluation.limit < 1:
        raise SpecError("batch_size and limit must be positive")
    if spec.evaluation.num_fewshot is not None and spec.evaluation.num_fewshot < 0:
        raise SpecError("num_fewshot must be nonnegative")
    if not math.isfinite(spec.evaluation.penalty) or spec.evaluation.penalty < 0:
        raise SpecError("evaluation.penalty must be finite and nonnegative")
    if any(not t.name or not t.metric or not math.isfinite(t.weight) or t.weight <= 0 for t in spec.evaluation.tasks):
        raise SpecError("Every task needs a name, metric, and positive finite weight")
    if any("/" in t.name or "\\" in t.name for t in spec.evaluation.tasks):
        raise SpecError("evaluation task names cannot contain path separators")
    screening = spec.evaluation.screening_limit
    top = spec.evaluation.promote_top
    if (screening is None) != (top is None):
        raise SpecError("evaluation.screening_limit and promote_top must be configured together")
    if screening is not None:
        if screening < 1 or top < 2 or top >= spec.population:
            raise SpecError("Staged evaluation requires screening_limit >=1 and 2 <= promote_top < population")
        if spec.evaluation.limit is not None and screening >= spec.evaluation.limit:
            raise SpecError("screening_limit must be smaller than final evaluation.limit")
    if spec.selection == "pareto" and len(spec.evaluation.tasks) < 2:
        raise SpecError("Pareto selection requires at least two evaluation tasks")
    return spec


def load_spec(path: str | Path) -> Spec:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_spec(raw)


def spec_to_dict(spec: Spec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "base_model": spec.base_model,
        "parents": [{"name": p.name, "model": p.model} for p in spec.parents],
        "method": spec.method,
        "population": spec.population,
        "genes": spec.genes,
        "mutation_sigma": spec.mutation_sigma,
        "seed": spec.seed,
        "out_dtype": spec.out_dtype,
        "density": spec.density,
        "task_scale": spec.task_scale,
        "selection": spec.selection,
        **({"search": spec.search} if spec.search != "evolution" else {}),
        "mode": spec.mode,
        "gene_groups": list(spec.gene_groups),
        "max_output_rank": spec.max_output_rank,
        "max_lora_output_mib": spec.max_lora_output_mib,
        **({"search_methods": list(spec.search_methods)} if spec.method == "auto" else {}),
        **({"tensor_rules": [{"match": m, "group": g} for m, g in spec.tensor_rules]} if spec.tensor_rules else {}),
        "evaluation": {
            "tasks": {t.name: {"metric": t.metric, "weight": t.weight, **({"task": json.loads(t.definition)} if t.definition else {})}
                  for t in spec.evaluation.tasks},
            "device": spec.evaluation.device,
            "batch_size": spec.evaluation.batch_size,
            "limit": spec.evaluation.limit,
            "penalty": spec.evaluation.penalty,
            "num_fewshot": spec.evaluation.num_fewshot,
            "apply_chat_template": spec.evaluation.apply_chat_template,
            "screening_limit": spec.evaluation.screening_limit,
            "promote_top": spec.evaluation.promote_top,
        },
    }


def save_spec(spec: Spec, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec_to_dict(spec), f, sort_keys=False, allow_unicode=True)
