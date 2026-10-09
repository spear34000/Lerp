"""Resumable, versioned, auditable disk-based merge search experiments.

`SIMULATED_TOY` scores are never confused with real evaluation results.
Actual heavyweight merge and scoring are delegated to MergeKit / lm-eval.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .compat import check_compatibility, resolve_spec_paths
from .genetics import breed_generation, demo_objective, initial_genomes, rank_entries
from .integrity import verify_frozen_inputs
from .artifacts import pin_artifact, verify_artifact
from .merge import mergekit_config
from .spec import Spec, load_spec, save_spec, spec_to_dict


# Scores whose evaluator is part of the verified toolchain: the lm-eval harness, or Lerp's own resident evaluator
# (lerp.resident), whose prompts, continuation tokenization and metrics are checked against lm-eval.
VERIFIED_SOURCES = frozenset({"lm_eval", "lerp_eval"})


class BreederError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def generation_dir(run: Path, gen: int) -> Path:
    if gen < 0:
        raise BreederError("Generation number cannot be negative")
    return run / "generations" / f"gen-{gen:03d}"


def candidate_dir(run: Path, gen: int, candidate: int) -> Path:
    if candidate < 0:
        raise BreederError("Candidate number cannot be negative")
    return generation_dir(run, gen) / f"cand-{candidate:03d}"


def load_run(run: Path) -> tuple[Spec, dict]:
    spec_path = run / "experiment.yaml"
    state_path = run / "state.json"
    if not spec_path.is_file() or not state_path.is_file():
        raise BreederError(f"Not a Lerp run: {run}")
    state = _read_json(state_path)
    if state.get("schema_version") not in (1, 2, 3):
        raise BreederError("Unsupported state schema (expected v1, v2 or v3)")
    if state.get("schema_version") in (2, 3) and _sha(spec_path.read_bytes()) != state.get("config_sha256"):
        raise BreederError("Run config changed after initialization. Restore experiment.yaml or start a new run")
    return load_spec(spec_path), state


def _write_candidate(spec: Spec, run: Path, gen: int, idx: int, genes: list[float], parents: list[str],
                     *, method: str | None = None, generation_override: Path | None = None) -> Path:
    path = (generation_override / f"cand-{idx:03d}") if generation_override else candidate_dir(run, gen, idx)
    path.mkdir(parents=True, exist_ok=False)
    uid = f"g{gen:03d}-c{idx:03d}"
    selected_method = method or spec.methods[0]
    if selected_method not in spec.methods:
        raise BreederError(f"Method {selected_method} not in configured search space")
    recipe = yaml.safe_dump(mergekit_config(spec, genes, method=selected_method), sort_keys=False, allow_unicode=True)
    recipe_sha = _sha(recipe.encode("utf-8"))
    _write_json(path / "genome.json", {
        "id": uid, "generation": gen, "candidate": idx, "genes": genes,
        "lineage": parents, "created_at": _now(), "recipe_sha256": recipe_sha,
        "method": selected_method, "mode": spec.mode, "gene_groups": list(spec.gene_groups),
    })
    (path / "merge.yaml").write_bytes(recipe.encode("utf-8"))  # bytes, not text: Windows text mode would turn \n into \r\n and break recipe_sha256
    return path


def init_run(source_yaml: Path, run: Path, *, check_remote: bool = False) -> tuple[dict, Path]:
    raw = load_spec(source_yaml)
    spec = resolve_spec_paths(raw, source_yaml.resolve().parent)
    report = check_compatibility(spec, remote=check_remote)
    if not report.ok:
        raise BreederError("Preflight failed:\n" + "\n".join(report.errors))
    if len(spec.parents) >= 3 and spec.population < len(spec.parents) + 1:
        report.warnings.append("Population too small to seed all single-parent specialists plus a balanced mixture")
    if (spec.method == "auto" and "linear" in spec.methods and "task_arithmetic" in spec.methods
            and spec.task_scale == 1.0):
        report.warnings.append("At task_scale=1, linear and task_arithmetic are mathematically identical for convex weights")
    if run.exists() and (not run.is_dir() or any(run.iterdir())):
        raise BreederError(f"Output directory already contains files: {run}")
    run.mkdir(parents=True, exist_ok=True)
    save_spec(spec, run / "experiment.yaml")
    state = {
        "schema_version": 3,
        "name": spec.name,
        "generation": 0,
        "created_at": _now(),
        "last_advance_at": None,
        "preflight_warnings": report.warnings,
        "config_sha256": _sha((run / "experiment.yaml").read_bytes()),
    }
    _write_json(run / "state.json", state)
    _write_json(run / "manifest.json", {
        "modelbreeder_version": "0.5.0",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "run_config_sha256": state["config_sha256"],
        "selection": spec.selection,
        "mode": spec.mode, "search_methods": list(spec.methods),
        "gene_groups": list(spec.gene_groups),
        "parent_references": {p.name: p.model for p in spec.parents},
        "base_model": spec.base_model,
        "seed": spec.seed,
        "created_at": state["created_at"],
        "disclaimer": "Local model files and mutable Hub branch names are not content-addressed; pin model revisions externally.",
    })
    genes = initial_genomes(spec.population, spec.genes, spec.seed, len(spec.parents), groups=len(spec.gene_groups))
    for idx, dna in enumerate(genes):
        _write_candidate(spec, run, 0, idx, dna, [], method=spec.methods[idx % len(spec.methods)])
    return state, run


def load_candidate(run: Path, gen: int, idx: int) -> dict:
    path = candidate_dir(run, gen, idx)
    if not path.is_dir():
        raise BreederError(f"Unknown candidate: g{gen:03d}-c{idx:03d}")
    info = _read_json(path / "genome.json")
    score_path = path / "score.json"
    if score_path.is_file():
        info["score"] = _read_json(score_path)
    screen_path = path / "screen_score.json"
    if screen_path.is_file():
        info["screen_score"] = _read_json(screen_path)
    return info


def _all_candidates(run: Path) -> list[dict]:
    result = []
    for gen_folder in sorted((run / "generations").glob("gen-*")):
        for cfolder in sorted(gen_folder.glob("cand-*")):
            if (cfolder / "genome.json").is_file():
                info = _read_json(cfolder / "genome.json")
                if (cfolder / "score.json").is_file():
                    info["score"] = _read_json(cfolder / "score.json")
                if (cfolder / "screen_score.json").is_file():
                    info["screen_score"] = _read_json(cfolder / "screen_score.json")
                result.append(info)
    return result


def compute_fitness(spec: Spec, metrics: dict[str, float]) -> float:
    expected = {task.name for task in spec.evaluation.tasks}
    if set(metrics) != expected or not expected:
        raise BreederError(f"Task scores must be exactly {sorted(expected)}, got {sorted(metrics)}")
    if not all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in metrics.values()):
        raise BreederError("All task metrics must be finite numbers between 0 and 1")
    weights = {task.name: task.weight for task in spec.evaluation.tasks}
    mean = sum(metrics[k] * weights[k] for k in metrics) / sum(weights.values())
    gap = max(metrics.values()) - min(metrics.values())
    return round(mean - spec.evaluation.penalty * gap, 8)


def _score_document(spec: Spec, metrics: dict[str, float], source: str, *,
                    evidence: str | None = None, runtime_device: str | None = None, backend: str = "hf") -> dict:
    if source not in {"manual", "lm_eval", "lerp_eval"}:
        raise BreederError("Score source must be manual, lm_eval or lerp_eval")
    return {
        "fitness": compute_fitness(spec, metrics),
        "metrics": metrics,
        "source": source,
        "status": ("unverified_manual" if source == "manual" else
                   "smoke_test" if spec.evaluation.limit is not None else "measured"),
        "recorded_at": _now(),
        "evidence": evidence,
        "evaluation_settings": {
            "tasks": {t.name: t.metric for t in spec.evaluation.tasks},
            "limit": spec.evaluation.limit,
            "fewshot": spec.evaluation.num_fewshot,
            "chat_template": spec.evaluation.apply_chat_template,
            "device": runtime_device or spec.evaluation.device,
            "batch_size": spec.evaluation.batch_size,
            "backend": backend,
        },
    }


def record_score(run: Path, gen: int, idx: int, metrics: dict[str, float], *, source: str, overwrite: bool = False,
                 evidence: str | None = None, runtime_device: str | None = None, backend: str = "hf") -> dict:
    spec, _ = load_run(run)
    load_candidate(run, gen, idx)
    doc = _score_document(spec, metrics, source, evidence=evidence, runtime_device=runtime_device, backend=backend)
    path = candidate_dir(run, gen, idx) / "score.json"
    if path.exists() and not overwrite:
        raise BreederError("Already scored. Use --overwrite only if intentionally replacing measurements")
    _write_json(path, doc)
    return doc


def simulate_generation(run: Path, gen: int) -> list[dict]:
    """Explicitly fake objective scores for offline CLI / search validation."""
    spec, _ = load_run(run)
    result = []
    if not spec.evaluation.tasks:
        raise BreederError("Demo requires evaluation.tasks")
    for idx in range(spec.population):
        info = load_candidate(run, gen, idx)
        path = candidate_dir(run, gen, idx) / "score.json"
        if path.exists():
            raise BreederError(f"Refusing to replace existing score for {info['id']}")
        toy = demo_objective(info["genes"], parents=len(spec.parents), points=spec.genes, groups=len(spec.gene_groups))
        metrics = {task.name: toy["toy_code" if j % 2 == 0 else "toy_reason"]
                   for j, task in enumerate(spec.evaluation.tasks)}
        doc = {
            "fitness": compute_fitness(spec, metrics), "metrics": metrics,
            "source": "SIMULATED_TOY", "status": "FAKE_DO_NOT_REPORT",
            "recorded_at": _now(), "toy_components": toy,
        }
        _write_json(path, doc)
        result.append({"id": info["id"], **doc})
    return result


def advance(run: Path, *, allow_simulated: bool = False, allow_manual: bool = False,
            retry_partial: bool = False) -> int:
    spec, state = load_run(run)
    verify_frozen_inputs(run, spec)
    prev = state["generation"]
    next_gen = prev + 1
    published = generation_dir(run, next_gen)
    # Crash after atomic directory publish but before state.json update: recover.
    if published.exists():
        commit = published / "generation_commit.json"
        if commit.is_file():
            doc = _read_json(commit)
            expected = {f"cand-{i:03d}" for i in range(spec.population)}
            found = {p.name for p in published.iterdir() if p.is_dir()}
            if (doc.get("previous_generation") == prev and doc.get("generation") == next_gen
                    and doc.get("config_sha256") == state.get("config_sha256") and found == expected
                    and all((published / folder / "genome.json").is_file() for folder in expected)):
                state["generation"] = next_gen
                state["last_advance_at"] = _now()
                _write_json(run / "state.json", state)
                return next_gen
        raise BreederError("Next-generation directory already exists but cannot be safely recovered")
    current = [load_candidate(run, prev, i) for i in range(spec.population)]
    if any("score" not in entry and "screen_score" not in entry for entry in current):
        raise BreederError("Every candidate in current generation must be scored or screened before advancing")
    if any("screen_score" in entry and entry.get("score", {}).get("source") == "SIMULATED_TOY" for entry in current):
        raise BreederError("Mixing staged screening and fake results is not allowed")
    if any("screen_score" in entry for entry in current) and sum("score" in e for e in current) < 2:
        raise BreederError("At least two candidates must complete full evaluation before advancement")
    if not allow_simulated and any(e.get("score", {}).get("source") == "SIMULATED_TOY" for e in current):
        raise BreederError("Simulated scores cannot advance real experiments. Use --allow-simulated for demos")
    if not allow_manual and any(e.get("score", {}).get("source") == "manual" for e in current):
        raise BreederError("Unverified manual scores cannot drive default evolution; use --allow-manual only if external measurements were independently verified")
    eligible = [c for c in _all_candidates(run) if "score" in c]
    if not allow_simulated:
        eligible = [c for c in eligible if c["score"]["source"] != "SIMULATED_TOY"]
    if not allow_manual:
        eligible = [c for c in eligible if c["score"]["source"] != "manual"]
    observed = list(eligible)  # every verified score so far; the GP search fits all of them
    # Pareto fronts + crowding distance preserve specialists, not just a scalar winner.
    eligible = rank_entries(eligible, spec.selection, [t.name for t in spec.evaluation.tasks])
    eligible = eligible[:max(2, spec.population * 2)]
    if len(eligible) < 2:
        raise BreederError("Need at least two scored candidates")
    if spec.search == "gp":
        from .gp import GPSearchError, propose_batch
        try:
            proposals = propose_batch([(c["genes"], c["score"]["fitness"]) for c in observed],
                                      spec, spec.population, spec.seed + next_gen * 100003)
        except GPSearchError as exc:
            raise BreederError(str(exc)) from exc
        children = [(dna, []) for dna in proposals]
    else:
        children = breed_generation(
            eligible, spec.population, spec.mutation_sigma, spec.seed + next_gen * 100003,
            parents=len(spec.parents), points=spec.genes, groups=len(spec.gene_groups),
            strategy=spec.selection, tasks=[t.name for t in spec.evaluation.tasks],
        )
    method_by_id = {e["id"]: e.get("method", spec.methods[0]) for e in eligible}
    pending = generation_dir(run, next_gen).with_name(f".gen-{next_gen:03d}.partial")
    if pending.exists():
        if not retry_partial:
            raise BreederError(f"Interrupted generation creation at {pending}; inspect, then pass --retry-partial")
        if pending.is_symlink():
            raise BreederError("Refusing to remove symlinked partial generation")
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    for idx, (dna, lineage) in enumerate(children):
        if spec.method == "auto":
            rng = random.Random(spec.seed + next_gen * 1010003 + idx)
            inherited = method_by_id.get(rng.choice(lineage), spec.methods[0])
            selected = rng.choice(spec.methods) if rng.random() < 0.25 else inherited
        else:
            selected = spec.method
        _write_candidate(spec, run, next_gen, idx, dna, lineage,
                         method=selected, generation_override=pending)
    _write_json(pending / "generation_commit.json", {
        "previous_generation": prev, "generation": next_gen,
        "population": spec.population, "config_sha256": state["config_sha256"],
        "created_at": _now(),
    })
    pending.replace(published)
    state["generation"] = next_gen
    state["last_advance_at"] = _now()
    _write_json(run / "state.json", state)
    return next_gen


def leaderboard(run: Path, *, include_simulated: bool = True, pareto: bool = False) -> list[dict]:
    spec, _ = load_run(run)
    records = [c for c in _all_candidates(run) if "score" in c]
    if not include_simulated:
        records = [c for c in records if c["score"]["source"] != "SIMULATED_TOY"]
    return rank_entries(records, "pareto" if pareto else "weighted", [t.name for t in spec.evaluation.tasks])


def _require_command(name: str, extra: str) -> None:
    if not shutil.which(name):
        raise BreederError(f"Command {name!r} not found. Install dependency: {extra}")


def _execute_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("Command: " + json.dumps(command, ensure_ascii=False) + "\n\n")
        f.flush()
        proc = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        raise BreederError(f"Command failed (exit {proc.returncode}). See {log}")


def _model_weights_present(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def build_candidate(run: Path, gen: int, idx: int, *, cuda: bool = False, retry_partial: bool = False, engine: str = "mergekit") -> Path:
    spec, _ = load_run(run)
    verify_frozen_inputs(run, spec)
    info = load_candidate(run, gen, idx)
    if engine not in ("mergekit", "lite", "lora"):
        raise BreederError("engine must be 'mergekit', 'lite' or 'lora'")
    method = info.get("method", spec.methods[0])
    if method not in spec.methods:
        raise BreederError("Candidate method is not allowed by experiment")
    if spec.mode == "lora" and engine != "lora":
        raise BreederError("LoRA experiments must use --engine lora")
    if spec.mode == "full" and engine == "lora":
        raise BreederError("Full checkpoint experiments cannot use --engine lora")
    if engine == "mergekit" and len(spec.gene_groups) > 1:
        raise BreederError("Module-group-specific genomes require --engine lite or --engine lora")
    if engine == "mergekit":
        _require_command("mergekit-yaml", 'pip install -e ".[merge]"')
    if engine in ("lite", "lora") and cuda:
        raise BreederError("The lite/lora engine is CPU-only; remove --cuda")
    path = candidate_dir(run, gen, idx)
    recipe = path / "merge.yaml"
    generated_recipe = yaml.safe_dump(mergekit_config(spec, info["genes"], method=method), sort_keys=False, allow_unicode=True)
    original_sha = info.get("recipe_sha256")
    if _sha(recipe.read_bytes()) != original_sha or _sha(generated_recipe.encode("utf-8")) != original_sha:
        raise BreederError(f"Merge recipe or genome was modified since candidate creation: {recipe}")
    output = path / "model"
    partial = path / ".model.partial"
    if output.exists():
        raise BreederError(f"Output model already exists; will not overwrite: {output}")
    if partial.exists():
        if not retry_partial:
            raise BreederError(f"Interrupted model build at {partial}; inspect it, then retry with --retry-partial")
        if partial.is_symlink():
            raise BreederError("Refusing to remove symlinked partial build")
        shutil.rmtree(partial)
    engine_details: dict = {}
    if engine == "lite":
        from .lite import LiteMergeError, build_lite
        try:
            engine_details = build_lite(spec, info["genes"], partial, method=method)
        except LiteMergeError as exc:
            raise BreederError(f"Lite backend error: {exc}") from exc
        (path / "build.log").write_text(json.dumps(engine_details, indent=2) + "\n", encoding="utf-8")
    elif engine == "lora":
        from .lora import LoRAMergeError, build_lora
        try:
            engine_details = build_lora(spec, info["genes"], partial, method=method)
        except LoRAMergeError as exc:
            raise BreederError(f"LoRA backend error: {exc}") from exc
        (path / "build.log").write_text(json.dumps(engine_details, indent=2) + "\n", encoding="utf-8")
    else:
        command = ["mergekit-yaml", str(recipe), str(partial)]
        if cuda:
            command.append("--cuda")
        _execute_logged(command, path / "build.log")
    if engine == "lora":
        finished_ok = (partial / "adapter_config.json").is_file() and (partial / "adapter_model.safetensors").is_file()
    else:
        finished_ok = (partial / "config.json").is_file() and _model_weights_present(partial)
    if not finished_ok:
        raise BreederError(f"Merge completed without expected checkpoint artifacts / weight tensors in {partial}; not marking it built")
    pin_artifact(partial)
    partial.replace(output)
    _write_json(path / "build.json", {
        "built_at": _now(), "model_path": str(output.resolve()),
        "recipe_sha256": _sha(recipe.read_bytes()), "cuda": cuda,
        "engine": engine, "method": method, "mode": spec.mode, "engine_details": engine_details,
        "weight_files": [p.name for p in sorted(output.glob("*.safetensors"))] +
                        [p.name for p in sorted(output.glob("pytorch_model*.bin"))],
    })
    return output


def _extract_lm_eval_scores(path: Path, spec: Spec) -> tuple[dict[str, float], Path]:
    json_paths = sorted(path.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate_path in json_paths:
        try:
            data = _read_json(candidate_path)
        except (ValueError, OSError, TypeError):
            continue
        table = data.get("results")
        if not isinstance(table, dict):
            continue
        values = {}
        for task in spec.evaluation.tasks:
            result = table.get(task.name)
            if not isinstance(result, dict) or task.metric not in result:
                break
            values[task.name] = result[task.metric]
        if len(values) == len(spec.evaluation.tasks):
            compute_fitness(spec, values)
            return values, candidate_path
    raise BreederError(f"Could not find valid requested task metrics in lm-eval output under {path}")


def _eval_model_args(spec: Spec, reference: str, *, adapter: bool = False) -> str:
    """LM Eval requires PEFT adapters to be loaded on their declared base."""
    base = spec.base_model if adapter else reference
    if "," in base or (adapter and "," in reference):
        raise BreederError("Comma in model reference is unsupported by lm-eval model_args syntax")
    return f"pretrained={base},peft={reference}" if adapter else f"pretrained={reference}"


def _run_evaluation(
    spec: Spec, checkpoint: str, folder: Path, *, device: str | None,
    overwrite: bool, retry_partial: bool, stage: str = "final", adapter: bool = False,
) -> tuple[dict[str, float], str]:
    if stage not in ("final", "screen"):
        raise BreederError("stage must be final or screen")
    if stage == "screen" and spec.evaluation.screening_limit is None:
        raise BreederError("Screening requires evaluation.screening_limit and promote_top")
    _require_command("lm-eval", 'pip install -e ".[eval]"')
    basename = "evaluation" if stage == "final" else "screening"
    finished = folder / basename
    partial = folder / f".{basename}.partial"
    if finished.exists() and not overwrite:
        raise BreederError(f"{basename} results already exist: {finished}; pass --overwrite to evaluate again")
    if partial.exists():
        if not retry_partial:
            raise BreederError(f"Interrupted {basename} at {partial}; inspect it then pass --retry-partial")
        if partial.is_symlink():
            raise BreederError("Refusing to remove symlinked partial evaluation")
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    cmd = [
        "lm-eval", "run", "--model", "hf", "--model_args", _eval_model_args(spec, checkpoint, adapter=adapter),
        "--tasks", ",".join(t.name for t in spec.evaluation.tasks),
        "--device", device or spec.evaluation.device,
        "--batch_size", str(spec.evaluation.batch_size),
        "--output_path", str(partial),
    ]
    limit = spec.evaluation.limit if stage == "final" else spec.evaluation.screening_limit
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if spec.evaluation.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(spec.evaluation.num_fewshot)])
    if spec.evaluation.apply_chat_template:
        cmd.append("--apply_chat_template")
    _write_json(partial / "modelbreeder_protocol.json", {
        "model_reference": checkpoint, "adapter": adapter, "stage": stage,
        "model_args": _eval_model_args(spec, checkpoint, adapter=adapter),
        "tasks": {t.name: t.metric for t in spec.evaluation.tasks},
        "device": device or spec.evaluation.device, "batch_size": spec.evaluation.batch_size,
        "limit": limit, "num_fewshot": spec.evaluation.num_fewshot,
        "apply_chat_template": spec.evaluation.apply_chat_template,
        "command": cmd,
    })
    _execute_logged(cmd, folder / f"{basename}.log")
    metrics, evidence_path = _extract_lm_eval_scores(partial, spec)
    relative_evidence = evidence_path.relative_to(partial)
    if finished.exists():
        backup = folder / f".{basename}.backup"
        if backup.exists():
            raise BreederError(f"Interrupted overwrite backup at {backup}; inspect before retry")
        finished.replace(backup)
        try:
            partial.replace(finished)
        except Exception:
            backup.replace(finished)
            raise
        if backup.is_symlink():
            raise BreederError("Refusing to remove symlinked evaluation backup")
        shutil.rmtree(backup)
    else:
        partial.replace(finished)
    return metrics, str(Path(basename) / relative_evidence)


def _model_complete(spec: Spec, folder: Path) -> bool:
    if spec.mode == "lora":
        return (folder / "adapter_config.json").is_file() and (folder / "adapter_model.safetensors").is_file()
    return (folder / "config.json").is_file() and _model_weights_present(folder)


def _protocol_device(folder: Path, stage: str, fallback: str) -> str:
    filename = "screening" if stage == "screen" else "evaluation"
    protocol = folder / filename / "modelbreeder_protocol.json"
    if not protocol.is_file():
        return fallback
    doc = _read_json(protocol)
    if doc.get("stage") != stage or not isinstance(doc.get("device"), str):
        raise BreederError("Evaluator protocol metadata is invalid")
    return doc["device"]


def _store_screen_score(spec: Spec, folder: Path, metrics: dict[str, float], evidence: str) -> dict:
    doc = _score_document(spec, metrics, "lm_eval", evidence=evidence,
                          runtime_device=_protocol_device(folder, "screen", spec.evaluation.device))
    doc["status"] = "screen_only_not_full_evaluation"
    doc["evaluation_settings"]["limit"] = spec.evaluation.screening_limit
    doc["evaluation_settings"]["stage"] = "screen"
    _write_json(folder / "screen_score.json", doc)
    return doc


def screen_candidate(run: Path, gen: int, idx: int, *, device: str | None = None,
                     overwrite: bool = False, retry_partial: bool = False) -> dict:
    spec, _ = load_run(run)
    verify_frozen_inputs(run, spec)
    if spec.evaluation.screening_limit is None or not spec.evaluation.tasks:
        raise BreederError("Screening requires screening_limit, promote_top and evaluation.tasks")
    load_candidate(run, gen, idx)
    folder = candidate_dir(run, gen, idx)
    if not _model_complete(spec, folder / "model"):
        raise BreederError("Cannot screen: complete model/adapter missing; run build first")
    verify_artifact(folder / "model")
    if (folder / "screen_score.json").exists() and not overwrite:
        raise BreederError("Candidate already screened")
    metrics, evidence = _run_evaluation(spec, str((folder / "model").resolve()), folder,
                                        device=device, overwrite=overwrite, retry_partial=retry_partial,
                                        stage="screen", adapter=spec.mode == "lora")
    return _store_screen_score(spec, folder, metrics, evidence)


def promotion_list(run: Path, gen: int) -> list[dict]:
    spec, state = load_run(run)
    if spec.evaluation.promote_top is None:
        raise BreederError("Staged evaluation is not configured")
    if gen < 0 or gen > state["generation"]:
        raise BreederError("Generation does not exist yet")
    candidates = [load_candidate(run, gen, i) for i in range(spec.population)]
    if any("screen_score" not in c for c in candidates):
        raise BreederError("Every candidate must be screened before promotion")
    ranked = rank_entries([{**c, "score": c["screen_score"]} for c in candidates],
                          spec.selection, [t.name for t in spec.evaluation.tasks])
    return ranked[:spec.evaluation.promote_top]


def evaluate_candidate(run: Path, gen: int, idx: int, *, device: str | None = None, overwrite: bool = False, retry_partial: bool = False) -> dict:
    spec, _ = load_run(run)
    verify_frozen_inputs(run, spec)
    if not spec.evaluation.tasks:
        raise BreederError("No evaluation.tasks configured")
    load_candidate(run, gen, idx)
    path = candidate_dir(run, gen, idx)
    model_dir = path / "model"
    if not _model_complete(spec, model_dir):
        raise BreederError(f"Missing complete model/adapter: {model_dir} (run build first)")
    verify_artifact(model_dir)
    if (path / "score.json").exists() and not overwrite:
        raise BreederError("Already scored. Pass --overwrite to recompute")
    metrics, evidence = _run_evaluation(spec, str(model_dir.resolve()), path, device=device, overwrite=overwrite,
                                        retry_partial=retry_partial, adapter=spec.mode == "lora")
    return record_score(run, gen, idx, metrics, source="lm_eval", overwrite=overwrite, evidence=evidence,
                        runtime_device=_protocol_device(path, "final", device or spec.evaluation.device))


def recover_candidate(run: Path, gen: int, idx: int) -> dict:
    """Recover scores when lm-eval finished but process crashed before score.json."""
    spec, _ = load_run(run)
    verify_frozen_inputs(run, spec)
    load_candidate(run, gen, idx)
    folder = candidate_dir(run, gen, idx)
    verify_artifact(folder / "model")
    if (folder / "score.json").is_file():
        raise BreederError("Candidate is already scored")
    results = folder / "evaluation"
    if not results.is_dir():
        raise BreederError("No completed evaluation/ folder. Use evaluate --retry-partial if only the temporary folder exists")
    metrics, evidence_file = _extract_lm_eval_scores(results, spec)
    evidence = str(Path("evaluation") / evidence_file.relative_to(results))
    return record_score(run, gen, idx, metrics, source="lm_eval", evidence=evidence,
                        runtime_device=_protocol_device(folder, "final", spec.evaluation.device))


def _baseline_ref(spec: Spec, name: str) -> str:
    if name == "base":
        return spec.base_model
    for parent in spec.parents:
        if name == parent.name:
            return parent.model
    raise BreederError(f"Unknown baseline {name!r}. Choose base or one of: {[p.name for p in spec.parents]}")


def record_baseline(run: Path, name: str, metrics: dict[str, float], *, source: str, evidence: str | None,
                    runtime_device: str, backend: str, overwrite: bool = False) -> dict:
    """Store a baseline score computed outside `evaluate_baseline` (for example by the resident evaluator)."""
    spec, _ = load_run(run)
    ref = _baseline_ref(spec, name)
    folder = run / "baselines" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "score.json").exists() and not overwrite:
        raise BreederError(f"Baseline {name} already scored; pass --overwrite")
    doc = {**_score_document(spec, metrics, source, evidence=evidence, runtime_device=runtime_device, backend=backend),
           "model_reference": ref, "baseline_name": name}
    _write_json(folder / "score.json", doc)
    return doc


def evaluate_baseline(run: Path, name: str, *, device: str | None = None, overwrite: bool = False, retry_partial: bool = False) -> dict:
    spec, _ = load_run(run)
    verify_frozen_inputs(run, spec)
    if not spec.evaluation.tasks:
        raise BreederError("No evaluation.tasks configured")
    ref = _baseline_ref(spec, name)
    folder = run / "baselines" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "score.json").exists() and not overwrite:
        raise BreederError(f"Baseline {name} already scored; pass --overwrite")
    metrics, evidence = _run_evaluation(spec, ref, folder, device=device, overwrite=overwrite,
                                        retry_partial=retry_partial, adapter=spec.mode == "lora" and name != "base")
    doc = {**_score_document(spec, metrics, "lm_eval", evidence=evidence,
                            runtime_device=_protocol_device(folder, "final", device or spec.evaluation.device)),
           "model_reference": ref, "baseline_name": name}
    _write_json(folder / "score.json", doc)
    return doc


def baselines(run: Path) -> dict[str, dict]:
    spec, _ = load_run(run)
    result = {}
    for name in ["base"] + [p.name for p in spec.parents]:
        path = run / "baselines" / name / "score.json"
        if path.is_file():
            result[name] = _read_json(path)
    return result


def comparisons(run: Path, *, include_simulated: bool = False) -> list[dict]:
    """Compare only measured evaluations with the *same* evaluation protocol.

    Unverified manual scores and fake demo scores must not be portrayed as
    improvements over lm-eval baselines, even when numeric fitness looks better.
    """
    candidates = leaderboard(run, include_simulated=include_simulated)
    measured = {name: score for name, score in baselines(run).items()
                if score.get("source") in VERIFIED_SOURCES}
    if not measured:
        raise BreederError("No verified-lm-eval parent baseline scores yet. Run: lerp baseline -r RUN --name all")
    comparisons_list = []
    for c in candidates:
        score = c["score"]
        if score.get("source") not in VERIFIED_SOURCES:
            continue
        comparable = [(name, baseline) for name, baseline in measured.items()
                      if score.get("evaluation_settings") == baseline.get("evaluation_settings")]
        if not comparable:
            continue
        best_baseline_name, best_baseline = max(comparable, key=lambda kv: kv[1]["fitness"])
        comparisons_list.append({
            "id": c["id"], "fitness": score["fitness"],
            "delta": round(score["fitness"] - best_baseline["fitness"], 8),
            "baseline": best_baseline_name, "score_source": score["source"],
            "status": score.get("status", "unknown"),
            "metrics": score["metrics"],
        })
    return comparisons_list


def cycle(run: Path, rounds: int, *, cuda: bool = False, device: str | None = None,
          retry_partial: bool = False, engine: str = "mergekit") -> list[int]:
    if not 1 <= rounds <= 100:
        raise BreederError("rounds must be between 1 and 100")
    completed = []
    for r in range(rounds):
        spec, state = load_run(run)
        verify_frozen_inputs(run, spec)
        if engine == "lite" and not set(spec.methods) <= {"linear", "task_arithmetic"}:
            raise BreederError("lite backend cannot execute all configured search_methods; use MergeKit")
        if spec.mode == "lora" and engine != "lora":
            raise BreederError("LoRA cycle requires --engine lora")
        if spec.mode == "full" and engine == "lora":
            raise BreederError("Full checkpoint cycle cannot use LoRA engine")
        if engine == "mergekit" and len(spec.gene_groups) > 1:
            raise BreederError("Grouped genomes require --engine lite/lora")
        gen = state["generation"]
        for idx in range(spec.population):
            item = load_candidate(run, gen, idx)
            if "score" in item:
                if item["score"]["source"] == "SIMULATED_TOY":
                    raise BreederError("Cannot auto-cycle candidates scored with synthetic data")
                continue
            folder = candidate_dir(run, gen, idx)
            if not _model_complete(spec, folder / "model"):
                build_candidate(run, gen, idx, cuda=cuda, retry_partial=retry_partial, engine=engine)
            if spec.evaluation.screening_limit is not None and "screen_score" not in item:
                if (folder / "screening").is_dir():
                    metrics, evidence_file = _extract_lm_eval_scores(folder / "screening", spec)
                    _store_screen_score(spec, folder, metrics, str(Path("screening") / evidence_file.relative_to(folder / "screening")))
                else:
                    screen_candidate(run, gen, idx, device=device, retry_partial=retry_partial)
        if spec.evaluation.screening_limit is not None:
            selected = {c["id"] for c in promotion_list(run, gen)}
        else:
            selected = {f"g{gen:03d}-c{i:03d}" for i in range(spec.population)}
        for idx in range(spec.population):
            item = load_candidate(run, gen, idx)
            if item["id"] not in selected or "score" in item:
                continue
            folder = candidate_dir(run, gen, idx)
            if (folder / "evaluation").is_dir():
                recover_candidate(run, gen, idx)
            else:
                evaluate_candidate(run, gen, idx, device=device, retry_partial=retry_partial)
        completed.append(gen)
        if r < rounds - 1:
            advance(run)
    return completed


def statuses(run: Path) -> list[dict]:
    load_run(run)
    output = []
    spec, _ = load_run(run)
    for info in _all_candidates(run):
        path = candidate_dir(run, info["generation"], info["candidate"])
        output.append({
            "id": info["id"], "generation": info["generation"],
            "built": _model_complete(spec, path / "model"),
            "screened": "screen_score" in info,
            "scored": "score" in info,
            "source": info.get("score", {}).get("source"),
            "partial": any((path / p).exists() for p in (".model.partial", ".evaluation.partial", ".evaluation.backup", ".screening.partial", ".screening.backup")),
        })
    return output


def lineage_dot(run: Path) -> str:
    load_run(run)
    records = _all_candidates(run)
    lines = ["digraph Lerp {", '  rankdir=LR;', '  node [shape=box,fontname="Arial"];']
    for cand in records:
        identifier = cand["id"].replace('"', "")
        label = identifier
        if "score" in cand:
            label += f"\\nfitness={cand['score']['fitness']:.4f}"
        lines.append(f'  "{identifier}" [label="{label}"];')
        for parent in cand.get("lineage", []):
            # Lineage is generated from safe candidate IDs, never user-controlled.
            lines.append(f'  "{parent}" -> "{identifier}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_recipe(run: Path, out: Path, *, candidate_id: str | None = None) -> Path:
    spec, _ = load_run(run)
    verify_frozen_inputs(run, spec)
    ranked = leaderboard(run, include_simulated=False)
    if candidate_id:
        matches = [c for c in ranked if c["id"] == candidate_id]
        if not matches:
            raise BreederError("Requested candidate has no real/manual evaluation score")
        winner = matches[0]
    else:
        if not ranked:
            raise BreederError("No scored real candidates to export")
        evaluated = [c for c in ranked if c["score"].get("source") in VERIFIED_SOURCES]
        if not evaluated:
            raise BreederError("No lm-eval-scored candidates: automatic champion selection refuses unverified manual scores")
        winner = evaluated[0]
    if out.exists() and any(out.iterdir()):
        raise BreederError(f"Export destination not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    folder = candidate_dir(run, winner["generation"], winner["candidate"])
    for filename in ("merge.yaml", "genome.json", "score.json"):
        shutil.copy2(folder / filename, out / filename)
    _write_json(out / "export.json", {
        "candidate": winner["id"],
        "recipe_sha256": _sha((out / "merge.yaml").read_bytes()),
        "config_sha256": _sha((run / "experiment.yaml").read_bytes()),
        "base": spec.base_model,
        "parents": {p.name: p.model for p in spec.parents},
        "method": winner.get("method", spec.methods[0]),
        "mode": spec.mode, "gene_groups": list(spec.gene_groups),
        "selection": spec.selection,
        "score": winner["score"],
        "baselines": baselines(run),
    })
    (out / "MODEL_CARD_DRAFT.md").write_text(
        f"# {spec.name} / {winner['id']} (RECIPE ONLY)\n\n"
        f"Merged from {len(spec.parents)} parent checkpoints/adapters using method `{winner.get('method', spec.methods[0])}`.\n\n"
        f"- Fitness: {winner['score']['fitness']:.6f}\n"
        f"- Source: {winner['score']['source']}\n"
        f"- Validation status: {winner['score'].get('status', 'unknown')}\n"
        "- Benchmark protocol and training data need independent review.\n"
        "- Check license compatibility and original model attribution before distributing.\n"
        "- No trained/merged weights are included. Run MergeKit with merge.yaml.\n"
        "- Model superiority to original parents is NOT assumed; see baselines.\n",
        encoding="utf-8",
    )
    return out


def validate_holdout(
    run: Path, holdout_config: Path, *, candidate_id: str | None = None,
    baseline: str = "all", device: str | None = None, retry_partial: bool = False,
    overwrite: bool = False,
) -> dict:
    """Independent evaluation protocol; results NEVER enter selection/leaderboard.

    Separate task names are required, but the user must additionally guarantee
    train/test data separation. This method does NOT assert statistical superiority.
    """
    from dataclasses import replace
    spec, state = load_run(run)
    verify_frozen_inputs(run, spec)
    raw = yaml.safe_load(holdout_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("evaluation"), dict):
        raise BreederError("Holdout config must have an evaluation: mapping")
    config = spec_to_dict(spec)
    # Holdout may have one metric even when the breeding selector uses Pareto.
    config["selection"] = "weighted"
    config["evaluation"] = raw["evaluation"]
    from .spec import parse_spec
    test_spec = parse_spec(config)
    if not test_spec.evaluation.tasks:
        raise BreederError("Holdout evaluation requires tasks")
    overlap = {t.name for t in spec.evaluation.tasks} & {t.name for t in test_spec.evaluation.tasks}
    if overlap:
        raise BreederError(f"Holdout tasks overlap optimization tasks: {sorted(overlap)}")
    if test_spec.evaluation.screening_limit is not None:
        raise BreederError("Holdout cannot use staged screening (evaluate all holdout tasks)")
    if baseline != "all" and baseline != "none" and baseline not in {"base", *(p.name for p in spec.parents)}:
        raise BreederError("--baseline must be all, none, base or one parent name")
    ranked = [c for c in leaderboard(run, include_simulated=False) if c["score"].get("source") in VERIFIED_SOURCES]
    if candidate_id:
        chosen = next((c for c in ranked if c["id"] == candidate_id), None)
        if chosen is None:
            raise BreederError("Holdout candidate must have a real lm-eval selection score")
    elif ranked:
        chosen = ranked[0]
    else:
        raise BreederError("Need at least one candidate with lm-eval results before holdout validation")
    model_dir = candidate_dir(run, chosen["generation"], chosen["candidate"]) / "model"
    if not _model_complete(spec, model_dir):
        raise BreederError("Selected candidate checkpoint missing; build it before holdout validation")
    verify_artifact(model_dir)
    raw_bytes = holdout_config.read_bytes()
    protocol_sha = _sha(raw_bytes)
    destination = run / "validation" / protocol_sha[:16]
    destination.mkdir(parents=True, exist_ok=True)
    protocol_snapshot = destination / "protocol.yaml"
    if protocol_snapshot.exists() and protocol_snapshot.read_bytes() != raw_bytes:
        raise BreederError("Holdout hash prefix collision or protocol mutated")
    protocol_snapshot.write_bytes(raw_bytes)
    references = {"candidate_" + chosen["id"]: (str(model_dir.resolve()), spec.mode == "lora")}
    names = ["base"] + [p.name for p in spec.parents] if baseline == "all" else ([] if baseline == "none" else [baseline])
    for name in names:
        references["baseline_" + name] = (_baseline_ref(spec, name), spec.mode == "lora" and name != "base")
    results = {}
    for name, (model_ref, adapter) in references.items():
        folder = destination / name
        folder.mkdir(parents=True, exist_ok=True)
        score_path = folder / "score.json"
        if score_path.is_file() and not overwrite:
            doc = _read_json(score_path)
            if doc.get("protocol_sha256") != protocol_sha or doc.get("model_reference") != model_ref:
                raise BreederError(f"Existing holdout result metadata mismatch for {name}")
        else:
            metrics, evidence = _run_evaluation(test_spec, model_ref, folder, device=device,
                                                overwrite=overwrite, retry_partial=retry_partial,
                                                adapter=adapter)
            doc = {**_score_document(test_spec, metrics, "lm_eval", evidence=evidence,
                                    runtime_device=_protocol_device(folder, "final", device or test_spec.evaluation.device)),
                   "model_reference": model_ref, "protocol_sha256": protocol_sha,
                   "dataset_independence": "UNVERIFIED_USER_RESPONSIBILITY",
                   "stage": "holdout_validation"}
            _write_json(score_path, doc)
        results[name] = doc
    candidate_fitness = results["candidate_"+chosen["id"]]["fitness"]
    baseline_scores = [(name, doc) for name, doc in results.items() if name.startswith("baseline_")]
    champion = max(baseline_scores, key=lambda kv: kv[1]["fitness"]) if baseline_scores else None
    summary = {
        "candidate": chosen["id"], "protocol_sha256": protocol_sha,
        "optimization_config_sha256": state["config_sha256"],
        "candidate_holdout_fitness": candidate_fitness,
        "best_baseline_name": champion[0] if champion else None,
        "best_baseline_holdout_fitness": champion[1]["fitness"] if champion else None,
        "delta_vs_best_baseline": round(candidate_fitness-champion[1]["fitness"], 8) if champion else None,
        "results": results,
        "disclaimer": "The holdout protocol is separate from model selection but independent training/evaluation data, statistical significance, and generalization were NOT verified.",
    }
    _write_json(destination / "summary.json", summary)
    return summary


def doctor(spec: Spec) -> dict:
    """Local lightweight diagnostics; never downloads model weights."""
    report = check_compatibility(spec, remote=False)
    sizes: dict[str, int | None] = {}
    for name, ref in [("base", spec.base_model)] + [(p.name, p.model) for p in spec.parents]:
        root = Path(ref)
        if not root.is_dir():
            sizes[name] = None
            continue
        sizes[name] = sum(p.stat().st_size for p in root.glob("*.safetensors") if p.is_file()) + \
                      sum(p.stat().st_size for p in root.glob("pytorch_model*.bin") if p.is_file())
    largest = (sum(sizes.get(p.name) or 0 for p in spec.parents) if spec.mode == "lora" else
               max((v for v in sizes.values() if v is not None), default=0))
    return {
        "model_compatibility": {"ok": report.ok, "errors": report.errors, "warnings": report.warnings, "checked": report.checked},
        "dependencies": {"mergekit-yaml": shutil.which("mergekit-yaml") is not None,
                         "lm-eval": shutil.which("lm-eval") is not None,
                         "torch": importlib.util.find_spec("torch") is not None,
                         "safetensors": importlib.util.find_spec("safetensors") is not None,
                         "PEFT: peft": importlib.util.find_spec("peft") is not None,
                         "PEFT: transformers": importlib.util.find_spec("transformers") is not None},
        "local_checkpoint_bytes": sizes,
        "approx_generation_output_bytes": largest * spec.population if largest else None,
        "warning": "Output estimate is approximate, neither a bound nor a peak-memory guarantee; adapter concatenation increases rank and full model weights may dominate inference RAM.",
    }
