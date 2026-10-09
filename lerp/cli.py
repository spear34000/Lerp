"""CLI entry points for reproducible, multiobjective model merge research."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compat import check_compatibility, resolve_spec_paths
from .experiment import (
    BreederError, advance, baselines, build_candidate, comparisons, cycle,
    doctor, evaluate_baseline, evaluate_candidate, export_recipe, init_run,
    leaderboard, lineage_dot, load_run, record_score, recover_candidate, simulate_generation, statuses,
    screen_candidate, promotion_list, validate_holdout,
)
from .report import create_report
from .integrity import IntegrityError, freeze_inputs, verify_frozen_inputs
from .statistics import StatisticsError, compare_samples
from .leakage import LeakageError, audit_split_overlap
from .locking import RunBusyError, locked_run
from .spec import SpecError, load_spec


def _candidate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", "-r", type=Path, required=True, help="Run folder")
    parser.add_argument("--generation", "-g", type=int, required=True)
    parser.add_argument("--candidate", "-i", type=int, required=True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lerp", description="Evolve multi-parent LLM checkpoint merges, with Pareto selection and reproducible evaluation")
    parser.add_argument("--version", action="version", version="Lerp 0.5.0")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Inspect HF checkpoint metadata; no model weights downloaded")
    check.add_argument("--config", "-c", type=Path, required=True)
    check.add_argument("--remote", action="store_true", help="Download HF config.json metadata only")

    diag = commands.add_parser("doctor", help="Show dependencies, compatibility and rough disk footprint")
    diag.add_argument("--config", "-c", type=Path, required=True)
    diag.add_argument("--json", action="store_true")

    init = commands.add_parser("init", help="Create generation zero and immutable experiment manifest")
    init.add_argument("--config", "-c", type=Path, required=True)
    init.add_argument("--out", "-o", type=Path, required=True)
    init.add_argument("--remote", action="store_true")

    pin = commands.add_parser("freeze", help="SHA-256 pin local model files BEFORE evaluation")
    pin.add_argument("--run", "-r", type=Path, required=True)
    pin.add_argument("--strict", action="store_true", help="Require every model's weight files locally")

    verify = commands.add_parser("verify-inputs", help="Check frozen model SHA-256 before using scores")
    verify.add_argument("--run", "-r", type=Path, required=True)

    stats = commands.add_parser("compare-samples", help="Paired sample-level bootstrap CI; accepts aligned JSONL")
    stats.add_argument("--candidate", type=Path, required=True)
    stats.add_argument("--baseline", type=Path, required=True)
    stats.add_argument("--out", "-o", type=Path, required=True)
    stats.add_argument("--id-field", default="id")
    stats.add_argument("--score-field", default="score")
    stats.add_argument("--replicates", type=int, default=5000)
    stats.add_argument("--seed", type=int, default=42)

    leak = commands.add_parser("audit-splits", help="Detect exact/normalized train/holdout text overlaps")
    leak.add_argument("--development", type=Path, required=True)
    leak.add_argument("--holdout", type=Path, required=True)
    leak.add_argument("--text-field", default="text")
    leak.add_argument("--id-field", default="id")
    leak.add_argument("--out", "-o", type=Path)

    simulate = commands.add_parser("simulate", help="FAKE synthetic toy scoring: NEVER report as LLM performance")
    simulate.add_argument("--run", "-r", type=Path, required=True)
    simulate.add_argument("--generation", "-g", type=int, default=None)

    step = commands.add_parser("advance", help="Select parents and evolve next-generation layer-wise recipes")
    step.add_argument("--retry-partial", action="store_true", help="Clear interrupted next-generation draft")
    step.add_argument("--run", "-r", type=Path, required=True)
    step.add_argument("--allow-simulated", action="store_true", help="Explicitly allow fake toy scoring for demos")
    step.add_argument("--allow-manual", action="store_true", help="Opt into evolving from external UNVERIFIED manual scores")

    build = commands.add_parser("build", help="Create a full HF checkpoint with MergeKit (CPU or CUDA)")
    _candidate_args(build)
    build.add_argument("--cuda", action="store_true")
    build.add_argument("--engine", choices=["mergekit", "lite", "lora"], default="mergekit", help="CPU Torch safetensors or full MergeKit")
    build.add_argument("--retry-partial", action="store_true", help="Delete ONLY own incomplete partial build and retry")

    automate = commands.add_parser("cycle", help="Resumable build/evaluate/breed loop (heavy compute/disk use)")
    automate.add_argument("--run", "-r", type=Path, required=True)
    automate.add_argument("--rounds", type=int, default=1)
    automate.add_argument("--cuda", action="store_true")
    automate.add_argument("--engine", choices=["mergekit", "lite", "lora"], default="mergekit")
    automate.add_argument("--device", help="Evaluator device override")
    automate.add_argument("--retry-partial", action="store_true", help="Clean interrupted temporary artifacts on retry")

    evaluation = commands.add_parser("evaluate", help="Evaluate one merged checkpoint with EleutherAI lm-eval")
    _candidate_args(evaluation)
    evaluation.add_argument("--device")
    evaluation.add_argument("--overwrite", action="store_true", help="Replace score only after a successful evaluation")
    evaluation.add_argument("--retry-partial", action="store_true")

    recover = commands.add_parser("recover", help="Recover a completed lm-eval result not yet committed to score.json")
    _candidate_args(recover)

    baseline = commands.add_parser("baseline", help="Evaluate base / unmerged parents for meaningful comparisons")
    baseline.add_argument("--run", "-r", type=Path, required=True)
    baseline.add_argument("--name", default="all", help="base, parent name, or all")
    baseline.add_argument("--device")
    baseline.add_argument("--overwrite", action="store_true")
    baseline.add_argument("--retry-partial", action="store_true")

    screen = commands.add_parser("screen", help="Run low-budget first-stage evaluation on one candidate")
    _candidate_args(screen)
    screen.add_argument("--device")
    screen.add_argument("--overwrite", action="store_true")
    screen.add_argument("--retry-partial", action="store_true")

    promote = commands.add_parser("promote", help="List top screened candidates selected for full evaluation")
    promote.add_argument("--run", "-r", type=Path, required=True)
    promote.add_argument("--generation", "-g", type=int, required=True)

    score = commands.add_parser("score", help="Record external manually reported metrics (UNVERIFIED)")
    _candidate_args(score)
    score.add_argument("--metric", action="append", required=True, metavar="TASK=VALUE")
    score.add_argument("--overwrite", action="store_true")

    board = commands.add_parser("board", help="Rank candidates by weighted fitness or Pareto fronts")
    board.add_argument("--run", "-r", type=Path, required=True)
    board.add_argument("--real-only", action="store_true", help="Exclude fake toy demo scores")
    board.add_argument("--pareto", action="store_true", help="Pareto fronts + crowding distance")
    board.add_argument("--top", type=int, default=20)

    compare = commands.add_parser("compare", help="Compare children with independently evaluated unmerged parents")
    compare.add_argument("--run", "-r", type=Path, required=True)
    compare.add_argument("--top", type=int, default=20)

    stat = commands.add_parser("status", help="Show build/evaluation status and incomplete artifacts")
    stat.add_argument("--run", "-r", type=Path, required=True)

    lineage = commands.add_parser("lineage", help="Export model ancestry as Graphviz DOT")
    lineage.add_argument("--run", "-r", type=Path, required=True)
    lineage.add_argument("--out", "-o", type=Path)

    report = commands.add_parser("report", help="Write a beautiful self-contained local HTML dashboard")
    report.add_argument("--run", "-r", type=Path, required=True)
    report.add_argument("--out", "-o", type=Path, help="HTML location, defaults to RUN/dashboard.html")

    validation = commands.add_parser("validate", help="Separate final held-out eval (cannot influence evolution)")
    validation.add_argument("--run", "-r", type=Path, required=True)
    validation.add_argument("--config", "-c", type=Path, required=True, help="Holdout evaluation YAML")
    validation.add_argument("--candidate", help="Optional lm-eval-scored candidate ID")
    validation.add_argument("--baseline", default="all", help="all, none, base or parent name")
    validation.add_argument("--device")
    validation.add_argument("--overwrite", action="store_true")
    validation.add_argument("--retry-partial", action="store_true")

    export = commands.add_parser("export", help="Export top recipe + audit metadata, without duplicating heavy weights")
    export.add_argument("--run", "-r", type=Path, required=True)
    export.add_argument("--out", "-o", type=Path, required=True)
    export.add_argument("--candidate", help="Optional explicit candidate ID; defaults to top measured candidate")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command in ("freeze", "verify-inputs"):
        spec, _ = load_run(args.run)
        result = (freeze_inputs(args.run, spec, strict=args.strict) if args.command == "freeze"
                  else verify_frozen_inputs(args.run, spec, require=True))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "compare-samples":
        result = compare_samples(args.candidate, args.baseline, id_field=args.id_field,
                                 score_field=args.score_field, replicates=args.replicates, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Wrote exploratory paired statistics:", args.out)
        return 0

    if args.command == "audit-splits":
        result = audit_split_overlap(args.development, args.holdout, id_field=args.id_field,
                                     text_field=args.text_field)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result["status"] == "FAIL_OVERLAP" else 0

    if args.command in ("check", "doctor"):
        spec = resolve_spec_paths(load_spec(args.config), args.config.resolve().parent)
        if args.command == "doctor":
            result = doctor(spec)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                for tool, available in result["dependencies"].items():
                    print(f"{'OK  ' if available else 'MISS'}  {tool}")
                print("Model metadata compatible:", result["model_compatibility"]["ok"])
                for warning in result["model_compatibility"]["warnings"]:
                    print("WARN ", warning)
                for error in result["model_compatibility"]["errors"]:
                    print("ERROR", error)
                if result["approx_generation_output_bytes"] is not None:
                    gib = result["approx_generation_output_bytes"] / 1024 ** 3
                    print(f"Approx output footprint PER generation: {gib:.2f} GiB (rough lower bound)")
            return 0 if result["model_compatibility"]["ok"] else 2
        summary = check_compatibility(spec, remote=args.remote)
        print(f"Checked {len(summary.checked)} checkpoint(s)")
        for item in summary.checked:
            print("  OK   ", item)
        for item in summary.warnings:
            print("  WARN ", item)
        for item in summary.errors:
            print("  FAIL ", item)
        return 0 if summary.ok else 2

    if args.command == "init":
        state, folder = init_run(args.config, args.out, check_remote=args.remote)
        spec, _ = load_run(folder)
        print(f"Initialized {spec.population} candidates, {len(spec.parents)} parents, selection={spec.selection}")
        print("Run folder:", folder)
        print("Config SHA-256:", state["config_sha256"])
        for item in state["preflight_warnings"]:
            print("WARN:", item)
        return 0

    if args.command == "simulate":
        _, state = load_run(args.run)
        gen = args.generation if args.generation is not None else state["generation"]
        for entry in simulate_generation(args.run, gen):
            print(f"FAKE_TOY {entry['id']} score={entry['fitness']:.6f}")
        return 0

    if args.command == "advance":
        gen = advance(args.run, allow_simulated=args.allow_simulated,
                      allow_manual=args.allow_manual, retry_partial=args.retry_partial)
        print(f"Generation {gen} created: use build/evaluate or simulate for synthetic demos")
        return 0

    if args.command == "build":
        output = build_candidate(args.run, args.generation, args.candidate, cuda=args.cuda, retry_partial=args.retry_partial, engine=args.engine)
        print("Built HF checkpoint:", output)
        return 0

    if args.command == "cycle":
        completed = cycle(args.run, args.rounds, cuda=args.cuda, device=args.device, retry_partial=args.retry_partial, engine=args.engine)
        print("Completed generations:", ", ".join(map(str, completed)))
        return 0

    if args.command == "evaluate":
        value = evaluate_candidate(args.run, args.generation, args.candidate,
                                   device=args.device, overwrite=args.overwrite, retry_partial=args.retry_partial)
        print("Measured score:", json.dumps(value, ensure_ascii=False))
        return 0

    if args.command == "recover":
        result = recover_candidate(args.run, args.generation, args.candidate)
        print("Recovered completed evaluation:", json.dumps(result, ensure_ascii=False))
        return 0

    if args.command == "baseline":
        spec, _ = load_run(args.run)
        names = (["base"] + [p.name for p in spec.parents]) if args.name == "all" else [args.name]
        existing = baselines(args.run)
        for name in names:
            if name in existing and not args.overwrite:
                print(f"Baseline {name}: already scored, skipped")
                continue
            result = evaluate_baseline(args.run, name, device=args.device,
                                       overwrite=args.overwrite, retry_partial=args.retry_partial)
            print(f"Baseline {name}: {result['fitness']:.5f} ({result['status']})")
        return 0

    if args.command == "screen":
        result = screen_candidate(args.run, args.generation, args.candidate,
                                  device=args.device, overwrite=args.overwrite, retry_partial=args.retry_partial)
        print("Screening score (NOT final):", json.dumps(result, ensure_ascii=False))
        return 0

    if args.command == "promote":
        selected = promotion_list(args.run, args.generation)
        print("Promoted candidate IDs:", ", ".join(item["id"] for item in selected))
        return 0

    if args.command == "score":
        metrics: dict[str, float] = {}
        for item in args.metric:
            task, sep, value = item.partition("=")
            if not sep or not task.strip() or task.strip() in metrics:
                raise BreederError(f"Invalid/duplicate score {item!r}; expected TASK=VALUE")
            try:
                metrics[task.strip()] = float(value)
            except ValueError as exc:
                raise BreederError(f"Invalid numeric value: {item}") from exc
        value = record_score(args.run, args.generation, args.candidate, metrics, source="manual", overwrite=args.overwrite)
        print("UNVERIFIED manual score:", json.dumps(value, ensure_ascii=False))
        return 0

    if args.command == "board":
        records = leaderboard(args.run, include_simulated=not args.real_only, pareto=args.pareto)
        print(f"{'Rank':<5} {'Candidate':<12} {'Fitness':>10} {'Source':<16} {'Status':<18} Genes")
        for i, item in enumerate(records[:max(0, args.top)], start=1):
            score = item["score"]
            g = ",".join(f"{v:.2f}" for v in item["genes"][:9])
            if len(item["genes"]) > 9:
                g += ",..."
            print(f"{i:<5} {item['id']:<12} {score['fitness']:>10.5f} {score['source']:<16} {score.get('status', '-'):<18} {g}")
        return 0

    if args.command == "compare":
        measured = baselines(args.run)
        print("Parent baselines:")
        for name, score in measured.items():
            print(f"  {name}: {score['fitness']:.5f}")
        records = comparisons(args.run)
        print(f"{'Candidate':<12} {'Fitness':>10} {'Delta':>11} {'Baseline':<14} {'Status':<18}")
        for item in records[:max(0, args.top)]:
            print(f"{item['id']:<12} {item['fitness']:>10.5f} {item['delta']:>+11.5f} {item['baseline']:<14} {item['status']:<18}")
        return 0

    if args.command == "status":
        rows = statuses(args.run)
        print(f"{'Candidate':<12} {'Built':<8} {'Scored':<8} {'Source':<18} {'Partial':<8}")
        for row in rows:
            print(f"{row['id']:<12} {str(row['built']):<8} {str(row['screened']):<10} {str(row['scored']):<8} {str(row['source'] or '-'):<18} {str(row['partial']):<8}")
        return 0

    if args.command == "lineage":
        dot = lineage_dot(args.run)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(dot, encoding="utf-8")
            print("Saved Graphviz lineage:", args.out)
        else:
            print(dot, end="")
        return 0

    if args.command == "report":
        output = args.out or args.run / "dashboard.html"
        create_report(args.run, output)
        print("Saved standalone dashboard:", output)
        return 0

    if args.command == "validate":
        result = validate_holdout(args.run, args.config, candidate_id=args.candidate,
                                  baseline=args.baseline, device=args.device, overwrite=args.overwrite,
                                  retry_partial=args.retry_partial)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export":
        output = export_recipe(args.run, args.out, candidate_id=args.candidate)
        print("Exported reproducible research recipe:", output)
        return 0

    raise AssertionError("unreachable")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    try:
        mutating = {"freeze", "simulate", "advance", "build", "cycle", "evaluate",
                    "recover", "baseline", "screen", "score", "validate"}
        if args.command in mutating:
            with locked_run(args.run):
                status = _run(args)
        else:
            status = _run(args)
    except (BreederError, SpecError, IntegrityError, StatisticsError, LeakageError, RunBusyError,
            OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        status = 2
    sys.exit(status)


if __name__ == "__main__":
    main()
