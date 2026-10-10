"""Cheap diagnostics for the gap between the evolution loop and plain training, reusing the founders, the control and their test scores
of a finished ablation run (only the new organisms are trained and scored).

Variants (same total training steps as the control):
  none-seq   crossover none, one child per generation, six generations: a sequence of 200-step segments with compression, selection and an
             optimizer restart between them. Differs from the control only by those; no waste from parallel children.
  graft-r32  graft crossover with children stored at rank 32 (less truncation loss per generation).
  blend-r32  blend crossover with children stored at rank 32.

    python experiments/evolution_followup.py run --base-run DIR_OF_A_FINISHED_ARM --out DIR [--variants none-seq graft-r32 blend-r32]
    python experiments/evolution_followup.py report --base-run DIR --out DIR
"""
import argparse
import dataclasses
import json
from pathlib import Path

from lerp.evolution.orchestrator import EvolutionConfig, run_evolution
from lerp.statistics import mcnemar

VARIANTS = {
    "none-seq": dict(crossover="none", children=1, generations=6),
    "graft-r32": dict(crossover="graft", store_rank=32),
    "blend-r32": dict(crossover="blend", store_rank=32),
}


def run(args) -> None:
    base_run = Path(args.base_run)
    src = json.loads((base_run / "result.json").read_text())["config"]
    for name in args.variants:
        out = Path(args.out) / name
        if (out / "result.json").is_file():
            continue
        cfg = EvolutionConfig(**{**src, **VARIANTS[name], "founders_from": str(base_run), "control_from": str(base_run)})
        print(f"=== {name} ===", flush=True)
        run_evolution(cfg, out, log=lambda m: print(m, flush=True))


def report(args) -> None:
    base_run = Path(args.base_run)
    ref = json.loads((base_run / "result.json").read_text())
    cfg = ref["config"]
    new, olds = cfg["new_family"], cfg["founders"]
    fams = olds + [new]
    ref_items = json.loads((base_run / "test_items.json").read_text())["items"]
    print(f"{'model':<28}" + "".join(f"{f:>9}" for f in fams) + f"{'lineage steps / discarded':>28}")
    for n in ("control-plain", "g0-mul"):
        print(f"{n:<28}" + "".join(f"{ref['test_accuracy'][n][f]:>9.3f}" for f in fams))
    earlier = {}
    for arm_dir in sorted(base_run.parent.glob("*")):
        if (arm_dir / "result.json").is_file():
            r = json.loads((arm_dir / "result.json").read_text())
            key = f"evolved:{r['best']}"
            earlier[f"earlier {arm_dir.name}"] = json.loads((arm_dir / "test_items.json").read_text())["items"][key]
            print(f"{'earlier ' + arm_dir.name:<28}" + "".join(f"{r['test_accuracy'][key][f]:>9.3f}" for f in fams)
                  + f"{r['steps_in_final_lineage']:>16} /{r['steps_discarded']:>5}")
    rows = {}
    for d in sorted(Path(args.out).glob("*")):
        if not (d / "result.json").is_file():
            continue
        r = json.loads((d / "result.json").read_text())
        key = f"evolved:{r['best']}"
        items = json.loads((d / "test_items.json").read_text())["items"][key]
        rows[d.name] = items
        print(f"{d.name + ' (' + r['best'] + ')':<28}" + "".join(f"{r['test_accuracy'][key][f]:>9.3f}" for f in fams)
              + f"{r['steps_in_final_lineage']:>16} /{r['steps_discarded']:>5}   verdict: {r['verdict']}")
    print(f"\nnew skill ({new}) paired McNemar, same test items: difference [95% CI] p (only-a vs only-b)")
    for name, items in rows.items():
        m = mcnemar(items[new], ref_items["control-plain"][new])
        print(f"  {name:<12} - control-plain  {m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}] p={m['p_exact_two_sided']:.4f} ({m['a_only']} vs {m['b_only']})")
        for other, oi in earlier.items():
            m = mcnemar(items[new], oi[new])
            print(f"  {name:<12} - {other:<18} {m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}] p={m['p_exact_two_sided']:.4f}")
    print("\nold skills: evolved minus best founder")
    for d in sorted(Path(args.out).glob("*")):
        if (d / "result.json").is_file():
            r = json.loads((d / "result.json").read_text())
            print(f"  {d.name:<12}" + "  ".join(f"{f} {v:+.3f}" for f, v in r["retention_vs_best_founder"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--base-run", required=True)
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)
