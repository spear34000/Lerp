"""Which part of the evolutionary loop helps? Crossover modes against each other and against compute-matched plain training.

All arms of a seed share the founders and the control (so they differ only in the loop), get the same number of training steps, the same
selection fixes and the same test items. Arms: ``graft`` (A + what B learned), ``blend`` (the original weighted average), ``none``
(no crossover: each child keeps training a survivor).

    python experiments/evolution_ablation.py run --config base.yaml --out DIR [--seeds 1] [--arms graft blend none]
    python experiments/evolution_ablation.py report --out DIR
"""
import argparse
import dataclasses
import json
from pathlib import Path

from lerp.evolution.orchestrator import load_config, run_evolution
from lerp.statistics import mcnemar


def run(args) -> None:
    base = load_config(Path(args.config))
    for seed in args.seeds:
        first = None
        for arm in args.arms:
            out = Path(args.out) / f"s{seed}" / arm
            if (out / "result.json").is_file():
                first = first or out
                continue
            extra = {"founders_from": str(first), "control_from": str(first)} if first else {}
            cfg = dataclasses.replace(base, seed=seed, crossover=arm, **extra)
            print(f"=== seed {seed} arm {arm} ===", flush=True)
            run_evolution(cfg, out, log=lambda m: print(m, flush=True))
            first = first or out


def report(args) -> None:
    root = Path(args.out)
    for seed_dir in sorted(root.glob("s*")):
        arms = {p.name: p for p in seed_dir.iterdir() if (p / "result.json").is_file()}
        if not arms:
            continue
        results = {a: json.loads((p / "result.json").read_text()) for a, p in arms.items()}
        items = {a: json.loads((p / "test_items.json").read_text())["items"] for a, p in arms.items()}
        any_arm = next(iter(results))
        cfg = results[any_arm]["config"]
        new, olds = cfg["new_family"], cfg["founders"]
        fams = olds + [new]
        print(f"\n=== {seed_dir.name}: test items per family {cfg['n_test']}, steps {results[any_arm]['training_steps']} ===")
        print(f"{'model':<26}" + "".join(f"{f:>9}" for f in fams) + f"{'steps in lineage / discarded':>32}")
        shared = results[any_arm]["test_accuracy"]
        for name in ("base", "g0-add", "g0-mul", "merge-only", "control-plain"):
            if name in shared:
                print(f"{name:<26}" + "".join(f"{shared[name][f]:>9.3f}" for f in fams))
        evolved = {}
        for arm, r in results.items():
            key = f"evolved:{r['best']}"
            evolved[arm] = key
            print(f"{'evolved/' + arm + ' (' + r['best'] + ')':<26}" + "".join(f"{r['test_accuracy'][key][f]:>9.3f}" for f in fams)
                  + f"{r['steps_in_final_lineage']:>16} /{r['steps_discarded']:>5}")
        print(f"\nnew skill ({new}), paired McNemar on the same test items [95% CI], p")
        def pair(label, a_items, b_items):
            m = mcnemar(a_items[new], b_items[new])
            print(f"  {label:<40} {m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}]  p={m['p_exact_two_sided']:.4f}"
                  f"  ({m['a_only']} vs {m['b_only']})")
        for arm in results:
            ev = items[arm][evolved[arm]]
            pair(f"{arm} - control-plain", ev, items[arm]["control-plain"])
            pair(f"{arm} - best founder (g0-mul)", ev, items[arm]["g0-mul"])
        names = sorted(results)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair(f"{a} - {b}", items[a][evolved[a]], items[b][evolved[b]])
        print("\nold skills: evolved minus best founder")
        for arm, r in results.items():
            print(f"  {arm:<8}" + "  ".join(f"{f} {v:+.3f}" for f, v in r["retention_vs_best_founder"].items()) + f"   verdict: {r['verdict']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", nargs="+", type=int, default=[1])
    r.add_argument("--arms", nargs="+", default=["graft", "blend", "none"])
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)
