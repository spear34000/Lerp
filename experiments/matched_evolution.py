"""Evolution (blend crossover) against plain training under MATCHED conditions, 3 seeds.

Matched: same base model, same start (the 0.5/0.5 combination of the founders), same exact data mix (chain 50% / add 25% / mul 25%), same training rank (32),
same learning rate, ONE global warm-up + cosine schedule per lineage (the control: 1,200 steps; the evolution lineage: 3 x 200 = 600 steps), no optimizer-state
transfer, the same fresh test items (1,000 per family, never used for training or selection), and the same training COST: the control trains as many steps as
all six children together (3 generations x 2 children x 200 = 1,200); selection evaluations are counted and reported. The evolution arm additionally compresses a cross
of two rank-32 parents back to rank 32 before training (extra step, energy kept recorded). The controls are the three continuous runs of the restart experiment
(`controls/s1..s3`, already trained, stream seeds 10001-10003).

Rule, fixed before running (new skill = `chain`; D = evolved - control on the fresh items, per seed s, and its mean over the 3 seeds):
  * EVOLUTION AHEAD    if mean D >= +0.02, D_s > 0 for all three seeds, and the 95% item-bootstrap interval of the mean excludes 0;
  * NO DIFFERENCE DETECTED  if |mean D| < 0.01 and the interval includes 0 (equivalence is NOT shown);
  * EVOLUTION BEHIND   if mean D <= -0.02, D_s < 0 for all three seeds, and the interval excludes 0;
  * otherwise NO VERDICT.
The old skills (`add`, `mul`) are reported separately with the same arithmetic but do not enter the rule. The bootstrap resamples items only (the same indices for
every seed), so training-seed variation is covered only through the all-seeds-agree requirement. Three seeds on one item set are a pilot, not three independent tests.

    python experiments/matched_evolution.py run --config CFG.yaml --out DIR [--seeds 1 2 3]
    python experiments/matched_evolution.py report --out DIR
"""
import argparse
import dataclasses
import json
import random
from pathlib import Path

import yaml

from lerp.evolution.matched import MatchedConfig, run_matched


def run(args) -> None:
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for seed in args.seeds:
        out = Path(args.out) / f"s{seed}"
        if (out / "result.json").is_file():
            continue
        print(f"=== seed {seed} ===", flush=True)
        run_matched(MatchedConfig(**{**raw, "seed": seed}), out, log=lambda m: print(m, flush=True))


def report(args) -> None:
    root = Path(args.out)
    seeds = sorted(p.name for p in root.glob("s*") if (p / "result.json").is_file())
    results = {s: json.loads((root / s / "result.json").read_text()) for s in seeds}
    items = {s: json.loads((root / s / "items.json").read_text())["items"] for s in seeds}
    cfg = next(iter(results.values()))["config"]
    new, fams = cfg["new_family"], cfg["founders"] + [cfg["new_family"]]
    print(f"seeds {seeds}; fresh items {cfg['n_fresh']} per family; cost per seed: {cfg['generations'] * cfg['children'] * cfg['child_steps']} training steps in total "
          f"(control: the same), lineage {cfg['generations'] * cfg['child_steps']} steps")
    print(f"\n{'seed':<6}{'model':<16}" + "".join(f"{f:>9}" for f in fams))
    diffs = {f: [] for f in fams}
    for s in seeds:
        acc, best = results[s]["fresh_accuracy"], results[s]["best"]
        for name, key in (("control", "control"), (f"evolved {best}", f"evolved:{best}"), ("start", "start"), ("g0-add", "g0-add"), ("g0-mul", "g0-mul")):
            print(f"{s:<6}{name:<16}" + "".join(f"{acc[key][f]:>9.3f}" for f in fams))
        for f in fams:
            diffs[f].append(acc[f"evolved:{best}"][f] - acc["control"][f])
    print(f"\nevolved - control, per seed [paired McNemar 95% CI], p, on the same fresh items")
    for s in seeds:
        for f in fams:
            m = results[s]["vs_control"][f]
            print(f"  {s} {f:<6}{m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}] p={m['p_exact_two_sided']:.3f} ({m['a_only']} vs {m['b_only']})")
    n = len(items[seeds[0]]["control"][new])
    rng = random.Random(0)
    paired = {s: [items[s][f"evolved:{results[s]['best']}"][new][i] - items[s]["control"][new][i] for i in range(n)] for s in seeds}
    boots = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(sum(paired[s][i] for i in idx) / n for s in seeds) / len(seeds))
    boots.sort()
    lo, hi = boots[49], boots[1949]
    mean = sum(diffs[new]) / len(seeds)
    signs_pos, signs_neg = all(d > 0 for d in diffs[new]), all(d < 0 for d in diffs[new])
    if mean >= 0.02 and signs_pos and lo > 0:
        verdict = "EVOLUTION AHEAD"
    elif mean <= -0.02 and signs_neg and hi < 0:
        verdict = "EVOLUTION BEHIND"
    elif abs(mean) < 0.01 and lo <= 0 <= hi:
        verdict = "NO DIFFERENCE DETECTED (equivalence not shown)"
    else:
        verdict = "NO VERDICT"
    print(f"\n`{new}`: mean D = {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] (item bootstrap), per seed " + " ".join(f"{d:+.3f}" for d in diffs[new]) + f"  -> {verdict}")
    for f in fams[:-1]:
        print(f"{f}: mean D = {sum(diffs[f]) / len(seeds):+.3f}, per seed " + " ".join(f"{d:+.3f}" for d in diffs[f]))
    print("\ncost per seed: " + "; ".join(f"{s}: steps {results[s]['cost']['training_steps']} (in final lineage {results[s]['cost']['steps_in_final_lineage']}), "
                                         f"dev evaluations {results[s]['cost']['dev_evaluations']} = {results[s]['cost']['dev_items_scored']} items" for s in seeds))
    print("pre-training compression (extra step, energy kept): " + "; ".join(
        f"{s}: " + ",".join(f"{c['energy_kept']:.2f}" for c in results[s]["cost"]["compressions"]) for s in seeds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": run, "report": report}[args.cmd](args)
