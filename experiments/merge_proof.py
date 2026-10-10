"""Does merging two skill LoRAs beat both parents on fresh items? One (pair, seed) per call, plus a verdict over all results.

Pre-registered criteria (fixed before any result was looked at). For a pair, the blend "proves" a gain when, in EVERY training seed, it beats
BOTH parents on the pooled accuracy of the two tasks on fresh items (never used in the search) by at least +0.03, with the exact McNemar
p < 0.01 and a 95% interval whose lower bound is above 0. Anything less is reported as not proven, naming the criterion that failed.

    python experiments/merge_proof.py run   --base DIR --adapters DIR --pair arcboolq --skills arc boolq --seed 1 --out DIR
    python experiments/merge_proof.py verdict --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

TASKS = {"arc": ("arc_easy", "acc_norm,none"), "boolq": ("boolq", "acc,none"), "piqa": ("piqa", "acc_norm,none"),
         "hellaswag": ("hellaswag", "acc_norm,none")}
SEARCH_ITEMS = 200
FRESH = (500, 1500)  # disjoint from the search window (0, 200)
MIN_GAIN, MAX_P = 0.03, 0.01


def make_spec(base, parents, tasks, limit, name):
    from lerp.spec import parse_spec
    raw = {"name": name, "base_model": base, "mode": "lora", "method": "linear", "search": "gp", "population": 4, "genes": 3,
           "seed": 7, "out_dtype": "float32", "gene_groups": ["attention", "mlp", "other"],
           "parents": [{"name": n, "model": p} for n, p in parents],
           "evaluation": {"device": "xpu", "limit": limit, "penalty": 0.1,
                          "tasks": {t: {"metric": TASKS[s][1], "weight": 1.0} for s in tasks for t in [TASKS[s][0]]}}}
    return parse_spec(raw)


def pooled(items, names):
    return [v for n in names for v in items[n]]


def compare(a, b, names):
    from lerp.statistics import mcnemar
    out = {"pooled": mcnemar(pooled(a, names), pooled(b, names))}
    for n in names:
        out[n] = mcnemar(a[n], b[n])
    return out


def run(args):
    import lerp.experiment as exp
    from lerp.integrity import freeze_inputs
    from lerp.resident import ResidentSession
    from lerp.search import search_run
    from lerp.spec import spec_to_dict
    out = Path(args.out) / f"{args.pair}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    adapters = Path(args.adapters)
    a, b = args.skills
    pa, pb = adapters / f"{args.pair}_{a}_s{args.seed}", adapters / f"{args.pair}_{b}_s{args.seed}"
    multi = adapters / f"{args.pair}_multi_s{args.seed}"
    spec = make_spec(args.base, [(a, str(pa)), (b, str(pb))], [a, b], SEARCH_ITEMS, f"{args.pair}-s{args.seed}")
    names = [TASKS[a][0], TASKS[b][0]]
    cfg = out / "experiment.yaml"
    cfg.write_text(yaml.safe_dump(spec_to_dict(spec)), encoding="utf-8")
    runs = out / "run"
    if not runs.exists():
        exp.init_run(cfg, runs)
        freeze_inputs(runs, spec, strict=True)
    t0 = time.time()
    session = ResidentSession(spec, "xpu", "bfloat16")
    search_run(runs, args.rounds, session=session, baselines=True, log=lambda m: print(m, flush=True))
    board = exp.leaderboard(runs, include_simulated=False)
    best = board[0]
    print(f"search done in {time.time() - t0:.0f}s; best {best['id']} fitness {best['score']['fitness']:.4f}", flush=True)

    results, items = {}, {}

    def record(label, metrics, owner=None):
        results[label] = metrics
        items[label] = {k: list(v) for k, v in (owner or session).last_items.items()}

    record("base", session.evaluate_reference("base", FRESH))
    record(a, session.evaluate_reference(a, FRESH))
    record(b, session.evaluate_reference(b, FRESH))
    record("blend_0.5", session.evaluate([0.5] * spec.genome_size, FRESH))
    record("blend_best", session.evaluate(best["genes"], FRESH))
    session.close()

    mspec = make_spec(args.base, [("multi", str(multi)), (a, str(pa))], [a, b], SEARCH_ITEMS, f"{args.pair}-multi-s{args.seed}")
    msession = ResidentSession(mspec, "xpu", "bfloat16")
    record("multitask", msession.evaluate_reference("multi", FRESH), msession)
    msession.close()

    comparisons = {}
    for label in ("blend_0.5", "blend_best"):
        comparisons[f"{label}_vs_{a}"] = compare(items[label], items[a], names)
        comparisons[f"{label}_vs_{b}"] = compare(items[label], items[b], names)
        comparisons[f"{label}_vs_multitask"] = compare(items[label], items["multitask"], names)
    doc = {"pair": args.pair, "seed": args.seed, "skills": [a, b], "tasks": names, "fresh_window": FRESH,
           "best_genes": best["genes"], "best_search_fitness": best["score"]["fitness"], "accuracy": results,
           "comparisons": comparisons, "search_evaluations": args.rounds * spec.population}
    (out / "result.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    pool = {k: sum(v.values()) / len(v) for k, v in results.items()}
    print(json.dumps({"pooled_accuracy": pool}, indent=2), flush=True)


def meets(c):
    p = c["pooled"]
    return p["difference"] >= MIN_GAIN and p["p_exact_two_sided"] < MAX_P and p["ci"]["lower"] > 0


def verdict(args):
    docs = [json.loads(p.read_text()) for p in sorted(Path(args.out).glob("*/result.json"))]
    if not docs:
        print("no results")
        return
    pairs = sorted({d["pair"] for d in docs})
    for pair in pairs:
        rows = [d for d in docs if d["pair"] == pair]
        a, b = rows[0]["skills"]
        print(f"\n=== {pair}  ({a} + {b}), {len(rows)} seed(s), fresh items {FRESH[0]}-{FRESH[1]} per task ===")
        print(f"{'seed':>4} {'model':<12}{'pooled':>8}" + "".join(f"{t:>10}" for t in rows[0]["tasks"]))
        for d in rows:
            for label, m in d["accuracy"].items():
                print(f"{d['seed']:>4} {label:<12}{sum(m.values()) / len(m):>8.3f}" + "".join(f"{m[t]:>10.3f}" for t in d["tasks"]))
        for blend in ("blend_best", "blend_0.5"):
            ok = []
            print(f"\n{blend}: pooled difference [95% CI], McNemar p")
            for d in rows:
                line = []
                for other in (a, b, "multitask"):
                    c = d["comparisons"][f"{blend}_vs_{other}"]
                    p = c["pooled"]
                    line.append(f"vs {other}: {p['difference']:+.3f} [{p['ci']['lower']:+.3f},{p['ci']['upper']:+.3f}] p={p['p_exact_two_sided']:.4f}")
                print(f"  seed {d['seed']}: " + "  |  ".join(line))
                ok.append(meets(d["comparisons"][f"{blend}_vs_{a}"]) and meets(d["comparisons"][f"{blend}_vs_{b}"]))
            proven = len(rows) >= 3 and all(ok)
            print(f"  -> criteria met in {sum(ok)}/{len(rows)} seeds: {'PROVEN' if proven else 'NOT PROVEN'} (needs all of >= 3 seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--adapters", required=True)
    r.add_argument("--pair", required=True)
    r.add_argument("--skills", nargs=2, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--out", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--out", required=True)
    args = ap.parse_args()
    {"run": run, "verdict": verdict}[args.cmd](args)
