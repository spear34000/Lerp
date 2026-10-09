"""Aggregate Lerp runs across training seeds: parents vs merged candidates (per task, mean +- sd)."""
import json
import statistics as st
import sys
from pathlib import Path

runs = {int(k): Path(v) for k, v in (a.split("=") for a in sys.argv[1:])}  # seed=runpath
base_run = runs[min(runs)]


def metrics(path: Path):
    f = path / "score.json"
    return json.loads(f.read_text(encoding="utf-8"))["metrics"] if f.is_file() else None


base = metrics(base_run / "baselines" / "base")
rows = {}
for seed, run in sorted(runs.items()):
    rows[seed] = {
        "arc_parent": metrics(run / "baselines" / "arc"),
        "hs_parent": metrics(run / "baselines" / "hellaswag"),
        "w0.25": metrics(run / "generations" / "gen-000" / "cand-000"),
        "w0.50": metrics(run / "generations" / "gen-000" / "cand-001"),
        "w0.75": metrics(run / "generations" / "gen-000" / "cand-002"),
    }
print("base (shared):", base)
cols = ["arc_parent", "hs_parent", "w0.25", "w0.50", "w0.75"]
print(f"{'seed':>4} " + " ".join(f"{c:>22}" for c in cols))
for seed, r in rows.items():
    print(f"{seed:>4} " + " ".join(f"{(str(round(r[c]['arc_easy'],2))+'/'+str(round(r[c]['hellaswag'],2))) if r[c] else 'n/a':>22}" for c in cols))


def ms(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.3f} +- {st.stdev(xs):.3f}" if len(xs) > 1 else (f"{xs[0]:.3f}" if xs else "n/a")


print("\nmean +- sd over seeds (arc_easy | hellaswag | mean of both)")
for c in cols:
    a = [r[c]["arc_easy"] for r in rows.values() if r[c]]
    h = [r[c]["hellaswag"] for r in rows.values() if r[c]]
    m = [(r[c]["arc_easy"] + r[c]["hellaswag"]) / 2 for r in rows.values() if r[c]]
    print(f"{c:>11}: {ms(a)} | {ms(h)} | {ms(m)}   (n={len(m)})")
print(f"{'base':>11}: {base['arc_easy']:.3f} | {base['hellaswag']:.3f} | {(base['arc_easy']+base['hellaswag'])/2:.3f}")

print("\nper-seed gain of the 0.50 merge over the BEST parent on each task (arc_easy, hellaswag) and over parent mean:")
for seed, r in rows.items():
    if not r["w0.50"]:
        continue
    ga = r["w0.50"]["arc_easy"] - max(r["arc_parent"]["arc_easy"], r["hs_parent"]["arc_easy"])
    gh = r["w0.50"]["hellaswag"] - max(r["arc_parent"]["hellaswag"], r["hs_parent"]["hellaswag"])
    mm = (r["w0.50"]["arc_easy"] + r["w0.50"]["hellaswag"]) / 2
    pm = sum(r[p][t] for p in ("arc_parent", "hs_parent") for t in ("arc_easy", "hellaswag")) / 4
    print(f"  seed {seed}: arc {ga:+.3f}  hellaswag {gh:+.3f}  merged-mean minus parents-mean {mm - pm:+.3f}")
