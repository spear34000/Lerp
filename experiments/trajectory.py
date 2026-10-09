"""Per-generation search scores of a Lerp run: best/mean fitness and mean ARC-parent weight."""
import glob, json, os, sys, statistics as st
run = sys.argv[1]
rows = []
for g in sorted(glob.glob(run + "/generations/gen-*/cand-*")):
    sp = os.path.join(g, "score.json")
    if not os.path.exists(sp): continue
    gen = int(os.path.basename(os.path.dirname(g))[4:]); cand = int(os.path.basename(g)[5:])
    s = json.load(open(sp)); gj = json.load(open(os.path.join(g, "genome.json")))
    genes = gj["genes"]
    rows.append(dict(gen=gen, cand=cand, fit=s["fitness"], metrics=s["metrics"], w=sum(genes) / len(genes), id=gj["id"]))
evals = 0
print(f"{'gen':>3} {'n':>3} {'best':>6} {'mean':>6} {'mean arc-weight':>16} {'evals so far':>13}")
for gen in sorted({r['gen'] for r in rows}):
    rs = [r for r in rows if r['gen'] == gen]; evals += len(rs)
    print(f"{gen:>3} {len(rs):>3} {max(r['fit'] for r in rs):>6.3f} {st.mean(r['fit'] for r in rs):>6.3f} {st.mean(r['w'] for r in rs):>16.2f} {evals:>13}")
best = sorted(rows, key=lambda r: -r['fit'])[:5]
print("top5:", [(r['id'], r['fit'], round(r['w'], 2)) for r in best])
json.dump(rows, open(run + "/trajectory.json", "w"))
