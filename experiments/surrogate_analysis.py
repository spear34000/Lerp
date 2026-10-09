"""Offline replay: how many evaluations does a GP/EI search need on the ARC+BoolQ candidate pool?

Pool = every candidate already scored by Lerp in the qb_evo and qb_rand runs (unique genomes).
Features = mean ARC-parent weight per module group (attention, mlp, other). Target = search fitness.
Compares: random sampling, GP + expected improvement (pool-based), against evolution's fixed budget.
"""
import glob
import json
import math
import random
import statistics as st
import sys

import numpy as np

runs = sys.argv[1:]
pool = {}
for run in runs:
    for g in sorted(glob.glob(run + "/generations/gen-*/cand-*")):
        try:
            score = json.load(open(g + "/score.json", encoding="utf-8"))
            genes = json.load(open(g + "/genome.json", encoding="utf-8"))["genes"]
        except FileNotFoundError:
            continue
        key = tuple(round(x, 6) for x in genes)
        pool[key] = (score["fitness"], score["metrics"])
keys = list(pool)
X = np.array([[np.mean(k[i * 3:(i + 1) * 3]) for i in range(3)] for k in keys])  # attn, mlp, other group means
y = np.array([pool[k][0] for k in keys])
arc = np.array([list(pool[k][1].values())[0] for k in keys])
boolq = np.array([list(pool[k][1].values())[1] for k in keys])
print(f"unique candidates: {len(keys)} | fitness mean {y.mean():.3f} sd {y.std():.3f} max {y.max():.3f}")


def loo_r2(X, y):
    A = np.c_[np.ones(len(X)), X]
    preds = []
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        preds.append(A[i] @ coef)
    preds = np.array(preds)
    return 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print("\n1) Which module group matters? OLS on group-mean weights (coef per +1.0 of ARC weight) and leave-one-out R^2")
for name, t in (("fitness", y), ("arc_easy", arc), ("boolq", boolq)):
    A = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    print(f"  {name:9s} attn {coef[1]:+.3f}  mlp {coef[2]:+.3f}  other {coef[3]:+.3f}   LOO R2 = {loo_r2(X, t):.2f}   (global-mean-only LOO R2 = {loo_r2(X.mean(1, keepdims=True), t):.2f})")


def rbf(a, b, ls):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d / ls ** 2)


def gp_ei_search(order_seed, budget, ls=0.25, noise=1e-4):
    rng = random.Random(order_seed)
    chosen = rng.sample(range(len(keys)), 2)  # two random starting evaluations
    best_trace = [max(y[chosen[:1]]), max(y[chosen])]
    while len(chosen) < budget:
        Xc, yc = X[chosen], y[chosen]
        mu0 = yc.mean()
        K = rbf(Xc, Xc, ls) + noise * np.eye(len(chosen))
        Ks = rbf(X, Xc, ls)
        alpha = np.linalg.solve(K, yc - mu0)
        mu = mu0 + Ks @ alpha
        v = np.linalg.solve(K, Ks.T)
        var = np.maximum(1.0 - (Ks * v.T).sum(1), 1e-9) * max(yc.var(), 1e-6)
        sd = np.sqrt(var)
        z = (mu - yc.max()) / sd
        ei = (mu - yc.max()) * 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2))) + sd * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        ei[chosen] = -1
        chosen.append(int(np.argmax(ei)))
        best_trace.append(max(y[chosen]))
    return best_trace


def random_search(order_seed, budget):
    rng = random.Random(order_seed)
    order = rng.sample(range(len(keys)), budget)
    return [max(y[order[:i + 1]]) for i in range(budget)]


print("\n2) Best fitness found after k evaluations (mean over 300 replays; pool max = %.3f)" % y.max())
budget = 24
gp = np.array([gp_ei_search(s, budget) for s in range(300)])
rd = np.array([random_search(s, budget) for s in range(300)])
print("   k     random    GP+EI")
for k in (3, 5, 8, 12, 16, 24):
    print(f"  {k:2d}    {rd[:, k - 1].mean():.4f}    {gp[:, k - 1].mean():.4f}")
for thr in (0.735, 0.74):
    def first_hit(tr):
        hit = np.where(tr >= thr)[0]
        return hit[0] + 1 if len(hit) else budget + 1
    print(f"  evaluations until fitness >= {thr}: random {np.mean([first_hit(t) for t in rd]):.1f}   GP+EI {np.mean([first_hit(t) for t in gp]):.1f}   (24 = evolution's whole budget)")
