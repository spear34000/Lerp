"""How much does the plain-training control itself vary? Every arm-versus-control comparison so far used ONE control adapter.

Trains extra controls of a finished evolution run with different training and sampling seeds (same data, same test items, same start: the 0.5/0.5
combination of the founders, same 50% replay mix, rank 16 at scale 2) plus one control with half the steps, and scores them on the run's stored test window
so they pair with its `test_items.json`.

Decision rules, fixed before running:
  * the extra controls spread by about +-0.02 or more around the original 0.115 on `chain`  -> the gap between loops and the control is inside the noise of the
    procedure; the honest conclusion is "evolution ~ plain training here" and this benchmark cannot separate loop designs;
  * the controls are tight (spread well below 0.02) and the half-step control is clearly below the full one  -> the gap is real; the optimizer / schedule restart
    hypothesis earns a direct test;
  * the half-step control ~ the full-step one  -> `chain` is near the floor of a 0.5B model: more steps barely help anyone and no loop change can show up here.

    python experiments/control_replicates.py --run DIR_OF_A_FINISHED_ARM --out DIR [--replicates 2]
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from lerp.evolution import problems
from lerp.evolution.combine import combine_adapters
from lerp.evolution.learning import compress_adapter, train_adapter
from lerp.evolution.orchestrator import Evaluator, EvolutionConfig
from lerp.statistics import mcnemar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--half", action="store_true", default=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    cfg = EvolutionConfig(**result["config"])
    families = cfg.founders + [cfg.new_family]
    total = result["training_steps"]["control"]
    train_rows = {f: [json.loads(line) for line in (run / "data" / f"{f}_train.jsonl").read_text(encoding="utf-8").splitlines()] for f in families}
    mix = {**{f: cfg.replay_fraction / len(cfg.founders) for f in cfg.founders}, cfg.new_family: 1.0 - cfg.replay_fraction}
    per_step = cfg.batch * cfg.accum

    def sft(count: int, seed: int):
        local = random.Random(seed)
        picked = []
        for fam, weight in mix.items():
            picked += [problems.sft_pair(r) for r in local.sample(train_rows[fam], min(round(count * weight), len(train_rows[fam])))]
        local.shuffle(picked)
        return picked

    out.mkdir(parents=True, exist_ok=True)
    start = out / "m05"
    if not start.exists():
        combine_adapters([(run / "archive" / "adapters" / f"g0-{f}", 0.5) for f in cfg.founders], start, out_scale=cfg.adapter_scale)
    plans = [(f"control-s{k + 2}", total, 7777 + k + 1, cfg.seed + 999 + 100 * (k + 1)) for k in range(args.replicates)]
    plans.append(("control-half", total // 2, 7777, cfg.seed + 999))   # the original seeds, half the steps, ONE cosine schedule over 600 steps
    adapters = {}
    for name, steps, sample_seed, train_seed in plans:
        final = out / name
        if not final.exists():
            tmp = out / f"{name}-trained"
            shutil.rmtree(tmp, ignore_errors=True)
            info = train_adapter(cfg.base_model, sft(steps * per_step, sample_seed * cfg.seed), tmp, init=start, steps=steps, lr=cfg.lr, seed=train_seed,
                                 device=cfg.device, dtype=cfg.dtype, batch=cfg.batch, accum=cfg.accum)
            compress_adapter(tmp, final, cfg.rank, cfg.adapter_scale)
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{name}: {steps} steps, final loss {info['final_loss']:.3f}", flush=True)
        adapters[name] = final
    evaluator = Evaluator(cfg, run / "data", families)
    window = (cfg.n_dev, cfg.n_dev + cfg.n_test)
    acc, items = evaluator.score(adapters, window)
    ref = json.loads((run / "test_items.json").read_text(encoding="utf-8"))["items"]
    new = cfg.new_family
    rows = {"control-plain (original)": (result["test_accuracy"]["control-plain"], ref["control-plain"])}
    rows.update({name: (acc[name], items[name]) for name in adapters})
    print(f"\n{'model':<28}" + "".join(f"{f:>9}" for f in families) + f"   {new} vs original control")
    for name, (a, it) in rows.items():
        m = mcnemar(it[new], ref["control-plain"][new]) if name != "control-plain (original)" else None
        tail = "" if m is None else f"{m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}] p={m['p_exact_two_sided']:.3f}"
        print(f"{name:<28}" + "".join(f"{a[f]:>9.3f}" for f in families) + f"   {tail}")
    chain = [a[new] for n, (a, _) in rows.items() if n != "control-half"]
    print(f"\nfull-length controls on {new}: min {min(chain):.3f} max {max(chain):.3f} spread {max(chain) - min(chain):.3f}")
    (out / "controls.json").write_text(json.dumps({"accuracy": {n: a for n, (a, _) in rows.items()}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
