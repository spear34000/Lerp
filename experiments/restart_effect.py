"""Does restarting the optimizer and the learning-rate schedule every 200 steps cost accuracy? Continuous (C1) against segmented-with-restart (C2).

Everything else is identical between the two conditions of a seed: the same starting adapter (the 0.5/0.5 combination of the founders, rank 32, scale 2),
the same prepared training stream (chain 50% / add 25% / mul 25% EXACTLY, 9,600 pairs, consumed in the same order), the same training rank (32), the same
number of steps (1,200), the same training seed, and the same final evaluation (rank 32 as trained, no compression). The only difference is that C2 stops
every 200 steps and starts the next segment with a fresh optimizer and its own warm-up + cosine schedule. Compression and selection are NOT part of this
comparison. C1 vs "C2 with carried state" is covered by the equivalence tests in tests/test_restart.py, not re-run here.

What the comparison can say: whether restarting optimizer + schedule together changes the result; it cannot say which of the two matters.

Reading the result (fixed before running):
  * -0.02 on `chain` (C2 minus C1) is a *practical threshold for deciding whether to pursue the restart explanation further*, not a significance test;
  * three seeds on the SAME 600 test items are a pilot: they are not three independent test sets, and with three seeds three negative signs happen 12.5% of
    the time by chance. Every seed's paired difference and 95% interval is printed; an unclear picture is reported as "no verdict".

    python experiments/restart_effect.py --run DIR_OF_A_FINISHED_ARM --out DIR [--seeds 1 2 3]
"""
import argparse
import json
from pathlib import Path

from lerp.evolution import problems
from lerp.evolution.combine import combine_adapters
from lerp.evolution.learning import train_adapter
from lerp.evolution.orchestrator import Evaluator, EvolutionConfig
from lerp.statistics import mcnemar

STEPS, SEGMENT = 1200, 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    cfg = EvolutionConfig(**json.loads((run / "result.json").read_text(encoding="utf-8"))["config"])
    families = cfg.founders + [cfg.new_family]
    rows = {f: [json.loads(line) for line in (run / "data" / f"{f}_train.jsonl").read_text(encoding="utf-8").splitlines()] for f in families}
    mix = {**{f: cfg.replay_fraction / len(cfg.founders) for f in cfg.founders}, cfg.new_family: 1.0 - cfg.replay_fraction}
    out.mkdir(parents=True, exist_ok=True)
    start = out / "start"
    if not start.exists():
        combine_adapters([(run / "archive" / "adapters" / f"g0-{f}", 0.5) for f in cfg.founders], start, out_scale=cfg.adapter_scale)
    common = dict(lr=cfg.lr, device=cfg.device, dtype=cfg.dtype, batch=cfg.batch, accum=cfg.accum, ordered=True)
    finals: dict[str, Path] = {}
    for seed in args.seeds:
        stream = problems.stream(rows, mix, STEPS * cfg.batch * cfg.accum, seed=10_000 + seed)
        c1 = out / f"c1_s{seed}"
        if not (c1 / "adapter_model.safetensors").is_file():
            info = train_adapter(cfg.base_model, stream, c1, init=start, steps=STEPS, seed=seed, **common)
            print(f"C1 seed {seed}: loss {info['final_loss']:.3f}", flush=True)
        finals[f"c1_s{seed}"] = c1
        previous = start
        for k in range(STEPS // SEGMENT):
            seg = out / f"c2_s{seed}_seg{k}"
            if not (seg / "adapter_model.safetensors").is_file():
                info = train_adapter(cfg.base_model, stream, seg, init=previous, steps=SEGMENT, data_offset_steps=k * SEGMENT, seed=seed + 100 * k, **common)
            previous = seg
        finals[f"c2_s{seed}"] = previous
        print(f"C2 seed {seed}: {STEPS // SEGMENT} segments done", flush=True)
    evaluator = Evaluator(cfg, run / "data", families)
    window = (cfg.n_dev, cfg.n_dev + cfg.n_test)
    acc, items = evaluator.score(finals, window)
    new = cfg.new_family
    print(f"\n{'model':<10}" + "".join(f"{f:>9}" for f in families))
    for name in finals:
        print(f"{name:<10}" + "".join(f"{acc[name][f]:>9.3f}" for f in families))
    diffs = []
    print(f"\npaired on {new} (C2 - C1), same {cfg.n_test} test items: difference [95% CI] p (only C2 vs only C1)")
    for seed in args.seeds:
        m = mcnemar(items[f"c2_s{seed}"][new], items[f"c1_s{seed}"][new])
        diffs.append(m["difference"])
        print(f"  seed {seed}: {m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}] p={m['p_exact_two_sided']:.3f} ({m['a_only']} vs {m['b_only']})")
    mean = sum(diffs) / len(diffs)
    signs = ["-" if d < 0 else "+" for d in diffs]
    print(f"\nmean C2 - C1 = {mean:+.3f}; signs {''.join(signs)}")
    if mean <= -0.02 and all(d < 0 for d in diffs):
        print("practical threshold reached in the same direction in every seed: worth testing the restart explanation further (pilot, not a verdict)")
    elif mean > -0.02 and any(d <= -0.02 for d in diffs) and any(d > 0 for d in diffs):
        print("mixed picture: no verdict")
    elif mean > -0.02:
        print("practical threshold not reached: no evidence for a restart cost in this pilot (equivalence not shown)")
    else:
        print("mean beyond the threshold but not consistent across seeds: no verdict")
    (out / "result.json").write_text(json.dumps({"accuracy": acc, "differences_chain_c2_minus_c1": dict(zip(map(str, args.seeds), diffs)), "mean": mean}, indent=2),
                                     encoding="utf-8")


if __name__ == "__main__":
    main()
