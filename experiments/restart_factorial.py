"""2x2 split of the restart cost: optimizer state (kept / fresh every segment) x learning-rate schedule (one global schedule / restarted every segment).

Cells (all: same start adapter, same 9,600-pair exact-mix stream consumed in the same order, rank 32, 1,200 steps in six 200-step segments, same evaluation):

  C1  state kept,  global schedule     (already trained by experiments/restart_effect.py: one continuous run)
  A   state kept,  restarted schedule  (optimizer state carried between segments; each segment has its own warm-up + cosine over 200 steps)
  B   fresh state, global schedule     (a new optimizer every segment; ONE warm-up + cosine over 1,200 steps, positioned by segment)
  C2  fresh state, restarted schedule  (already trained by experiments/restart_effect.py)

C1 and C2 are NOT retrained. A and B are trained here with the same three stream seeds. Randomness: the start adapter is given and LoRA dropout is 0, so training
randomness is the data order, which is identical across cells of a seed; A and B use the stream seed's own `seed` for every segment (C2's segments used
`seed + 100 k`, which has no effect under those conditions).

Evaluation: ALL twelve models (4 cells x 3 seeds) are scored on the same NEW 1,000 items per family, drawn after the rules below were written and excluding every
question of the training, dev and earlier test splits. The earlier 600-item results stay as diagnostics.

Rules, fixed before running:
  * restart cost replicates if C2 - C1 on `chain` is <= -0.02 on the seed mean and negative in all three seeds;
  * an effect (state, schedule or interaction; definitions in lerp/evolution/factorial.py) is called PRESENT if its seed mean is >= +0.02 on `chain`, it is positive
    in all three seeds, and the 95% item-bootstrap interval of the seed mean excludes 0; ABSENT if the seed mean is within +-0.01 and the interval includes 0; otherwise
    NO VERDICT;
  * the bootstrap resamples items only (same indices for all models); it does not capture training-seed variation, which is why the sign in each seed is also required.
  * the experiment cannot separate warm-up from cosine decay inside the schedule factor.

    python experiments/restart_factorial.py --run DIR_OF_A_FINISHED_ARM --prev DIR_OF_restart_effect --out DIR [--seeds 1 2 3]
"""
import argparse
import hashlib
import json
from pathlib import Path

from lerp.evolution import problems
from lerp.evolution.factorial import bootstrap
from lerp.evolution.learning import train_adapter
from lerp.evolution.orchestrator import Evaluator, EvolutionConfig
from lerp.statistics import mcnemar

STEPS, SEGMENT = 1200, 200
FRESH = 1000


def fresh_items(run: Path, families: list[str], out: Path) -> dict[str, str]:
    """1,000 new questions per family that appear in no earlier split of the run; written as the evaluation file the Evaluator reads."""
    hashes = {}
    for fam in families:
        seen = set()
        for name in (f"{fam}_train.jsonl", f"{fam}_eval.jsonl"):
            seen |= {json.loads(line)["q"] for line in (run / "data" / name).read_text(encoding="utf-8").splitlines()}
        rows = [r for r in problems.pool(fam, 20_000, seed=777) if r["q"] not in seen][:FRESH]
        if len(rows) < FRESH:
            raise RuntimeError(f"{fam}: not enough unseen questions")
        hashes[fam] = problems.write_jsonl(rows, out / f"{fam}_eval.jsonl")
    return hashes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    run, prev, out = Path(args.run), Path(args.prev), Path(args.out)
    cfg = EvolutionConfig(**json.loads((run / "result.json").read_text(encoding="utf-8"))["config"])
    families = cfg.founders + [cfg.new_family]
    rows = {f: [json.loads(line) for line in (run / "data" / f"{f}_train.jsonl").read_text(encoding="utf-8").splitlines()] for f in families}
    mix = {**{f: cfg.replay_fraction / len(cfg.founders) for f in cfg.founders}, cfg.new_family: 1.0 - cfg.replay_fraction}
    out.mkdir(parents=True, exist_ok=True)
    start = prev / "start"
    common = dict(lr=cfg.lr, device=cfg.device, dtype=cfg.dtype, batch=cfg.batch, accum=cfg.accum, ordered=True)
    cells: dict[str, Path] = {}
    for seed in args.seeds:
        stream = problems.stream(rows, mix, STEPS * cfg.batch * cfg.accum, seed=10_000 + seed)   # the same stream restart_effect.py used
        cells[f"c1_s{seed}"] = prev / f"c1_s{seed}"
        cells[f"c2_s{seed}"] = prev / f"c2_s{seed}_seg{STEPS // SEGMENT - 1}"
        for name, kwargs in (("a", dict(carry_state=True, global_schedule=False)), ("b", dict(carry_state=False, global_schedule=True))):
            previous, state = start, None
            for k in range(STEPS // SEGMENT):
                seg = out / f"{name}_s{seed}_seg{k}"
                new_state = out / f"{name}_s{seed}_state{k}.pt"
                if not (seg / "adapter_model.safetensors").is_file():
                    extra = {}
                    if kwargs["global_schedule"]:
                        extra.update(schedule_total=STEPS, schedule_offset=k * SEGMENT)
                    if kwargs["carry_state"]:
                        extra.update(state_out=new_state, **({"state_in": state} if state else {}))
                    train_adapter(cfg.base_model, stream, seg, init=previous, steps=SEGMENT, data_offset_steps=k * SEGMENT, seed=seed, **common, **extra)
                previous, state = seg, (new_state if kwargs["carry_state"] else None)
            cells[f"{name}_s{seed}"] = previous
            print(f"{name.upper()} seed {seed}: {STEPS // SEGMENT} segments done", flush=True)
    data = out / "data_fresh"
    hashes = fresh_items(run, families, data)
    evaluator = Evaluator(cfg, data, families)
    acc, items = evaluator.score(cells, (0, FRESH))
    new = cfg.new_family
    shas = {n: hashlib.sha256((p / "adapter_model.safetensors").read_bytes()).hexdigest() for n, p in cells.items()}
    print(f"\nfresh items: {FRESH} per family, never used for training, selection or earlier tests\n{'model':<8}" + "".join(f"{f:>9}" for f in families))
    for name in sorted(cells):
        print(f"{name:<8}" + "".join(f"{acc[name][f]:>9.3f}" for f in families))
    print(f"\n`{new}` per seed, paired McNemar against C1 (same fresh items): difference [95% CI] p")
    for seed in args.seeds:
        for cell in ("c2", "a", "b"):
            m = mcnemar(items[f"{cell}_s{seed}"][new], items[f"c1_s{seed}"][new])
            print(f"  seed {seed} {cell} - c1: {m['difference']:+.3f} [{m['ci']['lower']:+.3f},{m['ci']['upper']:+.3f}] p={m['p_exact_two_sided']:.3f}")
    boot = bootstrap([{k: items[f"{k}_s{s}"][new] for k in ("c1", "a", "b", "c2")} for s in args.seeds])
    print(f"\neffects on `{new}` (accuracy points; positive = favours keeping state / one global schedule), per seed and seed mean with item-bootstrap 95% interval")
    verdicts = {}
    for name in ("restart_cost", "state", "schedule", "interaction"):
        per = [e[name] for e in boot["per_seed"]]
        mean, (lo, hi) = boot["mean"][name], boot["interval"][name]
        threshold = -0.02 if name == "restart_cost" else 0.02
        if name == "restart_cost":
            # restart_cost = c1 - c2 is positive when restarting hurts; the rule states it as C2 - C1 <= -0.02
            present = mean >= 0.02 and all(v > 0 for v in per) and lo > 0
        else:
            present = mean >= 0.02 and all(v > 0 for v in per) and lo > 0
        absent = abs(mean) < 0.01 and lo <= 0 <= hi
        verdicts[name] = "PRESENT" if present else "ABSENT" if absent else "NO VERDICT"
        print(f"  {name:<13} mean {mean:+.3f} [{lo:+.3f},{hi:+.3f}]  per seed " + " ".join(f"{v:+.3f}" for v in per) + f"   -> {verdicts[name]}")
    (out / "result.json").write_text(json.dumps({"accuracy": acc, "effects": boot, "verdicts": verdicts, "fresh_data_sha256": hashes, "adapter_sha256": shas,
                                                 "config": {"steps": STEPS, "segment": SEGMENT, "seeds": args.seeds, "mix": mix, "fresh_items": FRESH,
                                                            "rank": 32, "stream_seeds": [10_000 + s for s in args.seeds]}}, indent=2), encoding="utf-8")
    (out / "items.json").write_text(json.dumps({"window": [0, FRESH], "items": items}), encoding="utf-8")


if __name__ == "__main__":
    main()
