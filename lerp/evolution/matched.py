"""Evolution arm of the matched-conditions comparison with plain training.

Everything that earlier comparisons let differ is fixed here to what the plain-training control uses:

* data: children train on pairs from ``problems.stream`` with the family shares EXACT (chain 50% / add 25% / mul 25%), however long the run;
* start: generation 1 crosses the founders (0.5/0.5) - the same adapter the control starts from;
* training rank: constant (``rank``, default 32). A cross of two rank-``rank`` parents has rank ``2 rank``; it is compressed back to ``rank`` BEFORE training
  (truncated SVD; the energy kept is recorded). This is a deliberate extra step, not part of the control, and is named as such in every report;
* learning-rate schedule: ONE global warm-up + cosine over the lineage (``generations * child_steps`` steps); a child of generation g trains the slice
  ``[(g-1) child_steps, g child_steps)`` of it with a fresh optimizer (the 2x2 restart experiment found the optimizer state immaterial under a global schedule).
  An optimizer-state transfer across a cross is therefore not needed and not attempted;
* cost: the control gets as many training steps as ALL children together (``generations * children * child_steps``); selection evaluations are counted too.

Selection uses the dev split only; the final comparison uses the shared fresh items. The decision rule is in ``experiments/matched_evolution.py``.
"""
from __future__ import annotations

import itertools
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import problems
from .archive import Archive, Organism
from .combine import combine_adapters
from .learning import compress_adapter, file_sha256, train_adapter
from .orchestrator import CRITERIA, Evaluator, EvolutionConfig, fitness, forgotten, rank_key


@dataclass
class MatchedConfig:
    base_model: str
    workdir: str                      # holds start/, founders/g0-*, data/*_train.jsonl + *_eval.jsonl (dev = first n_dev rows), data_fresh/, controls/s{seed}
    device: str = "cpu"
    dtype: str = "bfloat16"
    founders: list[str] = field(default_factory=lambda: ["add", "mul"])
    new_family: str = "chain"
    n_dev: int = 300
    n_fresh: int = 1000
    generations: int = 3
    children: int = 2
    survivors: int = 2
    child_steps: int = 200
    cross_weights: list[float] = field(default_factory=lambda: [0.5])
    replay_fraction: float = 0.5
    new_skill_weight: float = 2.0
    adapter_scale: float = 2.0
    rank: int = 32
    lr: float = 2e-4
    batch: int = 4
    accum: int = 2
    max_new_tokens: int = 12
    seed: int = 1

    @property
    def lineage_steps(self) -> int:
        return self.generations * self.child_steps

    @property
    def total_steps(self) -> int:
        return self.generations * self.children * self.child_steps


def _eval_cfg(cfg: MatchedConfig) -> EvolutionConfig:
    return EvolutionConfig(base_model=cfg.base_model, device=cfg.device, dtype=cfg.dtype, founders=cfg.founders, new_family=cfg.new_family,
                           n_dev=cfg.n_dev, rank=cfg.rank, adapter_scale=cfg.adapter_scale, new_skill_weight=cfg.new_skill_weight,
                           max_new_tokens=cfg.max_new_tokens, seed=cfg.seed, survivors=cfg.survivors)


def run_matched(cfg: MatchedConfig, out: Path, *, log: Callable[[str], None] = print, dev_evaluator=None, fresh_evaluator=None) -> dict:
    work, out = Path(cfg.workdir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    families = cfg.founders + [cfg.new_family]
    ecfg = _eval_cfg(cfg)
    rows = {f: [json.loads(line) for line in (work / "data" / f"{f}_train.jsonl").read_text(encoding="utf-8").splitlines()] for f in families}
    mix = {**{f: cfg.replay_fraction / len(cfg.founders) for f in cfg.founders}, cfg.new_family: 1.0 - cfg.replay_fraction}
    per_step = cfg.batch * cfg.accum
    dev_ev = dev_evaluator or Evaluator(ecfg, work / "data", families)
    fresh_ev = fresh_evaluator or Evaluator(ecfg, work / "data_fresh", families)
    archive = Archive(out / "archive")
    dev_window, fresh_window = (0, cfg.n_dev), (0, cfg.n_fresh)
    costs = {"training_steps": 0, "dev_evaluations": 0, "dev_items_scored": 0, "compressions": []}

    def adapter_dir(oid: str) -> Path:
        return archive.root / "adapters" / oid

    def score_dev(organisms: list[Organism]) -> None:
        acc, _ = dev_ev.score({o.id: adapter_dir(o.id) for o in organisms}, dev_window)
        costs["dev_evaluations"] += len(organisms)
        costs["dev_items_scored"] += len(organisms) * cfg.n_dev * len(families)
        for o in organisms:
            o.dev = acc[o.id]
            archive.record(o)

    founders = []
    for f in cfg.founders:
        oid = f"g0-{f}"
        shutil.copytree(work / "founders" / oid, adapter_dir(oid))
        founders.append(archive.record(Organism(id=oid, generation=0, op="founder", adapter=f"adapters/{oid}", status="survivor",
                                                adapter_sha256=file_sha256(adapter_dir(oid) / "adapter_model.safetensors"))))
    dev_ev.spare = [adapter_dir(o.id) for o in founders]
    score_dev(founders)
    anchor = {f: max(o.dev.get(f, 0.0) for o in founders) for f in cfg.founders}
    survivors = [o.id for o in founders]
    log(f"founders dev: " + "; ".join(f"{o.id} {o.dev}" for o in founders))

    for g in range(1, cfg.generations + 1):
        ranked = sorted((archive.organisms[i] for i in survivors), key=lambda o: rank_key(o, ecfg))
        ids = [o.id for o in ranked]
        combos = [((a, b), w) for a, b in itertools.combinations(ids, 2) for w in cfg.cross_weights]
        children: list[Organism] = []
        for k in range(cfg.children):
            (a, b), w = combos[k % len(combos)]
            oid = f"g{g}-c{k}"
            work_dirs = [out / "tmp" / f"{oid}-{name}" for name in ("start", "start-r", "trained")]
            for d in work_dirs:
                shutil.rmtree(d, ignore_errors=True)
            start, start_r, trained = work_dirs
            combine_adapters([(archive.path(a), w), (archive.path(b), 1 - w)], start, out_scale=cfg.adapter_scale)
            rank_in = sum(int(json.loads((archive.path(p) / "adapter_config.json").read_text())["r"]) for p in (a, b))
            use = start
            compression = {"rank_in": rank_in, "rank_out": rank_in, "energy_kept": 1.0}
            if rank_in > cfg.rank:   # keep the TRAINING rank at the control's rank: an explicit extra step, recorded
                compression = compress_adapter(start, start_r, cfg.rank, cfg.adapter_scale)
                use = start_r
            costs["compressions"].append({"child": oid, **compression})
            stream = problems.stream(rows, mix, cfg.lineage_steps * per_step, seed=20_000 + 1000 * cfg.seed + k)
            info = train_adapter(cfg.base_model, stream, adapter_dir(oid), init=use, steps=cfg.child_steps, lr=cfg.lr, seed=cfg.seed + 10 * g + k,
                                 device=cfg.device, dtype=cfg.dtype, batch=cfg.batch, accum=cfg.accum, ordered=True,
                                 data_offset_steps=(g - 1) * cfg.child_steps, schedule_total=cfg.lineage_steps,
                                 schedule_offset=(g - 1) * cfg.child_steps)
            info["pre_training_compression"] = compression
            costs["training_steps"] += cfg.child_steps
            for d in work_dirs:
                shutil.rmtree(d, ignore_errors=True)
            children.append(archive.record(Organism(id=oid, generation=g, op="cross+learn", adapter=f"adapters/{oid}", parents=[a, b],
                                                    weights=[w, 1 - w], training=info,
                                                    adapter_sha256=file_sha256(adapter_dir(oid) / "adapter_model.safetensors"))))
        score_dev(children)
        pool = [archive.organisms[i] for i in survivors]
        for c in children:
            lost = forgotten(c.dev, anchor, cfg.founders, CRITERIA["retention_tolerance"])
            c.status = "candidate" if not lost else f"rejected:forgot {','.join(lost)}"
            archive.record(c)
            log(f"  {c.id} <- {c.parents} w={c.weights[0]}  dev {c.dev}  fitness {fitness(c.dev, ecfg):.3f}  {c.status}  "
                f"(pre-training energy kept {c.training['pre_training_compression']['energy_kept']:.3f})")
        admitted = [c for c in children if c.status == "candidate"]
        ranked_all = sorted(pool + admitted, key=lambda o: rank_key(o, ecfg))
        keep = [o.id for o in ranked_all[:cfg.survivors]]
        for o in pool + children:
            o.status = "survivor" if o.id in keep else (o.status if o.status.startswith("rejected") else "dropped")
            archive.record(o)
        survivors = keep
        log(f"generation {g}: survivors {survivors}")

    best = min((archive.organisms[i] for i in survivors), key=lambda o: rank_key(o, ecfg))
    best.status = "final"
    archive.record(best)

    contenders: dict[str, Path | None] = {f"evolved:{best.id}": adapter_dir(best.id), "control": work / "controls" / f"s{cfg.seed}",
                                          "start": work / "start"}
    for o in founders:
        contenders[o.id] = adapter_dir(o.id)
    fresh_ev.spare = [adapter_dir(o.id) for o in founders]
    acc, items = fresh_ev.score(contenders, fresh_window)
    from ..statistics import mcnemar
    evolved = f"evolved:{best.id}"
    comparisons = {f: mcnemar(items[evolved][f], items["control"][f]) for f in families}
    lineage = archive.lineage(best.id)
    kept = sum(archive.organisms[i].training.get("steps", 0) for i in lineage if archive.organisms[i].op == "cross+learn")
    result = {
        "config": asdict(cfg), "best": best.id, "lineage": lineage, "fresh_accuracy": acc, "vs_control": comparisons,
        "cost": {**costs, "control_training_steps": cfg.total_steps, "steps_in_final_lineage": kept,
                 "steps_discarded": costs["training_steps"] - kept},
        "dev_anchor": anchor, "wall_seconds": time.time() - started,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "items.json").write_text(json.dumps({"window": list(fresh_window), "items": items}), encoding="utf-8")
    shutil.rmtree(out / "tmp", ignore_errors=True)
    log(json.dumps({"fresh_accuracy": acc, "cost": result["cost"]}, indent=2))
    return result
