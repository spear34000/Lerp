"""The generation loop: founders learn skills, survivors are crossed, the children learn a NEW verifiable skill, selection keeps the best,
and the survivors become the next parents. A plain-training control with the same number of training steps is the yardstick.

Selection uses the dev split only. The final comparison uses a test split that no training pair, no selection step and no replay ever saw.
Success is decided by criteria fixed in ``CRITERIA`` before any run.
"""
from __future__ import annotations

import itertools
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from . import problems
from .archive import Archive, Organism
from .learning import compress_adapter, file_sha256, train_adapter

CRITERIA = {
    "new_skill_gain": 0.05,       # evolved best beats the best founder on the new family by at least this (test accuracy)
    "new_skill_p": 0.01,          # exact McNemar p-value for that comparison
    "retention_tolerance": 0.05,  # on every founder family the evolved best may lose at most this against the best founder of that family
    "beats_control_p": 0.05,      # evolution "contributes" only if it beats the plain-training control on the new family with this p
}


@dataclass
class EvolutionConfig:
    base_model: str
    device: str = "cpu"
    dtype: str = "bfloat16"
    founders: list[str] = field(default_factory=lambda: ["add", "mul"])
    new_family: str = "chain"
    n_train: int = 2000
    n_dev: int = 100
    n_test: int = 300
    founder_steps: int = 150
    child_steps: int = 100
    generations: int = 3
    children: int = 4
    survivors: int = 2
    cross_weights: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    replay_fraction: float = 0.5
    rank: int = 16
    lr: float = 2e-4
    batch: int = 4
    accum: int = 2
    max_new_tokens: int = 12
    seed: int = 1


def load_config(path: Path) -> EvolutionConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        cfg = EvolutionConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid evolution config: {exc}") from exc
    if cfg.survivors < 2 or cfg.generations < 1 or cfg.children < 1 or not 0 <= cfg.replay_fraction < 1:
        raise ValueError("need survivors >= 2, generations >= 1, children >= 1 and 0 <= replay_fraction < 1")
    if cfg.new_family in cfg.founders or len(set(cfg.founders)) != len(cfg.founders) or len(cfg.founders) < 2:
        raise ValueError("founders must be at least two distinct families and must not include the new family")
    return cfg


class Evaluator:
    """Scores adapters on a family's problems with the resident generative evaluator; two adapters share one loaded base model."""

    def __init__(self, cfg: EvolutionConfig, data: Path, families: list[str]):
        self.cfg, self.families = cfg, families
        self.files = {f: (data / f"{f}_eval.jsonl").resolve() for f in families}

    def _spec(self, pair: list[tuple[str, Path]]):
        from ..spec import parse_spec
        raw = {"name": "evolution", "base_model": self.cfg.base_model, "mode": "lora", "method": "linear", "genes": 2, "population": 2,
               "seed": self.cfg.seed, "out_dtype": "float32",
               "parents": [{"name": n, "model": str(p)} for n, p in pair],
               "evaluation": {"device": self.cfg.device, "limit": self.cfg.n_dev,
                              "tasks": {f: {"metric": "exact_match,none", "task": problems.task_definition(p, self.cfg.max_new_tokens)}
                                        for f, p in self.files.items()}}}
        return parse_spec(raw)

    def score(self, adapters: dict[str, Path | None], window: tuple[int, int]) -> tuple[dict[str, dict], dict[str, dict]]:
        """{name: {family: accuracy}} and {name: {family: [0/1 per item]}}. ``None`` as the adapter scores the bare base model."""
        from ..resident import ResidentSession
        base_only = [n for n, p in adapters.items() if p is None]
        acc: dict[str, dict] = {}
        items: dict[str, dict] = {}
        alias = {f"m{i}": n for i, n in enumerate(n for n, p in adapters.items() if p is not None)}  # parent names must be folder-safe
        pool = [(a, adapters[n]) for a, n in alias.items()]
        if not pool:
            raise ValueError("at least one adapter is needed to build an evaluation session")
        for i in range(0, len(pool), 2):
            pair = pool[i:i + 2]
            if len(pair) == 1:  # the loader wants two distinct parents; the padding adapter is never scored here
                if len(pool) < 2:
                    raise ValueError("scoring needs at least two adapters in one call (add a second one)")
                pair.append(("padslot", pool[0][1]))
            session = ResidentSession(self._spec(pair), self.cfg.device, self.cfg.dtype)
            try:
                for name, _ in pair:
                    if name == "padslot":
                        continue
                    acc[alias[name]] = session.evaluate_reference(name, window)
                    items[alias[name]] = {k: list(v) for k, v in session.last_items.items()}
                if base_only and i == 0:
                    for name in base_only:
                        acc[name] = session.evaluate_reference("base", window)
                        items[name] = {k: list(v) for k, v in session.last_items.items()}
            finally:
                session.close()
                del session
        return acc, items


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _cross(cfg: EvolutionConfig, base: str, a: Path, b: Path, weight: float, dest: Path) -> None:
    from ..lora import build_lora
    from ..spec import parse_spec
    spec = parse_spec({"name": "cross", "base_model": base, "mode": "lora", "method": "linear", "genes": 2, "out_dtype": "float32",
                       "parents": [{"name": "a", "model": str(a)}, {"name": "b", "model": str(b)}]})
    build_lora(spec, [weight] * spec.genome_size, dest)


def run_evolution(cfg: EvolutionConfig, out: Path, *, log: Callable[[str], None] = print, evaluator=None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    families = cfg.founders + [cfg.new_family]
    started = time.time()

    # -- data: one JSONL per family and role, hashes recorded ---------------------------------------------------
    data = out / "data"
    train_rows: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    for fam in families:
        parts = problems.splits(fam, cfg.n_train, cfg.n_dev, cfg.n_test, cfg.seed)
        train_rows[fam] = parts["train"]
        hashes[f"{fam}_train"] = problems.write_jsonl(parts["train"], data / f"{fam}_train.jsonl")
        hashes[f"{fam}_eval"] = problems.write_jsonl(parts["dev"] + parts["test"], data / f"{fam}_eval.jsonl")
        questions = [{r["q"] for r in parts[k]} for k in ("train", "dev", "test")]
        if any(x & y for x, y in itertools.combinations(questions, 2)):
            raise RuntimeError(f"{fam}: train, dev and test share questions")
    dev_window, test_window = (0, cfg.n_dev), (cfg.n_dev, cfg.n_dev + cfg.n_test)
    evaluator = evaluator or Evaluator(cfg, data, families)
    archive = Archive(out / "archive")
    steps_spent = {"evolution": 0, "control": 0}
    pairs_per_step = cfg.batch * cfg.accum

    def sft(fams_weights: dict[str, float], count: int, seed: int) -> list[tuple[str, str]]:
        local = random.Random(seed)
        picked = []
        for fam, weight in fams_weights.items():
            k = round(count * weight)
            picked += [problems.sft_pair(r) for r in local.sample(train_rows[fam], min(k, len(train_rows[fam])))]
        local.shuffle(picked)
        return picked

    def learn_mix() -> dict[str, float]:
        old = cfg.replay_fraction / len(cfg.founders)
        return {**{f: old for f in cfg.founders}, cfg.new_family: 1.0 - cfg.replay_fraction}

    def adapter_dir(oid: str) -> Path:
        return archive.root / "adapters" / oid

    def register(o: Organism) -> Organism:
        o.adapter_sha256 = file_sha256(adapter_dir(o.id) / "adapter_model.safetensors")
        return archive.record(o)

    def evaluate_dev(organisms: list[Organism]) -> None:
        acc, _ = evaluator.score({o.id: adapter_dir(o.id) for o in organisms}, dev_window)
        for o in organisms:
            o.dev = acc[o.id]
            archive.record(o)

    # -- generation 0: founders ---------------------------------------------------------------------------------
    founders: list[Organism] = []
    for i, fam in enumerate(cfg.founders):
        oid = f"g0-{fam}"
        info = train_adapter(cfg.base_model, sft({fam: 1.0}, cfg.founder_steps * pairs_per_step, cfg.seed * 100 + i), adapter_dir(oid),
                             steps=cfg.founder_steps, lr=cfg.lr, rank=cfg.rank, seed=cfg.seed + i, device=cfg.device, dtype=cfg.dtype,
                             batch=cfg.batch, accum=cfg.accum)
        founders.append(register(Organism(id=oid, generation=0, op="founder", adapter=f"adapters/{oid}", training=info, status="survivor")))
        log(f"founder {oid}: trained {cfg.founder_steps} steps, loss {info['final_loss']:.3f}")
    evaluate_dev(founders)
    for o in founders:
        log(f"  {o.id} dev {o.dev}")
    survivors = [o.id for o in founders]

    # -- generations ----------------------------------------------------------------------------------------------
    for g in range(1, cfg.generations + 1):
        combos = [(a, b, w) for a, b in itertools.combinations(survivors, 2) for w in cfg.cross_weights]
        rng.shuffle(combos)
        children: list[Organism] = []
        for k, (a, b, w) in enumerate(combos[:cfg.children]):
            oid = f"g{g}-c{k}"
            merged = out / "tmp" / f"{oid}-merged"
            trained = out / "tmp" / f"{oid}-trained"
            for d in (merged, trained):
                shutil.rmtree(d, ignore_errors=True)
            _cross(cfg, cfg.base_model, archive.path(a), archive.path(b), w, merged)
            info = train_adapter(cfg.base_model, sft(learn_mix(), cfg.child_steps * pairs_per_step, cfg.seed * 1000 + g * 10 + k), trained,
                                 init=merged, steps=cfg.child_steps, lr=cfg.lr, seed=cfg.seed + g * 10 + k, device=cfg.device, dtype=cfg.dtype,
                                 batch=cfg.batch, accum=cfg.accum)
            info["compression"] = compress_adapter(trained, adapter_dir(oid), cfg.rank)
            shutil.rmtree(merged, ignore_errors=True)
            shutil.rmtree(trained, ignore_errors=True)
            steps_spent["evolution"] += cfg.child_steps
            children.append(register(Organism(id=oid, generation=g, op="cross+learn", adapter=f"adapters/{oid}", parents=[a, b],
                                              weights=[w, 1 - w], training=info)))
        evaluate_dev(children)
        pool = [archive.organisms[i] for i in survivors]
        old_best = {f: max(o.dev.get(f, 0.0) for o in pool) for f in cfg.founders}
        for c in children:
            lost = [f for f in cfg.founders if c.dev.get(f, 0.0) < old_best[f] - CRITERIA["retention_tolerance"]]
            c.status = "candidate" if not lost else f"rejected:forgot {','.join(lost)}"
            archive.record(c)
            log(f"  {c.id} <- {c.parents} w={c.weights[0]:.1f}  dev {c.dev}  {c.status}  (energy kept {c.training['compression']['energy_kept']:.3f})")
        admitted = [c for c in children if c.status == "candidate"]
        ranked = sorted(pool + admitted, key=lambda o: (-_mean(o.dev.get(f, 0.0) for f in families), o.id))
        keep = [o.id for o in ranked[:cfg.survivors]]
        for fam, oid in archive.niches([o.id for o in ranked], families).items():  # one best organism per niche survives as well
            if oid not in keep and len(keep) < cfg.survivors + len(families):
                keep.append(oid)
        for o in pool + children:
            if o.id in keep:
                o.status = "survivor"
            elif not o.status.startswith("rejected"):
                o.status = "dropped"
            archive.record(o)
        survivors = keep
        best = archive.organisms[keep[0]]
        log(f"generation {g}: survivors {survivors}; best {best.id} dev mean {_mean(best.dev.get(f, 0.0) for f in families):.3f}")

    best = max((archive.organisms[i] for i in survivors), key=lambda o: (_mean(o.dev.get(f, 0.0) for f in families), o.id))
    best.status = "final"
    archive.record(best)

    # -- controls ---------------------------------------------------------------------------------------------------
    merge_only = adapter_dir("control-merge-only")
    _cross(cfg, cfg.base_model, archive.path(founders[0]), archive.path(founders[1]), 0.5, out / "tmp" / "m05")
    compress_adapter(out / "tmp" / "m05", merge_only, cfg.rank)
    total_steps = steps_spent["evolution"]
    control = adapter_dir("control-plain")
    control_tmp = out / "tmp" / "control-trained"
    shutil.rmtree(control_tmp, ignore_errors=True)
    cinfo = train_adapter(cfg.base_model, sft(learn_mix(), total_steps * pairs_per_step, cfg.seed * 7777), control_tmp, init=out / "tmp" / "m05",
                          steps=total_steps, lr=cfg.lr, seed=cfg.seed + 999, device=cfg.device, dtype=cfg.dtype, batch=cfg.batch, accum=cfg.accum)
    compress_adapter(control_tmp, control, cfg.rank)
    steps_spent["control"] = total_steps
    log(f"control: plain training {total_steps} steps from the 0.5 merge of the founders, loss {cinfo['final_loss']:.3f}")

    # -- independent test ----------------------------------------------------------------------------------------------
    contenders: dict[str, Path | None] = {"base": None, "merge-only": merge_only, "control-plain": control, f"evolved:{best.id}": adapter_dir(best.id)}
    for o in founders:
        contenders[o.id] = adapter_dir(o.id)
    acc, items = evaluator.score(contenders, test_window)
    from ..statistics import mcnemar
    evolved = f"evolved:{best.id}"
    founder_ids = [o.id for o in founders]
    comparisons: dict[str, dict] = {}
    for fam in families:
        best_founder = max(founder_ids, key=lambda i: acc[i][fam])
        comparisons[f"{fam}: evolved vs best founder ({best_founder})"] = mcnemar(items[evolved][fam], items[best_founder][fam])
        comparisons[f"{fam}: evolved vs control-plain"] = mcnemar(items[evolved][fam], items["control-plain"][fam])
        comparisons[f"{fam}: evolved vs merge-only"] = mcnemar(items[evolved][fam], items["merge-only"][fam])
    gain = comparisons[f"{cfg.new_family}: evolved vs best founder ({max(founder_ids, key=lambda i: acc[i][cfg.new_family])})"]
    vs_control = comparisons[f"{cfg.new_family}: evolved vs control-plain"]
    retention = {f: acc[evolved][f] - max(acc[i][f] for i in founder_ids) for f in cfg.founders}
    checks = {
        "new_skill_learned": gain["difference"] >= CRITERIA["new_skill_gain"] and gain["p_exact_two_sided"] < CRITERIA["new_skill_p"]
                              and gain["ci"]["lower"] > 0,
        "old_skills_retained": all(v >= -CRITERIA["retention_tolerance"] for v in retention.values()),
        "evolution_beats_plain_training": vs_control["difference"] > 0 and vs_control["p_exact_two_sided"] < CRITERIA["beats_control_p"],
    }
    result = {
        "config": asdict(cfg), "criteria": CRITERIA, "data_sha256": hashes, "best": best.id, "lineage": archive.lineage(best.id),
        "test_accuracy": acc, "retention_vs_best_founder": retention, "comparisons": comparisons, "checks": checks,
        "verdict": ("PROVEN: learned the new skill, kept the old ones, and beat plain training" if all(checks.values()) else
                    "learned and retained, but NOT shown to beat plain training" if checks["new_skill_learned"] and checks["old_skills_retained"] else
                    "NOT PROVEN: " + ", ".join(k for k, v in checks.items() if not v)),
        "training_steps": steps_spent, "wall_seconds": time.time() - started,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.rmtree(out / "tmp", ignore_errors=True)
    log(json.dumps({"test_accuracy": acc, "checks": checks, "verdict": result["verdict"]}, indent=2))
    return result
