"""The generation loop: founders learn skills, survivors are crossed, the children learn a NEW verifiable skill, selection keeps the best,
and the survivors become the next parents. A plain-training control with the same number of training steps is the yardstick.

Selection uses the dev split only. The final comparison uses a test split that no training pair, no selection step and no replay ever saw.
Success is decided by criteria fixed in ``CRITERIA`` before any run.

Crossover modes (``EvolutionConfig.crossover``):

* ``blend`` - the original: ``w * A + (1 - w) * B`` (a convex average dilutes what each parent learned);
* ``graft`` - ``A + (B - init_B)``: A plus only what B learned in its own training step, so shared ancestry is counted once and learning accumulates;
* ``none``  - no crossover: each child continues training a survivor (lineage-only), the plain-training baseline *inside* the loop.
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
from .combine import combine_adapters, graft_parts
from .learning import compress_adapter, file_sha256, train_adapter

CRITERIA = {
    "new_skill_gain": 0.05,       # evolved best beats the best founder on the new family by at least this (test accuracy)
    "new_skill_p": 0.01,          # exact McNemar p-value for that comparison
    "retention_tolerance": 0.05,  # on every founder family the evolved best may lose at most this against the best founder of that family
    "beats_control_p": 0.05,      # evolution "contributes" only if it beats the plain-training control on the new family with this p
}

CROSSOVERS = ("blend", "graft", "none")
_SAME_FOR_REUSE = ("seed", "n_train", "n_dev", "n_test", "founders", "new_family", "founder_steps", "rank", "lr", "batch", "accum",
                   "adapter_scale", "dtype", "max_new_tokens")
_SAME_FOR_CONTROL = _SAME_FOR_REUSE + ("child_steps", "generations", "children", "replay_fraction")


@dataclass
class EvolutionConfig:
    base_model: str
    device: str = "cpu"
    dtype: str = "bfloat16"
    founders: list[str] = field(default_factory=lambda: ["add", "mul"])
    new_family: str = "chain"
    n_train: int = 2000
    n_dev: int = 300
    n_test: int = 1000
    founder_steps: int = 150
    child_steps: int = 200
    generations: int = 3
    children: int = 2
    survivors: int = 2
    crossover: str = "graft"
    cross_weights: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])   # blend mode only
    replay_fraction: float = 0.5
    new_skill_weight: float = 2.0     # fitness = weighted mean of the dev accuracies; the new skill counts this many times
    adapter_scale: float = 2.0        # alpha / r of every stored adapter; 2 = fresh adapters (1 reproduces the first run)
    rank: int = 16
    lr: float = 2e-4
    batch: int = 4
    accum: int = 2
    max_new_tokens: int = 12
    seed: int = 1
    founders_from: str | None = None  # a finished run whose founders are reused (ablations share them)
    control_from: str | None = None   # a finished run whose plain-training control is reused


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
    if cfg.crossover not in CROSSOVERS:
        raise ValueError(f"crossover must be one of {CROSSOVERS}")
    if cfg.adapter_scale <= 0 or cfg.new_skill_weight <= 0:
        raise ValueError("adapter_scale and new_skill_weight must be positive")
    return cfg


# -- selection helpers (module level so one definition serves every use and the tests) ---------------------------------
def fitness(dev: dict, cfg: EvolutionConfig) -> float:
    """Weighted mean of the dev accuracies; the new skill has weight ``new_skill_weight``, every founder skill 1."""
    weights = {f: 1.0 for f in cfg.founders}
    weights[cfg.new_family] = cfg.new_skill_weight
    total = sum(weights.values())
    return sum(w * dev.get(f, 0.0) for f, w in weights.items()) / total


def rank_key(o: Organism, cfg: EvolutionConfig) -> tuple:
    """The single ranking used for survivor selection AND for the final pick (best first, ties broken by id)."""
    return (-fitness(o.dev, cfg), o.id)


def forgotten(dev: dict, anchor: dict, founders: list[str], tolerance: float) -> list[str]:
    """Founder skills on which ``dev`` fell more than ``tolerance`` below the founders' own best (fixed at generation 0, so losses cannot ratchet)."""
    return [f for f in founders if dev.get(f, 0.0) < anchor[f] - tolerance]


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
        unique: dict[str, str] = {}   # adapter directory -> first name that uses it (the best organism may itself be a founder)
        duplicates: dict[str, str] = {}
        for n, p in adapters.items():
            if p is None:
                continue
            key = str(Path(p).resolve())
            if key in unique:
                duplicates[n] = unique[key]
            else:
                unique[key] = n
        alias = {f"m{i}": n for i, n in enumerate(unique.values())}  # parent names must be folder-safe
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
        for n, first in duplicates.items():
            acc[n], items[n] = acc[first], items[first]
        return acc, items


def _check_same(cfg: EvolutionConfig, other: dict, keys: tuple[str, ...], what: str) -> None:
    mine = asdict(cfg)
    diff = [k for k in keys if mine[k] != other["config"].get(k)]
    if diff:
        raise ValueError(f"cannot reuse {what}: the other run differs in {diff}")


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

    # -- generation 0: founders (trained here, or reused from a finished run so ablation arms share them) ------------
    founders: list[Organism] = []
    reuse = None
    if cfg.founders_from:
        src = Path(cfg.founders_from)
        reuse = json.loads((src / "result.json").read_text(encoding="utf-8"))
        _check_same(cfg, reuse, _SAME_FOR_REUSE, "the founders")
        old = Archive(src / "archive")
    for i, fam in enumerate(cfg.founders):
        oid = f"g0-{fam}"
        if reuse:
            shutil.copytree(old.path(oid), adapter_dir(oid))
            info = {**old.organisms[oid].training, "reused_from": str(src)}
        else:
            info = train_adapter(cfg.base_model, sft({fam: 1.0}, cfg.founder_steps * pairs_per_step, cfg.seed * 100 + i), adapter_dir(oid),
                                 steps=cfg.founder_steps, lr=cfg.lr, rank=cfg.rank, seed=cfg.seed + i, device=cfg.device, dtype=cfg.dtype,
                                 batch=cfg.batch, accum=cfg.accum)
        founders.append(register(Organism(id=oid, generation=0, op="founder", adapter=f"adapters/{oid}", training=info, status="survivor")))
        log(f"founder {oid}: {'reused' if reuse else 'trained'} {cfg.founder_steps} steps, loss {info.get('final_loss', float('nan')):.3f}")
    evaluate_dev(founders)
    for o in founders:
        log(f"  {o.id} dev {o.dev}")
    anchor = {f: max(o.dev.get(f, 0.0) for o in founders) for f in cfg.founders}  # the forgetting gate is anchored here for good
    survivors = [o.id for o in founders]

    # -- generations ----------------------------------------------------------------------------------------------
    for g in range(1, cfg.generations + 1):
        ranked_now = sorted((archive.organisms[i] for i in survivors), key=lambda o: rank_key(o, cfg))
        ids = [o.id for o in ranked_now]
        if cfg.crossover == "none":
            plans = [((ids[k % len(ids)],), None) for k in range(cfg.children)]
        elif cfg.crossover == "graft":
            plans = [((a, b), None) for a, b in itertools.permutations(ids, 2)]  # best-first: (best, second) is tried before the reverse
            plans = plans[:cfg.children]
        else:
            combos = [((a, b), w) for a, b in itertools.combinations(ids, 2) for w in cfg.cross_weights]
            rng.shuffle(combos)
            plans = combos[:cfg.children]
        children: list[Organism] = []
        for k, (parents, w) in enumerate(plans):
            oid = f"g{g}-c{k}"
            start = adapter_dir(f"{oid}-init")  # kept: a later graft needs exactly what this organism started from
            trained = out / "tmp" / f"{oid}-trained"
            shutil.rmtree(trained, ignore_errors=True)
            if cfg.crossover == "none":
                combine_adapters([(archive.path(parents[0]), 1.0)], start, out_scale=cfg.adapter_scale)
                weights = [1.0]
            elif cfg.crossover == "graft":
                secondary_init = adapter_dir(f"{parents[1]}-init")
                combine_adapters(graft_parts(archive.path(parents[0]), archive.path(parents[1]),
                                             secondary_init if secondary_init.is_dir() else None), start, out_scale=cfg.adapter_scale)
                weights = [1.0, 1.0]
            else:
                combine_adapters([(archive.path(parents[0]), w), (archive.path(parents[1]), 1 - w)], start, out_scale=cfg.adapter_scale)
                weights = [w, 1 - w]
            info = train_adapter(cfg.base_model, sft(learn_mix(), cfg.child_steps * pairs_per_step, cfg.seed * 1000 + g * 10 + k), trained,
                                 init=start, steps=cfg.child_steps, lr=cfg.lr, seed=cfg.seed + g * 10 + k, device=cfg.device, dtype=cfg.dtype,
                                 batch=cfg.batch, accum=cfg.accum)
            info["compression"] = compress_adapter(trained, adapter_dir(oid), cfg.rank, cfg.adapter_scale)
            info["start_adapter"] = f"adapters/{oid}-init"
            shutil.rmtree(trained, ignore_errors=True)
            steps_spent["evolution"] += cfg.child_steps
            children.append(register(Organism(id=oid, generation=g, op="cross+learn", adapter=f"adapters/{oid}", parents=list(parents),
                                              weights=weights, training=info)))
        evaluate_dev(children)
        pool = [archive.organisms[i] for i in survivors]
        for c in children:
            lost = forgotten(c.dev, anchor, cfg.founders, CRITERIA["retention_tolerance"])
            c.status = "candidate" if not lost else f"rejected:forgot {','.join(lost)}"
            archive.record(c)
            log(f"  {c.id} <- {c.parents} w={c.weights}  dev {c.dev}  fitness {fitness(c.dev, cfg):.3f}  {c.status}  "
                f"(energy kept {c.training['compression']['energy_kept']:.3f})")
        admitted = [c for c in children if c.status == "candidate"]
        ranked = sorted(pool + admitted, key=lambda o: rank_key(o, cfg))
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
        log(f"generation {g}: survivors {survivors}; best {ranked[0].id} fitness {fitness(ranked[0].dev, cfg):.3f}")

    best = min((archive.organisms[i] for i in survivors), key=lambda o: rank_key(o, cfg))  # the same key as the selection above
    best.status = "final"
    archive.record(best)

    # -- controls (reusable: one control serves every crossover arm of an ablation) ----------------------------------
    merge_only, control = adapter_dir("control-merge-only"), adapter_dir("control-plain")
    total_steps = steps_spent["evolution"]
    m05 = out / "tmp" / "m05"
    shutil.rmtree(m05, ignore_errors=True)
    combine_adapters([(archive.path(founders[0]), 0.5), (archive.path(founders[1]), 0.5)], m05, out_scale=cfg.adapter_scale)
    if cfg.control_from:
        src = Path(cfg.control_from)
        other = json.loads((src / "result.json").read_text(encoding="utf-8"))
        _check_same(cfg, other, _SAME_FOR_CONTROL, "the control")
        if other["training_steps"]["control"] != total_steps:
            raise ValueError("cannot reuse the control: it trained a different number of steps")
        shutil.copytree(src / "archive" / "adapters" / "control-merge-only", merge_only)
        shutil.copytree(src / "archive" / "adapters" / "control-plain", control)
        log(f"control: reused from {src}")
    else:
        compress_adapter(m05, merge_only, cfg.rank, cfg.adapter_scale)
        control_tmp = out / "tmp" / "control-trained"
        shutil.rmtree(control_tmp, ignore_errors=True)
        cinfo = train_adapter(cfg.base_model, sft(learn_mix(), total_steps * pairs_per_step, cfg.seed * 7777), control_tmp, init=m05, steps=total_steps,
                              lr=cfg.lr, seed=cfg.seed + 999, device=cfg.device, dtype=cfg.dtype, batch=cfg.batch, accum=cfg.accum)
        compress_adapter(control_tmp, control, cfg.rank, cfg.adapter_scale)
        log(f"control: plain training {total_steps} steps from the 0.5 merge of the founders, loss {cinfo['final_loss']:.3f}")
    steps_spent["control"] = total_steps

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
    new_best_founder = max(founder_ids, key=lambda i: acc[i][cfg.new_family])
    gain = comparisons[f"{cfg.new_family}: evolved vs best founder ({new_best_founder})"]
    vs_control = comparisons[f"{cfg.new_family}: evolved vs control-plain"]
    retention = {f: acc[evolved][f] - max(acc[i][f] for i in founder_ids) for f in cfg.founders}
    checks = {
        "new_skill_learned": gain["difference"] >= CRITERIA["new_skill_gain"] and gain["p_exact_two_sided"] < CRITERIA["new_skill_p"]
                              and gain["ci"]["lower"] > 0,
        "old_skills_retained": all(v >= -CRITERIA["retention_tolerance"] for v in retention.values()),
        "evolution_beats_plain_training": vs_control["difference"] > 0 and vs_control["p_exact_two_sided"] < CRITERIA["beats_control_p"],
    }
    lineage = archive.lineage(best.id)
    lineage_children = [i for i in lineage if archive.organisms[i].op == "cross+learn"]
    kept_steps = sum(archive.organisms[i].training.get("steps", 0) for i in lineage_children)
    result = {
        "config": asdict(cfg), "criteria": CRITERIA, "data_sha256": hashes, "best": best.id, "lineage": lineage, "anchor_dev": anchor,
        "test_accuracy": acc, "retention_vs_best_founder": retention, "comparisons": comparisons, "checks": checks,
        "verdict": ("PROVEN: learned the new skill, kept the old ones, and beat plain training" if all(checks.values()) else
                    "learned and retained, but NOT shown to beat plain training" if checks["new_skill_learned"] and checks["old_skills_retained"] else
                    "NOT PROVEN: " + ", ".join(k for k, v in checks.items() if not v)),
        "training_steps": steps_spent,
        "steps_in_final_lineage": kept_steps, "steps_discarded": steps_spent["evolution"] - kept_steps,
        "wall_seconds": time.time() - started,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "test_items.json").write_text(json.dumps({"test_window": test_window, "items": items}), encoding="utf-8")
    shutil.rmtree(out / "tmp", ignore_errors=True)
    log(json.dumps({"test_accuracy": acc, "checks": checks, "verdict": result["verdict"],
                    "steps_in_final_lineage": kept_steps, "steps_discarded": result["steps_discarded"]}, indent=2))
    return result
