"""`lerp search`: the cycle loop with the resident evaluator (seconds per candidate) instead of one harness run per candidate.

Candidates, generations, scores, baselines and the leaderboard live in the normal run folder, so `advance`, `board`, `compare`,
`build`, `validate` and `export` work unchanged. Searching only scores candidates; build the winner afterwards with
`lerp build -g G -i I` and confirm it with `lerp validate`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .experiment import (BreederError, advance, load_candidate, load_run, record_baseline, record_score)
from .integrity import verify_frozen_inputs

PROTOCOL_FILE = "resident_protocol.json"
_COMPARED = ("engine", "protocol_version", "mode", "items", "dtype", "device", "tasks", "softcap")


def _check_protocol(run: Path, protocol: dict) -> None:
    path = run / PROTOCOL_FILE
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    changed = [k for k in _COMPARED if existing.get(k) != protocol.get(k)]
    if changed:
        raise BreederError(f"The scoring protocol changed since this run started ({', '.join(changed)}); scores would not be comparable. "
                           f"Restore the settings or start a new run (recorded in {path.name})")


def search_run(run: Path, rounds: int = 1, *, device: str | None = None, dtype: str = "bfloat16", baselines: bool = False,
               session=None, log: Callable[[str], None] = print) -> list[int]:
    """Score every unscored candidate of the current generation, advance, repeat. Returns the generations completed."""
    from .resident import ResidentError, ResidentSession
    if not 1 <= rounds <= 100:
        raise BreederError("rounds must be between 1 and 100")
    spec, state = load_run(run)
    verify_frozen_inputs(run, spec)
    if spec.evaluation.limit is None:
        raise BreederError("search needs evaluation.limit (items per task); a whole benchmark split per candidate is too slow for a search")
    window = (0, spec.evaluation.limit)
    own = session is None
    try:
        if own:
            session = ResidentSession(spec, device or spec.evaluation.device, dtype)
    except ResidentError as exc:
        raise BreederError(str(exc)) from exc
    try:
        protocol = session.protocol(window)
        _check_protocol(run, protocol)
        backend = f"lerp-resident/{session.dtype}"
        if baselines:
            for name in ["base"] + [p.name for p in spec.parents]:
                if (run / "baselines" / name / "score.json").is_file():
                    continue
                t0 = time.time()
                metrics = session.evaluate_reference(name, window)
                doc = record_baseline(run, name, metrics, source="lerp_eval", evidence=PROTOCOL_FILE,
                                      runtime_device=str(session.device), backend=backend)
                log(f"baseline {name}: {doc['fitness']:.4f} {metrics} ({time.time() - t0:.1f}s)")
        completed: list[int] = []
        for r in range(rounds):
            spec, state = load_run(run)
            verify_frozen_inputs(run, spec)
            gen = state["generation"]
            for idx in range(spec.population):
                item = load_candidate(run, gen, idx)
                if "score" in item:
                    if item["score"]["source"] == "SIMULATED_TOY":
                        raise BreederError("Cannot search a run whose candidates were scored with synthetic data")
                    continue
                t0 = time.time()
                try:
                    metrics = session.evaluate(item["genes"], window, method=item.get("method"))
                except ResidentError as exc:
                    raise BreederError(str(exc)) from exc
                doc = record_score(run, gen, idx, metrics, source="lerp_eval", evidence=PROTOCOL_FILE,
                                   runtime_device=str(session.device), backend=backend)
                log(f"g{gen:03d}-c{idx:03d}  fitness {doc['fitness']:.4f}  {metrics}  ({time.time() - t0:.1f}s)")
            completed.append(gen)
            if r < rounds - 1:
                advance(run)
        return completed
    finally:
        if own:
            session.close()
