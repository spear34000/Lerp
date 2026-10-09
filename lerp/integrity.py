"""Content-addressed, opt-in input freezing for auditable experiments.

Local model artifacts are streamed through SHA-256. Remote references and
missing base weights are never silently called verified. This is intentionally
an I/O-heavy integrity check, not an authenticity or license certification.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from .compat import local_reference
from .spec import Spec


class IntegrityError(RuntimeError):
    pass


def _sha_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(root: Path, role: str, mode: str) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise IntegrityError(f"Not a safe local model directory: {root}")
    if role != "base" and mode == "lora":
        required = [root / "adapter_config.json", root / "adapter_model.safetensors"]
        if any(not path.is_file() for path in required):
            raise IntegrityError(f"Incomplete local PEFT adapter: {root}")
        return required
    # No arbitrary nested files, which may contain unrelated checkpoints or caches.
    names = {
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "generation_config.json", "vocab.json", "merges.txt", "added_tokens.json",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    }
    files = {path for path in root.iterdir() if path.is_file() and (
        path.name in names or path.name.endswith(".safetensors") or
        path.name.startswith("pytorch_model") and path.name.endswith(".bin") or
        path.name.endswith(".model")
    )}
    if not (root / "config.json").is_file():
        raise IntegrityError(f"Local base/checkpoint config.json missing: {root}")
    return sorted(files)


def snapshot_inputs(spec: Spec, *, strict: bool = False) -> dict:
    sources = [("base", spec.base_model)] + [(p.name, p.model) for p in spec.parents]
    records = []
    for name, reference in sources:
        if not local_reference(reference):
            if strict:
                raise IntegrityError(f"Strict input freeze requires a local, downloaded revision: {name}={reference}")
            records.append({"name": name, "reference": reference, "verified": False,
                            "reason": "Remote reference not content-hashed"})
            continue
        root = Path(reference)
        paths = _artifact_paths(root, name, spec.mode)
        has_weights = any(p.name.endswith((".safetensors", ".bin")) for p in paths)
        if strict and not has_weights:
            raise IntegrityError(f"Strict mode requires local weights for {name}: {root}")
        file_records = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise IntegrityError(f"Refusing symlink or nonregular input: {path}")
            stat_before = path.stat()
            if not os.path.isfile(path):
                raise IntegrityError(f"Not a regular input: {path}")
            digest = _sha_file(path)
            stat_after = path.stat()
            if ((stat_before.st_size, stat_before.st_mtime_ns, stat_before.st_ino) !=
                    (stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ino)):
                raise IntegrityError(f"Input changed during hashing: {path}")
            file_records.append({"file": path.name, "bytes": stat_after.st_size, "sha256": digest})
        records.append({"name": name, "reference": str(root.resolve()),
                        "verified": bool(has_weights), "files": file_records,
                        **({"reason": "Checkpoint weights missing (metadata only)"} if not has_weights else {})})
    return {"format_version": 1, "strict": bool(strict), "sources": records}


def freeze_inputs(run: Path, spec: Spec, *, strict: bool = False) -> dict:
    path = run / "input_fingerprints.json"
    if path.exists():
        raise IntegrityError("Inputs already frozen; never silently overwrite pinned hashes")
    if any((run / "generations").glob("gen-*/cand-*/score.json")) or any(
        (run / "baselines").glob("*/score.json")
    ):
        raise IntegrityError("Freeze inputs BEFORE recording scores; start a clean run if already scored")
    config_path = run / "experiment.yaml"
    if not config_path.is_file():
        raise IntegrityError("Run has no experiment.yaml")
    record = snapshot_inputs(spec, strict=strict)
    record["experiment_config_sha256"] = _sha_file(config_path)
    record["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    # O_EXCL: a second concurrent freezer cannot silently overwrite.
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise IntegrityError("Another process froze this experiment") from exc
    return record


def verify_frozen_inputs(run: Path, spec: Spec, *, require: bool = False) -> dict:
    path = run / "input_fingerprints.json"
    if not path.is_file():
        if require:
            raise IntegrityError("No frozen inputs; run 'lerp freeze --run ...' first")
        return {"frozen": False, "passed": False, "warning": "No input hashes pinned"}
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("format_version") != 1:
        raise IntegrityError("Unsupported input fingerprint format")
    config_hash = _sha_file(run / "experiment.yaml")
    if record.get("experiment_config_sha256") != config_hash:
        raise IntegrityError("Experiment config changed since inputs were frozen")
    current = snapshot_inputs(spec, strict=record.get("strict", False))
    if record.get("sources") != current["sources"]:
        raise IntegrityError("Model input files, weights, metadata, or source reference changed since freeze")
    return {"frozen": True, "passed": True, "strict": record["strict"],
            "fully_verified": all(source["verified"] for source in current["sources"]),
            "sources": len(current["sources"]),
            "sha256": _sha_file(path)}
