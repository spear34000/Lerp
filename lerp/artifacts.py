"""Detect accidental output checkpoint mutation between build and evaluation.

Not a signature or provenance attestation. A malicious party can edit both
model weights and this manifest, so still use trusted storage and access control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ArtifactError(RuntimeError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> list[dict]:
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError(f'Invalid checkpoint directory: {root}')
    result = []
    for path in sorted(root.iterdir()):
        if path.name == 'modelbreeder_artifact_integrity.json':
            continue
        if path.is_symlink():
            raise ArtifactError(f'Refusing symlink in artifact directory: {path}')
        if path.is_dir():
            raise ArtifactError(f'Unexpected nested directory in checkpoint: {path}')
        if not path.is_file():
            raise ArtifactError(f'Not a regular artifact: {path}')
        result.append({'file': path.name, 'size': path.stat().st_size, 'sha256': _hash(path)})
    if not any(f['file'].endswith(('.safetensors', '.bin')) for f in result):
        raise ArtifactError('Checkpoint has no supported weight files')
    return result


def pin_artifact(root: Path) -> dict:
    path = root / 'modelbreeder_artifact_integrity.json'
    if path.exists():
        raise ArtifactError(f'Artifact already pinned: {root}')
    data = {'schema_version': 1, 'files': _files(root)}
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    return data


def verify_artifact(root: Path, *, require: bool = False) -> dict:
    path = root / 'modelbreeder_artifact_integrity.json'
    if not path.is_file():
        if require:
            raise ArtifactError(f'Missing artifact integrity manifest: {path}')
        return {'pinned': False}
    recorded = json.loads(path.read_text(encoding='utf-8'))
    if recorded.get('schema_version') != 1 or recorded.get('files') != _files(root):
        raise ArtifactError(f'Output checkpoint was modified since build: {root}')
    return {'pinned': True, 'files': len(recorded['files']), 'manifest_sha256': _hash(path)}
