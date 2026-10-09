"""Cheap exact/normalized text overlap gate for development vs held-out sets.

No semantic near-duplicate guarantee: textual paraphrases and contamination of
training corpora remain out of scope. Does not emit confidential sample text.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path


class LeakageError(ValueError):
    pass


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text).casefold()
    normalized = re.sub(r'[^\w\s]', ' ', normalized, flags=re.UNICODE)
    return ' '.join(normalized.split())


def _index(path: Path, text_field: str, id_field: str) -> tuple[set[str], dict[str, str], int]:
    ids: set[str] = set()
    hashed: dict[str, str] = {}
    records = 0
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise LeakageError(f'{path}:{line_no}: invalid JSONL') from exc
            if not isinstance(record, dict) or text_field not in record or id_field not in record:
                raise LeakageError(f'{path}:{line_no}: missing {text_field!r}/{id_field!r}')
            if not isinstance(record[text_field], str) or not record[text_field].strip():
                raise LeakageError(f'{path}:{line_no}: nonempty text required')
            if not isinstance(record[id_field], (int, str)) or isinstance(record[id_field], bool):
                raise LeakageError(f'{path}:{line_no}: id must be a string/integer')
            sid = str(record[id_field])
            if sid in ids:
                raise LeakageError(f'{path}:{line_no}: duplicate ID')
            ids.add(sid)
            digest = hashlib.sha256(_normalize(record[text_field]).encode('utf-8')).hexdigest()
            hashed[digest] = sid
            records += 1
    return ids, hashed, records


def audit_split_overlap(development: Path, holdout: Path, *,
                        text_field: str = 'text', id_field: str = 'id') -> dict:
    a_ids, a_hashes, a_count = _index(development, text_field, id_field)
    b_ids, b_hashes, b_count = _index(holdout, text_field, id_field)
    matching_hashes = sorted(a_hashes.keys() & b_hashes.keys())
    repeated_ids = sorted(a_ids & b_ids)
    return {
        'status': 'FAIL_OVERLAP' if matching_hashes or repeated_ids else 'PASS_ONLY_EXACT_NORMALIZED_CHECK',
        'development_count': a_count, 'holdout_count': b_count,
        'normalized_duplicate_texts': len(matching_hashes),
        'repeated_ids': len(repeated_ids),
        'duplicate_hash_examples': matching_hashes[:10],
        'repeated_id_examples': repeated_ids[:10],
        'limitation': 'Does not detect paraphrases, content from model pretraining, or hidden data leakage.',
    }
