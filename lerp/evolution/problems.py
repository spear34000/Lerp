"""Verifiable arithmetic problem families, with exact answers and disjoint splits.

Every question has one correct integer answer, so a model's output is checked by comparison with a number computed here, never by another
model. Splits are built from one deterministic pool of unique questions, so train, dev and test never share a question.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

PROMPT = "Q: {q}\nA:"


def _add(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(100, 9999), rng.randint(100, 9999)
    return (f"{a} + {b} =", a + b) if rng.random() < .5 else (f"{a} - {b} =", a - b)


def _mul(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(12, 999), rng.randint(2, 99)
    return f"{a} * {b} =", a * b


def _chain(rng: random.Random) -> tuple[str, int]:
    """Two operations with precedence: multiplication first."""
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 99)
    return (f"{a} + {b} * {c} =", a + b * c) if rng.random() < .5 else (f"{a} * {b} - {c} =", a * b - c)


FAMILIES = {"add": _add, "mul": _mul, "chain": _chain}


def pool(family: str, count: int, seed: int) -> list[dict]:
    """``count`` unique questions of a family, in a fixed order for a given seed."""
    if family not in FAMILIES:
        raise ValueError(f"unknown problem family {family!r}; choose from {sorted(FAMILIES)}")
    rng = random.Random(f"{family}:{seed}")
    seen: set[str] = set()
    rows: list[dict] = []
    guard = 0
    while len(rows) < count:
        question, answer = FAMILIES[family](rng)
        guard += 1
        if guard > count * 50:
            raise ValueError(f"cannot draw {count} unique {family} questions")
        if question in seen:
            continue
        seen.add(question)
        rows.append({"q": question, "a": str(answer), "family": family})
    return rows


def splits(family: str, n_train: int, n_dev: int, n_test: int, seed: int) -> dict[str, list[dict]]:
    rows = pool(family, n_train + n_dev + n_test, seed)
    return {"train": rows[:n_train], "dev": rows[n_train:n_train + n_dev], "test": rows[n_train + n_dev:]}


def stream(rows: dict[str, list[dict]], mix: dict[str, float], count: int, seed: int) -> list[tuple[str, str]]:
    """``count`` training pairs with the family shares in ``mix`` EXACTLY (rounded per family), however large ``count`` is.

    Each family is drawn by walking through shuffled passes over its rows (a new shuffle per pass), so a long run repeats rows instead of silently
    changing the mix the way a cap at the pool size does; the result is shuffled once and then consumed in order."""
    local = random.Random(seed)
    picked: list[tuple[str, str]] = []
    for family, share in mix.items():
        need = round(count * share)
        pool_rows = list(rows[family])
        drawn: list[dict] = []
        while len(drawn) < need:
            local.shuffle(pool_rows)
            drawn.extend(pool_rows[:need - len(drawn)])
        picked += [sft_pair(r) for r in drawn]
    local.shuffle(picked)
    return picked


def verify(row: dict, text: str) -> bool:
    """Exact check of a model completion against the computed answer: the first integer in the text."""
    import re
    found = re.search(r"-?\d[\d,]*", text)
    return bool(found) and found.group(0).replace(",", "") == row["a"]


def sft_pair(row: dict) -> tuple[str, str]:
    return PROMPT.format(q=row["q"]), f" {row['a']}\n"


def write_jsonl(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_bytes(text.encode("utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_definition(path: Path, max_new_tokens: int = 12) -> dict:
    """Declarative generative task (see lerp.tasks) over a JSONL file of {q, a} rows."""
    return {"dataset": f"file:{path.as_posix()}", "split": "all", "prompt": PROMPT,
            "answer": {"field": "a"},
            "generate": {"max_new_tokens": max_new_tokens, "stop": ["\n", "Q:"]},
            "extract": {"regex": "(-?[0-9][0-9,]*)", "pick": "first"},
            "normalize": ["remove_commas", "number"]}
