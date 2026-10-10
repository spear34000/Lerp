"""Generative tasks for the resident evaluator: greedy decoding of a fixed window of prompts, answer extraction, exact match.

The prompts of a window are tokenized once. Each candidate generates from the blended weights already sitting in the model, the
completion is cut at the first stop string, the answer is pulled out with the task's ``extract`` regex and compared with the
gold answer after the task's ``normalize`` steps.
"""
from __future__ import annotations

from typing import Any, Sequence

from .spec import Spec
from .tasks import TaskDefinition, resolve_task


class GenerativeScorer:
    def __init__(self, spec: Spec, tokenizer, lo: int, hi: int, *, add_special_tokens: bool = False,
                 max_length: int = 4096, rows: dict[str, Sequence[Any]] | None = None):
        self.tokenizer = tokenizer
        self.tasks: dict[str, TaskDefinition] = {}
        self.items: list[tuple[str, int, list[int], str]] = []  # (task, doc, prompt ids, gold)
        self.counts: dict[str, int] = {}
        for task in spec.evaluation.tasks:
            definition = resolve_task(task.name, task.definition)
            if not definition.generative:
                continue
            docs = definition.documents(lo, hi, rows=None if rows is None else rows[task.name])
            if not docs:
                raise ValueError(f"task {task.name}: no documents in items [{lo}, {hi})")
            self.tasks[task.name] = definition
            self.counts[task.name] = len(docs)
            room = max_length - definition.generate.get("max_new_tokens", 256)
            for d, (prompt, gold) in enumerate(docs):
                ids = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
                self.items.append((task.name, d, ids[-room:] if room > 0 else ids[-1:], gold))
        self.order = sorted(range(len(self.items)), key=lambda i: (self.items[i][0], len(self.items[i][2])))

    def __bool__(self) -> bool:
        return bool(self.items)

    def batches(self, size: int):
        """Batches of similar length that never mix tasks (each task has its own generation limits)."""
        batch: list[int] = []
        for i in self.order:
            if batch and (len(batch) >= size or self.items[batch[0]][0] != self.items[i][0]):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def pad_id(self) -> int:
        tok = self.tokenizer
        for value in (tok.pad_token_id, tok.eos_token_id):
            if value is not None:
                return int(value)
        return 0

    def accuracy(self, texts: dict[int, str]) -> dict[str, float]:
        outcomes: dict[str, dict[int, int]] = {name: {} for name in self.tasks}
        for i, text in texts.items():
            name, doc, _, gold = self.items[i]
            outcomes[name][doc] = int(self.tasks[name].extract_answer(text) == gold)
        self.outcomes = {name: [outcomes[name][d] for d in sorted(outcomes[name])] for name in self.tasks}
        return {name: sum(v) / self.counts[name] for name, v in self.outcomes.items()}
