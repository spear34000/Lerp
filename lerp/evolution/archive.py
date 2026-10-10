"""Organisms, lineage and niches. Append-only: nothing recorded about an organism is rewritten, only superseded by a later line."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Organism:
    id: str
    generation: int
    op: str                      # founder | cross | cross+learn | learn
    adapter: str                 # path, relative to the archive directory
    parents: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)   # cross: weight of each parent
    training: dict = field(default_factory=dict)         # steps, seed, data hash, final loss, compression
    adapter_sha256: str = ""
    dev: dict = field(default_factory=dict)              # accuracy per problem family on the dev split (used for selection)
    status: str = "candidate"    # candidate | survivor | rejected:<reason> | final
    created: float = field(default_factory=time.time)


class Archive:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / "organisms.jsonl"
        self.organisms: dict[str, Organism] = {}
        if self.log.is_file():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    o = Organism(**json.loads(line))
                    self.organisms[o.id] = o

    def record(self, organism: Organism) -> Organism:
        self.organisms[organism.id] = organism
        with self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(organism), sort_keys=True) + "\n")
        return organism

    def path(self, organism: Organism | str) -> Path:
        o = self.organisms[organism] if isinstance(organism, str) else organism
        return self.root / o.adapter

    def lineage(self, organism_id: str) -> list[str]:
        """Ancestors, oldest first, ending with the organism."""
        seen: set[str] = set()
        order: list[str] = []

        def visit(i: str) -> None:
            if i in seen or i not in self.organisms:
                return
            seen.add(i)
            for p in self.organisms[i].parents:
                visit(p)
            order.append(i)
        visit(organism_id)
        return order

    def niches(self, ids: list[str], families: list[str]) -> dict[str, str]:
        """Best organism per family among ``ids`` (ties broken by id)."""
        best: dict[str, str] = {}
        for fam in families:
            scored = [(self.organisms[i].dev.get(fam, -1.0), i) for i in ids]
            if scored:
                best[fam] = sorted(scored, key=lambda x: (-x[0], x[1]))[0][1]
        return best
