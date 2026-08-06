"""Record types carried through the pipeline, with full provenance."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


def _uid(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Provenance:
    source: str = ""
    node_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)
    section_path: list[str] = field(default_factory=list)
    breadcrumb: str = ""
    generator: str = ""
    generated_at: float = field(default_factory=time.time)
    verification: list[dict] = field(default_factory=list)

    def merge(self, other: "Provenance") -> "Provenance":
        return Provenance(
            source=self.source or other.source,
            node_ids=list(dict.fromkeys(self.node_ids + other.node_ids)),
            pages=sorted(set(self.pages + other.pages)),
            bboxes=self.bboxes + other.bboxes,
            section_path=self.section_path or other.section_path,
            breadcrumb=" || ".join(x for x in {self.breadcrumb: 1, other.breadcrumb: 1} if x),
            generator=self.generator or other.generator,
        )


@dataclass
class Chunk:
    text: str
    kind: str = "section"
    prov: Provenance = field(default_factory=Provenance)
    tables: list[str] = field(default_factory=list)
    grids: list[list[list[str]]] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    figure_refs: list[dict] = field(default_factory=list)
    resolved_refs: str = ""
    tokens: int = 0
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = _uid(self.prov.source, self.text[:512])
        if not self.tokens:
            self.tokens = estimate_tokens(self.text)

    def context(self) -> str:
        head = f"[{self.prov.breadcrumb}]" if self.prov.breadcrumb else ""
        parts = [head, self.text]
        if self.tables:
            parts.append("Tables:\n" + "\n\n".join(self.tables))
        if self.equations:
            parts.append("Equations:\n" + "\n".join(self.equations))
        if self.resolved_refs:
            parts.append("Resolved references:\n" + self.resolved_refs)
        if self.figure_refs:
            parts.append("Figures:\n" + "\n".join(f"[{f['id']}] {f.get('caption', 'unlabelled figure')}" for f in self.figure_refs))
        return "\n\n".join(p for p in parts if p)

    def images(self) -> list[dict]:
        return [f for f in self.figure_refs if f.get("image_path")]


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class QARecord:
    question: str
    answer: str
    context: str = ""
    task: str = "qa"
    persona: str = "neutral"
    difficulty: str = "intermediate"
    hops: int = 1
    turns: list[Turn] = field(default_factory=list)
    rejected: str = ""
    rejection_mode: str = ""
    tool_trace: list[dict] = field(default_factory=list)
    tool_env: dict = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    prov: Provenance = field(default_factory=Provenance)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = _uid(self.prov.source, self.question, self.answer[:256])

    @property
    def accepted(self) -> bool:
        return not any(f.startswith("reject:") for f in self.flags)

    def messages(self) -> list[dict]:
        if self.turns:
            return [{"role": t.role, "content": t.content} for t in self.turns]
        return [{"role": "user", "content": self.question}, {"role": "assistant", "content": self.answer}]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d

    @staticmethod
    def from_dict(d: dict) -> "QARecord":
        d = dict(d)
        d["turns"] = [Turn(**t) for t in d.get("turns", [])]
        d["prov"] = Provenance(**d.get("prov", {}))
        return QARecord(**d)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
