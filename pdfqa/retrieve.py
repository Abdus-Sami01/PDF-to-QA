"""Hybrid retrieval over an exported corpus, and grounded answering on top of it.

The corpus this reads is the same chunk set the dataset was generated from, so a retrieval failure
here is a real failure of the chunking — not an artefact of a different splitter. That makes the
index worth evaluating against, and makes `ask` a way to sanity-check a corpus before training on it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .llm import Runtime, hashed_embedding
from .prompts import GROUNDED_ANSWER
from .select import cosine, normalize
from .verify import STOP

WORD = re.compile(r"[A-Za-z0-9_.%-]+")


def terms(text: str) -> list[str]:
    return [t.lower() for t in WORD.findall(text) if t.lower() not in STOP and len(t) > 1]


@dataclass
class Passage:
    id: str
    text: str
    source: str = ""
    breadcrumb: str = ""
    pages: list[int] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def citation(self) -> str:
        where = f"p{self.pages[0]}" if self.pages else self.breadcrumb.split(" > ")[-1]
        return f"{self.source}{(' ' + where) if where else ''}"


@dataclass
class Hit:
    passage: Passage
    score: float
    lexical: float = 0.0
    dense: float = 0.0


class Index:
    """BM25 over terms, fused with cosine over embeddings. Both halves are optional but rarely useless."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        self.vectors: list[list[float]] = []
        for i, p in enumerate(passages):
            counts = Counter(terms(p.text))
            self.lengths.append(sum(counts.values()) or 1)
            for term, n in counts.items():
                self.postings[term].append((i, n))
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        self.n = len(passages)

    def embed(self, runtime: Runtime | None = None) -> None:
        texts = [p.text[:2000] for p in self.passages]
        self.vectors = [normalize(v) for v in (runtime.embed(texts) if runtime else [hashed_embedding(t) for t in texts])]

    def bm25(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, freq in posting:
                norm = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += idf * (freq * (self.k1 + 1)) / norm
        return scores

    def search(self, query: str, k: int = 5, alpha: float = 0.5, runtime: Runtime | None = None, min_score: float = 0.12) -> list[Hit]:
        """min_score matters once embeddings are on: cosine makes every passage a candidate, so
        without a floor an off-topic question still retrieves five confident-looking passages."""
        lexical = self.bm25(query)
        top_lex = max(lexical.values(), default=0.0) or 1.0
        dense: dict[int, float] = {}
        if self.vectors:
            qv = normalize((runtime.embed([query]) if runtime else [hashed_embedding(query)])[0])
            dense = {i: cosine(qv, v) for i, v in enumerate(self.vectors)}

        candidates = set(lexical) | set(dense)
        hits = []
        for i in candidates:
            lex = lexical.get(i, 0.0) / top_lex
            den = max(0.0, dense.get(i, 0.0))
            hits.append(Hit(self.passages[i], alpha * lex + (1 - alpha) * den, lex, den))
        hits.sort(key=lambda h: -h.score)
        return [h for h in hits[:k] if h.score >= min_score]

    @staticmethod
    def load(path: str | Path) -> "Index":
        p = Path(path)
        files = sorted(p.rglob("corpus.jsonl")) if p.is_dir() else [p]
        passages: list[Passage] = []
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                passages.append(
                    Passage(
                        id=row.get("id", ""),
                        text=row.get("text", ""),
                        source=row.get("source", ""),
                        breadcrumb=row.get("breadcrumb", ""),
                        pages=row.get("pages", []),
                        node_ids=row.get("node_ids", []),
                        images=row.get("images", []),
                    )
                )
        if not passages:
            raise ValueError(f"no corpus.jsonl passages found under {path}")
        return Index(passages)

    @staticmethod
    def from_chunks(chunks) -> "Index":
        return Index(
            [
                Passage(
                    id=c.id,
                    text=c.context(),
                    source=c.prov.source,
                    breadcrumb=c.prov.breadcrumb,
                    pages=list(c.prov.pages),
                    node_ids=list(c.prov.node_ids),
                    images=[f["image_path"] for f in c.figure_refs if f.get("image_path")],
                )
                for c in chunks
            ]
        )


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    unanswerable: bool = False

    def citations(self) -> list[str]:
        return list(dict.fromkeys(h.passage.citation() for h in self.hits))


UNANSWERABLE = "UNANSWERABLE"


def answer(runtime: Runtime, index: Index, question: str, k: int = 5, alpha: float = 0.5, min_score: float = 0.12) -> Answer:
    hits = index.search(question, k, alpha, runtime if index.vectors else None, min_score)
    if not hits:
        return Answer("No passage in this corpus is relevant to that question.", [], unanswerable=True)
    passages = "\n\n".join(f"[{i + 1}] ({h.passage.citation()}) {h.passage.text[:2500]}" for i, h in enumerate(hits))
    text = runtime.complete(
        "verify",
        GROUNDED_ANSWER.format(question=question, passages=passages, marker=UNANSWERABLE),
        temperature=0.0,
        max_tokens=900,
    ).strip()
    return Answer(text, hits, unanswerable=is_refusal(text))


def is_refusal(text: str) -> bool:
    """The marker counts only as the whole answer; models quote the instruction back mid-sentence."""
    stripped = text.strip().strip("\"'`*. ").upper()
    return stripped == UNANSWERABLE or stripped.startswith(UNANSWERABLE + " ") or stripped.startswith(UNANSWERABLE + ".")
