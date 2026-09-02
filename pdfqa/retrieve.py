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

from .llm import Runtime, Vector, hashed_embedding
from .prompts import GROUNDED_ANSWER
from .select import EXACT_LIMIT, HyperplaneIndex, cosine, normalize
from .verify import STOP

WORD = re.compile(r"[A-Za-z0-9_.%-]+")

MAX_DOC_FRACTION = 0.5
"""A term in more than this share of passages is treated as a stop word for candidate generation."""

COMMON_TERM_FLOOR = 1000
"""Below this many passages, every term looks common; the corpus is small enough to scan anyway."""


def terms(text: str) -> list[str]:
    return [t.lower() for t in WORD.findall(text) if t.lower() not in STOP and len(t) > 1]


@dataclass
class Passage:
    id: str
    text: str
    source: str = ""
    breadcrumb: str = ""
    pages: list[int] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def citation(self) -> str:
        """Page for paged formats, anchor for markup ones, section name as a last resort."""
        if self.pages:
            where = f"p{self.pages[0]}"
        elif self.anchors:
            where = self.anchors[0]
        else:
            where = self.breadcrumb.split(" > ")[-1]
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
        self.vectors: list[Vector] = []
        for i, p in enumerate(passages):
            counts = Counter(terms(p.text))
            self.lengths.append(sum(counts.values()) or 1)
            for term, n in counts.items():
                self.postings[term].append((i, n))
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        self.n = len(passages)
        self.dense_index: HyperplaneIndex | None = None

    def embed(self, runtime: Runtime | None = None) -> None:
        texts = [p.text[:2000] for p in self.passages]
        self.vectors = [normalize(v) for v in (runtime.embed(texts) if runtime else [hashed_embedding(t) for t in texts])]
        self.dense_index = None
        if len(self.vectors) > EXACT_LIMIT:
            # Scoring every passage against the query is linear per query, which a corpus of a few
            # hundred papers turns into most of a second — and `eval` issues one query per record.
            self.dense_index = HyperplaneIndex(len(self.vectors[0]))
            for i, v in enumerate(self.vectors):
                self.dense_index.add(i, v)

    def bm25(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term, posting in self._query_postings(query):
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, freq in posting:
                norm = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                scores[i] += idf * (freq * (self.k1 + 1)) / norm
        return scores

    def _query_postings(self, query: str) -> list[tuple[str, list[tuple[int, int]]]]:
        """Query terms worth traversing, commonest ones dropped once anything selective remains.

        A term in nearly every passage has an IDF of about zero, so it changes no ranking — but it
        still makes every passage a candidate, and each candidate then costs a cosine and a Hit.
        That, not the vector scan, is what made a 20,000-passage corpus take a quarter of a second
        per query. Dropped only when a selective term survives, so a query built entirely of common
        words still retrieves something.
        """
        found = [(t, p) for t in set(terms(query)) if (p := self.postings.get(t))]
        if self.n < COMMON_TERM_FLOOR:
            return found
        selective = [(t, p) for t, p in found if len(p) / self.n < MAX_DOC_FRACTION]
        return selective or found

    def search(self, query: str, k: int = 5, alpha: float = 0.5, runtime: Runtime | None = None, min_score: float = 0.12) -> list[Hit]:
        """min_score matters once embeddings are on: cosine makes every passage a candidate, so
        without a floor an off-topic question still retrieves five confident-looking passages."""
        lexical = self.bm25(query)
        top_lex = max(lexical.values(), default=0.0) or 1.0
        dense: dict[int, float] = {}
        if self.vectors:
            qv = normalize((runtime.embed([query]) if runtime else [hashed_embedding(query)])[0])
            scan = self._dense_candidates(qv, lexical)
            dense = {i: cosine(qv, self.vectors[i]) for i in scan}

        candidates = set(lexical) | set(dense)
        hits = []
        for i in candidates:
            lex = lexical.get(i, 0.0) / top_lex
            den = max(0.0, dense.get(i, 0.0))
            hits.append(Hit(self.passages[i], alpha * lex + (1 - alpha) * den, lex, den))
        hits.sort(key=lambda h: -h.score)
        return [h for h in hits[:k] if h.score >= min_score]

    def _dense_candidates(self, qv: Vector, lexical: dict[int, float]) -> set[int]:
        """Which passages are worth a cosine. Bucket neighbours are approximate, so every passage
        BM25 already matched is scored too: a passage that shares the query's words must not be lost
        to a projection that happened to send it elsewhere."""
        if self.dense_index is None:
            return set(range(len(self.vectors)))
        return self.dense_index.candidates(qv) | set(lexical)

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
                        anchors=row.get("anchors", []),
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
                    anchors=list(c.prov.anchors),
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
