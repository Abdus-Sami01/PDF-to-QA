"""Deduplication and quality-diversity selection: MinHash/LSH, k-center, greedy DPP, hardness balancing."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict

from .llm import Runtime
from .records import Chunk, QARecord
from .verify import difficulty_bucket, hardness

WORD = re.compile(r"[a-z0-9]+")
MERSENNE = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[str]:
    toks = WORD.findall(text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


class MinHash:
    def __init__(self, num_perm: int = 128, seed: int = 17):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self.params = [(rng.randrange(1, MERSENNE), rng.randrange(0, MERSENNE)) for _ in range(num_perm)]

    def signature(self, text: str, k: int = 5) -> tuple[int, ...]:
        grams = shingles(text, k)
        if not grams:
            return tuple([MERSENNE] * self.num_perm)
        hashed = [int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little") for g in grams]
        return tuple(min((a * h + b) % MERSENNE for h in hashed) for a, b in self.params)


def jaccard_est(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


class LSHIndex:
    def __init__(self, num_perm: int = 128, bands: int = 32):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.minhash = MinHash(num_perm)
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self.signatures: dict[str, tuple[int, ...]] = {}

    def add(self, key: str, text: str) -> None:
        sig = self.minhash.signature(text)
        self.signatures[key] = sig
        for b in range(self.bands):
            band = sig[b * self.rows : (b + 1) * self.rows]
            self.buckets[(b, hash(band))].append(key)

    def candidates(self, key: str) -> set[str]:
        sig = self.signatures[key]
        out: set[str] = set()
        for b in range(self.bands):
            band = sig[b * self.rows : (b + 1) * self.rows]
            out.update(self.buckets.get((b, hash(band)), []))
        out.discard(key)
        return out


def dedup_lexical(items: list, text_of, threshold: float = 0.8, num_perm: int = 128, bands: int = 32) -> tuple[list, list]:
    """MinHash + LSH near-duplicate removal; keeps the first occurrence in input order."""
    index = LSHIndex(num_perm, bands)
    kept, dropped = [], []
    kept_keys: set[str] = set()
    for i, item in enumerate(items):
        key = f"i{i}"
        index.add(key, text_of(item))
        dup = None
        for cand in index.candidates(key):
            if cand in kept_keys and jaccard_est(index.signatures[key], index.signatures[cand]) >= threshold:
                dup = cand
                break
        if dup is None:
            kept.append(item)
            kept_keys.add(key)
        else:
            dropped.append(item)
    return kept, dropped


def dedup_chunks(chunks: list[Chunk], threshold: float = 0.85) -> tuple[list[Chunk], list[Chunk]]:
    return dedup_lexical(chunks, lambda c: c.text, threshold)


def dedup_questions_lexical(records: list[QARecord], threshold: float = 0.8) -> tuple[list[QARecord], list[QARecord]]:
    return dedup_lexical(records, lambda r: r.question, threshold)


# --------------------------------------------------------------------------- dense space


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def embed_records(runtime: Runtime, records: list[QARecord], batch: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(records), batch):
        chunk = [f"{r.question}\n{r.answer[:500]}" for r in records[i : i + batch]]
        vectors.extend(runtime.embed(chunk))
    return [normalize(v) for v in vectors]


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def dedup_semantic(records: list[QARecord], vectors: list[list[float]], threshold: float = 0.92) -> tuple[list[QARecord], list[QARecord]]:
    """k-center greedy: keep points that are far from everything already kept."""
    order = sorted(range(len(records)), key=lambda i: -records[i].scores.get("quality", 0.0))
    kept_idx: list[int] = []
    dropped: list[QARecord] = []
    for i in order:
        if all(cosine(vectors[i], vectors[j]) < threshold for j in kept_idx):
            kept_idx.append(i)
        else:
            dropped.append(records[i])
    kept_idx.sort()
    return [records[i] for i in kept_idx], dropped


def dpp_select(records: list[QARecord], vectors: list[list[float]], budget_tokens: int | None = None, k: int | None = None, quality_weight: float = 1.0) -> list[QARecord]:
    """Greedy submodular selection maximising quality plus marginal coverage (MAP-style DPP)."""
    if not records:
        return []
    quality = [max(0.05, r.scores.get("quality", 0.5)) ** quality_weight for r in records]
    selected: list[int] = []
    max_sim = [0.0] * len(records)
    remaining = set(range(len(records)))
    used_tokens = 0
    limit = k if k is not None else len(records)

    while remaining and len(selected) < limit:
        best, best_gain = None, -1.0
        for i in remaining:
            gain = quality[i] * (1.0 - max_sim[i])
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None or best_gain <= 0:
            break
        cost = len(records[best].question) // 4 + len(records[best].answer) // 4
        if budget_tokens is not None and used_tokens + cost > budget_tokens:
            remaining.discard(best)
            continue
        selected.append(best)
        remaining.discard(best)
        used_tokens += cost
        for i in remaining:
            sim = cosine(vectors[best], vectors[i])
            if sim > max_sim[i]:
                max_sim[i] = sim
    selected.sort()
    return [records[i] for i in selected]


# --------------------------------------------------------------------------- distribution


DEFAULT_MIX = {"simple": 0.25, "intermediate": 0.45, "complex": 0.30}


def balance_difficulty(records: list[QARecord], mix: dict[str, float] | None = None, total: int | None = None) -> list[QARecord]:
    """Trim over-represented difficulty buckets toward the target mix, best-quality first."""
    mix = mix or DEFAULT_MIX
    buckets: dict[str, list[QARecord]] = defaultdict(list)
    for r in records:
        bucket = difficulty_bucket(r)
        r.difficulty = bucket
        r.scores.update({f"hardness_{k}": v for k, v in hardness(r).items()})
        buckets[bucket].append(r)
    total = total or len(records)
    out: list[QARecord] = []
    for name, share in mix.items():
        pool = sorted(buckets.get(name, []), key=lambda r: -r.scores.get("quality", 0.0))
        out.extend(pool[: max(0, round(share * total))])
    chosen = {id(r) for r in out}
    leftovers = [r for r in records if id(r) not in chosen]
    if len(out) < total:
        out.extend(sorted(leftovers, key=lambda r: -r.scores.get("quality", 0.0))[: total - len(out)])
    return out


def distribution_report(records: list[QARecord]) -> dict:
    by_difficulty: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    by_persona: dict[str, int] = defaultdict(int)
    for r in records:
        by_difficulty[r.difficulty] += 1
        by_task[r.task] += 1
        by_persona[r.persona] += 1
    quality = [r.scores.get("quality", 0.0) for r in records] or [0.0]
    return {
        "count": len(records),
        "difficulty": dict(by_difficulty),
        "task": dict(by_task),
        "persona": dict(by_persona),
        "quality_mean": round(sum(quality) / len(quality), 4),
        "quality_min": round(min(quality), 4),
        "quality_max": round(max(quality), 4),
    }
