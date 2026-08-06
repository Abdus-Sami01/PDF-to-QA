"""End-to-end orchestration: extract -> chunk -> graph -> synthesise -> verify -> select -> export."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from . import export as exporters
from . import extract, synth, verify as verifier
from .cache import Store, file_fingerprint, fingerprint
from .chunking import chunk_tree
from .config import PARSER_VERSION, PROMPT_VERSION, Config
from .docast import DocumentTree
from .graph import KnowledgeGraph, build_graph
from .llm import Runtime
from .records import Chunk, Provenance, QARecord
from .select import (
    balance_difficulty,
    dedup_chunks,
    dedup_questions_lexical,
    dedup_semantic,
    distribution_report,
    dpp_select,
    embed_records,
)

Progress = Callable[[str, dict], None]


def _noop(stage: str, info: dict) -> None:
    pass


class Pipeline:
    def __init__(self, config: Config, progress: Progress | None = None):
        self.cfg = config
        self.store = Store(config.cache_dir, config.cache)
        self.runtime = Runtime(config.runtime_specs(), retries=config.runtime.retries)
        self.rng = random.Random(config.seed)
        self.progress = progress or _noop
        self.report: dict = {"documents": [], "stages": {}}

    # ---------------------------------------------------------------- stages

    def parse(self, path: str | Path) -> DocumentTree:
        assets = str(Path(self.cfg.outdir) / self.cfg.assets_dir) if self.cfg.assets_dir else None
        key = fingerprint(file_fingerprint(path), PARSER_VERSION, self.cfg.backend, assets, self.cfg.figure_dpi)
        cached = self.store.get("ast", key)
        if cached is not None:
            self.progress("parse", {"path": str(path), "cached": True})
            return DocumentTree.from_dict(cached)
        tree = extract.load(path, self.cfg.backend, assets, self.cfg.figure_dpi)
        self.store.put("ast", key, tree.as_dict())
        self.progress("parse", {"path": str(path), "nodes": len(tree.index), "refs": len(tree.references)})
        return tree

    def chunk(self, tree: DocumentTree) -> list[Chunk]:
        c = self.cfg.chunk
        key = fingerprint(tree.source, len(tree.index), PARSER_VERSION, c.max_tokens, c.min_tokens, c.overlap_headings)
        cached = self.store.get("chunks", key)
        if cached is not None:
            chunks = [_chunk_from_dict(d) for d in cached]
        else:
            chunks = chunk_tree(tree, c.max_tokens, c.min_tokens, c.overlap_headings)
            self.store.put("chunks", key, [_chunk_to_dict(x) for x in chunks])
        kept, dropped = dedup_chunks(chunks, c.dedup_threshold)
        self.progress("chunk", {"chunks": len(kept), "deduped": len(dropped)})
        return kept

    def graph(self, chunks: list[Chunk]) -> KnowledgeGraph:
        if not self.cfg.graph.enabled:
            kg = KnowledgeGraph()
            for c in chunks:
                kg.chunk_index[c.id] = c
            return kg
        key = fingerprint([c.id for c in chunks], PROMPT_VERSION, self.cfg.graph.llm_extract, self.cfg.runtime.generate)
        cached = self.store.get("graph", key)
        if cached is not None:
            kg = KnowledgeGraph.from_dict(cached, chunks)
        else:
            kg = build_graph(chunks, self.runtime, self.cfg.graph.llm_extract, self.cfg.graph.max_chunks)
            self.store.put("graph", key, kg.as_dict())
        self.progress("graph", kg.stats())
        return kg

    def synthesize(self, chunks: list[Chunk], kg: KnowledgeGraph) -> list[QARecord]:
        s = self.cfg.synth
        records: list[QARecord] = []

        records.extend(self._map(lambda c: synth.generate_qa(self.runtime, c, s.qa_per_chunk, s.temperature), chunks, flatten=True))
        self.progress("synth.qa", {"records": len(records)})

        if s.multihop_pairs and kg.entities:
            multihop = synth.generate_multihop(self.runtime, kg, s.multihop_pairs, s.multihop_per_pair, self.rng)
            records.extend(multihop)
            self.progress("synth.multihop", {"records": len(multihop)})

        if s.multiturn_per_doc:
            picks = self._sample(chunks, s.multiturn_per_doc)
            convos = self._map(
                lambda c: synth.generate_multiturn(self.runtime, c, self.rng.choice(s.personas), s.multiturn_length), picks
            )
            convos = [c for c in convos if c]
            records.extend(convos)
            self.progress("synth.multiturn", {"records": len(convos)})

        if s.figure_qa_per_doc and self.runtime.has_vision():
            picks = self._sample([c for c in chunks if c.images()], s.figure_qa_per_doc)
            figs = self._map(lambda c: synth.generate_figure_qa(self.runtime, c), picks, flatten=True)
            records.extend(figs)
            self.progress("synth.figure_qa", {"records": len(figs), "chunks_with_images": len(picks)})

        if s.react_per_doc:
            picks = self._sample([c for c in chunks if c.tables or _has_numbers(c)], s.react_per_doc) or self._sample(chunks, s.react_per_doc)
            traces = [t for t in self._map(lambda c: synth.generate_react(self.runtime, c), picks) if t]
            records.extend(traces)
            self.progress("synth.react", {"records": len(traces)})

        records.extend(self._augment(records))
        return records

    def _augment(self, records: list[QARecord]) -> list[QARecord]:
        s = self.cfg.synth
        base = [r for r in records if r.task in ("qa", "multihop")]
        extra: list[QARecord] = []

        for rec in self._sample(base, int(len(base) * s.persona_ratio)):
            persona, style = self.rng.choice(s.personas), self.rng.choice(s.styles)
            extra.append(synth.apply_persona(self.runtime, rec, persona, style))
        self.progress("synth.persona", {"records": len(extra)})

        evolved = [synth.evolve(self.runtime, r, rng=self.rng) for r in self._sample(base, int(len(base) * s.evol_ratio))]
        evolved = [e for e in evolved if e.id not in {r.id for r in records}]
        extra.extend(evolved)
        self.progress("synth.evol", {"records": len(evolved)})
        return extra

    def preference_pairs(self, records: list[QARecord]) -> None:
        ratio = self.cfg.synth.dpo_ratio
        if ratio <= 0:
            return
        targets = self._sample([r for r in records if not r.turns and r.accepted], int(len(records) * ratio))
        self._map(lambda r: synth.make_preference_pair(self.runtime, r, rng=self.rng), targets)
        self.progress("synth.dpo", {"records": sum(1 for r in records if r.rejected)})

    def verify(self, records: list[QARecord]) -> list[QARecord]:
        v = self.cfg.verify
        if not v.enabled:
            return records
        self._map(
            lambda r: verifier.verify(r, self.runtime, v.model_gates, v.symbolic, v.consistency_samples, v.allow_exec, v.z3),
            records,
        )
        kept = [r for r in records if r.accepted and r.scores.get("quality", 0.0) >= v.min_quality]
        self.progress("verify", {"kept": len(kept), "rejected": len(records) - len(kept)})
        self.report["stages"]["rejections"] = _rejection_counts(records)
        return kept

    def select(self, records: list[QARecord]) -> list[QARecord]:
        s = self.cfg.select
        records, lex_dropped = dedup_questions_lexical(records, s.lexical_threshold)
        vectors = embed_records(self.runtime, records)
        records, sem_dropped = dedup_semantic(records, vectors, s.semantic_threshold)
        self.progress("dedup", {"lexical_dropped": len(lex_dropped), "semantic_dropped": len(sem_dropped), "kept": len(records)})

        if s.dpp and records:
            keep_ids = {r.id for r in records}
            vectors = [v for r, v in zip(records, vectors)] if len(vectors) == len(records) else embed_records(self.runtime, records)
            records = dpp_select(records, vectors, s.budget_tokens, s.target_count)
            self.progress("coreset", {"kept": len(records), "of": len(keep_ids)})

        if s.balance:
            records = balance_difficulty(records, s.mix, s.target_count)
            self.progress("balance", {"kept": len(records)})
        return records

    # ---------------------------------------------------------------- driver

    def run(self, inputs: Iterable[str] | None = None) -> dict:
        paths = list(inputs or self.cfg.inputs)
        if not paths:
            raise ValueError("no inputs given")
        all_records: list[QARecord] = []

        for path in _expand(paths):
            tree = self.parse(path)
            chunks = self.chunk(tree)
            kg = self.graph(chunks)
            records = self.synthesize(chunks, kg)
            self.report["documents"].append(
                {"source": tree.source, "chunks": len(chunks), "graph": kg.stats(), "generated": len(records)}
            )
            all_records.extend(records)

        self.preference_pairs(all_records)
        kept = self.verify(all_records)
        kept = self.select(kept)

        written = exporters.export(kept, self.cfg.outdir, self.cfg.formats)
        stats = distribution_report(kept)
        card = exporters.write_dataset_card(kept, stats, Path(self.cfg.outdir) / "DATASET_CARD.md")

        self.report["stats"] = stats
        self.report["files"] = written | {"card": card}
        self.report["cache"] = self.store.stats()
        self.report["llm_calls"] = len(self.runtime.calls)
        self.progress("done", {"records": len(kept), "outdir": self.cfg.outdir})
        return self.report

    # ---------------------------------------------------------------- helpers

    def _map(self, fn, items: list, flatten: bool = False) -> list:
        workers = max(1, self.cfg.runtime.workers)
        if workers == 1 or len(items) <= 1:
            results = [_safe(fn, i) for i in items]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda i: _safe(fn, i), items))
        if not flatten:
            return [r for r in results if r is not None]
        out = []
        for r in results:
            if r:
                out.extend(r)
        return out

    def _sample(self, items: list, n: int) -> list:
        if n <= 0 or not items:
            return []
        if n >= len(items):
            return list(items)
        return self.rng.sample(items, n)


def _safe(fn, item):
    try:
        return fn(item)
    except Exception as exc:  # a single bad chunk must not kill a long run
        return None if not isinstance(item, QARecord) else item


def _has_numbers(chunk: Chunk) -> bool:
    return any(ch.isdigit() for ch in chunk.text)


def _rejection_counts(records: list[QARecord]) -> dict:
    counts: dict[str, int] = {}
    for r in records:
        for f in r.flags:
            if f.startswith("reject:"):
                counts[f[7:]] = counts.get(f[7:], 0) + 1
    return counts


def _expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in (".pdf", ".md", ".markdown", ".txt")))
        elif any(ch in raw for ch in "*?["):
            out.extend(sorted(Path().glob(raw)))
        else:
            out.append(p)
    return out


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "id": c.id,
        "text": c.text,
        "kind": c.kind,
        "tables": c.tables,
        "equations": c.equations,
        "figures": c.figures,
        "figure_refs": c.figure_refs,
        "resolved_refs": c.resolved_refs,
        "tokens": c.tokens,
        "prov": c.prov.__dict__,
    }


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(
        text=d["text"],
        kind=d.get("kind", "section"),
        prov=Provenance(**d["prov"]),
        tables=d.get("tables", []),
        equations=d.get("equations", []),
        figures=d.get("figures", []),
        figure_refs=d.get("figure_refs", []),
        resolved_refs=d.get("resolved_refs", ""),
        tokens=d.get("tokens", 0),
        id=d.get("id", ""),
    )
