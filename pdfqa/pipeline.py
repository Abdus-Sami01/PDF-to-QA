"""End-to-end orchestration: extract -> chunk -> graph -> synthesise -> verify -> select -> export."""

from __future__ import annotations

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from . import extract, synth, verify as verifier
from . import registry
from .export import export as export_dataset, split_is_document_wise, split_records, write_corpus, write_dataset_card
from .adapters import SUFFIXES as ADAPTER_SUFFIXES
from .cache import Store, file_fingerprint, fingerprint
from .chunking import chunk_tree
from .config import PARSER_VERSION, PROMPT_VERSION, Config
from .docast import DocumentTree
from .extract import ExtractError
from .graph import KnowledgeGraph, build_graph, merge_graphs
from .llm import Runtime
from .records import Chunk, Provenance, QARecord
from .select import (
    balance_difficulty,
    dedup_against,
    dedup_chunks,
    dedup_questions_lexical,
    dedup_semantic,
    distribution_report,
    dpp_select,
    embed_records,
    load_questions,
)

BUILTIN_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"} | set(ADAPTER_SUFFIXES)


def supported() -> set[str]:
    return BUILTIN_SUFFIXES | set(registry.ADAPTERS)

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
        if config.plugins:
            self.report["plugins"] = registry.load_plugins(config.plugins)
            self.report["registered"] = registry.summary()

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
            # Draw before dispatching: a shared RNG consumed inside worker threads is ordered by
            # scheduling, so the same seed would otherwise give different personas per run.
            personas = [self.rng.choice(s.personas) for _ in picks]
            convos = self._map(
                lambda pair: synth.generate_multiturn(self.runtime, pair[0], pair[1], s.multiturn_length),
                list(zip(picks, personas)),
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
            traces = [t for t in self._map(lambda c: synth.generate_react(self.runtime, c, chunks), picks) if t]
            if s.repair_traces:
                self._map(synth.repair_trace, traces)
            records.extend(traces)
            self.progress("synth.react", {"records": len(traces), "repaired": sum(1 for t in traces if any(v.get("stage") == "trace_repair" for v in t.prov.verification))})

        records.extend(self._custom_tasks(chunks, kg))
        records.extend(self._augment(records))
        return records

    def _custom_tasks(self, chunks: list[Chunk], kg: KnowledgeGraph) -> list[QARecord]:
        wanted = self.cfg.synth.tasks or list(registry.TASKS)
        out: list[QARecord] = []
        for name in wanted:
            fn = registry.TASKS.get(name)
            if fn is None:
                raise ValueError(f"unknown task {name!r}; registered: {sorted(registry.TASKS)}")
            produced = [r for r in (fn(self.runtime, chunks, kg, self.cfg) or []) if isinstance(r, QARecord)]
            for rec in produced:
                rec.task = rec.task if rec.task != "qa" else name
            out.extend(produced)
            self.progress(f"synth.{name}", {"records": len(produced)})
        return out

    def synthesize_cached(self, tree: DocumentTree, chunks: list[Chunk], kg: KnowledgeGraph) -> list[QARecord]:
        """Synthesis is the expensive stage; checkpoint it per document so a crashed run resumes."""
        key = fingerprint(tree.source, [c.id for c in chunks], PROMPT_VERSION, self.cfg.synth.__dict__, self.cfg.runtime.generate, self.cfg.seed)
        cached = self.store.get("synth", key)
        if cached is not None:
            self.progress("synth", {"source": tree.source, "records": len(cached), "cached": True})
            return [QARecord.from_dict(d) for d in cached]
        records = self.synthesize(chunks, kg)
        self.store.put("synth", key, [r.as_dict() for r in records])
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

    def cross_document(self, graphs: list[KnowledgeGraph], chunks: list[Chunk]) -> list[QARecord]:
        """Only meaningful once two documents share an entity, so it no-ops on a single input."""
        n = self.cfg.synth.cross_doc_pairs
        if n <= 0 or len(graphs) < 2:
            return []
        corpus = merge_graphs(graphs, chunks)
        self.report["corpus_graph"] = corpus.stats() | corpus.meta
        key = fingerprint(sorted({c.prov.source for c in chunks}), [c.id for c in chunks], PROMPT_VERSION, n, self.cfg.runtime.generate, self.cfg.seed)
        cached = self.store.get("cross_doc", key)
        if cached is not None:
            self.progress("synth.cross_document", {"records": len(cached), "cached": True})
            return [QARecord.from_dict(d) for d in cached]
        records = synth.generate_cross_document(self.runtime, corpus, n, self.cfg.synth.multihop_per_pair, self.rng)
        self.store.put("cross_doc", key, [r.as_dict() for r in records])
        self.progress("synth.cross_document", {"records": len(records), "documents": len(graphs)})
        return records

    def preference_pairs(self, records: list[QARecord]) -> None:
        ratio = self.cfg.synth.dpo_ratio
        if ratio <= 0:
            return
        targets = self._sample([r for r in records if not r.turns and r.accepted], int(len(records) * ratio))
        modes = [self.rng.choice(list(synth.REJECTION_MODES)) for _ in targets]
        self._map(lambda pair: synth.make_preference_pair(self.runtime, pair[0], mode=pair[1]), list(zip(targets, modes)))
        self.progress("synth.dpo", {"records": sum(1 for r in records if r.rejected)})

    def verify(self, records: list[QARecord]) -> list[QARecord]:
        v = self.cfg.verify
        self._generated = list(records)
        if not v.enabled:
            return records
        self._map(
            lambda r: verifier.verify(r, self.runtime, v.model_gates, v.symbolic, v.consistency_samples, v.allow_exec, v.z3),
            records,
        )
        kept = [r for r in records if r.accepted and r.scores.get("quality", 0.0) >= v.min_quality]
        self.progress("verify", {"kept": len(kept), "rejected": len(records) - len(kept)})
        self.report["stages"]["rejections"] = _rejection_counts(records)
        self._verified = list(kept)
        return kept

    def yield_report(self, final: list[QARecord]) -> dict:
        """Where records were lost, per shape.

        Composition alone cannot answer "why is there no multi-turn data" — a shape generated at
        full model cost and rejected wholesale looks exactly like a shape nobody asked for.
        """
        generated = getattr(self, "_generated", final)
        verified = getattr(self, "_verified", final)
        made, kept, out = Counter(r.task for r in generated), Counter(r.task for r in verified), Counter(r.task for r in final)

        self.report["gates"] = _gate_stats(generated)
        rows = {}
        for shape in sorted(made):
            reasons = Counter(
                flag[7:] for r in generated if r.task == shape for flag in r.flags if flag.startswith("reject:")
            )
            rows[shape] = {
                "generated": made[shape],
                "verified": kept.get(shape, 0),
                "final": out.get(shape, 0),
                "top_rejections": dict(reasons.most_common(3)),
            }
        return rows

    def select(self, records: list[QARecord]) -> list[QARecord]:
        s = self.cfg.select
        if s.against:
            reference = load_questions(s.against)
            records, seen_before = dedup_against(records, reference, s.lexical_threshold)
            self.progress("dedup.against", {"reference": len(reference), "dropped": len(seen_before), "kept": len(records)})
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
        graphs: list[KnowledgeGraph] = []
        corpus_chunks: list[Chunk] = []

        for path in _expand(paths):
            try:
                tree = self.parse(path)
            except (ExtractError, OSError) as exc:
                # One unreadable file in a large corpus must not discard the whole run.
                self.report.setdefault("skipped", []).append({"path": str(path), "reason": str(exc)[:300]})
                self.progress("skip", {"path": str(path), "reason": type(exc).__name__})
                continue
            chunks = self.chunk(tree)
            kg = self.graph(chunks)
            records = self.synthesize_cached(tree, chunks, kg)
            self.report["documents"].append(
                {"source": tree.source, "chunks": len(chunks), "graph": kg.stats(), "generated": len(records)}
            )
            all_records.extend(records)
            graphs.append(kg)
            corpus_chunks.extend(chunks)

        all_records.extend(self.cross_document(graphs, corpus_chunks))
        self.preference_pairs(all_records)
        kept = self.verify(all_records)
        kept = self.select(kept)

        out = Path(self.cfg.outdir)
        written = export_dataset(kept, out, self.cfg.formats)
        if self.cfg.corpus:
            written["corpus"] = write_corpus(corpus_chunks, out / "corpus.jsonl")
            self.progress("corpus", {"chunks": len(corpus_chunks)})
        if self.cfg.split:
            written |= self.write_splits(kept, out)

        stats = distribution_report(kept)
        card = write_dataset_card(kept, stats, out / "DATASET_CARD.md")

        self.report["stats"] = stats
        self.report["yield"] = self.yield_report(kept)
        self.report["files"] = written | {"card": card}
        self.report["files"]["run_report"] = str(_write_run_report(self.report, out))
        self.report["cache"] = self.store.stats()
        self.report["llm_calls"] = len(self.runtime.calls)
        self.progress("done", {"records": len(kept), "outdir": self.cfg.outdir})
        return self.report

    def write_splits(self, records: list[QARecord], out: Path) -> dict[str, str]:
        splits = split_records(records, tuple(self.cfg.split), self.cfg.seed)
        document_wise = split_is_document_wise(records)
        written: dict[str, str] = {}
        for name, subset in splits.items():
            if subset:
                written |= {f"{name}/{k}": v for k, v in export_dataset(subset, out / name, self.cfg.formats).items()}
        self.report["splits"] = {
            "sizes": {k: len(v) for k, v in splits.items()},
            "document_wise": document_wise,
            "note": "" if document_wise else "fewer than 3 source documents; split row-wise, so contexts overlap across splits",
        }
        self.progress("split", self.report["splits"]["sizes"] | {"document_wise": document_wise})
        return written

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


def _gate_stats(records: list[QARecord]) -> dict:
    """Pass rates across every record that was gated, including the ones that did not survive.

    Computing these from the exported dataset instead reports ~100% for every gate by construction,
    since rejected records are exactly the ones missing from it.
    """
    stats: dict[str, dict[str, int]] = {}
    for rec in records:
        for entry in rec.prov.verification:
            if "passed" not in entry:
                continue
            bucket = stats.setdefault(entry["stage"], {"pass": 0, "fail": 0})
            bucket["pass" if entry["passed"] else "fail"] += 1
    return dict(sorted(stats.items()))


def _write_run_report(report: dict, outdir: Path) -> Path:
    """Persist the run report so `pdfqa report` can show where records were lost, not just what survived."""
    path = Path(outdir) / "run_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


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
            out.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in supported()))
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
        "grids": c.grids,
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
        grids=d.get("grids", []),
        equations=d.get("equations", []),
        figures=d.get("figures", []),
        figure_refs=d.get("figure_refs", []),
        resolved_refs=d.get("resolved_refs", ""),
        tokens=d.get("tokens", 0),
        id=d.get("id", ""),
    )
