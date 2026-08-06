"""QA synthesis: single-hop, multi-hop over the KG, multi-turn, ReAct, personas, Evol, DPO."""

from __future__ import annotations

import random

from .graph import KnowledgeGraph, serialize_path
from .llm import Runtime, parse_json
from .prompts import (
    EVOL_INSTRUCT,
    MULTIHOP_GENERATE,
    MULTITURN_GENERATE,
    MUTATIONS,
    PERSONA_REWRITE,
    PERSONAS,
    QA_GENERATE,
    REACT_GENERATE,
    REJECT_MUTATE,
    REJECTION_MODES,
    STYLES,
)
from .records import Chunk, Provenance, QARecord, Turn


def _prov(chunk: Chunk, generator: str) -> Provenance:
    return Provenance(
        source=chunk.prov.source,
        node_ids=list(chunk.prov.node_ids),
        pages=list(chunk.prov.pages),
        bboxes=list(chunk.prov.bboxes),
        section_path=list(chunk.prov.section_path),
        breadcrumb=chunk.prov.breadcrumb,
        generator=generator,
    )


def generate_qa(runtime: Runtime, chunk: Chunk, n: int = 3, temperature: float = 0.8) -> list[QARecord]:
    prompt = QA_GENERATE.format(breadcrumb=chunk.prov.breadcrumb, context=chunk.context()[:8000], n=n)
    raw = runtime.complete("generate", prompt, temperature=temperature, max_tokens=2048)
    data = parse_json(raw, default={}) or {}
    pairs = data.get("pairs", data if isinstance(data, list) else [])
    out = []
    for p in pairs:
        if not isinstance(p, dict) or not p.get("question") or not p.get("answer"):
            continue
        rec = QARecord(
            question=str(p["question"]).strip(),
            answer=str(p["answer"]).strip(),
            context=chunk.context(),
            task="qa",
            difficulty=str(p.get("difficulty", "intermediate")).lower(),
            prov=_prov(chunk, "generate_qa"),
        )
        if p.get("evidence"):
            rec.prov.verification.append({"stage": "evidence", "quote": str(p["evidence"])[:400]})
        out.append(rec)
    return out


def generate_multihop(runtime: Runtime, kg: KnowledgeGraph, n_pairs: int = 8, per_pair: int = 1, rng: random.Random | None = None) -> list[QARecord]:
    rng = rng or random.Random(0)
    seeds = kg.bridging_pairs()
    rng.shuffle(seeds)
    out: list[QARecord] = []
    for ent_a, ent_b, path in seeds[:n_pairs]:
        chunk_a = _pick_chunk(kg, ent_a.chunk_ids)
        chunk_b = _pick_chunk(kg, ent_b.chunk_ids)
        if chunk_a is None or chunk_b is None or chunk_a.id == chunk_b.id:
            continue
        prompt = MULTIHOP_GENERATE.format(
            path=serialize_path(kg, path),
            breadcrumb_a=chunk_a.prov.breadcrumb,
            context_a=chunk_a.context()[:5000],
            breadcrumb_b=chunk_b.prov.breadcrumb,
            context_b=chunk_b.context()[:5000],
            n=per_pair,
        )
        raw = runtime.complete("generate", prompt, temperature=0.8, max_tokens=2048)
        data = parse_json(raw, default={}) or {}
        for p in data.get("pairs", []) or []:
            if not isinstance(p, dict) or not p.get("question"):
                continue
            prov = _prov(chunk_a, "generate_multihop").merge(_prov(chunk_b, "generate_multihop"))
            prov.verification.append({"stage": "kg_path", "path": serialize_path(kg, path)})
            out.append(
                QARecord(
                    question=str(p["question"]).strip(),
                    answer=str(p.get("answer", "")).strip(),
                    context=chunk_a.context() + "\n\n---\n\n" + chunk_b.context(),
                    task="multihop",
                    difficulty="complex",
                    hops=int(p.get("hops", 2) or 2),
                    prov=prov,
                )
            )
    return out


def _pick_chunk(kg: KnowledgeGraph, chunk_ids) -> Chunk | None:
    for cid in sorted(chunk_ids):
        chunk = kg.chunk_index.get(cid)
        if chunk is not None:
            return chunk
    return None


def generate_multiturn(runtime: Runtime, chunk: Chunk, persona: str = "practitioner", turns: int = 6) -> QARecord | None:
    prompt = MULTITURN_GENERATE.format(
        persona=f"{persona} — {PERSONAS.get(persona, persona)}",
        breadcrumb=chunk.prov.breadcrumb,
        context=chunk.context()[:8000],
        turns=turns,
    )
    raw = runtime.complete("generate", prompt, temperature=0.9, max_tokens=3000)
    data = parse_json(raw, default={}) or {}
    items = data.get("turns", data if isinstance(data, list) else [])
    parsed = [Turn(str(t["role"]), str(t["content"]).strip()) for t in items if isinstance(t, dict) and t.get("role") and t.get("content")]
    if len(parsed) < 2:
        return None
    first_user = next((t.content for t in parsed if t.role == "user"), "")
    last_assistant = next((t.content for t in reversed(parsed) if t.role == "assistant"), "")
    return QARecord(
        question=first_user,
        answer=last_assistant,
        context=chunk.context(),
        task="multiturn",
        persona=persona,
        difficulty="complex",
        turns=parsed,
        prov=_prov(chunk, "generate_multiturn"),
    )


def generate_react(runtime: Runtime, chunk: Chunk) -> QARecord | None:
    prompt = REACT_GENERATE.format(breadcrumb=chunk.prov.breadcrumb, context=chunk.context()[:8000])
    raw = runtime.complete("generate", prompt, temperature=0.7, max_tokens=2500)
    data = parse_json(raw, default={}) or {}
    trace = [t for t in (data.get("trace") or []) if isinstance(t, dict) and t.get("action")]
    if not data.get("question") or not trace:
        return None
    return QARecord(
        question=str(data["question"]).strip(),
        answer=str(data.get("answer", "")).strip(),
        context=chunk.context(),
        task="react",
        difficulty="complex",
        tool_trace=trace,
        prov=_prov(chunk, "generate_react"),
    )


def apply_persona(runtime: Runtime, rec: QARecord, persona: str, style: str) -> QARecord:
    prompt = PERSONA_REWRITE.format(
        persona=f"{persona} — {PERSONAS.get(persona, persona)}",
        style=f"{style} — {STYLES.get(style, style)}",
        question=rec.question,
        answer=rec.answer,
    )
    raw = runtime.complete("generate", prompt, temperature=0.8, max_tokens=1500)
    data = parse_json(raw, default={}) or {}
    if not data.get("question") or not data.get("answer"):
        return rec
    clone = QARecord(
        question=str(data["question"]).strip(),
        answer=str(data["answer"]).strip(),
        context=rec.context,
        task=rec.task,
        persona=persona,
        difficulty=rec.difficulty,
        hops=rec.hops,
        prov=_prov_copy(rec, f"persona:{persona}/{style}"),
    )
    clone.prov.verification.append({"stage": "persona", "from": rec.id, "style": style})
    return clone


def _prov_copy(rec: QARecord, generator: str) -> Provenance:
    p = rec.prov
    return Provenance(
        source=p.source,
        node_ids=list(p.node_ids),
        pages=list(p.pages),
        bboxes=list(p.bboxes),
        section_path=list(p.section_path),
        breadcrumb=p.breadcrumb,
        generator=generator,
    )


def evolve(runtime: Runtime, rec: QARecord, mutation: str | None = None, rng: random.Random | None = None) -> QARecord:
    rng = rng or random.Random()
    mutation = mutation or rng.choice(list(MUTATIONS))
    prompt = EVOL_INSTRUCT.format(
        mutation=f"{mutation} — {MUTATIONS[mutation]}",
        question=rec.question,
        answer=rec.answer,
        context=rec.context[:8000],
    )
    raw = runtime.complete("generate", prompt, temperature=0.9, max_tokens=1800)
    data = parse_json(raw, default={}) or {}
    if not data.get("applied", True) or not data.get("question") or not data.get("answer"):
        return rec
    evolved = QARecord(
        question=str(data["question"]).strip(),
        answer=str(data["answer"]).strip(),
        context=rec.context,
        task=rec.task,
        persona=rec.persona,
        difficulty="complex" if rec.difficulty != "complex" else "complex",
        hops=rec.hops,
        prov=_prov_copy(rec, f"evol:{mutation}"),
    )
    evolved.prov.verification.append({"stage": "evol", "mutation": mutation, "from": rec.id})
    return evolved


def make_preference_pair(runtime: Runtime, rec: QARecord, mode: str | None = None, rng: random.Random | None = None) -> QARecord:
    rng = rng or random.Random()
    mode = mode or rng.choice(list(REJECTION_MODES))
    prompt = REJECT_MUTATE.format(
        mode=mode,
        mode_desc=REJECTION_MODES[mode],
        question=rec.question,
        answer=rec.answer,
        context=rec.context[:8000],
    )
    raw = runtime.complete("generate", prompt, temperature=0.9, max_tokens=1500)
    data = parse_json(raw, default={}) or {}
    rejected = str(data.get("rejected", "")).strip()
    if not rejected or rejected == rec.answer:
        return rec
    rec.rejected = rejected
    rec.rejection_mode = mode
    rec.prov.verification.append({"stage": "dpo", "mode": mode, "injected": str(data.get("injected", ""))[:300]})
    return rec
