"""Exporters for the common fine-tuning stacks. Parquet is used when pyarrow is present, JSONL otherwise."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .records import Chunk, QARecord

DEFAULT_SYSTEM = "You answer questions strictly from the provided source material."


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def as_sharegpt(rec: QARecord) -> dict:
    mapping = {"user": "human", "assistant": "gpt"}
    return {"conversations": [{"from": mapping.get(m["role"], m["role"]), "value": m["content"]} for m in rec.messages()]}


def as_chatml(rec: QARecord, system: str = DEFAULT_SYSTEM) -> dict:
    return {"messages": [{"role": "system", "content": system}] + rec.messages()}


def as_alpaca(rec: QARecord) -> dict:
    return {"instruction": rec.question, "input": rec.context, "output": rec.answer}


def as_dpo(rec: QARecord, system: str = DEFAULT_SYSTEM) -> dict:
    return {
        "prompt": rec.question,
        "system": system,
        "chosen": rec.answer,
        "rejected": rec.rejected,
        "rejection_mode": rec.rejection_mode,
    }


def as_react(rec: QARecord) -> dict:
    steps = []
    for step in rec.tool_trace:
        steps.append(
            {
                "thought": step.get("thought", ""),
                "action": step.get("action", ""),
                "action_input": step.get("action_input", ""),
                "observation": step.get("observation", ""),
                "verified": bool(step.get("executed_ok")),
            }
        )
    return {"question": rec.question, "trace": steps, "answer": rec.answer, "tables": rec.tool_env.get("grids", [])}


def as_multimodal(rec: QARecord) -> dict:
    """Interleaved image+text turn, the shape LLaVA-style and OpenAI vision trainers expect."""
    content = [{"type": "image", "image": p} for p in rec.images]
    content.append({"type": "text", "text": rec.question})
    return {
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": rec.answer}],
        "images": rec.images,
    }


def _meta(rec: QARecord) -> dict:
    return {
        "id": rec.id,
        "images": rec.images,
        "task": rec.task,
        "persona": rec.persona,
        "difficulty": rec.difficulty,
        "hops": rec.hops,
        "quality": rec.scores.get("quality", 0.0),
        "source": rec.prov.source,
        "pages": rec.prov.pages,
        "node_ids": rec.prov.node_ids,
        "breadcrumb": rec.prov.breadcrumb,
    }


FORMATS = ("chatml", "sharegpt", "alpaca", "openai", "axolotl", "llamafactory", "unsloth", "dpo", "react", "multimodal", "raw")

SUBSET = {"dpo": lambda r: bool(r.rejected), "react": lambda r: bool(r.tool_trace), "multimodal": lambda r: bool(r.images)}


def export(records: list[QARecord], outdir: str | Path, formats: list[str] | None = None, system: str = DEFAULT_SYSTEM, with_meta: bool = True) -> dict[str, str]:
    out = Path(outdir)
    formats = formats or ["chatml", "sharegpt", "dpo", "raw"]
    written: dict[str, str] = {}

    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
        if fmt in ("chatml", "openai", "unsloth"):
            rows = [as_chatml(r, system) for r in records]
        elif fmt in ("sharegpt", "llamafactory"):
            rows = [as_sharegpt(r) for r in records]
        elif fmt in ("alpaca", "axolotl"):
            rows = [as_alpaca(r) for r in records]
        elif fmt == "dpo":
            rows = [as_dpo(r, system) for r in records if r.rejected]
        elif fmt == "react":
            rows = [as_react(r) for r in records if r.tool_trace]
        elif fmt == "multimodal":
            rows = [as_multimodal(r) for r in records if r.images]
        else:
            rows = [r.as_dict() for r in records]

        if with_meta and fmt != "raw":
            keep = SUBSET.get(fmt, lambda r: True)
            for row, rec in zip(rows, [r for r in records if keep(r)]):
                row["meta"] = _meta(rec)

        written[fmt] = str(_write_jsonl(out / f"{fmt}.jsonl", rows))

    parquet = write_parquet(records, out / "dataset.parquet")
    if parquet:
        written["parquet"] = parquet
    return written


def write_parquet(records: list[QARecord], path: str | Path) -> str | None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [r.id for r in records],
            "question": [r.question for r in records],
            "answer": [r.answer for r in records],
            "rejected": [r.rejected for r in records],
            "images": [json.dumps(r.images) for r in records],
            "context": [r.context for r in records],
            "messages": [json.dumps(r.messages(), ensure_ascii=False) for r in records],
            "task": [r.task for r in records],
            "persona": [r.persona for r in records],
            "difficulty": [r.difficulty for r in records],
            "hops": [r.hops for r in records],
            "quality": [r.scores.get("quality", 0.0) for r in records],
            "source": [r.prov.source for r in records],
            "pages": [json.dumps(r.prov.pages) for r in records],
            "node_ids": [json.dumps(r.prov.node_ids) for r in records],
            "breadcrumb": [r.prov.breadcrumb for r in records],
            "provenance": [json.dumps(r.prov.verification, ensure_ascii=False, default=str) for r in records],
        }
    )
    pq.write_table(table, path)
    return str(path)


def write_corpus(chunks: list[Chunk], path: str | Path) -> str:
    """The chunk set is already a retrieval corpus — breadcrumbed, provenanced, table-aware.

    Exporting it means one run yields both training data and the index you would evaluate against.
    """
    rows = []
    for c in chunks:
        rows.append(
            {
                "id": c.id,
                "text": c.context(),
                "body": c.text,
                "breadcrumb": c.prov.breadcrumb,
                "section_path": c.prov.section_path,
                "source": c.prov.source,
                "pages": c.prov.pages,
                "node_ids": c.prov.node_ids,
                "tokens": c.tokens,
                "tables": c.tables,
                "equations": c.equations,
                "images": [f["image_path"] for f in c.figure_refs if f.get("image_path")],
            }
        )
    return str(_write_jsonl(Path(path), rows))


def split_records(records: list[QARecord], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 7) -> dict[str, list[QARecord]]:
    """Split by source document, not by row.

    Rows from one paper share context and entities, so a row-wise split leaks the eval set into
    training. Whole documents move together instead.
    """
    if not records:
        return {"train": [], "validation": [], "test": []}
    by_source: dict[str, list[QARecord]] = {}
    for rec in records:
        by_source.setdefault(rec.prov.source or "unknown", []).append(rec)

    sources = sorted(by_source, key=lambda s: (-len(by_source[s]), s))
    random.Random(seed).shuffle(sources)
    total = len(records)
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    names = ["train", "validation", "test"]
    out: dict[str, list[QARecord]] = {n: [] for n in names}

    if len(sources) < 3:
        return _split_rows(records, ratios, seed)

    for src in sources:
        deficits = [targets[i] - len(out[names[i]]) for i in range(3)]
        pick = deficits.index(max(deficits))
        out[names[pick]].extend(by_source[src])
    return out


def _split_rows(records: list[QARecord], ratios: tuple[float, float, float], seed: int) -> dict[str, list[QARecord]]:
    """Fallback for corpora too small to split by document; the leakage caveat is reported."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * ratios[0])
    n_val = int(len(shuffled) * ratios[1])
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_is_document_wise(records: list[QARecord]) -> bool:
    return len({r.prov.source for r in records}) >= 3


def load_records(path: str | Path) -> list[QARecord]:
    """Read back a `raw` export — the only format that round-trips every field."""
    p = Path(path)
    files = sorted(p.rglob("raw.jsonl")) if p.is_dir() else [p]
    out: list[QARecord] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(QARecord.from_dict(json.loads(line)))
    return out


def audit(records: list[QARecord]) -> dict:
    """Post-hoc view of a produced dataset: what passed, what was repaired, where it came from."""
    gates: dict[str, dict[str, int]] = {}
    stages: dict[str, int] = {}
    per_source: dict[str, int] = {}
    quality = sorted(r.scores.get("quality", 0.0) for r in records)

    for rec in records:
        for src in rec.prov.sources or [rec.prov.source]:
            per_source[src] = per_source.get(src, 0) + 1
        for entry in rec.prov.verification:
            stage = entry.get("stage", "?")
            stages[stage] = stages.get(stage, 0) + 1
            if "passed" in entry:
                bucket = gates.setdefault(stage, {"pass": 0, "fail": 0})
                bucket["pass" if entry["passed"] else "fail"] += 1

    def pct(q: float) -> float:
        return round(quality[min(len(quality) - 1, int(q * len(quality)))], 4) if quality else 0.0

    return {
        "count": len(records),
        "quality": {"p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90)} if quality else {},
        "gates": dict(sorted(gates.items())),
        "stages": dict(sorted(stages.items())),
        "sources": dict(sorted(per_source.items(), key=lambda kv: -kv[1])),
        "with_rejected": sum(1 for r in records if r.rejected),
        "with_images": sum(1 for r in records if r.images),
        "with_tool_trace": sum(1 for r in records if r.tool_trace),
        "multi_turn": sum(1 for r in records if r.turns),
        "repaired_traces": sum(1 for r in records if any(v.get("stage") == "trace_repair" for v in r.prov.verification)),
        "mean_answer_chars": round(sum(len(r.answer) for r in records) / max(1, len(records)), 1),
    }


def write_dataset_card(records: list[QARecord], stats: dict, path: str | Path) -> str:
    sources = sorted({r.prov.source for r in records})
    lines = [
        "# Dataset card",
        "",
        f"Records: {stats.get('count', len(records))}",
        f"Sources: {', '.join(sources) or 'n/a'}",
        f"Mean quality score: {stats.get('quality_mean', 0.0)}",
        "",
        "## Composition",
        "",
        "| axis | breakdown |",
        "| --- | --- |",
        f"| task | {stats.get('task', {})} |",
        f"| difficulty | {stats.get('difficulty', {})} |",
        f"| persona | {stats.get('persona', {})} |",
        "",
        "## Provenance",
        "",
        "Every row carries source file, page indexes, AST node ids, section breadcrumb, and the full",
        "verification log for each gate it passed.",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)
