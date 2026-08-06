"""Exporters for the common fine-tuning stacks. Parquet is used when pyarrow is present, JSONL otherwise."""

from __future__ import annotations

import json
from pathlib import Path

from .records import QARecord

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
