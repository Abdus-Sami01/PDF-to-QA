"""Example plugin: a custom task, a custom export format, and a custom input adapter."""

from pdfqa.adapters import Builder
from pdfqa.records import Provenance, QARecord
from pdfqa.registry import register_adapter, register_format, register_task


@register_task("definition")
def definitions(runtime, chunks, kg, config):
    """One definition-style pair per entity the graph typed as a method."""
    out = []
    methods = [e for e in kg.entities.values() if e.type == "method"]
    for entity in methods[:3]:
        chunk = next((kg.chunk_index[c] for c in sorted(entity.chunk_ids) if c in kg.chunk_index), None)
        if chunk is None:
            continue
        out.append(
            QARecord(
                question=f"What is {entity.name} and where is it described?",
                answer=f"{entity.name} is described in {chunk.prov.breadcrumb}.",
                context=chunk.context(),
                task="definition",
                prov=Provenance(source=chunk.prov.source, node_ids=list(chunk.prov.node_ids),
                                breadcrumb=chunk.prov.breadcrumb, generator="plugin:definition"),
            )
        )
    return out


@register_format("minimal")
def minimal(records):
    return [{"q": r.question, "a": r.answer} for r in records]


@register_adapter(".log")
def read_log(path):
    builder = Builder(path.stem, path.name)
    builder.heading(path.stem, 1)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            builder.block("paragraph", line)
    return builder.build("log")
