import json
from pathlib import Path

import pytest

from pdfqa.cli import main
from pdfqa.export import audit, load_records
from pdfqa.pipeline import Pipeline
from pdfqa.records import Provenance, QARecord
from pdfqa.select import dedup_against, load_questions

FIXTURES = Path(__file__).parent / "fixtures"


def rec(question, answer="Some grounded answer about routing."):
    return QARecord(question=question, answer=answer, context="ctx", prov=Provenance(source="sample.md"))


# ------------------------------------------------------------------ dedup against earlier exports


def test_load_questions_reads_every_export_shape(tmp_path):
    (tmp_path / "raw.jsonl").write_text(json.dumps({"question": "How many experts?", "answer": "Four."}) + "\n")
    (tmp_path / "chatml.jsonl").write_text(
        json.dumps({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "What is the latency?"}]}) + "\n"
    )
    (tmp_path / "sharegpt.jsonl").write_text(
        json.dumps({"conversations": [{"from": "human", "value": "Which dataset was used?"}]}) + "\n"
    )
    (tmp_path / "alpaca.jsonl").write_text(json.dumps({"instruction": "Explain the router.", "output": "x"}) + "\n")
    (tmp_path / "multimodal.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "What does the chart show?"}]}]}) + "\n"
    )
    questions = load_questions([str(tmp_path)])
    assert len(questions) == 5
    assert "What does the chart show?" in questions


def test_load_questions_skips_malformed_lines(tmp_path):
    (tmp_path / "raw.jsonl").write_text('{"question": "Good one?"}\nnot json at all\n\n{"answer": "no question"}\n')
    assert load_questions([str(tmp_path)]) == ["Good one?"]


def test_dedup_against_drops_near_duplicates_only():
    reference = ["How many expert blocks does the SparseRoute router select per token in each layer?"]
    records = [
        rec("How many expert blocks does the SparseRoute router select per token in every layer?"),
        rec("What latency does the dense baseline record on the held-out split of RetrievalBench?"),
    ]
    kept, dropped = dedup_against(records, reference, threshold=0.5)
    assert len(dropped) == 1 and len(kept) == 1
    assert "latency" in kept[0].question


def test_dedup_against_is_a_noop_without_a_reference():
    records = [rec("How many experts does the router select?")]
    kept, dropped = dedup_against(records, [])
    assert kept == records and dropped == []


def test_second_run_against_the_first_adds_nothing(config, tmp_path):
    config.inputs = [str(FIXTURES / "sample.md")]
    config.formats = ["raw"]
    first = Pipeline(config).run()
    assert first["stats"]["count"] > 0

    config.select.against = [config.outdir]
    config.outdir = str(tmp_path / "second")
    second = Pipeline(config).run()
    assert second["stats"]["count"] == 0


# ------------------------------------------------------------------ audit / report


def test_audit_counts_gate_outcomes_and_sources(config):
    Pipeline(config).run()
    records = load_records(config.outdir)
    assert records
    summary = audit(records)
    assert summary["count"] == len(records)
    assert summary["gates"]["structural"]["pass"] == len(records)
    assert set(summary["quality"]) == {"p10", "p50", "p90"}
    assert summary["sources"]["sample.md"] == len(records)


def test_audit_separates_record_shapes(config):
    config.inputs = [str(FIXTURES / "sample.md"), str(FIXTURES / "sample2.md")]
    Pipeline(config).run()
    records = load_records(config.outdir)
    summary = audit(records)
    assert summary["with_tool_trace"] == sum(1 for r in records if r.tool_trace)
    assert summary["multi_turn"] == sum(1 for r in records if r.turns)
    assert summary["with_rejected"] == sum(1 for r in records if r.rejected)
    assert summary["mean_answer_chars"] > 0


def test_load_records_round_trips_provenance(config):
    Pipeline(config).run()
    records = load_records(config.outdir)
    assert all(r.prov.source for r in records)
    assert any(r.prov.verification for r in records)
    assert all(r.id for r in records)


def test_report_command_prints_a_summary(config, capsys):
    Pipeline(config).run()
    assert main(["report", config.outdir]) == 0
    out = capsys.readouterr().out
    assert "gate pass rates" in out and "per source" in out


def test_report_command_json_mode(config, capsys):
    Pipeline(config).run()
    assert main(["report", config.outdir, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["composition"]["count"] == payload["count"]


def test_report_on_an_empty_directory_fails_cleanly(tmp_path, capsys):
    assert main(["report", str(tmp_path)]) == 1
    assert "no raw.jsonl" in capsys.readouterr().err


# ------------------------------------------------------------------ cross-doc caching


def test_cross_document_stage_is_cached(config):
    config.inputs = [str(FIXTURES / "sample.md"), str(FIXTURES / "sample2.md")]
    first = Pipeline(config)
    first_report = first.run()
    assert "cross_doc" in first.store.stats()["entries"]
    first_ids = {r["id"] for r in _rows(config, "cross_document")}
    assert first_ids

    second = Pipeline(config)
    stages: list[str] = []
    second.progress = lambda stage, info: stages.append(stage) if not info.get("cached") else stages.append(f"{stage}:cached")
    second.run()
    assert "synth.cross_document:cached" in stages
    assert {r["id"] for r in _rows(config, "cross_document")} == first_ids
    assert first_report["corpus_graph"]["documents"] == 2


def _rows(config, task):
    path = Path(config.outdir) / "raw.jsonl"
    return [r for r in (json.loads(l) for l in path.read_text().splitlines()) if r["task"] == task]


# ------------------------------------------------------------------ pdfminer rendering


def test_pdfminer_backend_renders_figures(tmp_path):
    pytest.importorskip("pdfminer.high_level")
    pytest.importorskip("pypdfium2")
    from pdfqa.docast import FIGURE
    from pdfqa.extract import load

    pdf = FIXTURES / "sample.pdf"
    if not pdf.exists():
        import sys

        sys.path.insert(0, str(FIXTURES))
        from make_pdf import build

        build()

    tree = load(pdf, "pdfminer", tmp_path / "assets")
    figures = [n for n in tree.nodes([FIGURE]) if n.attrs.get("image_path")]
    assert figures
    for node in figures:
        path = Path(node.attrs["image_path"])
        assert path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert node.attrs["image_width"] > 0 and node.attrs["image_height"] > 0
    assert any(f.attrs.get("caption", "").startswith("Figure 1") for f in figures)


def test_png_writer_emits_a_valid_header(tmp_path):
    from pdfqa.extract import _write_png

    out = tmp_path / "x.png"
    _write_png(out, 2, 2, 6, bytes([0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 255]), 3)
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    assert int.from_bytes(data[16:20], "big") == 2
    assert data[-8:-4] == b"IEND"
