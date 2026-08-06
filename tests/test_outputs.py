import json
import sys
from pathlib import Path

import pytest

from pdfqa.chunking import chunk_tree
from pdfqa.export import split_is_document_wise, split_records, write_corpus
from pdfqa.pipeline import Pipeline
from pdfqa.records import Provenance, QARecord

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from make_docs import build_all  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def records_for(sources, per_source=10):
    out = []
    for src in sources:
        for i in range(per_source):
            out.append(QARecord(question=f"{src} question {i}", answer="a", context="c", prov=Provenance(source=src)))
    return out


# ------------------------------------------------------------------ retrieval corpus


def test_corpus_rows_carry_retrieval_metadata(tree, tmp_path):
    chunks = chunk_tree(tree, max_tokens=300)
    path = write_corpus(chunks, tmp_path / "corpus.jsonl")
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert len(rows) == len(chunks)
    for row in rows:
        assert row["id"] and row["text"] and row["source"] == "sample.md"
        assert isinstance(row["node_ids"], list) and row["node_ids"]
        assert row["tokens"] > 0
    assert any(row["tables"] for row in rows)


def test_corpus_text_includes_the_breadcrumb_for_retrieval(tree, tmp_path):
    chunks = chunk_tree(tree, max_tokens=300)
    rows = [json.loads(l) for l in Path(write_corpus(chunks, tmp_path / "c.jsonl")).read_text().splitlines()]
    with_crumb = [r for r in rows if r["breadcrumb"]]
    assert with_crumb
    assert with_crumb[0]["breadcrumb"] in with_crumb[0]["text"]


def test_pipeline_writes_a_corpus_alongside_the_dataset(config):
    report = Pipeline(config).run()
    corpus = Path(report["files"]["corpus"])
    assert corpus.exists()
    rows = [json.loads(l) for l in corpus.read_text().splitlines()]
    assert rows and all(r["source"] == "sample.md" for r in rows)


def test_corpus_can_be_disabled(config):
    config.corpus = False
    report = Pipeline(config).run()
    assert "corpus" not in report["files"]


# ------------------------------------------------------------------ splits


def test_split_keeps_each_document_in_one_split():
    records = records_for([f"doc{i}.pdf" for i in range(10)])
    splits = split_records(records, (0.6, 0.2, 0.2))
    seen: dict[str, str] = {}
    for name, subset in splits.items():
        for rec in subset:
            assert seen.setdefault(rec.prov.source, name) == name
    assert sum(len(v) for v in splits.values()) == len(records)


def test_split_roughly_honours_the_requested_ratios():
    records = records_for([f"doc{i}.pdf" for i in range(20)])
    splits = split_records(records, (0.7, 0.15, 0.15))
    assert 0.55 <= len(splits["train"]) / len(records) <= 0.85
    assert splits["validation"] and splits["test"]


def test_split_falls_back_to_rows_for_tiny_corpora():
    records = records_for(["only.pdf"], per_source=20)
    assert not split_is_document_wise(records)
    splits = split_records(records, (0.8, 0.1, 0.1))
    assert sum(len(v) for v in splits.values()) == 20
    assert splits["train"] and splits["test"]


def test_split_of_nothing_is_empty():
    assert split_records([]) == {"train": [], "validation": [], "test": []}


def test_split_is_deterministic_for_a_seed():
    records = records_for([f"doc{i}.pdf" for i in range(8)])
    a = split_records(records, (0.6, 0.2, 0.2), seed=3)
    b = split_records(records, (0.6, 0.2, 0.2), seed=3)
    assert {k: [r.id for r in v] for k, v in a.items()} == {k: [r.id for r in v] for k, v in b.items()}


def test_pipeline_writes_split_subdirectories(config, tmp_path):
    docs = build_all(tmp_path / "docs")
    config.inputs = [str(FIXTURES / "sample.md"), str(FIXTURES / "sample2.md")] + [str(p) for p in docs.values()]
    config.formats = ["chatml", "raw"]
    config.split = [0.6, 0.2, 0.2]
    report = Pipeline(config).run()

    out = Path(config.outdir)
    assert report["splits"]["sizes"]["train"] > 0
    for name, size in report["splits"]["sizes"].items():
        if size:
            rows = (out / name / "raw.jsonl").read_text().splitlines()
            assert len(rows) == size


def test_split_report_flags_row_wise_fallback(config):
    config.split = [0.8, 0.1, 0.1]
    report = Pipeline(config).run()
    assert report["splits"]["document_wise"] is False
    assert "overlap" in report["splits"]["note"]


# ------------------------------------------------------------------ mixed-format runs


def test_a_single_run_handles_six_formats_at_once(config, tmp_path):
    docs = build_all(tmp_path / "docs")
    config.inputs = [str(tmp_path / "docs")]
    config.formats = ["raw"]
    report = Pipeline(config).run()
    sources = {d["source"] for d in report["documents"]}
    assert sources == {p.name for p in docs.values()}
    assert report["stats"]["count"] > 0


def test_directory_expansion_picks_up_every_supported_suffix(tmp_path):
    from pdfqa.pipeline import _expand, supported

    build_all(tmp_path / "docs")
    (tmp_path / "docs" / "ignored.rtf").write_text("no")
    found = _expand([str(tmp_path / "docs")])
    assert found and all(p.suffix.lower() in supported() for p in found)
    assert not any(p.suffix == ".rtf" for p in found)
