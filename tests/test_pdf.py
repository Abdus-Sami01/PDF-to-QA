import json
from pathlib import Path

import pytest

from pdfqa.chunking import chunk_tree
from pdfqa.docast import FIGURE, TABLE
from pdfqa.extract import _absorb_labels, _cluster_regions, _plausible_table, _reject_table_overlap, load
from pdfqa.pipeline import Pipeline
from pdfqa.synth import generate_figure_qa
from pdfqa.verify import verify, visual_grounding, z3_available, z3_consistency

fitz = pytest.importorskip("fitz")

FIXTURE_DIR = Path(__file__).parent / "fixtures"
PDF = FIXTURE_DIR / "sample.pdf"


@pytest.fixture(scope="module")
def pdf_path() -> Path:
    if not PDF.exists():
        import sys

        sys.path.insert(0, str(FIXTURE_DIR))
        from make_pdf import build

        build()
    return PDF


@pytest.fixture
def pdf_tree(pdf_path, tmp_path):
    return load(pdf_path, "pymupdf", tmp_path / "assets", dpi=144)


def test_headings_survive_small_label_text(pdf_tree):
    titles = [n.text for n in pdf_tree.nodes(["heading"])]
    assert "1 Introduction" in titles and "2 Results" in titles
    assert not any(t.startswith("SparseRoute at k = 4") for t in titles)


def test_vector_chart_becomes_a_figure_not_a_table(pdf_tree):
    figures = pdf_tree.nodes([FIGURE])
    tables = pdf_tree.nodes([TABLE])
    assert len(figures) == 1 and len(tables) == 1
    assert figures[0].attrs["caption"].startswith("Figure 1")
    assert tables[0].attrs["caption"].startswith("Table 1")
    assert tables[0].attrs["grid"][0] == ["Model", "Params", "Accuracy", "Latency"]


def test_figure_render_covers_its_data_labels(pdf_tree):
    fig = pdf_tree.nodes([FIGURE])[0]
    path = Path(fig.attrs["image_path"])
    assert path.exists() and path.stat().st_size > 0
    page = fitz.open(PDF)[0]
    words = [w[4] for w in page.get_text("words") if fitz.Rect(w[:4]) in fitz.Rect(*fig.span.bbox) + (-6, -6, 6, 6)]
    assert {"61.2", "75.1", "Dense"} <= set(words)


def test_table_is_rendered_too(pdf_tree):
    table = pdf_tree.nodes([TABLE])[0]
    assert Path(table.attrs["image_path"]).exists()
    assert table.attrs["image_width"] > 0


def test_pdf_references_resolve(pdf_tree):
    resolved = {(r.kind, r.key) for r in pdf_tree.references if r.resolved}
    assert ("table", "1") in resolved and ("figure", "1") in resolved


def test_chunks_expose_images(pdf_tree):
    chunks = chunk_tree(pdf_tree)
    with_images = [c for c in chunks if c.images()]
    assert with_images
    assert all(Path(f["image_path"]).exists() for c in with_images for f in c.images())
    assert "Figures:" in with_images[0].context()


def test_figure_qa_attaches_image_and_provenance(pdf_tree, runtime):
    chunk = next(c for c in chunk_tree(pdf_tree) if c.images())
    records = generate_figure_qa(runtime, chunk)
    assert records
    rec = records[0]
    assert rec.task == "figure_qa" and rec.images
    assert Path(rec.images[0]).exists()
    assert any(v["stage"] == "visual_evidence" for v in rec.prov.verification)


def test_figure_rows_use_visual_gate_not_text_grounding(pdf_tree, runtime):
    chunk = next(c for c in chunk_tree(pdf_tree) if c.images())
    rec = verify(generate_figure_qa(runtime, chunk)[0], runtime)
    stages = {v["stage"] for v in rec.prov.verification}
    assert "visual_grounding" in stages
    assert "numeric_grounding" not in stages
    assert rec.accepted


def test_visual_grounding_passes_when_image_missing(runtime, pdf_tree):
    chunk = next(c for c in chunk_tree(pdf_tree) if c.images())
    rec = generate_figure_qa(runtime, chunk)[0]
    rec.images = ["/nonexistent/figure.png"]
    assert visual_grounding(runtime, rec).passed


def test_full_pdf_run_exports_multimodal_rows(config, pdf_path, tmp_path):
    config.inputs = [str(pdf_path)]
    config.formats = ["chatml", "multimodal", "raw"]
    report = Pipeline(config).run()
    rows = [json.loads(l) for l in (Path(config.outdir) / "multimodal.jsonl").read_text().splitlines()]
    assert rows
    content = rows[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and Path(content[0]["image"]).exists()
    assert rows[0]["meta"]["images"]
    assert report["stats"]["task"].get("figure_qa", 0) > 0


def test_plausible_table_rejects_sparse_grids():
    assert not _plausible_table([["70.1", "", "", "", ""]])
    assert _plausible_table([["Model", "Acc"], ["Dense", "61.2"]])


def test_clustering_merges_neighbours_and_drops_slivers():
    blocks = [
        {"kind": "image", "page": 0, "bbox": (10, 10, 60, 60)},
        {"kind": "image", "page": 0, "bbox": (65, 10, 120, 60)},
        {"kind": "image", "page": 0, "bbox": (400, 400, 405, 402)},
    ]
    merged = _cluster_regions(blocks)
    assert len(merged) == 1 and merged[0]["bbox"] == (10, 10, 120, 60)


def test_table_regions_are_not_reported_as_figures():
    figs = [{"kind": "image", "page": 0, "bbox": (10, 10, 110, 110)}]
    assert _reject_table_overlap(figs, [(0, 0, 200, 200)]) == []
    assert _reject_table_overlap(figs, [(500, 500, 600, 600)]) == figs


def test_absorb_labels_grows_region_but_skips_captions():
    fig = {"kind": "image", "page": 0, "bbox": (100, 100, 200, 200)}
    blocks = [
        {"kind": "text", "page": 0, "bbox": (100, 205, 130, 215), "text": "61.2"},
        {"kind": "text", "page": 0, "bbox": (100, 220, 300, 232), "text": "Figure 1: Accuracy by width."},
        {"kind": "text", "page": 0, "bbox": (100, 600, 300, 640), "text": "A long body paragraph that must not be absorbed at all."},
    ]
    consumed = _absorb_labels([fig], blocks)
    assert len(consumed) == 1
    assert fig["bbox"][3] >= 215
    assert fig["bbox"][3] < 220


@pytest.mark.skipif(not z3_available(), reason="z3 not installed")
def test_z3_gate_requires_negation_to_be_unsat(runtime):
    from pdfqa.records import QARecord

    rec = QARecord(question="What accuracy is reported?", answer="It reaches 74.8 accuracy.",
                   context="SparseRoute reaches 74.8 accuracy on the held-out split.")
    gate = z3_consistency(runtime, rec)
    assert gate.passed and "negation=unsat" in gate.detail


def test_z3_gate_is_a_noop_without_the_package(runtime, monkeypatch):
    import pdfqa.verify as v
    from pdfqa.records import QARecord

    monkeypatch.setattr(v, "z3_available", lambda: False)
    rec = QARecord(question="What accuracy?", answer="74.8", context="reaches 74.8 accuracy")
    assert v.z3_consistency(runtime, rec).passed
