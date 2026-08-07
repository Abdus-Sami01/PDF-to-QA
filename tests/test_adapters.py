import json
import sys
from pathlib import Path

import pytest

from pdfqa.adapters import from_csv, from_html, from_latex, from_notebook
from pdfqa.chunking import chunk_tree
from pdfqa.docast import CAPTION, CODE, EQUATION, FIGURE, HEADING, TABLE
from pdfqa.extract import load
from pdfqa.tools import run_sql

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from make_docs import build_all  # noqa: E402


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("docs"))


# ------------------------------------------------------------------ html


def test_html_headings_and_table(docs):
    tree = load(docs["html"])
    assert tree.meta["title"] == "Sparse Routing Docs"
    assert [n.text for n in tree.nodes([HEADING])] == ["Sparse Routing", "Results"]
    table = tree.nodes([TABLE])[0]
    assert table.attrs["grid"] == [["Model", "Accuracy"], ["Dense", "61.2"], ["k=4", "74.8"]]
    assert table.attrs["caption"].startswith("Table 1")


def test_html_drops_chrome_and_scripts(docs):
    text = load(docs["html"]).root.text_content()
    assert "navigation noise" not in text
    assert "ignored()" not in text
    assert "Sparse Routing Docs" not in text.split("Sparse Routing")[0]


def test_html_keeps_code_and_images(docs):
    tree = load(docs["html"])
    assert tree.nodes([CODE])[0].text == 'print("routing")'
    figure = tree.nodes([FIGURE])[0]
    assert figure.attrs["src"] == "chart.png"
    assert figure.attrs["alt"] == "accuracy by routing width"


def test_html_image_paths_resolve_against_the_document(docs):
    figures = load(docs["html"]).nodes([FIGURE])
    resolved = [f for f in figures if f.attrs["image_path"]]
    assert len(resolved) == 1
    assert Path(resolved[0].attrs["image_path"]).exists()
    assert resolved[0].attrs["caption"] == "accuracy by routing width"


def test_remote_and_missing_images_resolve_to_nothing(docs):
    by_src = {f.attrs["src"]: f.attrs["image_path"] for f in load(docs["html"]).nodes([FIGURE])}
    assert by_src["https://example.com/remote.png"] == ""
    assert by_src["missing.png"] == ""


def test_html_figures_reach_chunks_as_images(docs):
    chunks = chunk_tree(load(docs["html"]), max_tokens=300)
    assert any(c.images() for c in chunks)


def test_epub_images_are_extracted_from_the_archive(tmp_path):
    import zipfile

    from make_docs import build_png

    png = build_png(tmp_path / "fig.png")
    epub = tmp_path / "with_image.epub"
    chapter = '<html><body><h1>Ch</h1><p>Body text about routing widths.</p><img src="images/fig.png" alt="a chart"></body></html>'
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Illustrated</dc:title></metadata>'
           '<manifest><item id="c1" href="c1.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>')
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("content.opf", opf)
        z.writestr("c1.xhtml", chapter)
        z.writestr("images/fig.png", png.read_bytes())

    tree = load(epub, assets_dir=tmp_path / "assets")
    figure = tree.nodes([FIGURE])[0]
    assert Path(figure.attrs["image_path"]).exists()
    assert Path(figure.attrs["image_path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_html_nesting_gives_breadcrumbs(docs):
    tree = load(docs["html"])
    table = tree.nodes([TABLE])[0]
    assert table.breadcrumb() == ["Sparse Routing Docs", "Sparse Routing", "Results"]


def test_html_without_a_title_falls_back_to_h1():
    tree = from_html("<html><body><h1>Fallback Title</h1><p>Body text here.</p></body></html>", "x.html")
    assert tree.meta["title"] == "Fallback Title"


def test_html_ignores_single_row_tables():
    tree = from_html("<html><body><table><tr><td>only</td></tr></table></body></html>", "x.html")
    assert tree.nodes([TABLE]) == []


# ------------------------------------------------------------------ latex


def test_latex_sections_nest_by_depth(docs):
    tree = load(docs["latex"])
    assert tree.meta["title"] == "Sparse Routing for Retrieval"
    router = next(n for n in tree.nodes([HEADING]) if n.text == "Router")
    assert router.breadcrumb() == ["Sparse Routing for Retrieval", "Introduction"]


def test_latex_equation_is_parsed_not_just_stored(docs):
    eq = load(docs["latex"]).nodes([EQUATION])[0]
    assert eq.attrs["balanced"] is True
    assert "W" in eq.attrs["symbols"]
    assert eq.attrs["tree"]["kind"] == "seq"


def test_latex_table_float_yields_a_clean_grid(docs):
    tree = load(docs["latex"])
    table = tree.nodes([TABLE])[0]
    assert table.attrs["grid"] == [["Model", "Accuracy"], ["Dense", "61.2"], ["SparseRoute", "74.8"]]
    assert "tabular" not in table.text
    assert tree.nodes([CAPTION])[0].text == "Accuracy on the held-out split."


def test_latex_strips_markup_from_prose(docs):
    text = load(docs["latex"]).root.text_content()
    assert "\\cite" not in text and "~" not in text
    assert "SparseRoute" in text


def test_latex_without_a_document_environment_still_parses():
    tree = from_latex("\\section{Alone}\nSome body text about routing.", "x.tex")
    assert [n.text for n in tree.nodes([HEADING])] == ["Alone"]


# ------------------------------------------------------------------ docx / epub / notebook / csv


def test_docx_uses_its_document_title_not_the_filename(docs):
    tree = load(docs["docx"])
    assert tree.meta["title"] == "Sparse Routing Report"


def test_docx_styles_become_headings_and_captions(docs):
    tree = load(docs["docx"])
    assert [n.text for n in tree.nodes([HEADING])] == ["Sparse Routing", "Results"]
    assert tree.nodes([CAPTION])[0].text.startswith("Table 1")
    assert tree.nodes([TABLE])[0].attrs["n_cols"] == 2


def test_epub_reads_spine_order_and_metadata_title(docs):
    tree = load(docs["epub"])
    assert tree.meta["title"] == "Routing Book"
    assert tree.meta["documents"] == 1
    assert [n.text for n in tree.nodes([HEADING])] == ["Chapter One"]


def test_notebook_keeps_markdown_code_and_outputs(docs):
    tree = load(docs["notebook"])
    assert tree.meta["format"] == "notebook"
    code = [n.text for n in tree.nodes([CODE])]
    assert any("74.8 - 61.2" in c for c in code)
    assert any("13.6" in c for c in code)


def test_notebook_error_outputs_are_captured():
    nb = {"cells": [{"cell_type": "code", "source": ["boom()"],
                     "outputs": [{"output_type": "error", "ename": "NameError", "evalue": "boom"}]}]}
    tree = from_notebook(json.dumps(nb), "x.ipynb")
    assert any("NameError" in n.text for n in tree.nodes([CODE]))


def test_csv_becomes_one_queryable_table(docs):
    tree = load(docs["csv"])
    table = tree.nodes([TABLE])[0]
    assert tree.meta["rows"] == 2
    result = run_sql("SELECT model FROM t WHERE accuracy > 70", [table.attrs["grid"]])
    assert result.ok and "SparseRoute" in result.output


def test_csv_handles_tab_separated_input():
    tree = from_csv("a\tb\n1\t2\n", "x.tsv")
    assert tree.nodes([TABLE])[0].attrs["grid"] == [["a", "b"], ["1", "2"]]


# ------------------------------------------------------------------ anchors


def test_html_blocks_carry_element_ids_as_anchors():
    tree = from_html(
        '<html><body><h1 id="intro">Routing</h1><p id="p1">First body.</p><p>Second body.</p></body></html>',
        "doc.html",
    )
    anchors = [n.span.anchor for n in tree.root.walk() if n.kind != "document"]
    assert anchors[0] == "#intro"
    assert anchors[1] == "#p1"
    assert anchors[2].startswith("#b")


def test_anchors_are_unique_within_a_document(docs):
    anchors = [n.span.anchor for n in load(docs["html"]).root.walk() if n.span.anchor]
    assert len(anchors) == len(set(anchors))


def test_epub_anchors_name_the_spine_file(docs):
    anchors = [n.span.anchor for n in load(docs["epub"]).root.walk() if n.span.anchor]
    assert anchors and all(a.startswith("c1.xhtml#") for a in anchors)


def test_pageless_formats_do_not_claim_page_zero(docs):
    chunks = chunk_tree(load(docs["html"]), max_tokens=300)
    assert chunks
    assert all(c.prov.pages == [] for c in chunks)
    assert any(c.prov.anchors for c in chunks)


def test_citation_uses_an_anchor_when_there_is_no_page(docs):
    from pdfqa.retrieve import Index

    passage = Index.from_chunks(chunk_tree(load(docs["html"]), max_tokens=300)).passages[0]
    citation = passage.citation()
    assert citation.startswith("doc.html #")


def test_anchors_reach_the_exported_corpus(docs, tmp_path):
    import json

    from pdfqa.export import write_corpus

    path = write_corpus(chunk_tree(load(docs["html"]), max_tokens=300), tmp_path / "corpus.jsonl")
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    assert any(r["anchors"] for r in rows)


def test_anchors_survive_a_tree_round_trip(docs):
    from pdfqa.docast import DocumentTree

    tree = load(docs["html"])
    clone = DocumentTree.from_dict(tree.as_dict())
    assert [n.span.anchor for n in clone.root.walk()] == [n.span.anchor for n in tree.root.walk()]


# ------------------------------------------------------------------ downstream


@pytest.mark.parametrize("fmt", ["html", "latex", "docx", "epub", "notebook", "csv"])
def test_every_adapter_produces_chunkable_output(docs, fmt):
    chunks = chunk_tree(load(docs[fmt]), max_tokens=300)
    assert chunks
    for c in chunks:
        assert c.prov.source and c.prov.breadcrumb
        assert c.text.strip()


def test_adapter_tables_reach_the_sql_tool(docs):
    chunks = chunk_tree(load(docs["docx"]), max_tokens=300)
    grids = [g for c in chunks for g in c.grids]
    assert grids
    assert run_sql("SELECT count(*) FROM t", grids).output.strip() == "2"


def test_unsupported_suffix_names_what_is_supported(tmp_path):
    bad = tmp_path / "x.rtf"
    bad.write_text("hi")
    with pytest.raises(ValueError, match=r"\.docx"):
        load(bad)
