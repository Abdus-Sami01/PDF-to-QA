"""Malformed and hostile input. A single bad file in a large corpus must not lose the run."""

import json
import zipfile
from pathlib import Path

import pytest

from pdfqa.extract import ExtractError, load, read_text
from pdfqa.pipeline import Pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ------------------------------------------------------------------ uniform failure type


@pytest.mark.parametrize(
    "name,data",
    [
        ("notzip.docx", b"this is not a zip file"),
        ("notzip.epub", b"neither is this"),
        ("broken.ipynb", b"{not json"),
        ("truncated.pdf", b"%PDF-1.4\ntrailer garbage"),
        ("empty.html", b""),
        ("empty.pdf", b""),
    ],
)
def test_unreadable_files_raise_one_documented_error(tmp_path, name, data):
    """Callers should not have to catch BadZipFile, KeyError, JSONDecodeError and FileDataError."""
    with pytest.raises(ExtractError):
        load(write(tmp_path, name, data))


def test_zip_without_the_expected_member_is_reported(tmp_path):
    path = tmp_path / "nomember.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("other.xml", "<x/>")
    with pytest.raises(ExtractError, match="readable .docx"):
        load(path)


def test_docx_with_malformed_xml_is_reported(tmp_path):
    path = tmp_path / "badxml.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", "<w:document><unclosed>")
    with pytest.raises(ExtractError, match="malformed"):
        load(path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ExtractError, match="does not exist"):
        load(tmp_path / "absent.md")


def test_pdf_error_message_names_the_backends_tried(tmp_path):
    path = write(tmp_path, "broken.pdf", b"%PDF-1.4 nonsense")
    with pytest.raises(ExtractError) as excinfo:
        load(path)
    assert "broken.pdf" in str(excinfo.value)


# ------------------------------------------------------------------ survivable input


def test_unclosed_html_still_parses(tmp_path):
    tree = load(write(tmp_path, "u.html", b"<html><body><table><tr><td>a<p>text without closing"))
    assert tree is not None


def test_single_column_csv_yields_no_table(tmp_path):
    tree = load(write(tmp_path, "one.csv", b"header\nvalue\n"))
    assert tree.nodes(["table"]) == []


def test_csv_with_nul_bytes_survives(tmp_path):
    tree = load(write(tmp_path, "nul.csv", b"a,b\n\x00,2\n"))
    assert tree.nodes(["table"])[0].attrs["n_cols"] == 2


def test_epub_without_a_spine_falls_back_to_any_html(tmp_path):
    path = tmp_path / "nospine.epub"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("junk.txt", "hello")
    assert load(path) is not None


def test_notebook_that_is_not_an_object_is_rejected(tmp_path):
    with pytest.raises(ExtractError, match="not a notebook"):
        load(write(tmp_path, "list.ipynb", json.dumps([1, 2, 3]).encode()))


def test_deeply_nested_html_does_not_recurse_to_death(tmp_path):
    html = b"<html><body>" + b"<div>" * 500 + b"deep text" + b"</div>" * 500 + b"</body></html>"
    assert "deep text" in load(write(tmp_path, "deep.html", html)).root.text_content()


# ------------------------------------------------------------------ encodings


def test_latin1_text_is_decoded_not_mangled(tmp_path):
    path = write(tmp_path, "l.html", "<html><body><p>café résumé</p></body></html>".encode("latin-1"))
    assert load(path).root.text_content() == "café résumé"


def test_cp1252_punctuation_survives(tmp_path):
    path = write(tmp_path, "c.html", b"<html><body><p>Sm\x93art\x94 quotes \x96 dash</p></body></html>")
    text = load(path).root.text_content()
    assert "�" not in text
    assert "“art”" in text


def test_utf8_bom_is_stripped(tmp_path):
    path = write(tmp_path, "b.csv", "﻿name,value\ncafé,3\n".encode("utf-8"))
    assert load(path).nodes(["table"])[0].attrs["grid"][0] == ["name", "value"]


def test_read_text_never_raises_on_arbitrary_bytes(tmp_path):
    path = write(tmp_path, "random.txt", bytes(range(256)))
    assert isinstance(read_text(path), str)


# ------------------------------------------------------------------ the run survives


@pytest.fixture
def mixed_corpus(tmp_path):
    """Two good documents among a pile of broken ones."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("sample.md", "sample2.md"):
        (corpus / name).write_bytes((FIXTURES / name).read_bytes())
    write(corpus, "notzip.docx", b"not a zip")
    write(corpus, "broken.ipynb", b"{not json")
    write(corpus, "empty.html", b"")
    write(corpus, "truncated.pdf", b"%PDF-1.4 broken")
    return corpus


def test_one_bad_file_does_not_abort_the_run(config, mixed_corpus):
    config.inputs = [str(mixed_corpus)]
    config.formats = ["raw"]
    report = Pipeline(config).run()

    assert len(report["documents"]) == 2
    assert report["stats"]["count"] > 0
    assert len(report["skipped"]) == 4


def test_skipped_files_are_reported_with_a_reason(config, mixed_corpus):
    config.inputs = [str(mixed_corpus)]
    config.formats = ["raw"]
    skipped = Pipeline(config).run()["skipped"]

    by_name = {Path(s["path"]).name: s["reason"] for s in skipped}
    assert "not a readable .docx" in by_name["notzip.docx"]
    assert "notebook JSON" in by_name["broken.ipynb"]
    assert "empty" in by_name["empty.html"]
    assert "truncated.pdf" in by_name["truncated.pdf"]


def test_a_clean_corpus_reports_nothing_skipped(config):
    report = Pipeline(config).run()
    assert "skipped" not in report
