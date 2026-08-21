"""Malformed and hostile input. A single bad file in a large corpus must not lose the run."""

import json
import zipfile
from pathlib import Path

import pytest

from pdfqa.extract import ExtractError, load, read_text
from pdfqa.llm import Backend, Completion, LLMError, Runtime
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


# ------------------------------------------------------------------ backend failure


class Flaky(Backend):
    """A backend that fails a fixed number of times, in a way we control."""

    name = "flaky"

    def __init__(self, failures: int, status: int | None = None, retry_after: float | None = None):
        self.model, self.remaining, self.status, self.retry_after = "flaky", failures, status, retry_after
        self.attempts = 0

    def complete(self, prompt, system="", temperature=0.7, max_tokens=1024):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise LLMError("nope", status=self.status, retry_after=self.retry_after)
        return Completion("ok", self.model, usage={"prompt_tokens": 10, "completion_tokens": 5})


def runtime_with(backend: Backend, **kw) -> Runtime:
    rt = Runtime({}, backoff=0.0, **kw)
    rt.backends["generate"] = backend
    return rt


def test_a_rejected_api_key_is_not_retried():
    """A 401 fails identically every time; retrying it three times only delays the error."""
    backend = Flaky(failures=99, status=401)
    with pytest.raises(LLMError):
        runtime_with(backend, retries=3).complete("generate", "hi")
    assert backend.attempts == 1


def test_a_rate_limit_is_retried_and_then_succeeds():
    backend = Flaky(failures=2, status=429)
    assert runtime_with(backend, retries=3).complete("generate", "hi") == "ok"
    assert backend.attempts == 3


def test_a_rate_limit_waits_as_long_as_the_provider_asked(monkeypatch):
    slept = []
    monkeypatch.setattr("pdfqa.llm.time.sleep", slept.append)
    runtime_with(Flaky(failures=1, status=429, retry_after=7.5), retries=2).complete("generate", "hi")
    assert slept == [7.5]


def test_a_server_error_is_retried():
    backend = Flaky(failures=1, status=503)
    assert runtime_with(backend, retries=2).complete("generate", "hi") == "ok"
    assert backend.attempts == 2


def test_a_dead_backend_stops_being_called():
    """Otherwise a wrong base_url costs a full corpus of retries and backoff before an empty export."""
    backend = Flaky(failures=10**6, status=500)
    rt = runtime_with(backend, retries=0)
    for _ in range(50):
        with pytest.raises(LLMError):
            rt.complete("generate", "hi")
    assert backend.attempts == Runtime.dead_after


def test_failures_are_counted_per_role():
    rt = runtime_with(Flaky(failures=1, status=500), retries=0)
    with pytest.raises(LLMError):
        rt.complete("generate", "hi")
    rt.complete("generate", "hi")
    health = rt.health()
    assert health["calls"] == 1 and health["failures"] == 1
    assert health["by_role"]["generate"] == {"ok": 1, "failed": 1}


def test_token_spend_is_totalled_across_provider_field_names():
    rt = runtime_with(Flaky(failures=0), retries=0)
    rt.complete("generate", "hi")
    rt.calls.append({"role": "generate", "model": "m", "latency": 0.0, "usage": {"input_tokens": 4, "output_tokens": 1}})
    assert rt.tokens() == {"prompt": 14, "completion": 6, "total": 20}


def test_a_run_against_a_dead_backend_fails_loudly(config):
    """It used to exit zero, write an empty dataset and a dataset card, and report no error at all."""
    config.runtime.generate = {"backend": "openai", "model": "x", "base_url": "http://127.0.0.1:9/v1", "timeout": 1.0}
    config.runtime.verify = dict(config.runtime.generate)
    config.runtime.retries = 0
    config.cache = False

    with pytest.raises(LLMError) as exc:
        Pipeline(config).run()
    assert "no model call succeeded" in str(exc.value)


def test_one_failing_stage_does_not_discard_the_documents_already_paid_for(config, monkeypatch):
    """Multi-hop synthesis is one call standing in for a whole document; it used to take the run with it."""
    def boom(*a, **kw):
        raise RuntimeError("multihop exploded")

    monkeypatch.setattr("pdfqa.synth.generate_multihop", boom)
    config.cache = False
    report = Pipeline(config).run()

    assert report["stats"]["count"] > 0
    assert report["errors"]["by_type"]["RuntimeError"] == 1
    assert any(s["stage"] == "synth.multihop" for s in report["errors"]["samples"])


def test_swallowed_task_failures_are_reported_not_hidden(config, monkeypatch):
    calls = {"n": 0}

    def sometimes(*a, **kw):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("chunk exploded")
        return []

    monkeypatch.setattr("pdfqa.synth.generate_qa", sometimes)
    config.cache = False
    report = Pipeline(config).run()
    assert report["errors"]["count"] >= 1


def test_a_token_budget_stops_the_run_but_keeps_what_it_bought(config):
    config.cache = False
    config.runtime.budget_tokens = 1
    pipeline = Pipeline(config)
    pipeline.runtime.calls.append({"role": "generate", "model": "m", "latency": 0.0, "usage": {"prompt_tokens": 5000, "completion_tokens": 0}})

    report = pipeline.run()
    assert "halted" in report and "ceiling" in report["halted"]
    assert Path(report["files"]["raw"]).exists()


def test_a_scan_with_no_extractable_text_says_so(config, tmp_path):
    """Zero records from an image-only PDF is indistinguishable from a boring document without this."""
    blank = tmp_path / "scan.md"
    blank.write_text("   \n\n  \n")
    config.inputs = [str(blank)]
    config.cache = False

    report = Pipeline(config).run()
    assert any("no extractable text" in s["reason"] for s in report.get("skipped", []))
