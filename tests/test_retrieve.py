import random
import re
import json
from pathlib import Path

import pytest

from pdfqa.chunking import chunk_tree
from pdfqa.cli import main
from pdfqa.evaluate import evaluate, grade_one, numeric_match, token_f1
from pdfqa.export import load_records
from pdfqa.pipeline import Pipeline
from pdfqa.records import Provenance, QARecord
from pdfqa.select import EXACT_LIMIT
from pdfqa.retrieve import COMMON_TERM_FLOOR, Index, Passage, answer, terms

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def index(tree, tree2_or_none=None):
    return Index.from_chunks(chunk_tree(tree, max_tokens=300))


@pytest.fixture
def dataset(config):
    config.formats = ["raw"]
    Pipeline(config).run()
    return config


# ------------------------------------------------------------------ index


def test_index_is_built_from_chunks(index):
    assert index.passages
    assert all(p.source == "sample.md" for p in index.passages)
    assert index.n == len(index.passages)


def test_bm25_ranks_the_relevant_section_first(index):
    hits = index.search("how many experts does the router select", k=3)
    assert hits
    assert "expert" in hits[0].passage.text.lower()


def test_rare_terms_beat_common_ones(index):
    hits = index.search("load-balancing coefficient", k=3)
    assert "load-balancing" in hits[0].passage.text.lower()


def test_dense_scores_appear_only_after_embedding(index):
    assert all(h.dense == 0.0 for h in index.search("routing", k=3))
    index.embed()
    assert any(h.dense > 0.0 for h in index.search("routing", k=3))


def test_alpha_switches_between_lexical_and_dense(index):
    index.embed()
    lexical = index.search("routing temperature", k=3, alpha=1.0)
    dense = index.search("routing temperature", k=3, alpha=0.0)
    assert all(h.score == h.lexical for h in lexical)
    assert all(abs(h.score - h.dense) < 1e-9 for h in dense)


def test_query_with_no_overlap_returns_nothing(index):
    assert index.search("quarterly dividend policy for shareholders", k=3) == []


def test_score_floor_still_applies_once_embeddings_make_everything_a_candidate(index):
    index.embed()
    assert index.search("quarterly dividend policy for shareholders", k=5) == []
    assert index.search("quarterly dividend policy for shareholders", k=5, min_score=0.0)


def test_answer_refuses_when_every_hit_is_below_the_floor(runtime, index):
    index.embed()
    found = answer(runtime, index, "What is the quarterly dividend policy for shareholders?")
    assert found.unanswerable and found.hits == []


def test_terms_drop_stopwords():
    assert "the" not in terms("the router and the experts")
    assert "router" in terms("the router and the experts")


def test_index_loads_from_an_exported_corpus(dataset):
    loaded = Index.load(dataset.outdir)
    assert loaded.passages
    assert all(p.id and p.text for p in loaded.passages)


def test_index_load_reports_an_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="no corpus.jsonl"):
        Index.load(tmp_path)


def test_citation_prefers_page_then_section():
    with_page = Passage(id="a", text="x", source="paper.pdf", pages=[4], breadcrumb="Doc > Results")
    without = Passage(id="b", text="x", source="notes.md", breadcrumb="Doc > Method")
    assert with_page.citation() == "paper.pdf p4"
    assert without.citation() == "notes.md Method"


# ------------------------------------------------------------------ answering


def test_answer_cites_the_passages_it_used(runtime, index):
    found = answer(runtime, index, "How many experts does SparseRoute select?")
    assert not found.unanswerable
    assert "[1]" in found.text
    assert found.citations()


def test_answer_refuses_when_the_corpus_cannot_support_it(runtime, index):
    found = answer(runtime, index, "What is the company dividend policy for shareholders?")
    assert found.unanswerable


def test_refusal_needs_the_marker_to_be_the_whole_answer():
    from pdfqa.retrieve import is_refusal

    assert is_refusal("UNANSWERABLE")
    assert is_refusal("  unanswerable.  ")
    assert not is_refusal("The passages do not say UNANSWERABLE is required, they report 74.8.")
    assert not is_refusal("Reply with exactly UNANSWERABLE if the passages fall short; here they do not.")


def test_ask_command_prints_answer_and_sources(dataset, capsys):
    assert main(["ask", "How many experts does SparseRoute select?", dataset.outdir]) == 0
    out = capsys.readouterr().out
    assert "sources:" in out


def test_ask_command_json_mode(dataset, capsys):
    assert main(["ask", "How many experts does SparseRoute select?", dataset.outdir, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"] and "answer" in payload


# ------------------------------------------------------------------ grading


def test_token_f1_rewards_overlap():
    assert token_f1("four of 32 experts", "four of 32 experts") == pytest.approx(1.0)
    assert token_f1("four of 32 experts", "completely unrelated wording") == 0.0
    assert 0.0 < token_f1("four of 32 experts per token", "four of 32 experts") < 1.0


def test_numeric_match_is_all_or_nothing_per_number():
    assert numeric_match("accuracy is 74.8", "we measured 74.8") == 1.0
    assert numeric_match("accuracy is 74.8", "we measured 72.9") == 0.0
    assert numeric_match("no numbers here", "still none") == 1.0
    assert numeric_match("61.2 and 74.8", "only 74.8") == 0.5


def test_grade_downgrades_a_correct_verdict_that_lost_a_number(runtime):
    rec = QARecord(question="What accuracy?", answer="It reaches 74.8 accuracy.", context="c", prov=Provenance(source="s"))
    graded = grade_one(runtime, rec, "It reaches high accuracy on the benchmark.")
    assert graded.verdict in ("partial", "incorrect")
    assert graded.numeric_match == 0.0


def test_grade_accepts_a_matching_answer(runtime):
    rec = QARecord(question="What accuracy?", answer="It reaches 74.8 accuracy.", context="c", prov=Provenance(source="s"))
    graded = grade_one(runtime, rec, "The reported figure is 74.8 accuracy.")
    assert graded.verdict == "correct"
    assert graded.numeric_match == 1.0


def test_empty_prediction_is_incorrect_without_calling_the_judge(runtime):
    rec = QARecord(question="q", answer="It reaches 74.8 accuracy.", context="c", prov=Provenance(source="s"))
    before = len(runtime.calls)
    graded = grade_one(runtime, rec, "   ")
    assert graded.verdict == "incorrect" and graded.detail == "empty prediction"
    assert len(runtime.calls) == before


# ------------------------------------------------------------------ eval modes


def test_context_mode_grades_every_record(runtime, dataset):
    records = load_records(dataset.outdir)
    summary = evaluate(runtime, records, mode="context", limit=3)
    assert summary["count"] == min(3, len(records))
    assert set(summary["verdicts"]) <= {"correct", "partial", "incorrect"}
    assert "by_task" in summary


def test_retrieval_mode_reports_recall(runtime, dataset):
    records = load_records(dataset.outdir)
    index = Index.load(dataset.outdir)
    summary = evaluate(runtime, records, index, mode="retrieval", limit=3)
    assert "retrieval_recall" in summary
    assert all(r["citations"] is not None for r in summary["results"])


def test_retrieval_mode_requires_an_index(runtime, dataset):
    with pytest.raises(ValueError, match="needs an index"):
        evaluate(runtime, load_records(dataset.outdir), None, mode="retrieval", limit=1)


def test_closed_book_mode_scores_worse_than_context(runtime, dataset):
    records = load_records(dataset.outdir)
    with_context = evaluate(runtime, records, mode="context", limit=4)
    closed = evaluate(runtime, records, mode="closed", limit=4)
    assert closed["accuracy"] <= with_context["accuracy"]


def test_eval_command_prints_a_summary(dataset, capsys):
    assert main(["eval", dataset.outdir, "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "accuracy" in out and "by task" in out


def test_eval_command_on_an_empty_directory_fails_cleanly(tmp_path, capsys):
    assert main(["eval", str(tmp_path)]) == 1
    assert "no raw.jsonl" in capsys.readouterr().err


# ------------------------------------------------------------------ retrieval at corpus scale


def prose_passages(n: int, seed: int = 3):
    """Passages cut from this repository's own prose, so term frequencies are realistic.

    A synthetic vocabulary makes this measurement meaningless in both directions: uniform terms
    invent a bottleneck that real text does not have, and a hand-made Zipf curve invents stop words
    the STOP list would already have removed.
    """
    words = re.findall(
        r"[A-Za-z][A-Za-z'-]+",
        "\n".join(f.read_text(errors="ignore") for f in sorted(Path(__file__).resolve().parent.parent.rglob("*.py")) if ".git" not in str(f)),
    )
    rng = random.Random(seed)
    passages, queries = [], []
    for i in range(n):
        start = rng.randrange(0, max(1, len(words) - 130))
        passages.append(Passage(id=f"p{i}", text=" ".join(words[start : start + 120]), source="d.pdf"))
    for _ in range(20):
        start = rng.randrange(0, max(1, len(words) - 12))
        queries.append(" ".join(words[start : start + 9]))
    return passages, queries


def test_bucketed_search_returns_what_the_exact_scan_returns():
    """The fast path is approximate, so it is only worth having if it agrees with the slow one."""
    passages, queries = prose_passages(EXACT_LIMIT + 2000)
    index = Index(passages)
    index.embed()
    assert index.dense_index is not None, "corpus large enough should use the bucketed path"

    approximate = [[h.passage.id for h in index.search(q, k=5)] for q in queries]
    index.dense_index = None
    exact = [[h.passage.id for h in index.search(q, k=5)] for q in queries]

    found = sum(len(set(a) & set(e)) for a, e in zip(approximate, exact))
    total = sum(len(e) for e in exact)
    assert total and found / total >= 0.95, f"recall@5 fell to {found}/{total}"


def test_a_small_corpus_still_scans_exactly():
    passages, _ = prose_passages(50)
    index = Index(passages)
    index.embed()
    assert index.dense_index is None


def test_near_universal_terms_do_not_drag_in_every_passage():
    """A term in almost every passage has an IDF of about zero, so it changes no ranking — but it
    used to make every passage a candidate, and each candidate costs a cosine and a Hit."""
    passages = [Passage(id=f"p{i}", text=f"boilerplate header everywhere token{i} unique{i}", source="d.pdf")
                for i in range(COMMON_TERM_FLOOR + 500)]
    index = Index(passages)

    assert len(index.bm25("boilerplate header everywhere")) == len(passages)
    narrowed = index.bm25("boilerplate header everywhere token7")
    assert len(narrowed) < len(passages) / 10
    assert index.passages[7].id in {index.passages[i].id for i in narrowed}


def test_a_query_of_only_common_terms_still_retrieves():
    """Dropping the common terms unconditionally would return nothing for such a query."""
    passages = [Passage(id=f"p{i}", text=f"boilerplate header everywhere token{i}", source="d.pdf")
                for i in range(COMMON_TERM_FLOOR + 500)]
    index = Index(passages)
    assert index.bm25("boilerplate header everywhere")
    assert index.search("boilerplate header everywhere", k=3)
