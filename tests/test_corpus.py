import json
from pathlib import Path

import pytest

from pdfqa.chunking import chunk_tree
from pdfqa.extract import from_markdown
from pdfqa.graph import KnowledgeGraph, build_graph, cross_document_pairs, merge_graphs, source_of
from pdfqa.pipeline import Pipeline
from pdfqa.synth import generate_cross_document, generate_react, table_registry
from pdfqa.tools import describe_schema, run_sql

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tree2():
    return from_markdown((FIXTURES / "sample2.md").read_text(encoding="utf-8"), source="sample2.md")


def kg_for(tree, runtime):
    return build_graph(chunk_tree(tree, max_tokens=300), runtime)


# ------------------------------------------------------------------ canonicalisation


def test_spacing_and_punctuation_variants_merge():
    kg = KnowledgeGraph()
    for name in ("SparseRoute", "Sparse-Route", "sparse route"):
        kg.add_entity(name, "method")
    assert len(kg.entities) == 3
    kg.canonicalise()
    assert len(kg.entities) == 1
    survivor = next(iter(kg.entities.values()))
    assert survivor.aliases


def test_plurals_merge_but_short_words_are_left_alone():
    kg = KnowledgeGraph()
    kg.add_entity("expert blocks", "term")
    kg.add_entity("expert block", "term")
    kg.add_entity("loss", "metric")
    kg.add_entity("los", "term")
    kg.canonicalise()
    keys = set(kg.entities)
    assert len([k for k in keys if "expert" in k]) == 1
    assert "loss" in keys and "los" in keys


def test_acronyms_fold_into_their_expansion():
    kg = KnowledgeGraph()
    kg.add_entity("Sparse Routing Network", "method")
    kg.add_entity("SRN", "method")
    kg.canonicalise()
    assert "srn" not in kg.entities
    assert "SRN" in kg.entities["sparse routing network"].aliases


def test_merging_rewires_edges_and_drops_self_loops():
    kg = KnowledgeGraph()
    kg.add_entity("SparseRoute", "method")
    kg.add_entity("Sparse-Route", "method")
    kg.add_entity("RetrievalBench", "dataset")
    kg.add_edge("Sparse-Route", "RetrievalBench", "evaluated_on")
    kg.add_edge("SparseRoute", "Sparse-Route", "same_as")
    kg.canonicalise()
    assert len(kg.entities) == 2
    survivor = "sparseroute" if "sparseroute" in kg.entities else next(k for k in kg.entities if "sparse" in k)
    assert not any(e.src == e.dst for e in kg.edges)
    assert kg.paths(survivor, "retrievalbench", max_hops=2)


def test_entity_type_survives_a_merge():
    kg = KnowledgeGraph()
    kg.add_entity("RetrievalBench", "term")
    kg.add_entity("retrieval-bench", "dataset")
    kg.canonicalise()
    assert next(iter(kg.entities.values())).type == "dataset"


def test_a_specific_type_upgrades_a_generic_one():
    kg = KnowledgeGraph()
    kg.add_entity("SparseRoute", "term")
    kg.add_entity("SparseRoute", "method")
    assert kg.entities["sparseroute"].type == "method"
    kg.add_entity("SparseRoute", "term")
    assert kg.entities["sparseroute"].type == "method"


def test_aliases_survive_serialisation():
    kg = KnowledgeGraph()
    kg.add_entity("Sparse Routing Network", "method")
    kg.add_entity("SRN", "method")
    kg.canonicalise()
    clone = KnowledgeGraph.from_dict(kg.as_dict())
    assert "SRN" in clone.entities["sparse routing network"].aliases


def test_graph_typing_reaches_the_pipeline_report(tree, runtime):
    kg = kg_for(tree, runtime)
    assert "method" in kg.stats()["by_type"]


# ------------------------------------------------------------------ corpus graph


def test_merge_graphs_unifies_a_shared_entity(tree, tree2, runtime):
    chunks = chunk_tree(tree, max_tokens=300) + chunk_tree(tree2, max_tokens=300)
    corpus = merge_graphs([kg_for(tree, runtime), kg_for(tree2, runtime)], chunks)
    assert corpus.meta["documents"] == 2
    shared = [e for e in corpus.entities.values() if len(source_of(corpus, e)) > 1]
    assert shared, "no entity was found in both documents"
    assert {"sample.md", "sample2.md"} <= {c.prov.source for c in corpus.chunk_index.values()}


def test_cross_document_pairs_span_sources(tree, tree2, runtime):
    chunks = chunk_tree(tree, max_tokens=300) + chunk_tree(tree2, max_tokens=300)
    corpus = merge_graphs([kg_for(tree, runtime), kg_for(tree2, runtime)], chunks)
    pairs = cross_document_pairs(corpus)
    assert pairs
    for a, b, _ in pairs:
        assert len(source_of(corpus, a) | source_of(corpus, b)) > 1


def test_cross_document_records_cite_both_sources(tree, tree2, runtime):
    chunks = chunk_tree(tree, max_tokens=300) + chunk_tree(tree2, max_tokens=300)
    corpus = merge_graphs([kg_for(tree, runtime), kg_for(tree2, runtime)], chunks)
    records = generate_cross_document(runtime, corpus, n_pairs=3)
    assert records
    rec = records[0]
    assert rec.task == "cross_document"
    assert set(rec.prov.sources) == {"sample.md", "sample2.md"}
    entry = next(v for v in rec.prov.verification if v["stage"] == "cross_document")
    assert len(set(entry["sources"])) == 2


def test_cross_document_is_skipped_for_a_single_input(config):
    report = Pipeline(config).run()
    assert "corpus_graph" not in report
    assert report["stats"]["task"].get("cross_document", 0) == 0


def test_two_document_run_produces_cross_document_rows(config):
    config.inputs = [str(FIXTURES / "sample.md"), str(FIXTURES / "sample2.md")]
    report = Pipeline(config).run()
    assert report["corpus_graph"]["documents"] == 2
    rows = [json.loads(l) for l in (Path(config.outdir) / "raw.jsonl").read_text().splitlines()]
    cross = [r for r in rows if r["task"] == "cross_document"]
    assert cross
    assert len(cross[0]["prov"]["sources"]) == 2


# ------------------------------------------------------------------ cross-table sql


def test_table_registry_puts_the_local_table_first(tree):
    chunks = chunk_tree(tree, max_tokens=300)
    local = next(c for c in chunks if c.grids)
    grids, _ = table_registry(chunks, local)
    assert grids[0] == local.grids[0]


def test_schema_description_names_every_table(tree, tree2):
    chunks = chunk_tree(tree, max_tokens=300) + chunk_tree(tree2, max_tokens=300)
    grids, captions = table_registry(chunks)
    schema = describe_schema(grids, captions)
    assert schema.startswith("t(")
    assert "accuracy" in schema
    assert len(schema.splitlines()) == len([g for g in grids if len(g) > 1])


def test_sql_can_join_two_tables_from_different_documents(tree, tree2):
    chunks = chunk_tree(tree, max_tokens=300) + chunk_tree(tree2, max_tokens=300)
    grids, captions = table_registry(chunks)
    names = describe_schema(grids, captions).splitlines()
    second = names[1].split("(")[0]
    result = run_sql(
        f"SELECT t.accuracy, {second}.accuracy FROM t JOIN {second} ON t.model = {second}.model "
        f"WHERE t.model = 'SparseRoute k=4'",
        grids,
        captions=captions,
    )
    assert result.ok and "74.8" in result.output and "72.9" in result.output


def test_react_trace_env_carries_every_document_table(tree, tree2, runtime):
    chunks = chunk_tree(tree, max_tokens=300) + chunk_tree(tree2, max_tokens=300)
    local = next(c for c in chunks if c.grids)
    rec = generate_react(runtime, local, chunks)
    assert rec is not None
    assert len(rec.tool_env["grids"]) >= 2
    assert len(rec.tool_env["captions"]) == len(rec.tool_env["grids"])
