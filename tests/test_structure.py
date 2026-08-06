from pdfqa.chunking import chunk_tree
from pdfqa.docast import EQUATION, HEADING, TABLE, DocumentTree
from pdfqa.graph import build_graph
from pdfqa.select import LSHIndex, dedup_lexical, jaccard_est


def test_headings_nest_by_level(tree):
    intro = next(n for n in tree.nodes([HEADING]) if n.text.startswith("1 Introduction"))
    router = next(n for n in tree.nodes([HEADING]) if n.text.startswith("2.1"))
    assert router.breadcrumb() == ["Sparse Routing for Long-Context Retrieval", "2 Method"]
    assert intro.level == 2 and router.level == 3


def test_table_and_equation_typed(tree):
    table = tree.nodes([TABLE])[0]
    assert table.attrs["n_cols"] == 4
    assert "<table>" in table.attrs["html"]
    assert table.attrs["grid"][0][0] == "Model"
    eq = tree.nodes([EQUATION])[0]
    assert eq.attrs["number"] == "1"
    assert eq.attrs["balanced"] is True


def test_references_bind_to_targets(tree):
    kinds = {r.kind for r in tree.references}
    assert {"section", "table", "citation"} <= kinds
    section_refs = [r for r in tree.references if r.kind == "section" and r.key == "2.1"]
    assert section_refs and section_refs[0].resolved
    target = tree.index[section_refs[0].target_id]
    assert target.text.startswith("2.1")


def test_resolve_context_inlines_target(tree):
    node = next(n for n in tree.root.walk() if "Section 2.1" in n.text)
    resolved = tree.resolve_context(node)
    assert "Section 2.1" in resolved and "->" in resolved


def test_tree_roundtrip(tree):
    clone = DocumentTree.from_dict(tree.as_dict())
    assert len(clone.index) == len(tree.index)
    assert clone.nodes([TABLE])[0].attrs["grid"] == tree.nodes([TABLE])[0].attrs["grid"]


def test_chunks_carry_provenance_and_keep_tables_whole(tree):
    chunks = chunk_tree(tree, max_tokens=300)
    assert chunks
    for c in chunks:
        assert c.prov.source == "sample.md"
        assert c.prov.node_ids and c.prov.breadcrumb
    table_chunks = [c for c in chunks if c.tables]
    assert len(table_chunks) == 1
    assert "SparseRoute k=4" in table_chunks[0].tables[0]


def test_chunk_respects_token_budget(tree):
    chunks = chunk_tree(tree, max_tokens=200, min_tokens=10)
    oversized = [c for c in chunks if c.tokens > 200 and not c.tables and not c.equations]
    assert not oversized


def test_graph_links_across_chunks(tree, runtime):
    chunks = chunk_tree(tree, max_tokens=300)
    kg = build_graph(chunks, runtime)
    assert kg.stats()["entities"] > 10
    assert kg.paths("SparseRoute", "RetrievalBench", max_hops=3)
    pairs = kg.bridging_pairs()
    assert pairs
    a, b, path = pairs[0]
    assert not (a.chunk_ids & b.chunk_ids)


def test_minhash_finds_near_duplicates():
    a = "the router selects four of thirty two experts per token in every layer"
    b = "the router selects four of thirty two experts per token in each layer"
    index = LSHIndex()
    index.add("a", a)
    index.add("b", b)
    assert jaccard_est(index.signatures["a"], index.signatures["b"]) > 0.5
    kept, dropped = dedup_lexical([a, b, "an entirely different sentence about latency budgets"], lambda x: x, threshold=0.5)
    assert len(kept) == 2 and len(dropped) == 1
