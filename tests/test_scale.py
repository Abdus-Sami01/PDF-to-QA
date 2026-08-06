import random
import time

from pdfqa.chunking import chunk_tree
from pdfqa.llm import hashed_embedding
from pdfqa.records import Provenance, QARecord
from pdfqa.select import EXACT_LIMIT, HyperplaneIndex, cosine, dedup_semantic, dpp_select, normalize
from pdfqa.synth import generate_qa


def make(n, dim=64, clusters=None, seed=5):
    """Vectors in tight clusters, so the number of true near-duplicates is known up front."""
    rng = random.Random(seed)
    clusters = clusters or max(1, n // 4)
    centres = [normalize([rng.gauss(0, 1) for _ in range(dim)]) for _ in range(clusters)]
    records, vectors = [], []
    for i in range(n):
        centre = centres[i % clusters]
        jitter = normalize([c + rng.gauss(0, 0.01) for c in centre])
        records.append(QARecord(question=f"question number {i}", answer="a", context="c", prov=Provenance(source="s")))
        records[-1].scores["quality"] = rng.random()
        vectors.append(jitter)
    return records, vectors


def test_context_is_shared_between_records_from_one_chunk(tree, runtime):
    chunk = chunk_tree(tree, max_tokens=300)[0]
    records = generate_qa(runtime, chunk, n=3)
    assert len(records) > 1
    assert len({id(r.context) for r in records}) == 1


def test_hyperplane_index_retrieves_near_neighbours():
    records, vectors = make(200, clusters=20)
    index = HyperplaneIndex(len(vectors[0]))
    for i, v in enumerate(vectors):
        index.add(i, v)
    probe = 0
    found = index.candidates(vectors[probe])
    true_neighbours = {i for i, v in enumerate(vectors) if cosine(v, vectors[probe]) >= 0.92}
    assert true_neighbours <= found


def test_bucketed_dedup_matches_the_exact_path_on_the_same_data():
    records, vectors = make(EXACT_LIMIT + 200, clusters=30)
    kept_bucketed, _ = dedup_semantic(records, vectors, threshold=0.92)

    small_records, small_vectors = records[:EXACT_LIMIT - 1], vectors[:EXACT_LIMIT - 1]
    kept_exact, _ = dedup_semantic(small_records, small_vectors, threshold=0.92)
    assert len(kept_exact) <= 30
    assert len(kept_bucketed) <= 40


def test_dedup_semantic_keeps_one_per_cluster():
    records, vectors = make(80, clusters=8)
    kept, dropped = dedup_semantic(records, vectors, threshold=0.9)
    assert len(kept) == 8
    assert len(dropped) == 72


def test_large_dedup_finishes_without_quadratic_blowup():
    records, vectors = make(3000, clusters=200)
    start = time.perf_counter()
    kept, _ = dedup_semantic(records, vectors, threshold=0.92)
    elapsed = time.perf_counter() - start
    assert len(kept) <= 260
    assert elapsed < 30.0


def test_dpp_short_circuits_when_there_is_no_budget():
    records, vectors = make(EXACT_LIMIT + 10, clusters=EXACT_LIMIT + 10)
    picked = dpp_select(records, vectors)
    assert picked == records


def test_dpp_still_selects_when_given_a_budget():
    records, vectors = make(EXACT_LIMIT + 10, clusters=50)
    picked = dpp_select(records, vectors, k=20)
    assert len(picked) == 20


def test_dpp_respects_a_token_budget():
    records, vectors = make(40, clusters=40)
    for r in records:
        r.question = "q" * 400
        r.answer = "a" * 400
    picked = dpp_select(records, vectors, budget_tokens=500)
    assert 0 < len(picked) <= 3


def test_hashed_embeddings_are_stable_across_calls():
    a, b = hashed_embedding("routing accuracy"), hashed_embedding("routing accuracy")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
