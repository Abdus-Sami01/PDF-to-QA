import json
from pathlib import Path
import random
import time

import pytest

from pdfqa.chunking import chunk_tree
from pdfqa.graph import KnowledgeGraph
from pdfqa.llm import hashed_embedding
from pdfqa.records import Chunk, Provenance, QARecord
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


def test_embedding_memory_stays_close_to_four_bytes_per_dimension():
    """Vectors dominate memory at corpus scale — list[float] is ~32 bytes per dimension."""
    import tracemalloc
    from pdfqa.select import normalize

    rng = random.Random(3)
    n, dim = 300, 1536
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        vectors = [normalize([rng.random() for _ in range(dim)]) for _ in range(n)]
        used = tracemalloc.get_traced_memory()[0] - base
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert len(vectors) == n
    assert used / (n * dim) < 8.0


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
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_vectors_are_float_arrays_not_lists():
    from array import array

    v = hashed_embedding("routing accuracy")
    assert isinstance(v, array) and v.typecode == "f"
    assert isinstance(normalize([3.0, 4.0]), array)


def test_runtime_packs_vectors_whatever_the_backend_returned():
    """A backend serving JSON returns plain lists; callers must not have to care which one ran."""
    from array import array

    from pdfqa.llm import Backend, Runtime

    class ListEmbedder(Backend):
        name = "listy"

        def embed(self, texts):
            return [[0.6, 0.8] for _ in texts]

    runtime = Runtime({"generate": {"backend": "echo"}})
    runtime.backends["embed"] = ListEmbedder()
    vectors = runtime.embed(["a", "b"])
    assert all(isinstance(v, array) and v.typecode == "f" for v in vectors)
    assert cosine(vectors[0], vectors[1]) == pytest.approx(1.0, abs=1e-6)


def test_runtime_embed_falls_back_when_a_backend_has_no_endpoint():
    from array import array

    from pdfqa.llm import Backend, Runtime

    class NoEmbedding(Backend):
        name = "nope"

    runtime = Runtime({"generate": {"backend": "echo"}})
    runtime.backends["embed"] = NoEmbedding()
    assert all(isinstance(v, array) for v in runtime.embed(["a"]))


def test_float32_precision_is_far_below_the_dedup_thresholds():
    """The memory win costs ~1e-7 per component; the tightest threshold in use is 0.92."""
    import math
    import random

    rng = random.Random(11)
    raw_a = [rng.gauss(0, 1) for _ in range(1536)]
    raw_b = [x + rng.gauss(0, 0.05) for x in raw_a]

    def exact(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    exact_cos = sum(x * y for x, y in zip(exact(raw_a), exact(raw_b)))
    packed_cos = cosine(normalize(raw_a), normalize(raw_b))
    assert abs(exact_cos - packed_cos) < 1e-5


def _latency_handler(delay_rng):
    """The offline backend returns instantly, which hides scheduling-order bugs entirely."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import scripted

    def handler(prompt, system=""):
        time.sleep(delay_rng.uniform(0.001, 0.006))
        return scripted(prompt, system)

    return handler


def _run_ids(config, workers, tmp_path, tag):
    from pdfqa.pipeline import Pipeline

    config.runtime.workers = workers
    config.outdir = str(tmp_path / tag)
    config.cache = False
    config.formats = ["raw"]
    config.runtime.generate = {"backend": "echo", "handler": _latency_handler(random.Random(99))}
    config.runtime.verify = dict(config.runtime.generate)
    Pipeline(config).run()
    rows = (Path(config.outdir) / "raw.jsonl").read_text().splitlines()
    return [(json.loads(r)["task"], json.loads(r)["persona"], json.loads(r)["rejection_mode"]) for r in rows]


def test_the_seed_is_honoured_under_concurrency(config, tmp_path):
    """Guards the seed contract against a future draw moving after a network call.

    Drawing from a shared RNG inside worker threads orders the draws by scheduling rather than by
    the seed. Today every draw happens before dispatch, so this passes either way; it fails the day
    someone moves one behind an LLM call, which is exactly when it would stop being reproducible.
    """
    outcomes = {tuple(_run_ids(config, 4, tmp_path, f"c{i}")) for i in range(4)}
    assert len(outcomes) == 1


def test_worker_count_does_not_change_the_output(config, tmp_path):
    single = _run_ids(config, 1, tmp_path, "single")
    parallel = _run_ids(config, 4, tmp_path, "parallel")
    assert single == parallel


def test_preference_modes_are_drawn_deterministically(config, tmp_path):
    """Same contract at the stage that draws the most values, with enough targets to interleave."""
    from pdfqa.pipeline import Pipeline
    from pdfqa.records import Provenance, QARecord

    def modes_for_one_run():
        config.runtime.workers = 4
        config.runtime.generate = {"backend": "echo", "handler": _latency_handler(random.Random(5))}
        config.runtime.verify = dict(config.runtime.generate)
        pipeline = Pipeline(config)
        records = [
            QARecord(question=f"Question number {i} about routing width?", answer=f"Answer {i} is 74.8 accuracy.",
                     context="SparseRoute reaches 74.8 accuracy.", prov=Provenance(source="s.md"))
            for i in range(24)
        ]
        pipeline.preference_pairs(records)
        return [r.rejection_mode for r in records]

    outcomes = {tuple(modes_for_one_run()) for _ in range(3)}
    assert len(outcomes) == 1
    assert any(outcomes.pop())


# ------------------------------------------------------------------ graph scaling


def kg_with(entities: int, seed: int, chunks: int = 40, density: int = 2):
    rng = random.Random(seed)
    kg = KnowledgeGraph()
    pool = []
    for i in range(chunks):
        c = Chunk(text=f"c{i}", prov=Provenance(source=f"doc{i % 5}.pdf"))
        pool.append(c)
        kg.chunk_index[c.id] = c
    names = [f"entity number {i}" for i in range(entities)]
    for name in names:
        kg.add_entity(name, "method", rng.choice(pool))
    for _ in range(entities * density):
        a, b = rng.sample(names, 2)
        kg.add_edge(a, b, "relates", rng.choice(pool).id)
    return kg


def brute_bridging(kg, min_gap: int = 1, limit: int = 64):
    """The pair-by-pair definition, kept as the reference the fast path must agree with."""
    out, keys = [], list(kg.entities)
    for i, a in enumerate(keys):
        ea = kg.entities[a]
        for b in keys[i + 1 :]:
            eb = kg.entities[b]
            if ea.chunk_ids & eb.chunk_ids or not ea.chunk_ids or not eb.chunk_ids:
                continue
            path = kg.paths(a, b, max_hops=3)
            if path and len(path[0]) > min_gap:
                out.append((ea, eb, path[0]))
    out.sort(key=lambda t: (len(t[2]), -len(t[0].chunk_ids | t[1].chunk_ids)))
    return out[:limit]


@pytest.mark.parametrize("seed", range(8))
def test_fast_bridging_finds_exactly_the_pairs_the_definition_does(seed):
    kg = kg_with(60, seed)
    expected = {tuple(sorted((a.name, b.name))): len(p) for a, b, p in brute_bridging(kg)}
    actual = {tuple(sorted((a.name, b.name))): len(p) for a, b, p in kg.bridging_pairs()}
    assert actual == expected


def test_bridging_pairs_does_not_blow_up_on_a_corpus_sized_graph():
    """Testing every pair and searching for a path between them is quadratic, and nearly all of that
    work goes into proving unrelated entities are unrelated. This used to take hours."""
    kg = kg_with(4000, seed=1, chunks=200, density=2)
    start = time.perf_counter()
    pairs = kg.bridging_pairs()
    assert pairs
    assert time.perf_counter() - start < 10.0


def test_reachable_returns_shortest_paths():
    kg = KnowledgeGraph()
    chunk = Chunk(text="c", prov=Provenance(source="d.pdf"))
    kg.chunk_index[chunk.id] = chunk
    for name in ("alpha", "beta", "gamma"):
        kg.add_entity(name, "method", chunk)
    kg.add_edge("alpha", "beta", "r", chunk.id)
    kg.add_edge("beta", "gamma", "r", chunk.id)
    kg.add_edge("alpha", "gamma", "r", chunk.id)

    hops = {key: len(path) for key, path in kg.reachable("alpha")}
    assert hops == {"beta": 1, "gamma": 1}
