import json
from pathlib import Path

from pdfqa.pipeline import Pipeline
from pdfqa.records import Provenance, QARecord
from pdfqa.select import dpp_select, embed_records
from pdfqa.verify import (
    lexical_grounding,
    numeric_grounding,
    run_python,
    self_consistency,
    shortcut_leakage,
    standalone_question,
    verify,
)

CTX = "SparseRoute selects 4 of 32 experts per token and reaches 74.8 accuracy on the held-out split."


def rec(q, a, ctx=CTX):
    return QARecord(question=q, answer=a, context=ctx, prov=Provenance(source="sample.md"))


def test_numeric_grounding_catches_hallucinated_number():
    good = numeric_grounding(rec("What accuracy is reported?", "It reaches 74.8 accuracy."))
    bad = numeric_grounding(rec("What accuracy is reported?", "It reaches 84.8 accuracy."))
    assert good.passed and not bad.passed
    assert "84.8" in bad.detail


def test_shortcut_leakage_flags_answer_inside_question():
    leaked = shortcut_leakage(rec("Does SparseRoute reach 74.8 accuracy on the held-out split?", "74.8"))
    clean = shortcut_leakage(rec("How well does the routed model perform against the dense baseline?",
                                 "It reaches 74.8 accuracy on the held-out split."))
    assert not leaked.passed and clean.passed


def test_standalone_question_rejects_deixis():
    assert not standalone_question(rec("What does this passage say about routing?", "Four experts.")).passed
    assert standalone_question(rec("How many experts does SparseRoute select?", "Four of 32.")).passed


def test_lexical_grounding_rejects_off_topic_answer():
    assert not lexical_grounding(rec("How many experts?", "Quarterly revenue grew across the European segment.")).passed


def test_sandbox_blocks_side_effects():
    ok, out = run_python("print('PASS')")
    assert ok and "PASS" in out
    blocked, msg = run_python("import os\nos.system('echo hi')")
    assert not blocked and "blocked" in msg


def test_verify_records_gate_log(runtime):
    r = verify(rec("How many experts does SparseRoute select per token?",
                   "It selects 4 of 32 experts per token."), runtime)
    stages = [v["stage"] for v in r.prov.verification]
    assert {"structural", "nli_forward", "nli_reverse", "clarity", "symbolic"} <= set(stages)
    assert r.accepted and r.scores["quality"] > 0.6


def test_verify_rejects_ungrounded_answer(runtime):
    r = verify(rec("How many experts does SparseRoute select per token?",
                   "It selects 9 of 41 experts per token."), runtime)
    assert not r.accepted
    assert any(f == "reject:numeric_grounding" for f in r.flags)


def test_self_consistency_uses_verify_role(runtime):
    gate = self_consistency(runtime, rec("How many experts?", "Four of 32."), samples=2)
    assert gate.name == "self_consistency"


def test_dpp_prefers_diverse_high_quality(runtime):
    records = [
        rec("How many experts does the router select?", "Four of 32."),
        rec("How many experts does the router pick?", "Four of 32."),
        rec("What is the reported latency of the dense baseline?", "340 ms."),
    ]
    for i, r in enumerate(records):
        r.scores["quality"] = 0.9 if i != 1 else 0.5
    vectors = embed_records(runtime, records)
    picked = dpp_select(records, vectors, k=2)
    assert len(picked) == 2
    assert any("latency" in p.question for p in picked)


def test_full_run_produces_verified_dataset(config):
    report = Pipeline(config).run()
    assert report["stats"]["count"] > 0
    out = Path(config.outdir)
    chatml = [json.loads(l) for l in (out / "chatml.jsonl").read_text().splitlines()]
    assert chatml and chatml[0]["messages"][0]["role"] == "system"
    assert chatml[0]["meta"]["source"] == "sample.md"
    assert chatml[0]["meta"]["node_ids"]
    dpo = [json.loads(l) for l in (out / "dpo.jsonl").read_text().splitlines()]
    assert dpo and dpo[0]["chosen"] != dpo[0]["rejected"]
    assert (out / "DATASET_CARD.md").exists()
    tasks = report["stats"]["task"]
    assert "qa" in tasks


def test_second_run_hits_the_cache(config):
    Pipeline(config).run()
    second = Pipeline(config)
    second.run()
    assert second.store.hits > 0
