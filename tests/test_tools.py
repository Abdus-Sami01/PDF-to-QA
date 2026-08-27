import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pdfqa.records import Provenance, QARecord, Turn
from pdfqa.synth import repair_trace
from pdfqa.tools import execute, observations_agree, run_lookup, run_sql
from pdfqa.verify import trace_execution, turn_coherence, z3_available, z3_solve

GRID = [
    ["Model", "Params", "Accuracy", "Latency (ms)"],
    ["Dense baseline", "7.0B", "61.2", "340"],
    ["SparseRoute k=4", "7.0B", "74.8", "210"],
    ["SparseRoute k=8", "7.0B", "75.1", "295"],
]

CTX = "SparseRoute selects 4 of 32 experts per token and reaches 74.8 accuracy on the held-out split."


def react(trace, answer="It improves accuracy by 13.6 points.", context=CTX, grids=None):
    return QARecord(
        question="What is the accuracy gain over the dense baseline?",
        answer=answer,
        context=context,
        task="react",
        tool_trace=trace,
        tool_env={"grids": grids if grids is not None else [GRID]},
        prov=Provenance(source="sample.md"),
    )


def test_sql_runs_over_an_extracted_table():
    result = run_sql("SELECT model, accuracy FROM t ORDER BY accuracy DESC LIMIT 1", [GRID])
    assert result.ok and "SparseRoute k=8" in result.output and "75.1" in result.output


def test_sql_column_names_are_sanitised():
    result = run_sql("SELECT latency_ms FROM t WHERE model = 'Dense baseline'", [GRID])
    assert result.ok and result.output.strip() == "340"


def test_sql_values_are_typed_for_arithmetic():
    result = run_sql("SELECT max(accuracy) - min(accuracy) FROM t", [GRID])
    assert result.ok and abs(float(result.output.strip()) - 13.9) < 1e-6


def test_sql_rejects_writes_and_pragmas():
    assert not run_sql("DROP TABLE t", [GRID]).ok
    assert not run_sql("SELECT * FROM t; PRAGMA table_info(t)", [GRID]).ok
    assert not run_sql("ATTACH DATABASE '/etc/passwd' AS x", [GRID]).ok


def test_sql_without_a_table_fails_cleanly():
    result = run_sql("SELECT 1 FROM t", [])
    assert not result.ok and "no table" in result.output


def test_lookup_returns_the_surrounding_passage():
    result = run_lookup("32 experts", CTX)
    assert result.ok and "SparseRoute" in result.output
    assert run_lookup("quarterly revenue", CTX).output == "(not found in source)"


def test_execute_dispatches_python():
    result = execute("python", "print(74.8 - 61.2)")
    assert result.ok and result.output.startswith("13.5")
    assert not execute("browser", "http://x").ok


def test_observations_agree_uses_numeric_tolerance():
    assert observations_agree("13.6", "13.599999999999994")
    assert not observations_agree("13.6", "12.1")
    assert observations_agree("no rows", "(no rows)")


def test_trace_execution_accepts_a_truthful_trace():
    rec = react([{"thought": "subtract", "action": "python", "action_input": "print(74.8 - 61.2)", "observation": "13.6"}])
    gate = trace_execution(rec)
    assert gate.passed and gate.score == 1.0
    assert rec.tool_trace[0]["executed_ok"] is True


def test_trace_execution_catches_a_fabricated_observation():
    rec = react([{"thought": "subtract", "action": "python", "action_input": "print(74.8 - 61.2)", "observation": "21.4"}])
    gate = trace_execution(rec)
    assert not gate.passed
    assert "claimed" in gate.detail


def test_trace_execution_catches_a_broken_sql_action():
    rec = react([{"thought": "query", "action": "sql", "action_input": "SELECT nope FROM t", "observation": "74.8"}])
    gate = trace_execution(rec)
    assert not gate.passed and "failed" in gate.detail


def test_repair_trace_substitutes_the_real_observation():
    rec = react([{"thought": "subtract", "action": "python", "action_input": "print(74.8 - 61.2)", "observation": "21.4"}])
    repair_trace(rec)
    assert rec.tool_trace[0]["observation_claimed"] == "21.4"
    assert rec.tool_trace[0]["observation"].startswith("13.5")
    assert trace_execution(rec).passed
    assert any(v.get("stage") == "trace_repair" for v in rec.prov.verification)


def test_repair_leaves_failing_steps_alone_so_the_gate_still_rejects():
    rec = react([{"thought": "query", "action": "sql", "action_input": "SELECT nope FROM t", "observation": "74.8"}])
    repair_trace(rec)
    assert rec.tool_trace[0]["observation"] == "74.8"
    assert not trace_execution(rec).passed


def test_trace_gate_can_be_disabled():
    rec = react([{"thought": "x", "action": "python", "action_input": "print(1)", "observation": "999"}])
    assert trace_execution(rec, allow_exec=False).passed


def dialogue(turns, context=CTX):
    return QARecord(question=turns[0][1], answer=turns[-1][1], context=context, task="multiturn",
                    turns=[Turn(r, c) for r, c in turns], prov=Provenance(source="sample.md"))


def test_turn_coherence_accepts_a_normal_dialogue():
    rec = dialogue([
        ("user", "How many experts does SparseRoute select?"),
        ("assistant", "It selects 4 of 32 experts per token."),
        ("user", "So all 32 are active?"),
        ("assistant", "No, the source says 4 of 32 experts are selected per token."),
    ])
    assert turn_coherence(rec).passed


def test_turn_coherence_rejects_broken_alternation_and_empty_turns():
    assert not turn_coherence(dialogue([("assistant", "Hello there friend."), ("user", "Hi.")])).passed
    assert not turn_coherence(dialogue([
        ("user", "How many experts?"), ("assistant", "Four."), ("assistant", "   "),
    ])).passed


def test_turn_coherence_rejects_a_dialogue_that_drifts_off_source():
    rec = dialogue([
        ("user", "How many experts does SparseRoute select?"),
        ("assistant", "Quarterly revenue grew sharply across the European segment last year."),
    ])
    assert not turn_coherence(rec).passed


def test_synthesis_is_checkpointed(config):
    from pdfqa.pipeline import Pipeline

    first = Pipeline(config)
    first.run()
    assert first.store.misses > 0
    second = Pipeline(config)
    second.run()
    assert any("synth" in name for name in second.store.stats()["entries"])
    assert second.runtime.calls == [] or len(second.runtime.calls) < len(first.runtime.calls)


# ------------------------------------------------------------------ dialogue grounding


def test_generated_text_covers_every_assistant_turn():
    from pdfqa.verify import generated_text

    rec = dialogue([
        ("user", "What does the router select?"),
        ("assistant", "It keeps 4 of 32 experts per token at temperature 0.7."),
        ("user", "So all 32 are active?"),
        ("assistant", "No."),
    ])
    text = generated_text(rec)
    assert "temperature 0.7" in text and "No." in text
    single = QARecord(question="How many?", answer="Four of 32.", context=CTX, prov=Provenance(source="s"))
    assert generated_text(single) == "Four of 32."


def test_grounding_judges_the_whole_dialogue_not_the_last_line():
    """The final turn is often a one-word correction; judging only that rejects sound data."""
    from pdfqa.verify import lexical_grounding, numeric_grounding

    context = "SparseRoute keeps 4 of 32 experts per token and the routing temperature is 0.7."
    rec = dialogue([
        ("user", "What does SparseRoute select?"),
        ("assistant", "SparseRoute keeps 4 of 32 experts per token, routing temperature 0.7."),
        ("user", "Every expert then?"),
        ("assistant", "No."),
    ], context=context)
    assert lexical_grounding(rec).passed
    assert numeric_grounding(rec).passed


def test_a_dialogue_that_drifts_is_still_rejected():
    from pdfqa.verify import lexical_grounding

    rec = dialogue([
        ("user", "What does SparseRoute select?"),
        ("assistant", "Quarterly revenue grew across the European segment last year."),
        ("user", "Really?"),
        ("assistant", "Dividends were also raised."),
    ], context="SparseRoute keeps 4 of 32 experts per token.")
    assert not lexical_grounding(rec).passed


def test_multiturn_records_survive_verification(runtime, tree):
    """Regression: every dialogue was rejected, so the shape cost model calls and produced nothing."""
    from pdfqa.chunking import chunk_tree
    from pdfqa.synth import generate_multiturn
    from pdfqa.verify import verify

    accepted = 0
    for chunk in chunk_tree(tree, max_tokens=300):
        record = generate_multiturn(runtime, chunk)
        if record is not None:
            verify(record, runtime)
            accepted += record.accepted
    assert accepted > 0


def test_numeric_grounding_credits_a_verified_tool_computation():
    """A ReAct trace exists to derive figures the text does not contain."""
    from pdfqa.verify import numeric_grounding, verify

    rec = react(
        [{"thought": "subtract", "action": "python", "action_input": "print(74.8 - 61.2)", "observation": "13.6"}],
        answer="The routed variant improves accuracy by 13.6 points.",
        context="SparseRoute reaches 74.8 accuracy against the dense baseline's 61.2.",
    )
    assert not numeric_grounding(rec).passed, "13.6 is absent from the source, as expected"

    verify(rec, runtime=None, use_model_gates=False)
    assert rec.scores["numeric_grounding"] == 1.0
    assert "reject:numeric_grounding" not in rec.flags


def test_a_number_no_tool_produced_is_still_rejected():
    from pdfqa.verify import verify

    rec = react(
        [{"thought": "subtract", "action": "python", "action_input": "print(74.8 - 61.2)", "observation": "13.6"}],
        answer="The routed variant improves accuracy by 99.9 points.",
        context="SparseRoute reaches 74.8 accuracy against the dense baseline's 61.2.",
    )
    verify(rec, runtime=None, use_model_gates=False)
    assert "reject:numeric_grounding" in rec.flags


# ------------------------------------------------------------------ solver thread safety


@pytest.mark.skipif(not z3_available(), reason="z3 not installed")
def test_every_solver_call_runs_on_the_one_solver_thread(monkeypatch):
    """The invariant behind the fix, checked directly because the bug it prevents is probabilistic.

    Holding a lock around the solver still let it be destroyed later on whichever thread the
    collector was on, which aborts the process instead of raising — an intermittent SIGSEGV, about
    one run in three over a 120-document corpus. Creating and destroying every z3 object on one
    dedicated thread is what fixes it, so that is what this asserts.
    """
    import threading as th

    from pdfqa import verify as V

    seen: list[str] = []
    real = V._z3_solve_here

    def spy(constraints, timeout):
        seen.append(th.current_thread().name)
        return real(constraints, timeout)

    monkeypatch.setattr(V, "_z3_solve_here", spy)

    def work(k):
        V.z3_solve(f"(declare-const x{k} Real)(assert (> x{k} {k}.5))")

    threads = [th.Thread(target=work, args=(k,), name=f"caller-{k}") for k in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert len(seen) == 6
    assert len(set(seen)) == 1, f"solver ran on several threads: {sorted(set(seen))}"
    assert seen[0].startswith("pdfqa-z3")
    assert not seen[0].startswith("caller-")


@pytest.mark.skipif(not z3_available(), reason="z3 not installed")
def test_the_solver_still_decides_correctly():
    assert z3_solve("(declare-const x Real)(assert (> x 3.0))")[0] == "sat"
    assert z3_solve("(declare-const x Real)(assert (> x 3.0))(assert (< x 1.0))")[0] == "unsat"
    assert z3_solve("this is not smt-lib at all")[0] == "error"
