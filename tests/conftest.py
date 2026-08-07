import json
import re
from pathlib import Path

import pytest

from pdfqa.config import Config
from pdfqa.extract import from_markdown
from pdfqa.llm import Runtime

FIXTURE = Path(__file__).parent / "fixtures" / "sample.md"

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i == -1:
        return ""
    j = text.find(end, i + len(start))
    return text[i + len(start) : j if j != -1 else len(text)].strip()


def _context(prompt: str) -> str:
    parts = prompt.split("---")
    return parts[1] if len(parts) > 2 else prompt


def scripted(prompt: str, system: str = "") -> str:
    """Deterministic stand-in for a real model: returns schema-valid JSON grounded in the prompt."""
    ctx = _context(prompt)
    nums = NUM.findall(ctx)
    first = nums[0] if nums else "4"

    if "looking at a figure" in prompt:
        return json.dumps(
            {"pairs": [{"question": "How does accuracy change as SparseRoute widens from 2 to 8 experts?",
                        "answer": "Accuracy rises steeply from 61.2 to 74.8 and then flattens at 75.1.",
                        "visual_evidence": "the fourth bar is barely taller than the third"}]}
        )
    if "supported by the image" in prompt:
        return json.dumps({"label": "entailment", "confidence": 0.88, "unsupported": []})
    if "SMT-LIB" in prompt:
        return json.dumps({"applicable": True,
                           "constraints": "(declare-const a Real)(assert (= a 74.8))",
                           "negation": "(declare-const a Real)(assert (= a 74.8))(assert (not (= a 74.8)))"})
    if prompt.startswith("Answer using only this context"):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if NUM.search(s)]
        return sentences[0][:300] if sentences else "The context does not state that."
    if "numbered passages" in prompt:
        if "routing" in prompt.lower() or "sparseroute" in prompt.lower() or "accuracy" in prompt.lower():
            return "SparseRoute selects 4 of 32 experts per token and reaches 74.8 accuracy [1]."
        return "UNANSWERABLE"
    if "Grade a candidate answer" in prompt:
        ref = _between(prompt, "Reference answer:", "Candidate answer:")
        cand = _between(prompt, "Candidate answer:", "\n\n")
        ref_nums, cand_nums = set(NUM.findall(ref)), set(NUM.findall(cand))
        if ref_nums and not (ref_nums & cand_nums):
            return json.dumps({"verdict": "incorrect", "confidence": 0.9, "missing": sorted(ref_nums), "wrong": []})
        return json.dumps({"verdict": "correct", "confidence": 0.9, "missing": [], "wrong": []})
    if prompt.startswith("Answer the question from your own knowledge"):
        return "UNKNOWN"
    if "knowledge graph" in prompt:
        return json.dumps(
            {
                "entities": [
                    {"name": "SparseRoute", "type": "method", "value": ""},
                    {"name": "RetrievalBench", "type": "dataset", "value": ""},
                    {"name": "accuracy", "type": "metric", "value": first},
                ],
                "relations": [{"source": "SparseRoute", "relation": "evaluated_on", "target": "RetrievalBench"}],
            }
        )
    if "two different documents" in prompt:
        return json.dumps(
            {"pairs": [{"question": "Do the routing study and the latency re-evaluation agree on SparseRoute k=4 accuracy on RetrievalBench?",
                        "answer": "No. The original study reports 74.8 accuracy while the latency re-evaluation measures 72.9, "
                                  "because the second includes router overhead in its end-to-end measurement.",
                        "hops": 2, "evidence_a": "74.8", "evidence_b": "72.9"}]}
        )
    if "multi-hop questions" in prompt:
        return json.dumps(
            {"pairs": [{"question": "How does the router configuration in the method section relate to the reported accuracy?",
                        "answer": f"The router keeps 4 experts and that configuration reaches {first} on the held-out split.",
                        "hops": 2, "evidence_a": "k = 4", "evidence_b": first}]}
        )
    if "multi-turn conversation" in prompt:
        # Quote the context it was actually given. A stub that answers from a fixed script is not
        # grounded in the chunk under test, so the grounding gates reject it and the multi-turn
        # path can never be exercised end to end.
        facts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if NUM.search(s)][:2] or [ctx.strip()[:160]]
        return json.dumps(
            {"turns": [
                {"role": "user", "content": "What does this configuration actually specify?"},
                {"role": "assistant", "content": facts[0][:300]},
                {"role": "user", "content": "So none of those numbers matter in practice?"},
                {"role": "assistant", "content": f"They do. {facts[-1][:260]}"},
            ]}
        )
    if "agentic tool-use" in prompt:
        return json.dumps(
            {"question": "What is the accuracy gain of SparseRoute k=4 over the dense baseline?",
             "trace": [{"thought": "Subtract the two accuracies.", "action": "python",
                        "action_input": "print(74.8 - 61.2)", "observation": "13.6"}],
             "answer": "SparseRoute k=4 improves accuracy by 13.6 points over the dense baseline."}
        )
    if "Rewrite this exchange" in prompt:
        return json.dumps({"question": "In practical terms, what does SparseRoute change?",
                           "answer": "It selects 4 of 32 experts per token and reaches 74.8 accuracy."})
    if "Rewrite the question to be harder" in prompt:
        return json.dumps({"question": "Given the load-balancing coefficient of 0.01, how much accuracy is lost when the term is removed?",
                           "answer": "Accuracy drops from 74.8 to 68.4, a loss of 6.4 points.", "applied": True})
    if "plausible but wrong" in prompt:
        return json.dumps({"rejected": "SparseRoute selects 4 of 32 experts and reaches 84.8 accuracy on the held-out split.",
                           "injected": "changed the accuracy from 74.8 to 84.8"})
    if "premise entails the hypothesis" in prompt:
        return json.dumps({"label": "entailment", "confidence": 0.92, "unsupported": []})
    if "without the source document" in prompt:
        return json.dumps({"answerable_without_source": False, "my_answer": "unknown", "matches_source_answer": False, "confidence": 0.9})
    if "Rate this training pair" in prompt:
        return json.dumps({"standalone": 0.9, "specific": 0.9, "natural": 0.85, "answer_complete": 0.9, "issues": []})
    if "executable Python" in prompt:
        return json.dumps({"applicable": True, "code": "print('PASS')", "expected": "PASS"})
    if "question-answer pairs" in prompt:
        return json.dumps(
            {"pairs": [
                {"question": "How many experts does the SparseRoute router select per token?",
                 "answer": "The router selects 4 of 32 expert blocks per token, with routing temperature 0.7.",
                 "difficulty": "simple", "evidence": "selects 4 of 32 expert blocks per token"},
                {"question": "What accuracy does SparseRoute reach on the RetrievalBench held-out split?",
                 "answer": f"It reaches {first} accuracy on the held-out split of RetrievalBench.",
                 "difficulty": "intermediate", "evidence": first},
                {"question": "What training schedule was used for the reported RetrievalBench results?",
                 "answer": "Training ran for 12 epochs at learning rate 3e-4 with batch size 256 on 8 A100 GPUs.",
                 "difficulty": "intermediate", "evidence": "12 epochs"},
            ]}
        )
    return json.dumps({"pairs": []})


@pytest.fixture
def sample_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def tree(sample_text):
    return from_markdown(sample_text, source="sample.md")


ECHO = {"backend": "echo", "handler": scripted}


@pytest.fixture
def runtime() -> Runtime:
    return Runtime({"generate": dict(ECHO), "verify": dict(ECHO), "vision": dict(ECHO)})


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.inputs = [str(FIXTURE)]
    cfg.outdir = str(tmp_path / "out")
    cfg.cache_dir = str(tmp_path / "cache")
    cfg.runtime.generate = dict(ECHO)
    cfg.runtime.verify = dict(ECHO)
    cfg.runtime.vision = dict(ECHO)
    cfg.runtime.workers = 1
    cfg.synth.multihop_pairs = 3
    cfg.synth.multiturn_per_doc = 2
    cfg.synth.react_per_doc = 1
    return cfg
