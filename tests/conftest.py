import json
import re
from pathlib import Path

import pytest

from pdfqa.config import Config
from pdfqa.extract import from_markdown
from pdfqa.llm import Runtime

FIXTURE = Path(__file__).parent / "fixtures" / "sample.md"

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _context(prompt: str) -> str:
    parts = prompt.split("---")
    return parts[1] if len(parts) > 2 else prompt


def scripted(prompt: str, system: str = "") -> str:
    """Deterministic stand-in for a real model: returns schema-valid JSON grounded in the prompt."""
    ctx = _context(prompt)
    nums = NUM.findall(ctx)
    first = nums[0] if nums else "4"

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
    if "multi-hop questions" in prompt:
        return json.dumps(
            {"pairs": [{"question": "How does the router configuration in the method section relate to the reported accuracy?",
                        "answer": f"The router keeps 4 experts and that configuration reaches {first} on the held-out split.",
                        "hops": 2, "evidence_a": "k = 4", "evidence_b": first}]}
        )
    if "multi-turn conversation" in prompt:
        return json.dumps(
            {"turns": [
                {"role": "user", "content": "What routing configuration does SparseRoute use by default?"},
                {"role": "assistant", "content": "It keeps 4 of 32 experts per token, with routing temperature 0.7."},
                {"role": "user", "content": "So it activates all 32 of them?"},
                {"role": "assistant", "content": "No — the source states 4 of 32 experts are selected per token."},
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


@pytest.fixture
def runtime() -> Runtime:
    return Runtime({"generate": {"backend": "echo", "handler": scripted}, "verify": {"backend": "echo", "handler": scripted}})


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.inputs = [str(FIXTURE)]
    cfg.outdir = str(tmp_path / "out")
    cfg.cache_dir = str(tmp_path / "cache")
    cfg.runtime.generate = {"backend": "echo", "handler": scripted}
    cfg.runtime.verify = {"backend": "echo", "handler": scripted}
    cfg.runtime.workers = 1
    cfg.synth.multihop_pairs = 3
    cfg.synth.multiturn_per_doc = 2
    cfg.synth.react_per_doc = 1
    return cfg
