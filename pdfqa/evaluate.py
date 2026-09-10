"""Grade a model against a generated test split, optionally through the retrieval index.

The gates that accepted a row are reused to judge answers about it, so "the dataset says X is
grounded" and "the model's answer is graded against X" rest on the same machinery rather than two
unrelated notions of correct.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .llm import Runtime, parse_json
from .prompts import GRADE_ANSWER
from .records import QARecord
from .retrieve import Index, answer as retrieve_answer
from .verify import numbers_in, tokens

VERDICTS = ("correct", "partial", "incorrect")


@dataclass
class Graded:
    record: QARecord
    predicted: str
    verdict: str = "incorrect"
    token_f1: float = 0.0
    numeric_match: float = 0.0
    retrieved_correct_source: bool | None = None
    citations: list[str] = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.record.id,
            "question": self.record.question,
            "reference": self.record.answer,
            "predicted": self.predicted,
            "verdict": self.verdict,
            "token_f1": round(self.token_f1, 4),
            "numeric_match": round(self.numeric_match, 4),
            "retrieval_hit": self.retrieved_correct_source,
            "citations": self.citations,
            "task": self.record.task,
            "source": self.record.prov.source,
            "detail": self.detail[:300],
        }


def token_f1(reference: str, candidate: str) -> float:
    ref, cand = Counter(tokens(reference)), Counter(tokens(candidate))
    if not ref or not cand:
        return 0.0
    overlap = sum((ref & cand).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(cand.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def numeric_match(reference: str, candidate: str, tolerance: float = 0.005) -> float:
    """Quantitative answers live or die on their numbers, so score those separately from wording."""
    ref = numbers_in(reference)
    if not ref:
        return 1.0
    cand = numbers_in(candidate)
    matched = sum(1 for r in ref if any(abs(r - c) <= max(0.01, abs(r) * tolerance) for c in cand))
    return matched / len(ref)


def grade_one(runtime: Runtime, rec: QARecord, predicted: str) -> Graded:
    graded = Graded(record=rec, predicted=predicted)
    graded.token_f1 = token_f1(rec.answer, predicted)
    graded.numeric_match = numeric_match(rec.answer, predicted)
    if not predicted.strip():
        graded.detail = "empty prediction"
        return graded
    data = parse_json(
        runtime.complete(
            "verify",
            GRADE_ANSWER.format(question=rec.question[:2000], reference=rec.answer[:2000], candidate=predicted[:2000]),
            temperature=0.0,
            max_tokens=600,
        ),
        default={},
    ) or {}
    verdict = str(data.get("verdict", "incorrect")).lower()
    graded.verdict = verdict if verdict in VERDICTS else "incorrect"
    if graded.verdict == "correct" and graded.numeric_match < 1.0:
        graded.verdict = "partial"
        graded.detail = "judge said correct but a reference number is missing from the answer; "
    graded.detail += f"missing={data.get('missing', [])[:2]} wrong={data.get('wrong', [])[:2]}"
    return graded


CLOSED_BOOK = "Answer the question from your own knowledge. If you do not know, reply exactly UNKNOWN.\n\nQuestion: {question}\n\nAnswer:"


def evaluate(
    runtime: Runtime,
    records: list[QARecord],
    index: Index | None = None,
    mode: str = "context",
    k: int = 5,
    limit: int | None = None,
) -> dict:
    """mode: `context` gives the model the gold passage, `retrieval` makes it find one, `closed` gives it nothing."""
    subset = records[:limit] if limit else records
    graded: list[Graded] = []

    for rec in subset:
        if mode == "closed":
            predicted = runtime.complete("verify", CLOSED_BOOK.format(question=rec.question), temperature=0.0, max_tokens=700).strip()
            result = grade_one(runtime, rec, predicted)
        elif mode == "retrieval":
            if index is None:
                raise ValueError("retrieval mode needs an index")
            found = retrieve_answer(runtime, index, rec.question, k)
            result = grade_one(runtime, rec, found.text)
            result.citations = found.citations()
            result.retrieved_correct_source = any(h.passage.source == rec.prov.source for h in found.hits)
        else:
            prompt = f"Answer using only this context.\n\nContext:\n---\n{rec.context[:9000]}\n---\n\nQuestion: {rec.question}\n\nAnswer:"
            predicted = runtime.complete("verify", prompt, temperature=0.0, max_tokens=700).strip()
            result = grade_one(runtime, rec, predicted)
        graded.append(result)

    return summarize(graded, mode)


def summarize(graded: list[Graded], mode: str) -> dict:
    n = max(1, len(graded))
    verdicts = Counter(g.verdict for g in graded)
    by_task: dict[str, Counter] = defaultdict(Counter)
    for g in graded:
        by_task[g.record.task][g.verdict] += 1

    retrieval = [g.retrieved_correct_source for g in graded if g.retrieved_correct_source is not None]
    summary = {
        "mode": mode,
        "count": len(graded),
        "accuracy": round(verdicts["correct"] / n, 4),
        "partial_rate": round(verdicts["partial"] / n, 4),
        "verdicts": dict(verdicts),
        "token_f1": round(sum(g.token_f1 for g in graded) / n, 4),
        "numeric_match": round(sum(g.numeric_match for g in graded) / n, 4),
        "by_task": {task: dict(counts) for task, counts in sorted(by_task.items())},
        "results": [g.as_dict() for g in graded],
    }
    if retrieval:
        summary["retrieval_recall"] = round(sum(retrieval) / len(retrieval), 4)
    return summary
