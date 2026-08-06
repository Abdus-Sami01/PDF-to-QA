"""Dry-run estimator: what a run will cost before you pay for it.

Parsing and chunking are free, so they are actually performed; everything past that is projected
from the config. Token figures are estimates from real chunk sizes, not guesses at document length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .records import Chunk, estimate_tokens

PROMPT_OVERHEAD = 400
ANSWER_TOKENS = {"qa": 260, "multihop": 320, "cross_document": 340, "multiturn": 700, "react": 600,
                 "figure_qa": 300, "persona": 220, "evol": 260, "dpo": 240, "graph": 350}
VERIFY_TOKENS = 200


@dataclass
class StageCost:
    stage: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    role: str = "generate"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "role": self.role,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class Plan:
    documents: list[dict] = field(default_factory=list)
    stages: list[StageCost] = field(default_factory=list)
    chunks: int = 0

    def add(self, stage: str, calls: int, prompt: int, completion: int, role: str = "generate") -> None:
        if calls > 0:
            self.stages.append(StageCost(stage, calls, prompt, completion, role))

    def totals(self) -> dict:
        return {
            "calls": sum(s.calls for s in self.stages),
            "prompt_tokens": sum(s.prompt_tokens for s in self.stages),
            "completion_tokens": sum(s.completion_tokens for s in self.stages),
        }

    def by_role(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for s in self.stages:
            bucket = out.setdefault(s.role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            bucket["calls"] += s.calls
            bucket["prompt_tokens"] += s.prompt_tokens
            bucket["completion_tokens"] += s.completion_tokens
        return out

    def cost(self, price_in: float, price_out: float) -> float:
        """Prices are per million tokens; the caller supplies them because published rates change."""
        totals = self.totals()
        return (totals["prompt_tokens"] * price_in + totals["completion_tokens"] * price_out) / 1_000_000

    def as_dict(self, price_in: float = 0.0, price_out: float = 0.0) -> dict:
        out = {
            "documents": self.documents,
            "chunks": self.chunks,
            "stages": [s.as_dict() for s in self.stages],
            "totals": self.totals(),
            "by_role": self.by_role(),
        }
        if price_in or price_out:
            out["estimated_cost"] = round(self.cost(price_in, price_out), 4)
            out["price_per_million"] = {"input": price_in, "output": price_out}
        return out


def estimate(config: Config, chunks_by_doc: dict[str, list[Chunk]]) -> Plan:
    plan = Plan()
    s, v = config.synth, config.verify
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    plan.chunks = len(all_chunks)
    for source, chunks in chunks_by_doc.items():
        plan.documents.append({"source": source, "chunks": len(chunks), "tokens": sum(c.tokens for c in chunks)})

    if not all_chunks:
        return plan

    context_tokens = sum(c.tokens for c in all_chunks)
    mean_chunk = context_tokens // max(1, len(all_chunks))
    docs = max(1, len(chunks_by_doc))

    if config.graph.enabled and config.graph.llm_extract:
        n = len(all_chunks) if config.graph.max_chunks is None else min(len(all_chunks), config.graph.max_chunks * docs)
        plan.add("graph", n, n * (mean_chunk + PROMPT_OVERHEAD), n * ANSWER_TOKENS["graph"])

    qa_calls = len(all_chunks)
    plan.add("qa", qa_calls, context_tokens + qa_calls * PROMPT_OVERHEAD, qa_calls * s.qa_per_chunk * ANSWER_TOKENS["qa"])
    records = qa_calls * s.qa_per_chunk

    for stage, per_doc, answer_key, context_multiplier in (
        ("multihop", s.multihop_pairs, "multihop", 2),
        ("multiturn", s.multiturn_per_doc, "multiturn", 1),
        ("react", s.react_per_doc, "react", 1),
    ):
        calls = per_doc * docs
        plan.add(stage, calls, calls * (mean_chunk * context_multiplier + PROMPT_OVERHEAD), calls * ANSWER_TOKENS[answer_key])
        records += calls * (s.multihop_per_pair if stage == "multihop" else 1)

    if s.cross_doc_pairs and docs > 1:
        plan.add("cross_document", s.cross_doc_pairs, s.cross_doc_pairs * (mean_chunk * 2 + PROMPT_OVERHEAD),
                 s.cross_doc_pairs * ANSWER_TOKENS["cross_document"])
        records += s.cross_doc_pairs

    if s.figure_qa_per_doc and config.runtime.vision:
        calls = s.figure_qa_per_doc * docs
        plan.add("figure_qa", calls, calls * (mean_chunk + PROMPT_OVERHEAD), calls * ANSWER_TOKENS["figure_qa"], role="vision")
        records += calls

    base = qa_calls * s.qa_per_chunk + s.multihop_pairs * docs
    for stage, ratio in (("persona", s.persona_ratio), ("evol", s.evol_ratio)):
        calls = int(base * ratio)
        plan.add(stage, calls, calls * (mean_chunk + PROMPT_OVERHEAD), calls * ANSWER_TOKENS[stage])
        records += calls

    dpo_calls = int(records * s.dpo_ratio)
    plan.add("dpo", dpo_calls, dpo_calls * (mean_chunk + PROMPT_OVERHEAD), dpo_calls * ANSWER_TOKENS["dpo"])

    if v.enabled and v.model_gates:
        gates = 3 + (1 if v.symbolic else 0) + (1 if v.z3 else 0) + v.consistency_samples
        calls = records * gates
        plan.add("verify", calls, calls * (mean_chunk + PROMPT_OVERHEAD), calls * VERIFY_TOKENS, role="verify")

    return plan


def collect_chunks(config: Config, paths: list[Path]) -> dict[str, list[Chunk]]:
    from .chunking import chunk_tree
    from .extract import load

    out: dict[str, list[Chunk]] = {}
    for path in paths:
        tree = load(path, config.backend, None, config.figure_dpi)
        out[tree.source] = chunk_tree(tree, config.chunk.max_tokens, config.chunk.min_tokens, config.chunk.overlap_headings)
    return out


def format_plan(plan: Plan, price_in: float = 0.0, price_out: float = 0.0) -> str:
    lines = [f"documents: {len(plan.documents)}   chunks: {plan.chunks}   "
             f"source tokens: {sum(d['tokens'] for d in plan.documents):,}", ""]
    lines.append(f"{'stage':<16}{'role':<10}{'calls':>8}{'prompt tok':>14}{'output tok':>14}")
    for s in plan.stages:
        lines.append(f"{s.stage:<16}{s.role:<10}{s.calls:>8,}{s.prompt_tokens:>14,}{s.completion_tokens:>14,}")
    totals = plan.totals()
    lines.append(f"{'total':<26}{totals['calls']:>8,}{totals['prompt_tokens']:>14,}{totals['completion_tokens']:>14,}")
    if price_in or price_out:
        lines.append("")
        lines.append(f"estimated cost at ${price_in}/M in, ${price_out}/M out: ${plan.cost(price_in, price_out):,.2f}")
    lines.append("")
    lines.append("Estimates assume no cache hits and full generation; a resumed run costs less.")
    return "\n".join(lines)
