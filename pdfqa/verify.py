"""Quality gating: bidirectional NLI, leakage detection, symbolic execution, self-consistency."""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .llm import Runtime, parse_json
from .llm import LLMError
from .prompts import CLARIFY_CHECK, FIGURE_GROUND, NLI_FORWARD, NLI_REVERSE, SYMBOLIC_CHECK, Z3_CHECK
from .tools import execute, observations_agree
from .records import QARecord

WORD = re.compile(r"[A-Za-z0-9_.%-]+")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CONTEXT_DEIXIS = re.compile(
    r"\b(this|the)\s+(passage|section|paper|document|text|excerpt|table above|figure above)\b|\bthe above\b|\bas (?:stated|mentioned|shown) (?:above|here)\b",
    re.I,
)

STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were", "for", "on", "by", "with", "as",
    "that", "this", "it", "at", "from", "be", "has", "have", "had", "which", "not", "we", "our", "its", "their",
    "what", "how", "why", "when", "where", "does", "do", "did", "can", "could", "would", "should", "than", "then",
}


@dataclass
class Gate:
    name: str
    passed: bool
    score: float = 0.0
    detail: str = ""
    inconclusive: bool = False
    """The judge never returned a verdict. Distinct from a verdict of "fail": rejecting a record
    because the verifier emitted prose instead of JSON destroys good data and blames the gate."""

    def as_dict(self) -> dict:
        out = {"stage": self.name, "passed": self.passed, "score": round(self.score, 4), "detail": self.detail[:400]}
        if self.inconclusive:
            out["inconclusive"] = True
        return out


def unjudged(name: str, raw: str) -> Gate:
    """A reply the judge could not be read from. Non-blocking, but counted and flagged, because a
    model that cannot emit JSON is a backend problem and must not read as a data problem."""
    return Gate(name, True, 0.0, f"verifier reply did not parse: {raw.strip()[:120]!r}", inconclusive=True)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD.findall(text) if t.lower() not in STOP and len(t) > 1]


def generated_text(rec: QARecord) -> str:
    """Everything the model asserted, which is what grounding must be measured against.

    `rec.answer` on a dialogue is only the final assistant turn — frequently a one-line correction
    — so judging it against the whole context rejects sound multi-turn data outright.
    """
    if rec.turns:
        return "\n".join(t.content for t in rec.turns if t.role == "assistant") or rec.answer
    return rec.answer


# --------------------------------------------------------------------------- cheap gates


def lexical_grounding(rec: QARecord, threshold: float = 0.25) -> Gate:
    """A coarse topic-drift filter, deliberately not a precision check.

    Token overlap cannot separate faithful from unfaithful once an answer reuses the context's
    vocabulary — a wrong claim built from the right words scores higher than a faithful but wordy
    one. Set high, it becomes a verbosity filter that discards good data; the precise work belongs
    to `numeric_grounding` (wrong figures) and `nli_forward` (contradictions).
    """
    ans, ctx = set(tokens(generated_text(rec))), set(tokens(rec.context))
    if not ans:
        return Gate("lexical_grounding", False, 0.0, "empty answer")
    overlap = len(ans & ctx) / len(ans)
    return Gate("lexical_grounding", overlap >= threshold, overlap, f"{len(ans & ctx)}/{len(ans)} answer tokens in context")


def supported_numbers(rec: QARecord) -> list[float]:
    """Numbers the record may legitimately state: those in the source, plus any a tool actually
    computed. A ReAct trace exists to derive figures the text does not contain, and an executed
    observation is stronger evidence than finding the digits in the prose."""
    values = [float(x) for x in NUMBER.findall(rec.context)]
    for step in rec.tool_trace:
        if step.get("executed_ok"):
            values += [float(x) for x in NUMBER.findall(str(step.get("executed_observation", "")))]
    return values


def numeric_grounding(rec: QARecord) -> Gate:
    """Every number in the answer must be in the source or computed by a verified tool call."""
    ans_nums = [float(x) for x in NUMBER.findall(generated_text(rec))]
    ctx_nums = supported_numbers(rec)
    if not ans_nums:
        return Gate("numeric_grounding", True, 1.0, "no numbers")
    missing = []
    for n in ans_nums:
        if not any(abs(n - c) <= max(0.01, abs(c) * 0.005) for c in ctx_nums):
            missing.append(n)
    score = 1.0 - len(missing) / len(ans_nums)
    return Gate("numeric_grounding", not missing, score, f"unsupported: {missing[:6]}" if missing else "all grounded")


def shortcut_leakage(rec: QARecord, max_ratio: float = 0.6, min_shared: int = 4) -> Gate:
    """Fails when the question already contains the distinctive tokens of its own answer."""
    q, a = set(tokens(rec.question)), tokens(rec.answer)
    if not a:
        return Gate("shortcut_leakage", False, 0.0, "empty answer")
    counts = Counter(a)
    distinctive = {t for t in a if counts[t] <= 3}
    shared = distinctive & q
    ratio = len(shared) / max(1, len(distinctive))
    leaked = ratio >= max_ratio and len(shared) >= min_shared
    if len(rec.answer) < 40 and rec.answer.strip().lower() in rec.question.lower():
        leaked = True
    return Gate("shortcut_leakage", not leaked, 1.0 - ratio, f"shared distinctive tokens: {sorted(shared)[:8]}")


def standalone_question(rec: QARecord) -> Gate:
    m = CONTEXT_DEIXIS.search(rec.question)
    return Gate("standalone_question", m is None, 0.0 if m else 1.0, f"deictic phrase: {m.group(0)}" if m else "self-contained")


def structural(rec: QARecord, min_q: int = 15, min_a: int = 10) -> Gate:
    issues = []
    if len(rec.question.strip()) < min_q:
        issues.append("question too short")
    if len(rec.answer.strip()) < min_a:
        issues.append("answer too short")
    if rec.question.strip() == rec.answer.strip():
        issues.append("question equals answer")
    if rec.task != "multiturn" and rec.question.count("?") > 3:
        issues.append("question is a list of questions")
    return Gate("structural", not issues, 1.0 if not issues else 0.0, "; ".join(issues))


# --------------------------------------------------------------------------- model gates


def nli_forward(runtime: Runtime, rec: QARecord, min_confidence: float = 0.5) -> Gate:
    prompt = NLI_FORWARD.format(context=rec.context[:9000], claim=generated_text(rec)[:3000])
    raw = runtime.complete("verify", prompt, temperature=0.0, max_tokens=700)
    data = parse_json(raw, default=None)
    if not isinstance(data, dict) or not data:
        return unjudged("nli_forward", raw)
    label = str(data.get("label", "neutral")).lower()
    conf = float(data.get("confidence", 0.0) or 0.0)
    passed = label == "entailment" and conf >= min_confidence
    detail = f"{label} conf={conf:.2f}"
    if data.get("unsupported"):
        detail += f" unsupported={data['unsupported'][:3]}"
    return Gate("nli_forward", passed, conf if label == "entailment" else 0.0, detail)


def nli_reverse(runtime: Runtime, rec: QARecord) -> Gate:
    """Passes when the question genuinely needs the document — no external-knowledge shortcut."""
    prompt = NLI_REVERSE.format(question=rec.question[:2000], answer=rec.answer[:2000])
    raw = runtime.complete("verify", prompt, temperature=0.0, max_tokens=700)
    data = parse_json(raw, default=None)
    if not isinstance(data, dict) or not data:
        return unjudged("nli_reverse", raw)
    leaked = bool(data.get("answerable_without_source")) and bool(data.get("matches_source_answer"))
    conf = float(data.get("confidence", 0.0) or 0.0)
    return Gate("nli_reverse", not leaked, 1.0 - conf if leaked else 1.0, "answerable from general knowledge" if leaked else "requires source")


def clarity(runtime: Runtime, rec: QARecord, threshold: float = 0.6) -> Gate:
    prompt = CLARIFY_CHECK.format(question=rec.question[:2000], answer=rec.answer[:3000])
    raw = runtime.complete("verify", prompt, temperature=0.0, max_tokens=600)
    data = parse_json(raw, default=None)
    if not isinstance(data, dict) or not data:
        return unjudged("clarity", raw)
    keys = ("standalone", "specific", "natural", "answer_complete")
    vals = [float(data.get(k, 0.0) or 0.0) for k in keys]
    score = sum(vals) / len(keys)
    return Gate("clarity", score >= threshold, score, f"{dict(zip(keys, [round(v, 2) for v in vals]))} {data.get('issues', [])}")


def self_consistency(runtime: Runtime, rec: QARecord, samples: int = 3, agreement: float = 0.6) -> Gate:
    """Re-answer the question from context several times; low agreement means the pair is unstable."""
    prompt = f"Answer using only this context. If the context does not answer it, reply exactly UNANSWERABLE.\n\nContext:\n---\n{rec.context[:9000]}\n---\n\nQuestion: {rec.question}\n\nAnswer:"
    answers = []
    for i in range(samples):
        try:
            answers.append(runtime.complete("verify", prompt, temperature=0.0 if i == 0 else 0.7, max_tokens=600).strip())
        except Exception:
            continue
    if not answers:
        return Gate("self_consistency", False, 0.0, "no samples")
    if sum("UNANSWERABLE" in a.upper() for a in answers) > len(answers) / 2:
        return Gate("self_consistency", False, 0.0, "majority says unanswerable from context")
    ref = set(tokens(rec.answer))
    sims = [_jaccard(ref, set(tokens(a))) for a in answers]
    score = sum(sims) / len(sims)
    return Gate("self_consistency", score >= agreement * 0.5, score, f"mean overlap {score:.2f} over {len(answers)} samples")


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def visual_grounding(runtime: Runtime, rec: QARecord) -> Gate:
    """Re-checks a figure-derived answer against the rendered crop it came from."""
    images = [p for p in rec.images if Path(p).exists()]
    if not images:
        return Gate("visual_grounding", True, 1.0, "no image attached")
    caption = (rec.context.splitlines() or [""])[0]
    prompt = FIGURE_GROUND.format(claim=rec.answer[:2000], caption=caption[:300])
    try:
        raw = runtime.complete_vision(prompt, images[:1], temperature=0.0, max_tokens=600)
    except LLMError as exc:
        return Gate("visual_grounding", True, 0.5, f"vision unavailable: {exc}")
    data = parse_json(raw, default={}) or {}
    label = str(data.get("label", "neutral")).lower()
    conf = float(data.get("confidence", 0.0) or 0.0)
    return Gate("visual_grounding", label == "entailment", conf if label == "entailment" else 0.0, f"{label} conf={conf:.2f}")


def trace_execution(rec: QARecord, timeout: float = 10.0, allow_exec: bool = True) -> Gate:
    """Replays every action in a ReAct trace and compares the real result to the claimed one."""
    if not rec.tool_trace:
        return Gate("trace_execution", True, 1.0, "no trace")
    if not allow_exec:
        # Not a pass: nothing was checked. Recording it as one puts traces whose observations were
        # never reproduced into the dataset wearing the same badge as traces that were.
        return Gate("trace_execution", True, 0.0, "execution disabled; observations not reproduced", inconclusive=True)

    grids = rec.tool_env.get("grids") or []
    captions = rec.tool_env.get("captions") or []
    checked, agreed, notes = 0, 0, []
    for i, step in enumerate(rec.tool_trace):
        action = str(step.get("action", "")).strip().lower()
        if action not in ("python", "sql", "lookup"):
            notes.append(f"step {i}: unknown tool {action!r}")
            continue
        result = execute(action, str(step.get("action_input", "")), rec.context, grids, timeout, captions)
        checked += 1
        step["executed_observation"] = result.output[:600]
        step["executed_ok"] = result.ok
        if not result.ok:
            notes.append(f"step {i} ({action}) failed: {result.output[:120]}")
            continue
        if observations_agree(str(step.get("observation", "")), result.output):
            agreed += 1
        else:
            notes.append(f"step {i} ({action}) claimed {str(step.get('observation', ''))[:60]!r}, got {result.output[:60]!r}")

    if not checked:
        return Gate("trace_execution", False, 0.0, "; ".join(notes) or "no runnable steps")
    score = agreed / checked
    return Gate("trace_execution", agreed == checked, score, "; ".join(notes) or f"{agreed}/{checked} observations reproduced")


def turn_coherence(rec: QARecord, min_overlap: float = 0.12) -> Gate:
    """Dialogue-shaped checks: alternation, no empty turns, no assistant turn adrift from the source."""
    turns = rec.turns
    if not turns:
        return Gate("turn_coherence", True, 1.0, "not a dialogue")
    issues = []
    if turns[0].role != "user":
        issues.append("does not open with a user turn")
    for i, t in enumerate(turns):
        if not t.content.strip():
            issues.append(f"turn {i} is empty")
        if i and t.role == turns[i - 1].role:
            issues.append(f"turn {i} repeats role {t.role}")

    ctx = set(tokens(rec.context))
    assistant = [t for t in turns if t.role == "assistant"]
    if not assistant:
        issues.append("no assistant turn")
    drifting = 0
    for t in assistant:
        ans = set(tokens(t.content))
        if ans and len(ans & ctx) / len(ans) < min_overlap:
            drifting += 1
    if assistant and drifting > len(assistant) / 2:
        issues.append(f"{drifting}/{len(assistant)} assistant turns unrelated to the source")

    score = 0.0 if issues else 1.0 - (drifting / max(1, len(assistant))) * 0.5
    return Gate("turn_coherence", not issues, score, "; ".join(issues) or f"{len(turns)} turns, {drifting} drifting")


# --------------------------------------------------------------------------- symbolic


def looks_quantitative(rec: QARecord) -> bool:
    return bool(NUMBER.search(rec.answer)) and bool(NUMBER.search(rec.context))


def symbolic_check(runtime: Runtime, rec: QARecord, timeout: float = 10.0, allow_exec: bool = True) -> Gate:
    if not looks_quantitative(rec):
        return Gate("symbolic", True, 1.0, "not quantitative")
    prompt = SYMBOLIC_CHECK.format(question=rec.question[:2000], answer=rec.answer[:2000], context=rec.context[:9000])
    data = parse_json(runtime.complete("verify", prompt, temperature=0.0, max_tokens=1200), default={}) or {}
    if not data.get("applicable", False) or not data.get("code"):
        return Gate("symbolic", True, 1.0, "solver deemed claim non-quantitative")
    if not allow_exec:
        return Gate("symbolic", True, 0.0, "execution disabled; claim not checked", inconclusive=True)
    ok, output = run_python(str(data["code"]), timeout)
    passed = ok and "PASS" in output and "FAIL" not in output
    return Gate("symbolic", passed, 1.0 if passed else 0.0, output[:400])


SOLVER_MODULES = ("math", "itertools", "fractions", "statistics", "decimal", "cmath", "numbers", "random", "re")

BLOCKED_BUILTINS = (
    "eval exec compile open input breakpoint help __import__ __loader__ __spec__ memoryview "
    "globals locals vars"
).split()
"""Removed from the child's builtins outright, so no amount of string-building reaches them."""

MEMORY_LIMIT = 512 * 1024 * 1024
OUTPUT_LIMIT = 20000

SANDBOX_RUNNER = '''
import resource, sys, builtins

resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except (ValueError, OSError):
    pass

_real_import = builtins.__import__
_ALLOWED = set({modules!r})
_BLOCKED = set({blocked!r})

# Everything except the dangerous names, rather than a short allowlist: module internals look their
# builtins up through this same mapping, so a mapping missing KeyError or getattr breaks re and
# random rather than the attacker.
_safe = {{k: getattr(builtins, k) for k in dir(builtins) if k not in _BLOCKED}}

_MODULES = {{}}
for _name in _ALLOWED:
    try:
        _m = _real_import(_name)
    except ImportError:
        continue
    # An imported module carries the *real* builtins on __builtins__, so `random.__builtins__["open"]`
    # handed back everything this sandbox removes. Point it at the restricted mapping instead.
    _m.__builtins__ = _safe
    _MODULES[_name] = _m


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _MODULES:
        raise ImportError("module %r is not available to solver code" % name)
    return _MODULES[root]


_safe["__import__"] = _guarded_import

with open({payload!r}, "r", encoding="utf-8") as fh:
    _source = fh.read()

_code = compile(_source, "<solver>", "exec")
del builtins, resource, _real_import
exec(_code, {{"__builtins__": _safe, "__name__": "__main__"}})
'''


def run_python(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Run solver code in a separate interpreter stripped of the means to affect this machine.

    Solver code is written by a model from document text, so a document is an input to it. The
    interpreter it runs in has no eval, exec, compile, open or getattr, an __import__ that serves
    only pure-computation modules, and kernel limits on address space, CPU time, subprocesses and
    file size — file size zero, so it cannot write at all.

    There is deliberately no blocklist over the source text. One used to guard this, and it did not
    work in either direction: `b['ev'+'al']` reached eval without ever spelling it, while
    `from fractions import Fraction` was rejected as an illegal import, so honest solver code failed
    for the same reason dishonest code got through.
    """
    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp) / "solver.py"
        payload.write_text(code, encoding="utf-8")
        runner = Path(tmp) / "runner.py"
        runner.write_text(
            SANDBOX_RUNNER.format(
                mem=MEMORY_LIMIT,
                cpu=max(1, int(timeout) + 1),
                modules=list(SOLVER_MODULES),
                blocked=BLOCKED_BUILTINS,
                payload=str(payload),
            ),
            encoding="utf-8",
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(runner)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        out = (proc.stdout + proc.stderr).strip()[:OUTPUT_LIMIT]
        return proc.returncode == 0, out


def z3_available() -> bool:
    try:
        import z3  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


_Z3_POOL: ThreadPoolExecutor | None = None
_Z3_POOL_LOCK = threading.Lock()


def _z3_pool() -> ThreadPoolExecutor:
    """One thread, created once, that owns every z3 object for its whole lifetime.

    A plain lock around the solver is not enough, and that is not a theoretical gap: the solver
    outlives the critical section, so its C++ teardown runs later on whichever thread the garbage
    collector happens to be on, and z3 aborts the process instead of raising. It reproduced as an
    intermittent SIGSEGV — about one run in three over a 120-document corpus with four workers.
    Confining creation and destruction to a single thread is what actually fixes it.
    """
    global _Z3_POOL
    with _Z3_POOL_LOCK:
        if _Z3_POOL is None:
            _Z3_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdfqa-z3")
        return _Z3_POOL


def _z3_solve_here(constraints: str, timeout: float) -> tuple[str, str]:
    import z3  # type: ignore

    solver = z3.Solver()
    solver.set("timeout", int(timeout * 1000))
    solver.from_string(constraints if "(check-sat)" not in constraints else constraints.replace("(check-sat)", ""))
    # str() before returning: a z3 object handed back to the caller would be released off-thread,
    # which is the whole failure this indirection exists to prevent.
    return str(solver.check()), ""


def z3_solve(constraints: str, timeout: float = 10.0) -> tuple[str, str]:
    """Return (sat|unsat|unknown|error, detail) for an SMT-LIB 2 fragment."""
    if not z3_available():
        return "unknown", "z3 not installed"
    try:
        return _z3_pool().submit(_z3_solve_here, constraints, timeout).result(timeout=timeout + 30.0)
    except Exception as exc:
        return "error", str(exc)[:300]


def z3_consistency(runtime: Runtime, rec: QARecord, timeout: float = 10.0) -> Gate:
    """A claim holds when its constraints are satisfiable and its negation is not."""
    if not z3_available():
        return Gate("z3", True, 1.0, "z3 not installed")
    if not looks_quantitative(rec):
        return Gate("z3", True, 1.0, "not quantitative")
    prompt = Z3_CHECK.format(question=rec.question[:2000], answer=rec.answer[:2000], context=rec.context[:9000])
    data = parse_json(runtime.complete("verify", prompt, temperature=0.0, max_tokens=1400), default={}) or {}
    if not data.get("applicable", False) or not data.get("constraints"):
        return Gate("z3", True, 1.0, "no checkable structure")

    claim, detail = z3_solve(str(data["constraints"]), timeout)
    if claim == "error":
        return Gate("z3", True, 0.5, f"encoding rejected: {detail}")
    if claim != "sat":
        return Gate("z3", False, 0.0, f"claim constraints {claim} against the source values")
    if not data.get("negation"):
        return Gate("z3", True, 0.7, "consistent; no negation supplied")
    negated, neg_detail = z3_solve(str(data["negation"]), timeout)
    if negated == "error":
        return Gate("z3", True, 0.6, f"negation rejected: {neg_detail}")
    entailed = negated == "unsat"
    return Gate("z3", entailed, 1.0 if entailed else 0.0, f"claim=sat negation={negated}")


# --------------------------------------------------------------------------- orchestration

CHEAP_GATES = (structural, standalone_question, shortcut_leakage, lexical_grounding, numeric_grounding)
VISUAL_CHEAP_GATES = (structural, standalone_question, shortcut_leakage)


def verify(
    rec: QARecord,
    runtime: Runtime | None = None,
    use_model_gates: bool = True,
    use_symbolic: bool = True,
    consistency_samples: int = 0,
    allow_exec: bool = True,
    use_z3: bool = True,
) -> QARecord:
    """Figure-derived rows swap text grounding for visual grounding — their numbers are read off axes."""
    visual = rec.task == "figure_qa" and bool(rec.images)
    gates: list[Gate] = []
    if rec.tool_trace:
        # Runs first so the numeric gate can credit figures a tool actually computed.
        gates.append(trace_execution(rec, allow_exec=allow_exec))
    gates += [g(rec) for g in (VISUAL_CHEAP_GATES if visual else CHEAP_GATES)]
    if rec.turns:
        gates.append(turn_coherence(rec))
    if use_model_gates and runtime is not None and all(g.passed for g in gates):
        if visual:
            gates.append(visual_grounding(runtime, rec))
        else:
            gates.append(nli_forward(runtime, rec))
        gates.append(nli_reverse(runtime, rec))
        gates.append(clarity(runtime, rec))
        if consistency_samples > 0 and not visual:
            gates.append(self_consistency(runtime, rec, consistency_samples))
        if use_symbolic and not visual and all(g.passed for g in gates):
            gates.append(symbolic_check(runtime, rec, allow_exec=allow_exec))
            if use_z3 and all(g.passed for g in gates):
                gates.append(z3_consistency(runtime, rec))

    for g in gates:
        rec.scores[g.name] = round(g.score, 4)
        rec.prov.verification.append(g.as_dict())
        if g.inconclusive:
            rec.flags.append(f"unverified:{g.name}")
        elif not g.passed:
            rec.flags.append(f"reject:{g.name}")
    rec.scores["quality"] = quality_score(rec, gates)
    return rec


def quality_score(rec: QARecord, gates: list[Gate]) -> float:
    weights = {
        "nli_forward": 3.0,
        "nli_reverse": 2.0,
        "clarity": 2.0,
        "symbolic": 2.0,
        "z3": 2.0,
        "visual_grounding": 3.0,
        "trace_execution": 3.0,
        "turn_coherence": 2.0,
        "self_consistency": 1.5,
        "numeric_grounding": 1.5,
        "lexical_grounding": 1.0,
        "shortcut_leakage": 1.0,
        "standalone_question": 1.0,
        "structural": 0.5,
    }
    judged = [g for g in gates if not g.inconclusive]
    num = sum(weights.get(g.name, 1.0) * (g.score if g.passed else 0.0) for g in judged)
    den = sum(weights.get(g.name, 1.0) for g in judged) or 1.0
    return round(num / den, 4)


# --------------------------------------------------------------------------- hardness


def hardness(rec: QARecord) -> dict[str, float]:
    q, a = tokens(rec.question), tokens(rec.answer)
    return {
        "entropy": round(_entropy(a), 4),
        "q_len": float(len(q)),
        "a_len": float(len(a)),
        "novelty": round(1.0 - _jaccard(set(q), set(tokens(rec.context))), 4),
        "numeric_density": round(len(NUMBER.findall(rec.answer)) / max(1, len(a)), 4),
        "hops": float(rec.hops),
    }


def _entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def difficulty_bucket(rec: QARecord) -> str:
    h = hardness(rec)
    score = h["hops"] * 1.5 + h["numeric_density"] * 4 + h["entropy"] / 4 + h["a_len"] / 60
    if rec.task in ("multihop", "react"):
        score += 1.5
    if score < 2.2:
        return "simple"
    if score < 4.0:
        return "intermediate"
    return "complex"
