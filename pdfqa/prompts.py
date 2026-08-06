"""Prompt templates. Every generation prompt asks for strict JSON so parsing stays mechanical."""

JSON_RULE = "Return only JSON. No prose before or after. No markdown fences."

KG_EXTRACT = """You are building a knowledge graph from one section of a technical document.

Section: {breadcrumb}
---
{context}
---

Extract entities and relations that are explicitly supported by this text.
Entity types: method, dataset, metric, quantity, claim, hyperparameter, term.
For quantity entities, put the numeric value with its unit in "value".

Schema:
{{"entities":[{{"name":"...","type":"...","value":""}}],
 "relations":[{{"source":"...","relation":"...","target":"..."}}]}}

""" + JSON_RULE

QA_GENERATE = """You write training data from technical documents.

Section: {breadcrumb}
Source context:
---
{context}
---

Write {n} question-answer pairs. Hard requirements:
- Every answer must be fully derivable from the context above and nothing else.
- Questions must be self-contained: never say "the passage", "this section", "the above", or "the document".
- Do not copy a distinctive answer phrase verbatim into its own question.
- Vary difficulty across the set: some lookup, some reasoning, some quantitative.
- If the context has tables or numbers, at least one pair must require reading them.

Schema:
{{"pairs":[{{"question":"...","answer":"...","difficulty":"simple|intermediate|complex","evidence":"exact quote from context"}}]}}

""" + JSON_RULE

MULTIHOP_GENERATE = """You write multi-hop questions that cannot be answered from any single section.

Reasoning path through the document graph:
{path}

Section A ({breadcrumb_a}):
---
{context_a}
---

Section B ({breadcrumb_b}):
---
{context_b}
---

Write {n} questions that each require combining a fact from A with a fact from B.
A reader holding only A, or only B, must be unable to answer.
State the bridging reasoning explicitly in the answer.

Schema:
{{"pairs":[{{"question":"...","answer":"...","hops":2,"evidence_a":"...","evidence_b":"..."}}]}}

""" + JSON_RULE

CROSS_DOC_GENERATE = """You write questions that require reading two different documents.

Shared concept linking them: {anchor}
Graph link: {path}

Document A — {source_a} ({breadcrumb_a}):
---
{context_a}
---

Document B — {source_b} ({breadcrumb_b}):
---
{context_b}
---

Write {n} questions that compare, reconcile, or combine what the two documents say.
Rules:
- Name the documents by their subject matter, not as "Document A" or "the second paper".
- The answer must attribute each fact to the document it came from.
- If the two documents disagree, say so explicitly rather than picking one.
- A reader holding only one of the two must be unable to answer.

Schema:
{{"pairs":[{{"question":"...","answer":"...","hops":2,"evidence_a":"...","evidence_b":"..."}}]}}

""" + JSON_RULE

MULTITURN_GENERATE = """Write a realistic multi-turn conversation grounded in one source section.

Persona: {persona}
Section: {breadcrumb}
Source context:
---
{context}
---

Rules for the {turns}-turn dialogue:
- Turn 1 is an ordinary question in this persona's voice.
- Include one follow-up that depends on the previous answer (pronouns, ellipsis).
- Include one turn where the user states something the context contradicts; the assistant corrects it, citing the context.
- If the persona would ask something ambiguous, the assistant asks a clarifying question before answering.
- The assistant never asserts anything the context does not support; it says so when the context is silent.

Schema:
{{"turns":[{{"role":"user|assistant","content":"..."}}]}}

""" + JSON_RULE

REACT_GENERATE = """Write an agentic tool-use trajectory that answers a question about this source.

Section: {breadcrumb}
Source context:
---
{context}
---

Available tools:
- python(code): executes Python, returns stdout. Use for arithmetic and unit conversion.
- sql(query): read-only SQLite over the tables extracted from this document. Schema:
{schema}
- lookup(term): returns the passage of the source document mentioning the term.

Query only tables and columns listed in the schema above; the query is really executed and a
wrong column name fails the trace. Joining two tables is allowed and encouraged when it answers
something neither table answers alone.

Write one question that genuinely needs computation or a table query, then the trajectory.
Every observation must be the true result of the given action against the real context.
Final answer must follow from the observations.

Schema:
{{"question":"...",
  "trace":[{{"thought":"...","action":"python|sql|lookup","action_input":"...","observation":"..."}}],
  "answer":"..."}}

""" + JSON_RULE

PERSONA_REWRITE = """Rewrite this exchange for a different audience without changing any fact.

Persona: {persona}
Response format: {style}

Question: {question}
Answer: {answer}

Keep every number, name, and claim identical. Change only voice, framing, and structure.
If the format is JSON, emit the answer as a JSON object inside the "answer" string.

Schema:
{{"question":"...","answer":"..."}}

""" + JSON_RULE

EVOL_INSTRUCT = """Rewrite the question to be harder, using this mutation: {mutation}

Original question: {question}
Reference answer: {answer}
Source context:
---
{context}
---

The rewritten question must still be answerable from the same context alone.
Update the answer so it fully addresses the rewritten question.
If the mutation cannot be applied without leaving the context, return the original unchanged and set "applied" to false.

Schema:
{{"question":"...","answer":"...","applied":true}}

""" + JSON_RULE

MUTATIONS = {
    "add_constraint": "add a specific constraint or qualifier that narrows what counts as a valid answer",
    "counterfactual": "turn it into a counterfactual: what would follow if one stated condition were different",
    "deepen_math": "require an additional quantitative step, such as a ratio, delta, or unit conversion",
    "implicit": "remove the explicit keywords so the reader must infer which part of the source applies",
    "comparative": "require comparing two things described in the source rather than reporting one",
    "multi_part": "split into two dependent sub-questions where the second builds on the first",
    "adversarial_premise": "embed a subtly false premise the answer must identify and correct",
}

REJECT_MUTATE = """Produce a plausible but wrong version of this answer, for preference training.

Failure mode to inject: {mode} — {mode_desc}

Question: {question}
Correct answer: {answer}
Source context:
---
{context}
---

The wrong answer must read as confident and fluent, stay on topic, and be roughly the same length.
Inject exactly the named failure mode; keep everything else faithful so the flaw is subtle.

Schema:
{{"rejected":"...","injected":"one sentence naming exactly what was corrupted"}}

""" + JSON_RULE

REJECTION_MODES = {
    "number_hallucination": "change one quantity to a nearby fabricated value",
    "unit_mismatch": "keep the number but report it in the wrong unit or scale",
    "off_by_one": "shift an index, count, rank, or ordinal by one",
    "causal_inversion": "reverse the direction of a causal or comparative claim",
    "overgeneralization": "state a conclusion far broader than the evidence supports",
    "unsupported_detail": "add a specific detail that is not in the source at all",
    "conflation": "attribute a property of one entity to a different entity in the source",
    "stale_scope": "answer a related question the user did not ask",
}

FIGURE_QA = """You are looking at a figure extracted from a technical document.

Section: {breadcrumb}
Caption: {caption}
Surrounding text:
---
{context}
---

Write {n} question-answer pairs about what the image actually shows — trends, axis values,
comparisons between series, the shape of a curve, or what a diagram's arrows mean.
Rules:
- Only state what is visible in the image or stated in the caption and surrounding text.
- Do not ask about anything you cannot read off the image.
- Questions must name the subject, not "the figure" or "this image".
- If the image is unreadable or carries no information, return an empty list.

Schema:
{{"pairs":[{{"question":"...","answer":"...","visual_evidence":"what in the image supports this"}}]}}

""" + JSON_RULE

FIGURE_GROUND = """Check whether this claim is supported by the image shown.

Claim: {claim}
Caption: {caption}

"entailment" only if the image visibly supports every part of the claim.
"contradiction" if the image shows otherwise. "neutral" if it is not readable from the image.

Schema:
{{"label":"entailment|neutral|contradiction","confidence":0.0,"unsupported":["..."]}}

""" + JSON_RULE

Z3_CHECK = """Encode this claim as SMT-LIB 2 constraints so a solver can check it for consistency.

Question: {question}
Answer: {answer}
Source context (the ground truth numbers):
---
{context}
---

Declare the quantities from the context as constants with their real values asserted, then assert
the claim the answer makes. A satisfiable system means the claim is consistent with the source.
Also emit "negation", which is the same system with the claim's assertion negated — that one must
be unsatisfiable for the claim to actually follow.
Use only (declare-const ...), (assert ...), and Real/Int/Bool sorts. Do not include (check-sat).
If the claim carries no checkable arithmetic or boolean structure, set "applicable" to false.

Schema:
{{"applicable":true,"constraints":"...","negation":"..."}}

""" + JSON_RULE

NLI_FORWARD = """Judge whether the premise entails the hypothesis.

Premise:
---
{context}
---

Hypothesis: {claim}

"entailment" only if every part of the hypothesis is supported by the premise.
"contradiction" if the premise states otherwise. "neutral" if the premise is silent.

Schema:
{{"label":"entailment|neutral|contradiction","confidence":0.0,"unsupported":["..."]}}

""" + JSON_RULE

NLI_REVERSE = """Decide whether this question can be answered without the source document.

Question: {question}
Answer given in the source: {answer}

Answer from general knowledge only. If you can produce the same answer without any document,
the question leaks external knowledge and is not a useful grounded training example.

Schema:
{{"answerable_without_source":true,"my_answer":"...","matches_source_answer":true,"confidence":0.0}}

""" + JSON_RULE

SYMBOLIC_CHECK = """Convert this quantitative claim into executable Python that verifies it.

Question: {question}
Answer: {answer}
Source context (contains the ground-truth numbers):
---
{context}
---

Write a self-contained script that pulls the relevant numbers from the context as literals,
recomputes the claim, and prints "PASS" or "FAIL: <reason>". No imports beyond math.
If the claim is not quantitative, set "applicable" to false.

Schema:
{{"applicable":true,"code":"...","expected":"PASS"}}

""" + JSON_RULE

CLARIFY_CHECK = """Rate this training pair for usability.

Question: {question}
Answer: {answer}

Score 0-1 on each: standalone (question makes sense with no context handed to the reader),
specific (one defensible answer), natural (a real person would ask this),
answer_complete (the answer actually resolves the question).

Schema:
{{"standalone":0.0,"specific":0.0,"natural":0.0,"answer_complete":0.0,"issues":["..."]}}

""" + JSON_RULE

PERSONAS = {
    "domain_expert": "a specialist in this field who wants precision and cites prior work",
    "non_technical_stakeholder": "a manager who needs the practical implication, no jargon",
    "student": "a learner who needs the underlying concept explained before the answer",
    "skeptical_reviewer": "a hostile peer reviewer probing for weaknesses and overclaims",
    "practitioner": "an engineer who wants to reproduce or apply the work",
    "journalist": "a reporter who needs a plain-language summary with the key number",
    "neutral": "a direct question with no particular framing",
}

STYLES = {
    "prose": "two to five sentences of plain prose",
    "bullets": "a short bulleted list",
    "json": "a JSON object with keys the question implies",
    "step_by_step": "numbered reasoning steps ending in a conclusion",
    "formal_proof": "a formal argument with stated assumptions and a conclusion",
    "table": "a small markdown table",
}
