# PDF-to-QA

Turns technical documents into verified fine-tuning datasets. It parses them into a typed AST,
builds a knowledge graph over them, generates SFT / DPO / multi-turn / ReAct / multimodal data
across that graph, then gates every row through grounding checks before export.

Most document-to-dataset tools flatten to text, slide a window over it, and ask a model for
questions. That loses tables, section structure, and cross-references, and it can only ever
produce single-paragraph lookup questions. This one keeps the structure and generates against it.

```bash
pip install -e ".[all]"
pdfqa run papers/ --backend ollama --model qwen2.5:7b -o dataset/
```

**Inputs:** PDF, Markdown, plain text, HTML, DOCX, EPUB, LaTeX source, Jupyter notebooks, CSV/TSV.
Point it at a directory and it takes whatever it recognises — one run can span six formats. Only
PDF needs a third-party parser; every other adapter is standard library (DOCX and EPUB are zip
archives of XML, notebooks are JSON).

**Outputs:** a fine-tuning dataset in ten formats, a retrieval corpus, and leak-free train/val/test
splits — from the same pass over the documents. The corpus is then queryable (`ask`) and the splits
gradeable (`eval`), so generate → retrieve → evaluate all run on one parse of the source.

## What it does

**Structure, not flat text.** Documents parse into a typed AST — headings, tables, equations,
captions, footnotes — with parent/child/sibling links intact. Tables come out as HTML plus a cell
grid. PDF nodes carry page index and bounding box. Every downstream stage works on the AST, so
each new input format inherits the entire pipeline: DOCX heading styles, HTML `<h2>`, and LaTeX
`\subsection` all become the same node type, and a `<table>`, a `w:tbl` and a `tabular` all become
the same queryable grid.

**Equations are parsed, not just stored.** LaTeX goes through a real tokeniser and parser into a
syntax tree, so a dropped superscript, an unclosed brace, a `\frac` missing an argument, or a
mismatched `\begin`/`\end` is reported with a reason instead of silently poisoning a chunk. A
truncated equation ending on a bare operator is caught the same way. `pdfqa inspect` lists the
malformed ones — a spike there usually means the extractor, not the paper, is at fault.

**Figures come out as images.** Charts and diagrams are usually *drawn*, not embedded, so they
never show up as image blocks — the engine reads the vector drawing regions, merges neighbouring
strokes into one figure, pulls in the axis ticks and data labels that sit just outside the drawing
bbox, and renders the region to PNG at your chosen DPI. Detected tables get rendered too. Each
crop is bound to its caption and carries page, bbox, and pixel dimensions. Chart-shaped false
positives from table detection are filtered out by cell fill ratio, so a bar chart stays a figure.

**Tables are found by alignment, not by ruling.** pdfminer reports no table structure at all, and
PyMuPDF's detector keys on ruled cells — so it misses booktabs-style tables, which have horizontal
rules only and hold their columns together purely by alignment. Rows are rebuilt from text
fragments (a cell is often its own text object) and grouped into a table when consecutive rows split
into the same number of x-aligned columns. It runs on both backends: as the only detector on
pdfminer, and as a fallback for regions PyMuPDF missed. On the bundled fixture both backends
return identical grids for both the ruled and the borderless table.

Alignment alone is not enough, though: **two-column prose aligns perfectly too**, and it is the
dominant layout for papers. Left and right column lines share a y band and start at identical x on
every line, so a purely geometric detector turns a page of text into a two-column "table" and
deletes the prose underneath it. Cell *content* is what separates them — tables hold short values,
prose holds sentences — so a candidate is rejected unless most cells are short (≤30 chars, ≤5
words). One wordy column out of four still passes.

**References resolve.** `[3]`, `Table 2`, `Eq. 4`, `Section 3.1` get bound to their actual targets
in the tree. When a chunk says "as shown in Table 2", the generator sees Table 2.

**Chunks follow the document.** Splits happen at section and block boundaries, never mid-table or
mid-equation. Each chunk keeps its breadcrumb (`Doc > Section 3 > Subsection 3.2`), its page
numbers, and its source node ids.

HTML and EPUB have no pages, so those blocks carry **anchors** instead — the element `id` when the
markup has one (`doc.html #results-table`, a real deep link) and a stable ordinal otherwise; EPUB
anchors name the spine file. Citations use page, then anchor, then section name, and a pageless
document reports no pages rather than claiming page 0.

**A graph across the whole document, then across the corpus.** Entities, methods, datasets, claims
and quantities get extracted into a document-level knowledge graph before any question is written.
Surface variants are folded together — spacing and punctuation (`Sparse-Route` / `sparse route`),
plurals, and acronyms matched to their expansion (`SRN` → `Sparse Routing Network`) — so one concept
is one node. Multi-hop questions are then sampled from entity pairs that are graph-connected but
live in *different* sections, so answering requires combining two places in the paper.

Run more than one document and the per-document graphs merge into a corpus graph. Any concept that
turns up in two papers becomes a cross-document seed: the generator gets a chunk from each source
and is told to compare, reconcile, or flag disagreement — attributing each fact to the document it
came from. On the bundled fixtures that produces, unprompted, a question about the two papers
reporting 74.8 and 72.9 for the same configuration and why they differ.

**Several data shapes, not just Q&A:**

| Shape | What you get |
| --- | --- |
| `qa` | Single-hop grounded pairs, mixed difficulty |
| `multihop` | Questions requiring two disjoint sections, sampled via KG paths |
| `cross_document` | Questions spanning two source documents that share a concept |
| `multiturn` | Dialogue trees with follow-ups, clarification, and corrected user misassumptions |
| `figure_qa` | Multimodal pairs — the rendered figure crop is sent to a vision model with its caption and section text |
| `react` | Tool-calling traces whose observations are actually executed, not asserted |
| DPO pairs | Chosen answer plus a mutated rejected one — number hallucination, unit mismatch, off-by-one, causal inversion, and five more |

Plus persona conditioning (domain expert, stakeholder, student, skeptical reviewer, ...), output
style conditioning (prose, bullets, JSON, step-by-step, proof), and an Evol-Instruct engine that
mutates questions to be harder — added constraints, counterfactuals, extra math steps, or
implicit phrasing.

**Tool traces are executed, not trusted.** A model asked to write a ReAct trace will happily invent
the observation it wishes it had got. So every action gets replayed: `python` runs in the sandbox,
`sql` runs against real in-memory SQLite tables built from **every** table in the document — header
rows become sanitised columns (`Latency (ms)` → `latency_ms`), cells get typed so
`max(accuracy) - min(accuracy)` actually works, captions become readable aliases (`table_1`), and
joins across two tables in different sections are allowed. The schema is written into the prompt,
so the model queries columns that exist. Read-only `SELECT`/`WITH` only. `lookup` searches the source. Claimed observations are compared to the
real ones with numeric tolerance. Mismatches get **repaired** — the real output replaces the invented
one and the original is kept in provenance — and steps whose tool genuinely errored are left alone
so the gate still rejects them.

**Everything gets gated.** Rows pass through cheap deterministic checks first, then model checks:

- *Forward NLI* — is the answer entailed by the source
- *Reverse NLI* — could the question be answered *without* the source (kills general-knowledge leakage)
- *Numeric grounding* — every number in the answer traced back to the source, with rounding tolerance
- *Shortcut detection* — rejects questions that already contain their own answer tokens
- *Symbolic check* — quantitative claims get compiled to Python and executed in a restricted
  subprocess
- *Z3* — the same claim is encoded as SMT-LIB constraints; it passes only if the constraints are
  satisfiable **and** their negation is unsatisfiable, which is entailment rather than mere consistency
- *Trace execution* — every tool call in a ReAct trace re-run and compared
- *Turn coherence* — dialogues checked for alternation, empty turns, and assistant turns that drift off source
- *Self-consistency* — optional re-answer voting

Figure-derived rows take a different route: text grounding would falsely reject numbers read off
an axis, so they're checked against the rendered crop by a vision model instead.

**Then it's deduped and coreset-selected.** MinHash + LSH kills near-identical source text and
near-identical questions; k-center greedy drops semantic restatements; a greedy submodular (DPP-style)
pass maximises diversity per token of budget; a final pass balances the simple/intermediate/complex mix.

Semantic dedup is exact all-pairs on small runs and switches to random-projection buckets past
~2000 records, where the quadratic path stops being affordable — measured 3.7× faster at 6000
records with identical output, and the gap widens from there. Below the crossover the projection
overhead costs more than it saves, so it stays exact.

Embeddings are held as packed float32 arrays rather than Python lists. Measured at 1536 dimensions
that is 6.6 KB per vector instead of 49 KB — a 7.5× difference, and the thing that decides whether
a large corpus fits in RAM at all (200k records: ~1.3 GB instead of ~9.9 GB). The cost is float32
precision, about 1e-7 per component, which is five orders of magnitude below the 0.92 cosine
threshold it feeds.

**Provenance survives to the export.** Every row carries source file, page indexes, bounding boxes,
AST node ids, section breadcrumb, and the full pass/fail log of every gate it went through.

## Runtime

Roles route independently, so bulk generation can run locally while verification hits a stronger
cloud model:

```yaml
runtime:
  generate: { backend: ollama, model: qwen2.5:7b, base_url: http://localhost:11434 }
  verify:   { backend: anthropic, model: claude-sonnet-4-5 }
  vision:   { backend: anthropic, model: claude-sonnet-4-5 }
  workers: 8
```

Figure QA only runs when a `vision` role is configured; without one the pipeline skips it and
everything else proceeds. Images are sent as base64 in whatever shape the backend expects —
Anthropic image blocks, OpenAI `image_url` parts, or Ollama's `images` array.

Backends: `ollama`, `vllm` and any OpenAI-compatible server, `openai`, `anthropic`, and `echo`
(offline, deterministic — the whole pipeline runs with no API key).

Parsing prefers PyMuPDF and falls back to pdfminer.six. pdfminer can't rasterise, so figure
rendering on that path goes through pypdfium2, with the PNG written directly — no imaging library
needed. Table detection is backend-independent (see below), so both paths produce queryable grids.

The core has **zero required dependencies**; PDF parsing, rendering, Parquet, YAML config, and Z3
are all optional extras.

## Exports

`chatml`, `sharegpt`, `alpaca`, `openai`, `axolotl`, `llamafactory`, `unsloth`, `dpo`, `react`,
`multimodal`, `raw`, plus a Parquet file for `datasets.load_dataset` and a generated dataset card.
Rendered figure crops land in `<outdir>/assets/<pdf-stem>/` and the `multimodal` rows point at them.

```bash
pdfqa run docs/ -f chatml dpo multimodal --limit 2000 --vision anthropic
pdfqa run docs/ --no-figures        # skip rendering and multimodal synthesis entirely
```

## More than one thing out of one run

The chunk set is already a retrieval corpus — breadcrumbed, provenanced, table-aware — so it gets
written out as `corpus.jsonl` next to the dataset. Same pass, no extra model calls. That gives you
the index to evaluate a model against and the data to train it with, built from identical parsing,
so retrieval failures can't be blamed on a different chunker.

```bash
pdfqa run papers/ -o dataset/ --split 0.8 0.1 0.1
```

Splits move **whole documents**, never rows. Rows from one paper share context passages and
entities, so a row-wise split puts the eval answer in the training set. With fewer than three
source documents there's nothing to split document-wise, so it falls back to rows and says so in
the report rather than quietly leaking:

```json
"splits": {"sizes": {"train": 3, "validation": 1, "test": 2},
           "document_wise": false,
           "note": "fewer than 3 source documents; split row-wise, so contexts overlap across splits"}
```

## Knowing the cost first

Verification is by far the most expensive stage, and that isn't obvious until you're billed for it.
`plan` parses and chunks for real — both free — then projects everything past that from your config
and actual chunk sizes:

```bash
pdfqa plan papers/ --price-in 3 --price-out 15
```

```
stage           role         calls    prompt tok    output tok
qa              generate        10         4,792         7,800
multihop        generate        16         8,928         5,120
dpo             generate        36        17,244         8,640
verify          verify         450       215,550        90,000
total                          566       272,854       132,020

estimated cost at $3.0/M in, $15.0/M out: $2.80
```

450 of 566 calls are verification. That's the knob to turn if a run is too expensive — drop
`verify.z3` or `verify.symbolic` and re-plan before committing. Prices are yours to supply rather
than baked in, since published rates change and a stale table would be worse than none.

## Extending it without forking

Three registration points, all additive — built-ins keep working whether or not anything is
registered:

```python
from pdfqa import register_task, register_format, register_adapter

@register_task("definition")
def definitions(runtime, chunks, kg, config):
    return [...]           # extra QARecords, generated however you like

@register_format("minimal")
def minimal(records):
    return [{"q": r.question, "a": r.answer} for r in records]

@register_adapter(".log")
def read_log(path):
    return ...             # a DocumentTree; inherits chunking, graph, verification, export
```

Point `plugins` in the config (or `--config`) at the module or file path and the pipeline loads it.
A registered adapter joins directory expansion, so `.log` files get picked up alongside PDFs.

## Querying the corpus

Because the corpus is exported, it can be searched — hybrid BM25 plus embeddings, answered with
citations back to source and page:

```bash
pdfqa ask "How many experts does the router select per token?" dataset/
```

```
SparseRoute selects 4 of 32 experts per token, with routing temperature 0.7 [1].

sources:
  0.544  sample.md p0    Sparse Routing > 2 Method > 2.1 Router
  0.523  sample.md p0    Sparse Routing > 4 Results
```

Hits below a score floor are dropped rather than passed to the model. That matters once embeddings
are on: cosine gives *every* passage a nonzero score, so without a floor an off-topic question
still retrieves five confident-looking passages and invites a confident wrong answer. Below the
floor the answer is a refusal and the exit code is 2.

## Evaluating a model on what came out

```bash
pdfqa eval dataset/test --mode retrieval --corpus dataset/
```

Three modes, and the gap between them is the interesting number:

- `context` — the gold passage is handed over. Measures reading, not retrieval.
- `retrieval` — the model has to find the passage itself. Also reports how often the correct source
  document made the top-k.
- `closed` — nothing is provided. Whatever it scores here, it already knew; that portion of your
  dataset is teaching it nothing.

Grading is a model judge plus two mechanical checks, and the mechanical ones can overrule: an answer
the judge calls correct is downgraded to partial if a number from the reference is missing, since
judges are lenient about digits and a wrong figure is not a wording difference.

## Auditing what came out

```bash
pdfqa report dataset/
```

```
records: 1842   mean quality: 0.79   p10/p50/p90: 0.62/0.81/0.93
task: {'qa': 902, 'multihop': 341, 'multiturn': 210, 'react': 156, 'figure_qa': 143, 'cross_document': 90}

gate pass rates:
  nli_forward          1691/1842  (92%)
  nli_reverse          1553/1842  (84%)
  trace_execution       141/156   (90%)
  z3                    398/402   (99%)
```

Gate pass rates are the useful signal: a low `nli_reverse` rate means the generator is writing
questions answerable from general knowledge, and a low `trace_execution` rate means it is inventing
tool observations. Both are prompt problems you can see and fix, rather than guess at.

## Growing a corpus over time

Re-running over a directory that gained new papers would otherwise re-emit the same questions for
the documents that didn't change. Point a run at its previous output and near-duplicates are dropped
before selection:

```bash
pdfqa run papers/ -o dataset/2026-02 --against dataset/2026-01
```

The reference side reads any export shape — `raw`, `chatml`, `sharegpt`, `alpaca`, `multimodal` —
so it works against datasets you exported for a trainer rather than kept in native form.

## Incremental

Every stage is content-addressed, including synthesis — the expensive one. Change a prompt and only
synthesis re-runs; change the parser and only extraction re-runs. Adding a PDF to a directory of 500
doesn't reprocess the other 499, and a run that dies halfway resumes from the documents it finished
instead of paying for them twice.

```bash
pdfqa cache            # what's stored
pdfqa cache --clear --stage synth
```

## Commands

```bash
pdfqa plan <inputs>      # estimate calls, tokens and cost before spending anything
pdfqa run <inputs>       # full pipeline (pdf, md, html, docx, epub, tex, ipynb, csv)
pdfqa inspect <file>     # AST outline, resolved references, rendered images, chunk plan
pdfqa graph <file>       # knowledge graph stats and multi-hop seed pairs
pdfqa ask <q> <dir>      # answer from the exported corpus, with citations
pdfqa eval <dir>         # grade a model against a generated split
pdfqa report <dir>       # audit a produced dataset
pdfqa cache              # cache state
pdfqa init pdfqa.yaml    # starter config
```

`inspect` on the bundled sample:

```
source: sample.md  title: Sparse Routing for Long-Context Retrieval
nodes: 23  references: 6 (6 resolved)  chunks: 7

outline:
  - Sparse Routing for Long-Context Retrieval
    - 1 Introduction
    - 2 Method
      - 2.1 Router
      - 2.2 Training
```

## Library use

```python
from pdfqa import Config, Pipeline

cfg = Config.load("pdfqa.yaml")
cfg.select.budget_tokens = 500_000
report = Pipeline(cfg).run(["paper.pdf"])
print(report["stats"])
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run fully offline against a scripted backend — no API key, no network. The PDF suite builds
its own fixture (`tests/fixtures/make_pdf.py`) containing a vector bar chart and a bordered table,
then asserts the chart is extracted as a figure, the table as a table, and both crops render.

## License

MIT
