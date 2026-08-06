# PDF-to-QA

Turns technical PDFs into verified fine-tuning datasets. It parses documents into a typed AST,
builds a knowledge graph over them, generates SFT / DPO / multi-turn / ReAct data across that
graph, then gates every row through grounding checks before export.

Most PDF-to-dataset tools flatten the document to text, slide a window over it, and ask a model
for questions. That loses tables, section structure, and cross-references, and it can only ever
produce single-paragraph lookup questions. This one keeps the structure and generates against it.

```bash
pip install -e ".[all]"
pdfqa run papers/ --backend ollama --model qwen2.5:7b -o dataset/
```

## What it does

**Structure, not flat text.** PDFs parse into a typed AST — headings, tables, equations,
captions, footnotes — with parent/child/sibling links intact. Tables come out as HTML plus a cell
grid, equations keep their LaTeX and numbering. Every node carries its page index and bounding box.

**References resolve.** `[3]`, `Table 2`, `Eq. 4`, `Section 3.1` get bound to their actual targets
in the tree. When a chunk says "as shown in Table 2", the generator sees Table 2.

**Chunks follow the document.** Splits happen at section and block boundaries, never mid-table or
mid-equation. Each chunk keeps its breadcrumb (`Doc > Section 3 > Subsection 3.2`), its page
numbers, and its source node ids.

**A graph across the whole document.** Entities, methods, datasets, claims and quantities get
extracted into a document-level knowledge graph before any question is written. Multi-hop
questions are then sampled from entity pairs that are graph-connected but live in *different*
sections — so answering genuinely requires combining two places in the paper.

**Several data shapes, not just Q&A:**

| Shape | What you get |
| --- | --- |
| `qa` | Single-hop grounded pairs, mixed difficulty |
| `multihop` | Questions requiring two disjoint sections, sampled via KG paths |
| `multiturn` | Dialogue trees with follow-ups, clarification, and corrected user misassumptions |
| `react` | Tool-calling traces (python / sql / lookup) with real observations |
| DPO pairs | Chosen answer plus a mutated rejected one — number hallucination, unit mismatch, off-by-one, causal inversion, and five more |

Plus persona conditioning (domain expert, stakeholder, student, skeptical reviewer, ...), output
style conditioning (prose, bullets, JSON, step-by-step, proof), and an Evol-Instruct engine that
mutates questions to be harder — added constraints, counterfactuals, extra math steps, or
implicit phrasing.

**Everything gets gated.** Rows pass through cheap deterministic checks first, then model checks:

- *Forward NLI* — is the answer entailed by the source
- *Reverse NLI* — could the question be answered *without* the source (kills general-knowledge leakage)
- *Numeric grounding* — every number in the answer traced back to the source, with rounding tolerance
- *Shortcut detection* — rejects questions that already contain their own answer tokens
- *Symbolic check* — quantitative claims get compiled to Python and executed in a restricted
  subprocess; Z3 is used for boolean constraints when installed
- *Self-consistency* — optional re-answer voting

**Then it's deduped and coreset-selected.** MinHash + LSH kills near-identical source text and
near-identical questions; k-center greedy drops semantic restatements; a greedy submodular (DPP-style)
pass maximises diversity per token of budget; a final pass balances the simple/intermediate/complex mix.

**Provenance survives to the export.** Every row carries source file, page indexes, bounding boxes,
AST node ids, section breadcrumb, and the full pass/fail log of every gate it went through.

## Runtime

Roles route independently, so bulk generation can run locally while verification hits a stronger
cloud model:

```yaml
runtime:
  generate: { backend: ollama, model: qwen2.5:7b, base_url: http://localhost:11434 }
  verify:   { backend: anthropic, model: claude-sonnet-4-5 }
  workers: 8
```

Backends: `ollama`, `vllm` and any OpenAI-compatible server, `openai`, `anthropic`, and `echo`
(offline, deterministic — the whole pipeline runs with no API key).

Parsing prefers PyMuPDF, falls back to pdfminer.six. The core has **zero required dependencies**;
PDF parsing, Parquet, YAML config, and Z3 are all optional extras.

## Exports

`chatml`, `sharegpt`, `alpaca`, `openai`, `axolotl`, `llamafactory`, `unsloth`, `dpo`, `react`,
`raw`, plus a Parquet file for `datasets.load_dataset` and a generated dataset card.

```bash
pdfqa run docs/ -f chatml dpo parquet --limit 2000
```

## Incremental

Every stage is content-addressed. Change a prompt and only synthesis re-runs; change the parser
and only extraction re-runs. Adding a PDF to a directory of 500 doesn't reprocess the other 499.

```bash
pdfqa cache            # what's stored
pdfqa cache --clear --stage graph
```

## Commands

```bash
pdfqa run <inputs>       # full pipeline
pdfqa inspect <file>     # AST outline, resolved references, chunk plan
pdfqa graph <file>       # knowledge graph stats and multi-hop seed pairs
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

Tests run fully offline against a scripted backend — no API key, no network.

## License

MIT
