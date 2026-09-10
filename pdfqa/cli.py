"""Command line entry point: `pdfqa plan`, `run`, `inspect`, `graph`, `ask`, `eval`, `report`, `cache`, `init`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cache import Store
from .chunking import chunk_tree
from .config import Config
from .docast import EQUATION, HEADING
from .export import audit, load_records
from .evaluate import evaluate
from .extract import ExtractError, load
from .graph import build_graph, dump
from .llm import LLMError, Runtime
from .pipeline import Pipeline, _expand
from .plan import collect_chunks, estimate, format_plan, rates_from_report
from .retrieve import Index, answer
from .select import distribution_report


def _progress(stage: str, info: dict) -> None:
    print(f"[{stage}] " + " ".join(f"{k}={v}" for k, v in info.items()), file=sys.stderr, flush=True)


def _config(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    if args.inputs:
        cfg.inputs = args.inputs
    if args.out:
        cfg.outdir = args.out
    if getattr(args, "formats", None):
        cfg.formats = args.formats
    if getattr(args, "split", None):
        cfg.split = list(args.split)
    if getattr(args, "no_corpus", False):
        cfg.corpus = False
    if getattr(args, "against", None):
        cfg.select.against = args.against
    if getattr(args, "no_cache", False):
        cfg.cache = False
    if getattr(args, "backend", None):
        cfg.runtime.generate = {"backend": args.backend, **({"model": args.model} if args.model else {})}
        cfg.runtime.verify = dict(cfg.runtime.generate)
    if getattr(args, "workers", None):
        cfg.runtime.workers = args.workers
    if getattr(args, "limit", None):
        cfg.select.target_count = args.limit
    if getattr(args, "vision", None):
        cfg.runtime.vision = {"backend": args.vision, **({"model": args.vision_model} if args.vision_model else {})}
    if getattr(args, "no_figures", False):
        cfg.assets_dir = None
        cfg.synth.figure_qa_per_doc = 0
    if getattr(args, "no_exec", False):
        cfg.verify.allow_exec = False
    if getattr(args, "offline", False):
        cfg.verify.model_gates = False
        cfg.graph.llm_extract = False
    return cfg


def cmd_run(args) -> int:
    cfg = _config(args)
    if getattr(args, "budget_tokens", None):
        cfg.runtime.budget_tokens = args.budget_tokens
    report = Pipeline(cfg, _progress).run()
    print(json.dumps(report, indent=2, default=str))
    _warn_about_losses(report)
    return 0


def _warn_about_losses(report: dict) -> None:
    """Failures that were survived are still failures, and stderr is where they get noticed."""
    errors = report.get("errors")
    if errors:
        kinds = ", ".join(f"{k} x{v}" for k, v in sorted(errors["by_type"].items()))
        print(f"[warn] {errors['count']} task(s) failed and were skipped: {kinds}", file=sys.stderr)
        for sample in errors["samples"][:3]:
            print(f"       {sample['stage']}: {sample['error']}", file=sys.stderr)
    unparsed = (report.get("runtime") or {}).get("unparsed")
    if unparsed:
        print(f"[warn] {sum(unparsed.values())} model reply(s) did not parse as JSON: "
              + ", ".join(f"{k} x{v}" for k, v in unparsed.items()), file=sys.stderr)
    if report.get("halted"):
        print(f"[warn] run stopped early: {report['halted']}", file=sys.stderr)
    for skip in report.get("skipped", []):
        print(f"[warn] skipped {skip['path']}: {skip['reason']}", file=sys.stderr)


def cmd_inspect(args) -> int:
    tree = load(args.input, args.backend or "auto", args.assets, args.dpi)
    chunks = chunk_tree(tree)
    if args.json:
        print(json.dumps({"tree": tree.as_dict(), "chunks": [c.context() for c in chunks]}, indent=2))
        return 0
    print(f"source: {tree.source}  title: {tree.meta.get('title', '')}")
    print(f"nodes: {len(tree.index)}  references: {len(tree.references)} "
          f"({sum(1 for r in tree.references if r.resolved)} resolved)  chunks: {len(chunks)}")
    print("\noutline:")
    for node in tree.nodes([HEADING]):
        print("  " * node.level + f"- {node.text[:90]}  [{node.id}]")
    print("\nchunks:")
    for c in chunks:
        print(f"  {c.id}  {c.tokens:>5}tok  p{c.prov.pages}  {c.prov.breadcrumb[:70]}")
    rendered = [n for n in tree.root.walk() if n.attrs.get("image_path")]
    if rendered:
        print(f"\nrendered images: {len(rendered)}")
        for n in rendered:
            print(f"  {n.id}  p{n.span.page}  {n.attrs['image_width']}x{n.attrs['image_height']}  "
                  f"{n.attrs.get('caption', '')[:50]}  {n.attrs['image_path']}")
    broken = [n for n in tree.nodes([EQUATION]) if n.attrs.get("issues")]
    if broken:
        print(f"\nmalformed equations: {len(broken)} of {len(tree.nodes([EQUATION]))}")
        for n in broken[:10]:
            print(f"  [{n.id}] {n.text[:60]!r}\n      {'; '.join(n.attrs['issues'])}")
    unresolved = [r for r in tree.references if not r.resolved]
    if unresolved:
        print(f"\nunresolved references: {len(unresolved)}")
        for r in unresolved[:10]:
            print(f"  {r.surface} ({r.kind}) from {r.source_id}")
    return 0


def cmd_graph(args) -> int:
    cfg = _config(args)
    tree = load(args.input, cfg.backend)
    chunks = chunk_tree(tree, cfg.chunk.max_tokens, cfg.chunk.min_tokens)
    runtime = Runtime(cfg.runtime_specs())
    kg = build_graph(chunks, runtime, cfg.graph.llm_extract and not args.offline)
    if args.json:
        print(dump(kg))
        return 0
    print(json.dumps(kg.stats(), indent=2))
    print("\nbridging pairs (multi-hop seeds):")
    for a, b, path in kg.bridging_pairs()[:20]:
        print(f"  {a.name}  <->  {b.name}   ({len(path)} hops)")
    return 0


def cmd_report(args) -> int:
    records = load_records(args.path)
    if not records:
        print(f"no raw.jsonl records under {args.path}", file=sys.stderr)
        return 1
    summary = audit(records) | {"composition": distribution_report(records)}
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    comp = summary["composition"]
    print(f"records: {summary['count']}   mean quality: {comp['quality_mean']}   "
          f"p10/p50/p90: {summary['quality'].get('p10')}/{summary['quality'].get('p50')}/{summary['quality'].get('p90')}")
    print(f"task: {comp['task']}\ndifficulty: {comp['difficulty']}\npersona: {comp['persona']}")
    print(f"\ndpo pairs: {summary['with_rejected']}   multimodal: {summary['with_images']}   "
          f"tool traces: {summary['with_tool_trace']} ({summary['repaired_traces']} repaired)   "
          f"dialogues: {summary['multi_turn']}")
    gates, scope = _gate_rates(args.path, summary)
    if gates:
        print(f"\ngate pass rates ({scope}):")
        for name, counts in gates.items():
            total = counts["pass"] + counts["fail"]
            print(f"  {name:<20} {counts['pass']}/{total}  ({100 * counts['pass'] / max(1, total):.0f}%)")
    print("\nper source:")
    for src, n in list(summary["sources"].items())[:20]:
        print(f"  {n:>6}  {src}")
    _print_yield(args.path)
    _print_spend(args.path)
    return 0


def _print_spend(path: str) -> None:
    """What the dataset cost, and what was lost on the way — both are invisible in the records."""
    report = _run_report(path)
    health, errors = report.get("runtime") or {}, report.get("errors")
    if health:
        tokens = health["tokens"]
        print(f"\nspend: {health['calls']} model calls   "
              f"{tokens['prompt']} prompt + {tokens['completion']} completion = {tokens['total']} tokens")
        if health["failures"]:
            print(f"  {health['failures']} call(s) failed: {health['first_errors'][0] if health['first_errors'] else ''}")
        if health.get("unparsed"):
            total = sum(health["unparsed"].values())
            print(f"  {total} reply(s) were paid for but could not be parsed: "
                  + ", ".join(f"{k} x{v}" for k, v in health["unparsed"].items()))
            for sample in health.get("first_unparsed", [])[:2]:
                print(f"       {sample}")
    if errors:
        print(f"  {errors['count']} task(s) skipped after errors: "
              + ", ".join(f"{k} x{v}" for k, v in sorted(errors["by_type"].items())))
    if report.get("halted"):
        print(f"  run stopped early: {report['halted']}")


def _run_report(path: str) -> dict:
    found = sorted(Path(path).rglob("run_report.json")) if Path(path).is_dir() else []
    return json.loads(found[0].read_text()) if found else {}


def _gate_rates(path: str, summary: dict) -> tuple[dict, str]:
    """Prefer the run report: the exported dataset holds only survivors, so rates taken from it
    are ~100% for every gate and say nothing about what the gates rejected."""
    gates = _run_report(path).get("gates")
    if gates:
        return gates, "all generated records"
    return summary["gates"], "surviving records only — run report not found"


def _print_yield(path: str) -> None:
    """Composition says what survived; yield says what was paid for and lost."""
    rows = _run_report(path).get("yield") or {}
    if not rows:
        return
    print(f"\n{'shape':<16}{'generated':>10}{'verified':>10}{'final':>7}   lost to")
    for shape, row in rows.items():
        reasons = ", ".join(f"{name} ({n})" for name, n in row["top_rejections"].items()) or "-"
        print(f"  {shape:<14}{row['generated']:>10}{row['verified']:>10}{row['final']:>7}   {reasons}")


def cmd_plan(args) -> int:
    cfg = _config(args)
    paths = _expand(list(args.inputs or cfg.inputs))
    if not paths:
        print("no inputs given", file=sys.stderr)
        return 1
    rates = rates_from_report(_run_report(args.from_run)) if args.from_run else None
    estimated = estimate(cfg, collect_chunks(cfg, paths), rates)
    if args.json:
        print(json.dumps(estimated.as_dict(args.price_in, args.price_out), indent=2))
        return 0
    print(format_plan(estimated, args.price_in, args.price_out))
    return 0


def cmd_ask(args) -> int:
    cfg = _config(args)
    index = Index.load(args.corpus)
    runtime = Runtime(cfg.runtime_specs())
    if not args.lexical_only:
        index.embed(runtime if cfg.runtime.embed else None)
    found = answer(runtime, index, args.question, args.k, args.alpha)
    if args.json:
        print(json.dumps({
            "question": args.question,
            "answer": found.text,
            "unanswerable": found.unanswerable,
            "citations": found.citations(),
            "hits": [{"id": h.passage.id, "source": h.passage.source, "breadcrumb": h.passage.breadcrumb,
                      "score": round(h.score, 4), "lexical": round(h.lexical, 4), "dense": round(h.dense, 4)}
                     for h in found.hits],
        }, indent=2))
        return 0
    print(found.text + "\n")
    print("sources:")
    for h in found.hits:
        print(f"  {h.score:.3f}  {h.passage.citation():<28} {h.passage.breadcrumb[:60]}")
    return 0 if not found.unanswerable else 2


def cmd_eval(args) -> int:
    cfg = _config(args)
    records = load_records(args.path)
    if not records:
        print(f"no raw.jsonl records under {args.path}", file=sys.stderr)
        return 1
    runtime = Runtime(cfg.runtime_specs())
    index = None
    if args.mode == "retrieval":
        index = Index.load(args.corpus or Path(args.path).parent)
        index.embed(runtime if cfg.runtime.embed else None)
    summary = evaluate(runtime, records, index, args.mode, args.k, args.limit)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"mode: {summary['mode']}   graded: {summary['count']}")
    print(f"accuracy: {summary['accuracy']:.1%}   partial: {summary['partial_rate']:.1%}   "
          f"token F1: {summary['token_f1']:.3f}   numeric match: {summary['numeric_match']:.3f}")
    if "retrieval_recall" in summary:
        print(f"retrieval recall (correct source in top-{args.k}): {summary['retrieval_recall']:.1%}")
    print("\nby task:")
    for task, counts in summary["by_task"].items():
        total = sum(counts.values())
        print(f"  {task:<16} {counts.get('correct', 0)}/{total} correct")
    return 0


def cmd_cache(args) -> int:
    store = Store(args.dir)
    if args.clear:
        n = store.invalidate(args.stage)
        print(f"cleared {n} entries" + (f" from {args.stage}" if args.stage else ""))
        return 0
    print(json.dumps(store.stats(), indent=2))
    return 0


def cmd_init(args) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    cfg = Config()
    cfg.inputs = ["docs/"]
    cfg.runtime.generate = {"backend": "ollama", "model": "qwen2.5:7b", "base_url": "http://localhost:11434"}
    cfg.runtime.verify = {"backend": "anthropic", "model": "claude-sonnet-4-5"}
    data = cfg.as_dict()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdfqa", description="Neuro-symbolic document to fine-tuning dataset engine")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline")
    run.add_argument("inputs", nargs="*", help="PDF/markdown files, directories, or globs")
    run.add_argument("-c", "--config")
    run.add_argument("-o", "--out")
    run.add_argument("-f", "--formats", nargs="+")
    run.add_argument("--backend", help="generate backend: openai, anthropic, ollama, vllm, echo")
    run.add_argument("--model")
    run.add_argument("--workers", type=int)
    run.add_argument("--limit", type=int, help="target record count after selection")
    run.add_argument("--vision", help="vision backend for figure QA, e.g. anthropic or ollama")
    run.add_argument("--vision-model")
    run.add_argument("--no-figures", action="store_true", help="skip figure rendering and multimodal synthesis")
    run.add_argument("--offline", action="store_true", help="skip all LLM-dependent gates and extraction")
    run.add_argument("--split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                     help="also write train/validation/test subdirectories, grouped by source document")
    run.add_argument("--no-corpus", action="store_true", help="skip the retrieval corpus export")
    run.add_argument("--against", nargs="+", help="drop questions that near-duplicate these earlier exports")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--no-exec", action="store_true",
                     help="never run model-written code; trace and symbolic gates report inconclusive instead")
    run.add_argument("--budget-tokens", type=int,
                     help="stop generating once this many model tokens are spent, and export what is done")
    run.set_defaults(func=cmd_run)

    ins = sub.add_parser("inspect", help="show the parsed AST, references, and chunk plan")
    ins.add_argument("input")
    ins.add_argument("--backend")
    ins.add_argument("--assets", help="render figures and tables as PNG crops into this directory")
    ins.add_argument("--dpi", type=int, default=144)
    ins.add_argument("--json", action="store_true")
    ins.set_defaults(func=cmd_inspect)

    gr = sub.add_parser("graph", help="build and print the document knowledge graph")
    gr.add_argument("input")
    gr.add_argument("-c", "--config")
    gr.add_argument("--backend")
    gr.add_argument("--model")
    gr.add_argument("--offline", action="store_true")
    gr.add_argument("--json", action="store_true")
    gr.set_defaults(func=cmd_graph, inputs=None, out=None)

    rp = sub.add_parser("report", help="audit an exported dataset: composition, quality, gate pass rates")
    rp.add_argument("path", help="output directory or a raw.jsonl file")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=cmd_report)

    pl = sub.add_parser("plan", help="estimate calls, tokens and cost without contacting a model")
    pl.add_argument("inputs", nargs="*")
    pl.add_argument("-c", "--config")
    pl.add_argument("--backend")
    pl.add_argument("--price-in", type=float, default=0.0, help="input price per million tokens")
    pl.add_argument("--price-out", type=float, default=0.0, help="output price per million tokens")
    pl.add_argument("--json", action="store_true")
    pl.add_argument("--from-run", metavar="DIR",
                    help="price verification from this earlier run's gate counts instead of assuming nothing is filtered")
    pl.set_defaults(func=cmd_plan, out=None)

    ask = sub.add_parser("ask", help="answer a question from an exported corpus, with citations")
    ask.add_argument("question")
    ask.add_argument("corpus", nargs="?", default="out", help="output directory or a corpus.jsonl file")
    ask.add_argument("-c", "--config")
    ask.add_argument("--backend")
    ask.add_argument("--model")
    ask.add_argument("-k", type=int, default=5, help="passages to retrieve")
    ask.add_argument("--alpha", type=float, default=0.5, help="1.0 pure BM25, 0.0 pure embedding")
    ask.add_argument("--lexical-only", action="store_true")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask, inputs=None, out=None)

    ev = sub.add_parser("eval", help="grade a model against a generated split")
    ev.add_argument("path", help="dataset directory or raw.jsonl (usually the test split)")
    ev.add_argument("-c", "--config")
    ev.add_argument("--backend")
    ev.add_argument("--model")
    ev.add_argument("--mode", choices=["context", "retrieval", "closed"], default="context")
    ev.add_argument("--corpus", help="corpus for retrieval mode; defaults to the dataset's parent")
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("--limit", type=int)
    ev.add_argument("--json", action="store_true")
    ev.set_defaults(func=cmd_eval, inputs=None, out=None)

    ca = sub.add_parser("cache", help="inspect or clear the incremental cache")
    ca.add_argument("--dir", default=".pdfqa-cache")
    ca.add_argument("--stage", help="ast, chunks, graph, or synth")
    ca.add_argument("--clear", action="store_true")
    ca.set_defaults(func=cmd_cache)

    ini = sub.add_parser("init", help="write a starter config file")
    ini.add_argument("path", nargs="?", default="pdfqa.yaml")
    ini.add_argument("--force", action="store_true")
    ini.set_defaults(func=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, ExtractError, ValueError) as exc:
        # These are the ways a run fails on the user's configuration rather than on a bug, and a
        # traceback buries the one line that says which.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
