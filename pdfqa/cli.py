"""Command line entry point: `pdfqa run`, `inspect`, `graph`, `cache`, `init`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cache import Store
from .chunking import chunk_tree
from .config import Config
from .docast import HEADING
from .extract import load
from .graph import build_graph, dump
from .llm import Runtime
from .pipeline import Pipeline


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
    if getattr(args, "offline", False):
        cfg.verify.model_gates = False
        cfg.graph.llm_extract = False
    return cfg


def cmd_run(args) -> int:
    cfg = _config(args)
    report = Pipeline(cfg, _progress).run()
    print(json.dumps(report, indent=2, default=str))
    return 0


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
    p = argparse.ArgumentParser(prog="pdfqa", description="Neuro-symbolic PDF to fine-tuning dataset engine")
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
    run.add_argument("--no-cache", action="store_true")
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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
