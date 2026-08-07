"""pdfqa — neuro-symbolic, graph-aware synthetic data engine for technical documents."""

from .chunking import chunk_tree
from .config import Config
from .docast import DocumentTree, Node
from .evaluate import evaluate as evaluate_model
from .export import audit, load_records, split_records, write_corpus
from .export import export as export_dataset
from .extract import from_markdown, load
from .graph import KnowledgeGraph, build_graph, merge_graphs
from .llm import Runtime
from .pipeline import Pipeline
from .plan import estimate
from .records import Chunk, Provenance, QARecord
from .registry import register_adapter, register_format, register_task
from .retrieve import Index, answer
from .tools import execute

__version__ = "1.1.0"

__all__ = [
    "Chunk",
    "Config",
    "DocumentTree",
    "Index",
    "KnowledgeGraph",
    "Node",
    "Pipeline",
    "Provenance",
    "QARecord",
    "Runtime",
    "answer",
    "audit",
    "build_graph",
    "chunk_tree",
    "estimate",
    "evaluate_model",
    "execute",
    "export_dataset",
    "from_markdown",
    "load",
    "load_records",
    "merge_graphs",
    "register_adapter",
    "register_format",
    "register_task",
    "split_records",
    "write_corpus",
]
