"""pdfqa — neuro-symbolic, graph-aware synthetic dataset engine for technical documents."""

from .chunking import chunk_tree
from .config import Config
from .docast import DocumentTree, Node
from .extract import from_markdown, load
from .graph import KnowledgeGraph, build_graph
from .llm import Runtime
from .pipeline import Pipeline
from .records import Chunk, Provenance, QARecord

__version__ = "0.9.0"

__all__ = [
    "Chunk",
    "Config",
    "DocumentTree",
    "KnowledgeGraph",
    "Node",
    "Pipeline",
    "Provenance",
    "QARecord",
    "Runtime",
    "build_graph",
    "chunk_tree",
    "from_markdown",
    "load",
]
