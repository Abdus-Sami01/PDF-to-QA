"""Typed document AST: nodes, tree, breadcrumbs, and reference binding."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

HEADING = "heading"
PARAGRAPH = "paragraph"
TABLE = "table"
EQUATION = "equation"
CAPTION = "caption"
FOOTNOTE = "footnote"
LIST = "list"
CODE = "code"
FIGURE = "figure"
DOCUMENT = "document"

BLOCK_KINDS = (PARAGRAPH, TABLE, EQUATION, CAPTION, FOOTNOTE, LIST, CODE, FIGURE)


@dataclass
class Span:
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None

    def as_dict(self) -> dict:
        return {"page": self.page, "bbox": list(self.bbox) if self.bbox else None}


@dataclass
class Node:
    kind: str
    text: str = ""
    level: int = 0
    span: Span = field(default_factory=Span)
    attrs: dict = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    parent: Optional["Node"] = field(default=None, repr=False, compare=False)
    id: str = ""

    def add(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def ancestors(self) -> list["Node"]:
        out, cur = [], self.parent
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        out.reverse()
        return out

    def breadcrumb(self) -> list[str]:
        out: list[str] = []
        for a in self.ancestors():
            t = a.title()
            if t and (not out or out[-1] != t):
                out.append(t)
        return out

    def title(self) -> str:
        if self.kind == DOCUMENT:
            return self.attrs.get("title", "Document")
        if self.kind == HEADING:
            return self.text.strip()
        return ""

    def walk(self) -> Iterator["Node"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def siblings(self) -> list["Node"]:
        if self.parent is None:
            return []
        return [c for c in self.parent.children if c is not self]

    def text_content(self, limit: int | None = None) -> str:
        parts = [n.text for n in self.walk() if n.text]
        joined = "\n\n".join(parts)
        return joined[:limit] if limit else joined

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "level": self.level,
            "span": self.span.as_dict(),
            "attrs": {k: v for k, v in self.attrs.items() if not k.startswith("_")},
            "children": [c.as_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(data: dict, parent: "Node | None" = None) -> "Node":
        span = Span(
            page=data.get("span", {}).get("page", 0),
            bbox=tuple(data["span"]["bbox"]) if data.get("span", {}).get("bbox") else None,
        )
        node = Node(
            kind=data["kind"],
            text=data.get("text", ""),
            level=data.get("level", 0),
            span=span,
            attrs=data.get("attrs", {}),
            id=data.get("id", ""),
            parent=parent,
        )
        for c in data.get("children", []):
            node.children.append(Node.from_dict(c, node))
        return node


REF_PATTERNS = [
    (r"\[(\d{1,3})\]", "citation"),
    (r"\bTable\s+(\d{1,3}[A-Za-z]?)\b", "table"),
    (r"\bFigure\s+(\d{1,3}[A-Za-z]?)\b", "figure"),
    (r"\bFig\.\s*(\d{1,3}[A-Za-z]?)\b", "figure"),
    (r"\bEq(?:uation|\.)\s*\(?(\d{1,3})\)?", "equation"),
    (r"\bSection\s+(\d+(?:\.\d+)*)\b", "section"),
    (r"\bAppendix\s+([A-Z])\b", "section"),
]

LABEL_PATTERNS = [
    (r"^\s*Table\s+(\d{1,3}[A-Za-z]?)\b", "table"),
    (r"^\s*Figure\s+(\d{1,3}[A-Za-z]?)\b", "figure"),
    (r"^\s*Fig\.\s*(\d{1,3}[A-Za-z]?)\b", "figure"),
    (r"^\s*\[(\d{1,3})\]", "citation"),
    (r"^\s*\((\d{1,3})\)\s*$", "equation"),
]


def _captioned(caption: "Node", kind: str) -> "Node | None":
    """A caption labels the nearest matching block among its siblings, either side."""
    want = TABLE if kind == "table" else FIGURE
    if caption.parent is None:
        return None
    sibs = caption.parent.children
    i = sibs.index(caption)
    for j in list(range(i + 1, len(sibs))) + list(range(i - 1, -1, -1)):
        if sibs[j].kind == want:
            return sibs[j]
    return None


@dataclass
class Reference:
    source_id: str
    target_id: str | None
    kind: str
    key: str
    surface: str
    resolved: bool = False


class DocumentTree:
    """Owns the root node plus derived indexes: ids, labels, and reference edges."""

    def __init__(self, root: Node, source: str = "", meta: dict | None = None):
        self.root = root
        self.source = source
        self.meta = meta or {}
        self.index: dict[str, Node] = {}
        self.labels: dict[tuple[str, str], str] = {}
        self.references: list[Reference] = []
        self.assign_ids()

    def assign_ids(self) -> None:
        self.index.clear()
        counter: dict[str, int] = {}
        for node in self.root.walk():
            if not node.id:
                n = counter.get(node.kind, 0)
                counter[node.kind] = n + 1
                seed = f"{self.source}:{node.kind}:{n}:{node.text[:64]}"
                digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
                node.id = f"{node.kind[:3]}-{digest}"
            self.index[node.id] = node

    def nodes(self, kinds: Iterable[str] | None = None) -> list[Node]:
        kinds = set(kinds) if kinds else None
        return [n for n in self.root.walk() if kinds is None or n.kind in kinds]

    def section_path(self, node: Node) -> str:
        trail = node.breadcrumb()
        return " > ".join(trail) if trail else self.meta.get("title", "Document")

    def build_labels(self) -> None:
        self.labels.clear()
        for node in self.root.walk():
            probe = node.attrs.get("label") or node.text
            if node.kind == HEADING:
                m = re.match(r"^\s*(\d+(?:\.\d+)*)\s", node.text)
                if m:
                    self.labels[("section", m.group(1))] = node.id
                m = re.match(r"^\s*Appendix\s+([A-Z])\b", node.text)
                if m:
                    self.labels[("section", m.group(1))] = node.id
            for pattern, kind in LABEL_PATTERNS:
                m = re.match(pattern, probe)
                if m:
                    target = node
                    if kind in ("table", "figure") and node.kind == CAPTION:
                        target = _captioned(node, kind) or node
                    self.labels.setdefault((kind, m.group(1)), target.id)
            if node.kind == EQUATION and node.attrs.get("number"):
                self.labels.setdefault(("equation", str(node.attrs["number"])), node.id)

    def bind_references(self) -> list[Reference]:
        self.build_labels()
        self.references.clear()
        for node in self.root.walk():
            if not node.text:
                continue
            for pattern, kind in REF_PATTERNS:
                for m in re.finditer(pattern, node.text):
                    key = m.group(1)
                    target = self.labels.get((kind, key))
                    if target == node.id:
                        continue
                    self.references.append(
                        Reference(
                            source_id=node.id,
                            target_id=target,
                            kind=kind,
                            key=key,
                            surface=m.group(0),
                            resolved=target is not None,
                        )
                    )
        return self.references

    def outbound(self, node_id: str) -> list[Reference]:
        return [r for r in self.references if r.source_id == node_id and r.resolved]

    def inbound(self, node_id: str) -> list[Reference]:
        return [r for r in self.references if r.target_id == node_id]

    def resolve_context(self, node: Node, max_chars: int = 1200) -> str:
        """Inline the targets of a node's references so implicit mentions become explicit."""
        parts = []
        for ref in self.outbound(node.id):
            target = self.index.get(ref.target_id or "")
            if target is None:
                continue
            body = target.attrs.get("html") or target.text
            parts.append(f"[{ref.surface} -> {target.id}] {body.strip()}")
        return "\n".join(parts)[:max_chars]

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "meta": self.meta,
            "root": self.root.as_dict(),
            "references": [r.__dict__ for r in self.references],
        }

    @staticmethod
    def from_dict(data: dict) -> "DocumentTree":
        tree = DocumentTree(Node.from_dict(data["root"]), data.get("source", ""), data.get("meta", {}))
        tree.references = [Reference(**r) for r in data.get("references", [])]
        tree.build_labels()
        return tree
