"""Semantic chunking along AST boundaries. Never splits a table or equation."""

from __future__ import annotations

from .docast import CAPTION, EQUATION, FIGURE, HEADING, TABLE, DocumentTree, Node
from .records import Chunk, Provenance, estimate_tokens

ATOMIC = (TABLE, EQUATION, FIGURE)


def chunk_tree(tree: DocumentTree, max_tokens: int = 900, min_tokens: int = 60, overlap_headings: bool = True) -> list[Chunk]:
    sections = _section_units(tree)
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(_split_section(tree, section, max_tokens, overlap_headings))
    return _merge_small(chunks, min_tokens, max_tokens)


def _section_units(tree: DocumentTree) -> list[Node]:
    """Leaf-most headings plus any content hanging directly off the document root."""
    units = []
    for node in tree.root.walk():
        if node.kind == HEADING and not any(c.kind == HEADING for c in node.children):
            units.append(node)
        elif node.kind == HEADING and any(c.kind != HEADING for c in node.children):
            units.append(node)
    direct = [c for c in tree.root.children if c.kind != HEADING]
    if direct:
        units.insert(0, tree.root)
    return units or [tree.root]


def _own_blocks(node: Node) -> list[Node]:
    return [c for c in node.children if c.kind != HEADING]


def _split_section(tree: DocumentTree, section: Node, max_tokens: int, overlap_headings: bool) -> list[Chunk]:
    blocks = _own_blocks(section)
    if not blocks:
        return []
    breadcrumb = _breadcrumb(tree, section)
    out: list[Chunk] = []
    buf: list[Node] = []
    budget = 0

    def emit():
        nonlocal buf, budget
        if buf:
            out.append(_make_chunk(tree, section, buf, breadcrumb, overlap_headings))
        buf, budget = [], 0

    for block in blocks:
        cost = estimate_tokens(block.text)
        if block.kind in ATOMIC and cost > max_tokens:
            emit()
            out.append(_make_chunk(tree, section, [block], breadcrumb, overlap_headings))
            continue
        if budget + cost > max_tokens and buf:
            emit()
        buf.append(block)
        budget += cost
    emit()
    return out


def _breadcrumb(tree: DocumentTree, section: Node) -> str:
    trail = section.breadcrumb()
    own = section.title()
    if own:
        trail = trail + [own]
    return " > ".join(t for t in trail if t) or tree.meta.get("title", "Document")


def _make_chunk(tree: DocumentTree, section: Node, blocks: list[Node], breadcrumb: str, overlap_headings: bool) -> Chunk:
    text_parts = []
    tables, equations, figures, refs = [], [], [], []
    grids: list[list[list[str]]] = []
    figure_refs: list[dict] = []
    node_ids, pages, bboxes, anchors = [], [], [], []

    for block in blocks:
        node_ids.append(block.id)
        if block.span.anchor:
            anchors.append(block.span.anchor)
        else:
            pages.append(block.span.page)
        if block.span.bbox:
            bboxes.append(list(block.span.bbox))
        if block.kind == TABLE:
            tables.append(block.attrs.get("html") or block.text)
            if block.attrs.get("grid"):
                grids.append(block.attrs["grid"])
            text_parts.append(f"[{block.id}] {block.attrs.get('caption', '')}\n{block.text}".strip())
            if block.attrs.get("image_path"):
                figure_refs.append(_figure_ref(block))
        elif block.kind == EQUATION:
            equations.append(block.attrs.get("latex") or block.text)
            text_parts.append(f"[{block.id}] {block.text}")
        elif block.kind == FIGURE:
            figures.append(block.attrs.get("image_path", block.id))
            figure_refs.append(_figure_ref(block))
        else:
            text_parts.append(block.text)
        resolved = tree.resolve_context(block)
        if resolved:
            refs.append(resolved)

    body = "\n\n".join(p for p in text_parts if p.strip())
    if overlap_headings and breadcrumb:
        body = f"{breadcrumb}\n\n{body}"

    prov = Provenance(
        source=tree.source,
        node_ids=node_ids + [section.id],
        pages=sorted(set(pages)),
        anchors=anchors,
        bboxes=bboxes,
        section_path=[t for t in breadcrumb.split(" > ") if t],
        breadcrumb=breadcrumb,
        generator="chunker",
    )
    return Chunk(
        text=body,
        kind="table" if tables and len(blocks) == 1 else "section",
        prov=prov,
        tables=tables,
        grids=grids,
        equations=equations,
        figures=figures,
        figure_refs=figure_refs,
        resolved_refs="\n".join(refs),
    )


def _figure_ref(block: Node) -> dict:
    return {
        "id": block.id,
        "kind": block.kind,
        "caption": block.attrs.get("caption", ""),
        "page": block.span.page,
        "bbox": list(block.span.bbox) if block.span.bbox else None,
        "image_path": block.attrs.get("image_path", ""),
        "width": block.attrs.get("image_width", 0),
        "height": block.attrs.get("image_height", 0),
    }


def _merge_small(chunks: list[Chunk], min_tokens: int, max_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for chunk in chunks:
        if out and chunk.tokens < min_tokens and out[-1].tokens + chunk.tokens <= max_tokens and out[-1].prov.breadcrumb == chunk.prov.breadcrumb:
            prev = out[-1]
            merged = Chunk(
                text=prev.text + "\n\n" + chunk.text.replace(chunk.prov.breadcrumb + "\n\n", "", 1),
                kind=prev.kind,
                prov=prev.prov.merge(chunk.prov),
                tables=prev.tables + chunk.tables,
                grids=prev.grids + chunk.grids,
                equations=prev.equations + chunk.equations,
                figures=prev.figures + chunk.figures,
                figure_refs=prev.figure_refs + chunk.figure_refs,
                resolved_refs="\n".join(x for x in (prev.resolved_refs, chunk.resolved_refs) if x),
            )
            out[-1] = merged
        else:
            out.append(chunk)
    return out
