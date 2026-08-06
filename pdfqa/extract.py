"""Document extraction into a typed AST. PDF backends are optional; text/markdown always works."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .docast import (
    CAPTION,
    CODE,
    DOCUMENT,
    EQUATION,
    FOOTNOTE,
    HEADING,
    LIST,
    PARAGRAPH,
    TABLE,
    DocumentTree,
    Node,
    Span,
)

CAPTION_RE = re.compile(r"^\s*(Table|Figure|Fig\.)\s*\d", re.I)
FOOTNOTE_RE = re.compile(r"^\s*(\[\d{1,3}\]|\d{1,2}\s{2,}|†|‡|\*)\s*\S")
EQ_NUM_RE = re.compile(r"\((\d{1,3})\)\s*$")
MATH_CHARS = set("∑∫∂√≈≠≤≥±×÷αβγδθλμσπΣΩ∈∀∃→⇒^_=")


def load(path: str | Path, backend: str = "auto") -> DocumentTree:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return from_markdown(p.read_text(encoding="utf-8", errors="replace"), source=p.name)
    if suffix != ".pdf":
        raise ValueError(f"unsupported input type: {suffix}")
    if backend in ("auto", "pymupdf"):
        try:
            return _from_pymupdf(p)
        except ImportError:
            if backend == "pymupdf":
                raise
    if backend in ("auto", "pdfminer"):
        try:
            return _from_pdfminer(p)
        except ImportError:
            if backend == "pdfminer":
                raise
    raise RuntimeError("no PDF backend available; install pymupdf or pdfminer.six")


# --------------------------------------------------------------------------- markdown


def from_markdown(text: str, source: str = "inline") -> DocumentTree:
    root = Node(DOCUMENT, attrs={"title": _guess_title(text) or source})
    stack: list[Node] = [root]
    blocks = _split_markdown_blocks(text)
    pending_caption: Node | None = None

    for block in blocks:
        kind, payload, attrs = block
        if kind == HEADING:
            level = attrs["level"]
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            node = stack[-1].add(Node(HEADING, text=payload, level=level))
            stack.append(node)
            continue
        node = Node(kind, text=payload, attrs=attrs)
        stack[-1].add(node)
        if kind == CAPTION:
            pending_caption = node
        elif pending_caption is not None and kind in (TABLE, EQUATION):
            node.attrs["caption"] = pending_caption.text
            pending_caption = None

    tree = DocumentTree(root, source=source, meta={"title": root.attrs["title"], "format": "markdown"})
    tree.bind_references()
    return tree


def _guess_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line:
            return line[:120]
    return ""


def _split_markdown_blocks(text: str) -> list[tuple[str, str, dict]]:
    lines = text.splitlines()
    out: list[tuple[str, str, dict]] = []
    buf: list[str] = []
    mode = PARAGRAPH
    i = 0

    def flush():
        nonlocal buf, mode
        if buf:
            body = "\n".join(buf).strip()
            if body:
                out.append(_classify_text_block(body) if mode == PARAGRAPH else (mode, body, {}))
        buf = []
        mode = PARAGRAPH

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            fence, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fence.append(lines[i])
                i += 1
            out.append((CODE, "\n".join(fence), {}))
            i += 1
            continue
        if stripped.startswith("$$"):
            flush()
            body, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("$$"):
                body.append(lines[i])
                i += 1
            latex = "\n".join(body).strip()
            out.append((EQUATION, latex, _equation_attrs(latex)))
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush()
            out.append((HEADING, m.group(2).strip(), {"level": len(m.group(1))}))
            i += 1
            continue
        if "|" in stripped and stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            out.append((TABLE, "\n".join(rows), _table_attrs(rows)))
            continue
        if not stripped:
            flush()
            i += 1
            continue
        if re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            if mode != LIST:
                flush()
            mode = LIST
            buf.append(line)
            i += 1
            continue
        if mode == LIST:
            flush()
        buf.append(line)
        i += 1
    flush()
    return out


def _classify_text_block(body: str) -> tuple[str, str, dict]:
    if CAPTION_RE.match(body) and len(body) < 400:
        return (CAPTION, body, {})
    if FOOTNOTE_RE.match(body) and len(body) < 400:
        return (FOOTNOTE, body, {})
    first = body.splitlines()[0]
    if len(body) < 200 and sum(c in MATH_CHARS for c in first) >= 3:
        return (EQUATION, body, _equation_attrs(body))
    return (PARAGRAPH, body, {})


def _equation_attrs(latex: str) -> dict:
    attrs: dict = {"latex": latex}
    m = EQ_NUM_RE.search(latex.strip())
    if m:
        attrs["number"] = m.group(1)
    attrs["symbols"] = sorted({s for s in re.findall(r"\\[a-zA-Z]+|[A-Za-z]\w*", latex)})[:32]
    attrs["balanced"] = _balanced(latex)
    return attrs


def _balanced(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _table_attrs(rows: list[str]) -> dict:
    grid = []
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        grid.append(cells)
    return {"grid": grid, "html": _grid_to_html(grid), "n_rows": len(grid), "n_cols": max((len(r) for r in grid), default=0)}


def _grid_to_html(grid: list[list[str]]) -> str:
    if not grid:
        return "<table></table>"
    head = "".join(f"<th>{c}</th>" for c in grid[0])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in grid[1:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------- pdf


def _from_pymupdf(path: Path) -> DocumentTree:
    import fitz  # type: ignore

    doc = fitz.open(path)
    raw: list[dict] = []
    tables: list[dict] = []
    for pno, page in enumerate(doc):
        for tbl in _pymupdf_tables(page):
            tables.append({"page": pno, **tbl})
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                raw.append({"kind": "image", "page": pno, "bbox": bbox, "text": "", "size": 0.0, "bold": False})
                continue
            lines, sizes, bold = [], [], 0
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                if text.strip():
                    lines.append(text)
                for s in line.get("spans", []):
                    sizes.append(round(s.get("size", 0.0), 1))
                    bold += 1 if "bold" in s.get("font", "").lower() else 0
            body = "\n".join(lines).strip()
            if not body:
                continue
            raw.append(
                {
                    "kind": "text",
                    "page": pno,
                    "bbox": tuple(block.get("bbox", (0, 0, 0, 0))),
                    "text": body,
                    "size": max(sizes) if sizes else 0.0,
                    "bold": bold > 0,
                }
            )
    doc.close()
    return _assemble(raw, tables, source=path.name, title=_pdf_title(raw, path))


def _pymupdf_tables(page) -> list[dict]:
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return []
    try:
        found = finder()
    except Exception:
        return []
    out = []
    for t in getattr(found, "tables", []):
        try:
            grid = [[("" if c is None else str(c)).strip() for c in row] for row in t.extract()]
        except Exception:
            continue
        if not grid:
            continue
        out.append({"bbox": tuple(t.bbox), "grid": grid, "html": _grid_to_html(grid)})
    return out


def _from_pdfminer(path: Path) -> DocumentTree:
    from pdfminer.high_level import extract_pages  # type: ignore
    from pdfminer.layout import LTTextContainer, LTChar, LTFigure, LTImage  # type: ignore

    raw: list[dict] = []
    for pno, layout in enumerate(extract_pages(str(path))):
        for element in layout:
            if isinstance(element, (LTFigure, LTImage)):
                raw.append({"kind": "image", "page": pno, "bbox": tuple(element.bbox), "text": "", "size": 0.0, "bold": False})
                continue
            if not isinstance(element, LTTextContainer):
                continue
            body = element.get_text().strip()
            if not body:
                continue
            sizes = [round(c.size, 1) for line in element for c in line if isinstance(c, LTChar)]
            fonts = [c.fontname.lower() for line in element for c in line if isinstance(c, LTChar)]
            raw.append(
                {
                    "kind": "text",
                    "page": pno,
                    "bbox": tuple(element.bbox),
                    "text": body,
                    "size": max(sizes) if sizes else 0.0,
                    "bold": any("bold" in f for f in fonts),
                }
            )
    return _assemble(raw, [], source=path.name, title=_pdf_title(raw, path))


def _pdf_title(raw: list[dict], path: Path) -> str:
    first = [b for b in raw if b["page"] == 0 and b["text"]]
    if not first:
        return path.stem
    return max(first, key=lambda b: b["size"])["text"].splitlines()[0][:200]


def _assemble(raw: list[dict], tables: list[dict], source: str, title: str) -> DocumentTree:
    body_size = _body_size(raw)
    root = Node(DOCUMENT, attrs={"title": title})
    stack = [root]
    pending_caption: Node | None = None
    consumed = _table_regions(tables)

    for block in raw:
        if block["kind"] == "image":
            node = Node("figure", span=Span(block["page"], block["bbox"]))
            stack[-1].add(node)
            pending_caption = None
            continue
        if _inside(block, consumed):
            continue
        text = block["text"]
        level = _heading_level(block, body_size)
        span = Span(block["page"], block["bbox"])
        if level:
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            stack.append(stack[-1].add(Node(HEADING, text=text, level=level, span=span)))
            continue
        kind, payload, attrs = _classify_text_block(text)
        node = stack[-1].add(Node(kind, text=payload, attrs=attrs, span=span))
        if kind == CAPTION:
            pending_caption = node

    for tbl in tables:
        node = Node(
            TABLE,
            text=_grid_to_text(tbl["grid"]),
            span=Span(tbl["page"], tbl["bbox"]),
            attrs={"grid": tbl["grid"], "html": tbl["html"], "n_rows": len(tbl["grid"]), "n_cols": max((len(r) for r in tbl["grid"]), default=0)},
        )
        _nearest_section(root, tbl["page"]).add(node)

    tree = DocumentTree(root, source=source, meta={"title": title, "format": "pdf", "pages": 1 + max((b["page"] for b in raw), default=0)})
    tree.bind_references()
    return tree


def _grid_to_text(grid: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in grid)


def _body_size(raw: list[dict]) -> float:
    sizes = Counter(round(b["size"], 1) for b in raw if b["kind"] == "text" and b["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _heading_level(block: dict, body_size: float) -> int:
    text = block["text"].strip()
    if not text or len(text) > 160 or text.count("\n") > 1:
        return 0
    numbered = re.match(r"^(\d+(?:\.\d+)*)\s+\S", text)
    size = block["size"]
    if size >= body_size * 1.45:
        return 1
    if size >= body_size * 1.2:
        return 2
    if numbered:
        return min(1 + numbered.group(1).count("."), 6)
    if block["bold"] and size >= body_size and len(text) < 90 and not text.endswith("."):
        return 3
    return 0


def _table_regions(tables: list[dict]) -> dict[int, list[tuple]]:
    regions: dict[int, list[tuple]] = {}
    for t in tables:
        regions.setdefault(t["page"], []).append(t["bbox"])
    return regions


def _inside(block: dict, regions: dict[int, list[tuple]]) -> bool:
    for x0, y0, x1, y1 in regions.get(block["page"], []):
        bx0, by0, bx1, by1 = block["bbox"]
        if bx0 >= x0 - 2 and by0 >= y0 - 2 and bx1 <= x1 + 2 and by1 <= y1 + 2:
            return True
    return False


def _nearest_section(root: Node, page: int) -> Node:
    best, best_page = root, -1
    for node in root.walk():
        if node.kind == HEADING and node.span.page <= page and node.span.page >= best_page:
            best, best_page = node, node.span.page
    return best
