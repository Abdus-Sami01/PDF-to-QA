"""Document extraction into a typed AST. PDF backends are optional; text/markdown always works."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .latex import validate
from .docast import (
    CAPTION,
    CODE,
    DOCUMENT,
    EQUATION,
    FIGURE,
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


def load(path: str | Path, backend: str = "auto", assets_dir: str | Path | None = None, dpi: int = 144) -> DocumentTree:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return from_markdown(p.read_text(encoding="utf-8", errors="replace"), source=p.name)
    if suffix != ".pdf":
        from .adapters import SUFFIXES, load as load_adapter

        if suffix in SUFFIXES:
            return load_adapter(p, assets_dir)
        raise ValueError(f"unsupported input type: {suffix}; supported: .pdf .md .txt {' '.join(sorted(SUFFIXES))}")
    if backend in ("auto", "pymupdf"):
        try:
            return _from_pymupdf(p, assets_dir, dpi)
        except ImportError:
            if backend == "pymupdf":
                raise
    if backend in ("auto", "pdfminer"):
        try:
            return _from_pdfminer(p, assets_dir, dpi)
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
    checked = validate(latex)
    attrs["symbols"] = checked["symbols"][:32]
    attrs["tree"] = checked["tree"]
    attrs["depth"] = checked["depth"]
    attrs["balanced"] = checked["valid"]
    if checked["issues"]:
        attrs["issues"] = checked["issues"][:6]
    return attrs


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


def _from_pymupdf(path: Path, assets_dir: str | Path | None = None, dpi: int = 144) -> DocumentTree:
    import fitz  # type: ignore

    doc = fitz.open(path)
    assets = Path(assets_dir) / path.stem if assets_dir else None
    if assets:
        assets.mkdir(parents=True, exist_ok=True)
    raw: list[dict] = []
    tables: list[dict] = []
    for pno, page in enumerate(doc):
        page_tables = _pymupdf_tables(page)
        page_dict = page.get_text("dict")
        page_blocks: list[dict] = []
        images: list[dict] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                images.append({"kind": "image", "page": pno, "bbox": bbox, "text": "", "size": 0.0, "bold": False})
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
            page_blocks.append(
                {
                    "kind": "text",
                    "page": pno,
                    "bbox": tuple(block.get("bbox", (0, 0, 0, 0))),
                    "text": body,
                    "size": max(sizes) if sizes else 0.0,
                    "bold": bold > 0,
                }
            )
        images.extend(_vector_regions(page, pno))
        figures = _reject_table_overlap(_cluster_regions(images, _cluster_gap(page_blocks)), [t["bbox"] for t in page_tables])
        consumed = _absorb_labels(figures, page_blocks)
        page_blocks = [b for b in page_blocks if id(b) not in consumed]
        if assets:
            for i, fig in enumerate(figures):
                _render_clip(page, fig["bbox"], assets / f"p{pno:03d}-fig{i:02d}.png", dpi, fig)
            for i, tbl in enumerate(page_tables):
                _render_clip(page, tbl["bbox"], assets / f"p{pno:03d}-tbl{i:02d}.png", dpi, tbl)
        page_blocks.extend(figures)
        page_blocks.sort(key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
        raw.extend(page_blocks)
        tables.extend({"page": pno, **t} for t in page_tables)
    doc.close()
    return _assemble(raw, tables, source=path.name, title=_pdf_title(raw, path))


MIN_FIGURE_SIDE = 24.0
MAX_PAGE_COVERAGE = 0.92
CLIP_PAD = 6.0


MIN_STROKE_SIDE = 3.0


def _vector_regions(page, pno: int) -> list[dict]:
    """Charts and diagrams are drawn, not embedded, so they never appear as image blocks."""
    getter = getattr(page, "get_drawings", None)
    if getter is None:
        return []
    try:
        drawings = getter()
    except Exception:
        return []
    out = []
    page_area = max(1.0, page.rect.width * page.rect.height)
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        w, h = rect.width, rect.height
        if max(w, h) < MIN_STROKE_SIDE:
            continue
        if (w * h) / page_area > MAX_PAGE_COVERAGE:
            continue
        out.append({"kind": "image", "page": pno, "bbox": tuple(rect), "text": "", "size": 0.0, "bold": False})
    return out


MAX_LABEL_CHARS = 40


def _absorb_labels(figures: list[dict], blocks: list[dict], margin: float = 16.0) -> set[int]:
    """Axis ticks and data labels are text, so the drawing bbox misses them; pull them in."""
    consumed: set[int] = set()
    for fig in figures:
        grew = True
        while grew:
            grew = False
            zone = (fig["bbox"][0] - margin, fig["bbox"][1] - margin, fig["bbox"][2] + margin, fig["bbox"][3] + margin)
            for block in blocks:
                if id(block) in consumed or block["page"] != fig["page"]:
                    continue
                text = block["text"].strip()
                if len(text) > MAX_LABEL_CHARS or CAPTION_RE.match(text):
                    continue
                if _intersect_area(block["bbox"], zone) / max(1.0, _area(block["bbox"])) < 0.6:
                    continue
                fig["bbox"] = _union(fig["bbox"], block["bbox"])
                consumed.add(id(block))
                grew = True
    return consumed


def _reject_table_overlap(figures: list[dict], table_boxes: list[tuple], overlap: float = 0.5) -> list[dict]:
    """Table rules are drawings too; drop any region a detected table already covers."""
    out = []
    for fig in figures:
        area = max(1.0, _area(fig["bbox"]))
        if any(_intersect_area(fig["bbox"], tb) / area > overlap for tb in table_boxes):
            continue
        out.append(fig)
    return out


def _area(b: tuple) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersect_area(a: tuple, b: tuple) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _cluster_gap(page_blocks: list[dict], default: float = 14.0) -> float:
    """A dense two-column page needs a tighter gap or two adjacent plots merge into one crop."""
    sizes = [b["size"] for b in page_blocks if b["kind"] == "text" and b["size"] > 0]
    if not sizes:
        return default
    line_height = sorted(sizes)[len(sizes) // 2] * 1.2
    return max(4.0, min(default, line_height))


def _cluster_regions(images: list[dict], gap: float = 14.0) -> list[dict]:
    """Vector figures arrive as many small blocks; merge neighbours into one figure region."""
    boxes = [dict(b) for b in images]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _near(boxes[i]["bbox"], boxes[j]["bbox"], gap):
                    boxes[i]["bbox"] = _union(boxes[i]["bbox"], boxes[j]["bbox"])
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        if (x1 - x0) < MIN_FIGURE_SIDE or (y1 - y0) < MIN_FIGURE_SIDE:
            continue
        out.append(b)
    return out


def _near(a: tuple, b: tuple, gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _union(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _render_clip(page, bbox: tuple, out_path: Path, dpi: int, sink: dict) -> None:
    import fitz  # type: ignore

    rect = (fitz.Rect(*bbox) + (-CLIP_PAD, -CLIP_PAD, CLIP_PAD, CLIP_PAD)) & page.rect
    if rect.is_empty or rect.width < MIN_FIGURE_SIDE or rect.height < MIN_FIGURE_SIDE:
        return
    if (rect.width * rect.height) / max(1.0, page.rect.width * page.rect.height) > MAX_PAGE_COVERAGE:
        return
    scale = dpi / 72.0
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_path)
    except Exception:
        return
    sink["image_path"] = str(out_path)
    sink["image_width"] = pix.width
    sink["image_height"] = pix.height
    sink["image_dpi"] = dpi


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
        if not _plausible_table(grid):
            continue
        out.append({"bbox": tuple(t.bbox), "grid": grid, "html": _grid_to_html(grid)})
    return out


def _plausible_table(grid: list[list[str]], min_fill: float = 0.5) -> bool:
    """Chart axes and boxed callouts get detected as tables; real tables are mostly full."""
    if len(grid) < 2 or max((len(r) for r in grid), default=0) < 2:
        return False
    cells = [c for row in grid for c in row]
    filled = sum(1 for c in cells if c.strip())
    return bool(cells) and filled / len(cells) >= min_fill


def _from_pdfminer(path: Path, assets_dir: str | Path | None = None, dpi: int = 144) -> DocumentTree:
    from pdfminer.high_level import extract_pages  # type: ignore
    from pdfminer.layout import LTChar, LTCurve, LTFigure, LTImage, LTLine, LTRect, LTTextContainer  # type: ignore

    assets = Path(assets_dir) / path.stem if assets_dir else None
    if assets:
        assets.mkdir(parents=True, exist_ok=True)
    raw: list[dict] = []

    for pno, layout in enumerate(extract_pages(str(path))):
        page_blocks: list[dict] = []
        images: list[dict] = []

        def visit(container):
            for element in container:
                if isinstance(element, (LTFigure, LTImage, LTLine, LTRect, LTCurve)):
                    x0, y0, x1, y1 = element.bbox
                    if max(x1 - x0, y1 - y0) >= MIN_STROKE_SIDE:
                        images.append({"kind": "image", "page": pno, "bbox": tuple(element.bbox), "text": "", "size": 0.0, "bold": False})
                    if isinstance(element, LTFigure):
                        visit(element)
                    continue
                if not isinstance(element, LTTextContainer):
                    continue
                body = element.get_text().strip()
                if not body:
                    continue
                sizes = [round(c.size, 1) for line in element for c in line if isinstance(c, LTChar)]
                fonts = [c.fontname.lower() for line in element for c in line if isinstance(c, LTChar)]
                page_blocks.append(
                    {
                        "kind": "text",
                        "page": pno,
                        "bbox": tuple(element.bbox),
                        "text": body,
                        "size": max(sizes) if sizes else 0.0,
                        "bold": any("bold" in f for f in fonts),
                    }
                )

        visit(layout)
        figures = _cluster_regions(images, _cluster_gap(page_blocks))
        consumed = _absorb_labels(figures, page_blocks)
        page_blocks = [b for b in page_blocks if id(b) not in consumed]
        if assets:
            page_size = (layout.bbox[2], layout.bbox[3])
            for i, fig in enumerate(figures):
                render_region(path, pno, fig["bbox"], page_size, assets / f"p{pno:03d}-fig{i:02d}.png", dpi, fig)
        page_blocks.extend(figures)
        page_blocks.sort(key=lambda b: (-round(b["bbox"][3], 1), b["bbox"][0]))
        raw.extend(page_blocks)

    return _assemble(raw, [], source=path.name, title=_pdf_title(raw, path))


def render_region(pdf: Path, page_no: int, bbox: tuple, page_size: tuple, out_path: Path, dpi: int, sink: dict) -> None:
    """Rasterise a PDF region with pypdfium2 — the renderer for backends that cannot draw, like pdfminer."""
    try:
        import pypdfium2  # type: ignore
    except ImportError:
        return
    width, height = page_size
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0.0, x0 - CLIP_PAD), max(0.0, y0 - CLIP_PAD)
    x1, y1 = min(width, x1 + CLIP_PAD), min(height, y1 + CLIP_PAD)
    if x1 - x0 < MIN_FIGURE_SIDE or y1 - y0 < MIN_FIGURE_SIDE:
        return
    if ((x1 - x0) * (y1 - y0)) / max(1.0, width * height) > MAX_PAGE_COVERAGE:
        return
    try:
        doc = pypdfium2.PdfDocument(str(pdf))
        page = doc[page_no]
        bitmap = page.render(scale=dpi / 72.0, crop=(x0, y0, width - x1, height - y1))
        _write_png(out_path, bitmap.width, bitmap.height, bitmap.stride, bytes(bitmap.buffer), bitmap.n_channels)
        doc.close()
    except Exception:
        return
    sink["image_path"] = str(out_path)
    sink["image_width"] = bitmap.width
    sink["image_height"] = bitmap.height
    sink["image_dpi"] = dpi


def _write_png(path: Path, width: int, height: int, stride: int, buf: bytes, channels: int) -> None:
    """Minimal PNG writer so rendering needs no imaging library; pdfium hands back BGR rows."""
    import struct
    import zlib

    raw = bytearray()
    for y in range(height):
        row = buf[y * stride : y * stride + width * channels]
        raw.append(0)
        if channels >= 3:
            rgb = bytearray(width * 3)
            rgb[0::3] = row[2::channels]
            rgb[1::3] = row[1::channels]
            rgb[2::3] = row[0::channels]
            raw += rgb
        else:
            raw += row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2 if channels >= 3 else 0, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    )


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
            node = Node(FIGURE, span=Span(block["page"], block["bbox"]), attrs=_image_attrs(block))
            stack[-1].add(node)
            if pending_caption is not None:
                node.attrs["caption"] = pending_caption.text
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
        attrs = {
            "grid": tbl["grid"],
            "html": tbl["html"],
            "n_rows": len(tbl["grid"]),
            "n_cols": max((len(r) for r in tbl["grid"]), default=0),
            **_image_attrs(tbl),
        }
        node = Node(TABLE, text=_grid_to_text(tbl["grid"]), span=Span(tbl["page"], tbl["bbox"]), attrs=attrs)
        _nearest_section(root, tbl["page"]).add(node)
        caption = _caption_near(root, tbl["page"], tbl["bbox"])
        if caption:
            node.attrs["caption"] = caption

    tree = DocumentTree(root, source=source, meta={"title": title, "format": "pdf", "pages": 1 + max((b["page"] for b in raw), default=0)})
    tree.bind_references()
    return tree


def _image_attrs(block: dict) -> dict:
    return {k: block[k] for k in ("image_path", "image_width", "image_height", "image_dpi") if k in block}


def _caption_near(root: Node, page: int, bbox: tuple, max_gap: float = 60.0) -> str:
    """Match a rendered table to the caption sitting directly above or below it."""
    best, best_gap = "", max_gap
    for node in root.walk():
        if node.kind != CAPTION or node.span.page != page or not node.span.bbox:
            continue
        gap = min(abs(node.span.bbox[1] - bbox[3]), abs(bbox[1] - node.span.bbox[3]))
        if gap < best_gap:
            best, best_gap = node.text, gap
    return best


def _grid_to_text(grid: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in grid)


def _body_size(raw: list[dict]) -> float:
    """Weight by characters, not blocks — a page of tiny axis labels must not define body size."""
    sizes: Counter = Counter()
    for b in raw:
        if b["kind"] == "text" and b["text"]:
            sizes[round(b["size"], 1)] += len(b["text"])
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
