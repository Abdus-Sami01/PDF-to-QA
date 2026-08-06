"""Input adapters: HTML, DOCX, EPUB, LaTeX source, notebooks, and CSV into the same document AST.

Everything downstream — chunking, the graph, synthesis, verification — works on the AST, so any
format that can be turned into one inherits the whole pipeline. All of these use only the standard
library: DOCX and EPUB are zip archives of XML, notebooks are JSON.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from .docast import (
    CAPTION,
    CODE,
    DOCUMENT,
    EQUATION,
    FIGURE,
    HEADING,
    LIST,
    PARAGRAPH,
    TABLE,
    DocumentTree,
    Node,
)


class Builder:
    """Keeps the heading stack so adapters only have to emit blocks in reading order."""

    def __init__(self, title: str, source: str):
        self.root = Node(DOCUMENT, attrs={"title": title or source})
        self.stack: list[Node] = [self.root]
        self.source = source
        self._pending_caption: Node | None = None

    def heading(self, text: str, level: int) -> None:
        text = text.strip()
        if not text:
            return
        while len(self.stack) > 1 and self.stack[-1].level >= level:
            self.stack.pop()
        self.stack.append(self.stack[-1].add(Node(HEADING, text=text, level=level)))

    def block(self, kind: str, text: str = "", attrs: dict | None = None) -> Node:
        node = self.stack[-1].add(Node(kind, text=text.strip(), attrs=attrs or {}))
        if kind == CAPTION:
            self._pending_caption = node
        elif self._pending_caption is not None and kind in (TABLE, FIGURE, EQUATION):
            node.attrs.setdefault("caption", self._pending_caption.text)
            self._pending_caption = None
        return node

    def build(self, fmt: str, meta: dict | None = None) -> DocumentTree:
        tree = DocumentTree(
            self.root,
            source=self.source,
            meta={"title": self.root.attrs["title"], "format": fmt, **(meta or {})},
        )
        tree.bind_references()
        return tree


def grid_attrs(grid: list[list[str]]) -> dict:
    from .extract import _grid_to_html

    return {"grid": grid, "html": _grid_to_html(grid), "n_rows": len(grid), "n_cols": max((len(r) for r in grid), default=0)}


def grid_text(grid: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in grid)


def equation_attrs(latex: str) -> dict:
    from .extract import _equation_attrs

    return _equation_attrs(latex)


# --------------------------------------------------------------------------- html


BLOCK_TAGS = {"p", "div", "section", "article", "li", "blockquote", "dd", "dt"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "head", "title"}


class _HTMLReader(HTMLParser):
    def __init__(self, builder: Builder):
        super().__init__(convert_charrefs=True)
        self.b = builder
        self.buf: list[str] = []
        self.skip_depth = 0
        self.mode: list[str] = []
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.title: str = ""

    # -- helpers

    def flush(self, kind: str = PARAGRAPH, attrs: dict | None = None) -> None:
        text = re.sub(r"[ \t]+", " ", "".join(self.buf)).strip()
        self.buf = []
        if not text:
            return
        if kind == HEADING:
            self.b.heading(text, (attrs or {}).get("level", 1))
        else:
            self.b.block(kind, text, attrs)

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "table":
            self.flush()
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.flush()
            self.mode.append(f"h{tag[1]}")
        elif tag in ("pre", "code") and "pre" not in self.mode:
            self.flush()
            self.mode.append("pre")
        elif tag in ("figcaption", "caption"):
            self.flush()
            self.mode.append("caption")
        elif tag == "img":
            src = dict(attrs).get("src", "")
            alt = dict(attrs).get("alt", "")
            self.flush()
            self.b.block(FIGURE, alt, {"image_path": src, "alt": alt})
        elif tag == "br":
            self.buf.append("\n")
        elif tag in BLOCK_TAGS:
            self.flush(LIST if tag == "li" else PARAGRAPH)

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            if len(self.table) >= 2:
                self.b.block(TABLE, grid_text(self.table), grid_attrs(self.table))
            self.table = None
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self.mode and self.mode[-1].startswith("h"):
            level = int(self.mode.pop()[1:])
            self.flush(HEADING, {"level": level})
        elif tag in ("pre", "code") and self.mode and self.mode[-1] == "pre":
            self.mode.pop()
            self.flush(CODE)
        elif tag in ("figcaption", "caption") and self.mode and self.mode[-1] == "caption":
            self.mode.pop()
            self.flush(CAPTION)
        elif tag in BLOCK_TAGS:
            self.flush(LIST if tag == "li" else PARAGRAPH)

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.table is not None:
            return
        else:
            self.buf.append(data)


TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def from_html(html: str, source: str = "inline") -> DocumentTree:
    m = TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    if not title:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.I | re.S)
        title = re.sub(r"<[^>]+>|\s+", " ", h1.group(1)).strip() if h1 else source
    builder = Builder(title, source)
    reader = _HTMLReader(builder)
    reader.feed(html)
    reader.flush()
    return builder.build("html")


# --------------------------------------------------------------------------- docx

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def from_docx(path: Path) -> DocumentTree:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
        title = _docx_title(z) or path.stem
    body = ET.fromstring(xml).find(f"{W}body")
    builder = Builder(title, path.name)
    if body is None:
        return builder.build("docx")

    for element in body:
        if element.tag == f"{W}p":
            text = _docx_text(element)
            if not text.strip():
                continue
            level = _docx_heading_level(element)
            if level:
                builder.heading(text, level)
            elif _docx_is_caption(element) or re.match(r"^\s*(Table|Figure)\s*\d", text):
                builder.block(CAPTION, text)
            else:
                builder.block(PARAGRAPH, text)
        elif element.tag == f"{W}tbl":
            grid = [[_docx_text(cell) for cell in row.findall(f"{W}tc")] for row in element.findall(f"{W}tr")]
            grid = [row for row in grid if any(c.strip() for c in row)]
            if len(grid) >= 2:
                builder.block(TABLE, grid_text(grid), grid_attrs(grid))
    return builder.build("docx")


def _docx_title(archive: zipfile.ZipFile) -> str:
    """Prefer the authored document title; the filename is usually a build artefact."""
    import xml.etree.ElementTree as ET

    try:
        core = ET.fromstring(archive.read("docProps/core.xml").decode("utf-8", "replace"))
    except (KeyError, ET.ParseError):
        return ""
    found = core.find("{http://purl.org/dc/elements/1.1/}title")
    return (found.text or "").strip() if found is not None else ""


def _docx_text(element) -> str:
    return "".join(t.text or "" for t in element.iter(f"{W}t")).strip()


def _docx_style(element) -> str:
    style = element.find(f"{W}pPr/{W}pStyle")
    return (style.get(f"{W}val") or "").lower() if style is not None else ""


def _docx_heading_level(element) -> int:
    style = _docx_style(element)
    m = re.match(r"heading(\d)", style)
    if m:
        return int(m.group(1))
    if style in ("title", "subtitle"):
        return 1
    outline = element.find(f"{W}pPr/{W}outlineLvl")
    if outline is not None and outline.get(f"{W}val", "").isdigit():
        return int(outline.get(f"{W}val")) + 1
    return 0


def _docx_is_caption(element) -> bool:
    return _docx_style(element) == "caption"


# --------------------------------------------------------------------------- epub


def from_epub(path: Path) -> DocumentTree:
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        opf = next((n for n in names if n.endswith(".opf")), None)
        order = []
        title = path.stem
        if opf:
            root = ET.fromstring(z.read(opf).decode("utf-8", "replace"))
            ns = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
            found = root.find(".//dc:title", ns)
            if found is not None and found.text:
                title = found.text.strip()
            base = str(Path(opf).parent)
            manifest = {item.get("id"): item.get("href") for item in root.findall(".//opf:manifest/opf:item", ns)}
            for ref in root.findall(".//opf:spine/opf:itemref", ns):
                href = manifest.get(ref.get("idref"))
                if href:
                    candidate = str(Path(base) / href) if base not in (".", "") else href
                    if candidate in names:
                        order.append(candidate)
        if not order:
            order = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))]

        builder = Builder(title, path.name)
        reader = _HTMLReader(builder)
        for name in order:
            reader.feed(z.read(name).decode("utf-8", "replace"))
            reader.flush()
            reader.mode.clear()
    return builder.build("epub", {"documents": len(order)})


# --------------------------------------------------------------------------- latex source

TEX_SECTIONS = {"part": 1, "chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4, "paragraph": 5}
TEX_COMMAND = re.compile(r"\\(" + "|".join(TEX_SECTIONS) + r")\*?\s*\{")
TEX_ENV = re.compile(r"\\begin\{(equation\*?|align\*?|displaymath|gather\*?|tabular|table|figure|verbatim|lstlisting)\}(.*?)\\end\{\1\}", re.S)
TEX_TITLE = re.compile(r"\\title\s*\{(.*?)\}", re.S)
TEX_CAPTION = re.compile(r"\\caption\s*\{(.*?)\}", re.S)


def from_latex(text: str, source: str = "inline") -> DocumentTree:
    text = re.sub(r"(?<!\\)%.*", "", text)
    body = text.split(r"\begin{document}", 1)[-1].split(r"\end{document}")[0]
    m = TEX_TITLE.search(text)
    builder = Builder(_tex_plain(m.group(1)) if m else source, source)

    for kind, payload, raw in _tex_blocks(body):
        if kind == HEADING:
            builder.heading(_tex_plain(payload), raw)
        elif kind == TABLE:
            caption = TEX_CAPTION.search(raw)
            if caption:
                builder.block(CAPTION, _tex_plain(caption.group(1)))
            grid = _tabular_grid(_inner_tabular(payload))
            if len(grid) >= 2:
                builder.block(TABLE, grid_text(grid), grid_attrs(grid))
        elif kind == EQUATION:
            builder.block(EQUATION, payload.strip(), equation_attrs(payload.strip()))
        elif kind == FIGURE:
            caption = TEX_CAPTION.search(raw)
            builder.block(CAPTION, _tex_plain(caption.group(1))) if caption else None
            builder.block(FIGURE, "", {"caption": _tex_plain(caption.group(1)) if caption else ""})
        elif kind == CODE:
            builder.block(CODE, payload.strip())
        else:
            plain = _tex_plain(payload)
            if plain:
                builder.block(PARAGRAPH, plain)
    return builder.build("latex")


def _tex_blocks(body: str):
    pos = 0
    while pos < len(body):
        sec = TEX_COMMAND.search(body, pos)
        env = TEX_ENV.search(body, pos)
        nxt = min([m.start() for m in (sec, env) if m], default=None)
        if nxt is None:
            yield (PARAGRAPH, body[pos:], "")
            return
        for para in re.split(r"\n\s*\n", body[pos:nxt]):
            if para.strip():
                yield (PARAGRAPH, para, "")
        if sec and sec.start() == nxt:
            name = sec.group(1)
            title, end = _match_brace(body, sec.end() - 1)
            yield (HEADING, title, TEX_SECTIONS[name])
            pos = end
        else:
            name, inner = env.group(1), env.group(2)
            if name.startswith(("equation", "align", "displaymath", "gather")):
                yield (EQUATION, inner, env.group(0))
            elif name in ("tabular", "table"):
                yield (TABLE, inner, env.group(0))
            elif name == "figure":
                yield (FIGURE, inner, env.group(0))
            else:
                yield (CODE, inner, env.group(0))
            pos = env.end()


def _match_brace(text: str, start: int) -> tuple[str, int]:
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], len(text)


TABULAR_INNER = re.compile(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", re.S)


def _inner_tabular(payload: str) -> str:
    """A `table` float wraps the real `tabular`; the caption and placement specs are not rows."""
    m = TABULAR_INNER.search(payload)
    return m.group(1) if m else payload


def _tabular_grid(inner: str) -> list[list[str]]:
    inner = TEX_CAPTION.sub("", inner)
    inner = re.sub(r"\\(centering|label\{[^}]*\}|small|footnotesize)", "", inner)
    inner = re.sub(r"\\(hline|toprule|midrule|bottomrule|cline\{[^}]*\})", "", inner)
    inner = re.sub(r"^\s*\{[^}]*\}", "", inner.strip(), count=1)
    grid = []
    for row in inner.split(r"\\"):
        cells = [_tex_plain(c) for c in row.split("&")]
        if any(cells):
            grid.append(cells)
    return grid


def _tex_plain(text: str) -> str:
    text = re.sub(r"\\(?:label|ref|cite[a-z]*|index)\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|mathrm|text)\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("~", " ").replace("\\&", "&").replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+", " ", text).strip()


# --------------------------------------------------------------------------- notebook


def from_notebook(text: str, source: str = "inline") -> DocumentTree:
    from .extract import from_markdown

    nb = json.loads(text)
    parts: list[str] = []
    for cell in nb.get("cells", []):
        body = "".join(cell.get("source", []))
        if not body.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(body)
        else:
            parts.append(f"```\n{body}\n```")
            for out in cell.get("outputs", []):
                rendered = _notebook_output(out)
                if rendered:
                    parts.append(f"```\n{rendered}\n```")
    tree = from_markdown("\n\n".join(parts), source=source)
    tree.meta["format"] = "notebook"
    return tree


def _notebook_output(out: dict) -> str:
    if out.get("output_type") == "stream":
        return "".join(out.get("text", []))[:2000]
    data = out.get("data", {})
    for key in ("text/plain", "text/markdown"):
        if key in data:
            return "".join(data[key])[:2000]
    if out.get("output_type") == "error":
        return f"{out.get('ename', '')}: {out.get('evalue', '')}"
    return ""


# --------------------------------------------------------------------------- csv


def from_csv(text: str, source: str = "inline", max_rows: int = 500) -> DocumentTree:
    dialect = csv.Sniffer().sniff(text[:4096]) if text.strip() else csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))[: max_rows + 1]
    grid = [[c.strip() for c in row] for row in rows if any(c.strip() for c in row)]
    builder = Builder(Path(source).stem, source)
    builder.heading(Path(source).stem, 1)
    if len(grid) >= 2:
        builder.block(TABLE, grid_text(grid), grid_attrs(grid))
    return builder.build("csv", {"rows": max(0, len(grid) - 1)})


# --------------------------------------------------------------------------- dispatch

SUFFIXES = {
    ".html": "html", ".htm": "html", ".xhtml": "html",
    ".docx": "docx",
    ".epub": "epub",
    ".tex": "latex",
    ".ipynb": "notebook",
    ".csv": "csv", ".tsv": "csv",
}


def load(path: str | Path) -> DocumentTree:
    p = Path(path)
    fmt = SUFFIXES.get(p.suffix.lower())
    if fmt is None:
        raise ValueError(f"no adapter for {p.suffix}")
    if fmt == "docx":
        return from_docx(p)
    if fmt == "epub":
        return from_epub(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    if fmt == "html":
        return from_html(text, p.name)
    if fmt == "latex":
        return from_latex(text, p.name)
    if fmt == "notebook":
        return from_notebook(text, p.name)
    return from_csv(text, p.name)
