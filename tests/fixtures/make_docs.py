"""Builds the non-PDF fixtures (docx, epub) used by the adapter tests. Everything here is stdlib."""

import json
import zipfile
from pathlib import Path

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

HTML = """<html><head><title>Sparse Routing Docs</title></head><body>
<nav>navigation noise</nav>
<h1>Sparse Routing</h1>
<p>The router selects 4 of 32 experts per token.</p>
<h2>Results</h2>
<caption>Table 1: Accuracy on the held-out split.</caption>
<table><tr><th>Model</th><th>Accuracy</th></tr><tr><td>Dense</td><td>61.2</td></tr><tr><td>k=4</td><td>74.8</td></tr></table>
<pre>print("routing")</pre>
<img src="chart.png" alt="accuracy by routing width">
<script>ignored()</script>
</body></html>"""

LATEX = r"""\documentclass{article}
\title{Sparse Routing for Retrieval}
\begin{document}
\section{Introduction}
We introduce SparseRoute, selecting 4 of 32 experts~\cite{chen}.
\subsection{Router}
\begin{equation}
s_i = x^\top W_r e_i + b_i
\end{equation}
\section{Results}
\begin{table}
\centering
\caption{Accuracy on the held-out split.}
\begin{tabular}{lr}
\hline
Model & Accuracy \\
Dense & 61.2 \\
SparseRoute & 74.8 \\
\end{tabular}
\end{table}
\end{document}
"""

CSV = "Model,Accuracy,Latency (ms)\nDense,61.2,340\nSparseRoute,74.8,210\n"

NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "source": ["# Routing Analysis\n", "\n", "We compare routing widths.\n"]},
        {"cell_type": "code", "source": ["print(74.8 - 61.2)"], "outputs": [{"output_type": "stream", "text": ["13.6\n"]}]},
    ]
}


def _p(text, style=None):
    pr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{pr}<w:r><w:t>{text}</w:t></w:r></w:p>"


def _cell(text):
    return f"<w:tc><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>"


def build_docx(path: Path) -> Path:
    rows = "".join(
        "<w:tr>" + "".join(_cell(c) for c in row) + "</w:tr>"
        for row in [["Model", "Accuracy"], ["Dense", "61.2"], ["k=4", "74.8"]]
    )
    body = (
        _p("Sparse Routing", "Heading1")
        + _p("The router selects 4 of 32 experts per token.")
        + _p("Results", "Heading2")
        + _p("Table 1: Accuracy on the held-out split.", "Caption")
        + f"<w:tbl>{rows}</w:tbl>"
    )
    core = ('<?xml version="1.0"?><cp:coreProperties xmlns:cp="x" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "<dc:title>Sparse Routing Report</dc:title></cp:coreProperties>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')
        z.writestr("docProps/core.xml", core)
    return path


def build_epub(path: Path) -> Path:
    chapter = ('<html><head><title>c1</title></head><body><h1>Chapter One</h1>'
               "<p>Routing selects four of thirty two experts.</p></body></html>")
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
           '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Routing Book</dc:title></metadata>'
           '<manifest><item id="c1" href="c1.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("content.opf", opf)
        z.writestr("c1.xhtml", chapter)
    return path


def build_all(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    made = {
        "html": directory / "doc.html",
        "latex": directory / "doc.tex",
        "csv": directory / "doc.csv",
        "notebook": directory / "doc.ipynb",
        "docx": build_docx(directory / "doc.docx"),
        "epub": build_epub(directory / "doc.epub"),
    }
    made["html"].write_text(HTML, encoding="utf-8")
    made["latex"].write_text(LATEX, encoding="utf-8")
    made["csv"].write_text(CSV, encoding="utf-8")
    made["notebook"].write_text(json.dumps(NOTEBOOK), encoding="utf-8")
    return made
