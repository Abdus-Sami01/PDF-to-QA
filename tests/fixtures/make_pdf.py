"""Builds the PDF fixture used by the PyMuPDF extraction tests. Run: python tests/fixtures/make_pdf.py"""

import sys
from pathlib import Path

import fitz

OUT = Path(__file__).parent / "sample.pdf"

BODY = [
    ("Long-context transformers degrade on retrieval-heavy workloads. We introduce SparseRoute, a", 11, False),
    ("router that selects 4 of 32 expert blocks per token. Prior work [1] reported 61.2 accuracy.", 11, False),
]

BARS = [("Dense", 61.2), ("k=2", 70.1), ("k=4", 74.8), ("k=8", 75.1)]


def build() -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.insert_text((72, 90), "Sparse Routing for Long-Context Retrieval", fontsize=18, fontname="hebo")
    page.insert_text((72, 130), "1 Introduction", fontsize=13, fontname="hebo")
    y = 155
    for line, size, _ in BODY:
        page.insert_text((72, y), line, fontsize=size)
        y += 16

    page.insert_text((72, 215), "2 Results", fontsize=13, fontname="hebo")
    page.insert_text((72, 240), "Figure 1: Accuracy by routing width on RetrievalBench.", fontsize=9)

    base_y, scale = 430, 2.2
    for i, (label, value) in enumerate(BARS):
        x = 90 + i * 60
        height = (value - 55) * scale
        page.draw_rect(fitz.Rect(x, base_y - height, x + 38, base_y), color=(0.1, 0.2, 0.5), fill=(0.2, 0.4, 0.8))
        page.insert_text((x + 2, base_y + 14), label, fontsize=8)
        page.insert_text((x + 2, base_y - height - 5), str(value), fontsize=8)
    page.draw_line(fitz.Point(85, base_y), fitz.Point(340, base_y), color=(0, 0, 0))

    page.insert_text((72, 500), "Table 1: Accuracy on the held-out split.", fontsize=9)
    rows = [["Model", "Params", "Accuracy", "Latency"],
            ["Dense baseline", "7.0B", "61.2", "340"],
            ["SparseRoute k=4", "7.0B", "74.8", "210"]]
    top, left, rw, ch = 515, 72, 110, 20
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(left + c * rw, top + r * ch, left + (c + 1) * rw, top + (r + 1) * ch)
            page.draw_rect(rect, color=(0, 0, 0))
            page.insert_text((rect.x0 + 4, rect.y0 + 14), cell, fontsize=9)

    page.insert_text((72, 620), "SparseRoute at k = 4 improves accuracy by 13.6 points over the dense", fontsize=11)
    page.insert_text((72, 636), "baseline, as shown in Table 1 and Figure 1.", fontsize=11)

    _booktabs_page(doc)
    doc.save(OUT)
    doc.close()
    return OUT


BOOKTABS = [["Dataset", "Documents", "Mean tokens"],
            ["RetrievalBench", "48000", "9400"],
            ["LongQA", "12000", "3100"]]


def _booktabs_page(doc) -> None:
    """A borderless table: horizontal rules only, columns held together by alignment alone."""
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 90), "3 Datasets", fontsize=13, fontname="hebo")
    page.insert_text((72, 120), "Table 2: Corpus statistics.", fontsize=9)

    top, left, col_x, row_h = 140, 72, [72, 220, 340], 18
    page.draw_line(fitz.Point(left, top - 12), fitz.Point(460, top - 12), color=(0, 0, 0))
    for r, row in enumerate(BOOKTABS):
        for c, cell in enumerate(row):
            page.insert_text((col_x[c], top + r * row_h), cell, fontsize=10)
        if r == 0:
            page.draw_line(fitz.Point(left, top + 5), fitz.Point(460, top + 5), color=(0, 0, 0))
    page.draw_line(fitz.Point(left, top + len(BOOKTABS) * row_h - 7), fitz.Point(460, top + len(BOOKTABS) * row_h - 7), color=(0, 0, 0))

    page.insert_text((72, 260), "RetrievalBench is the larger of the two corpora used in this study.", fontsize=11)


if __name__ == "__main__":
    print(build())
    sys.exit(0)
