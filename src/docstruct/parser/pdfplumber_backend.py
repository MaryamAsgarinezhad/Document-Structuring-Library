"""PDF parsing backend built on ``pdfplumber``.

Unlike PyMuPDF, pdfplumber does not reorder characters for RTL scripts — it
reports words/glyphs in the order they appear in the PDF's content stream.
Whether that comes out correct for Persian depends entirely on how the PDF
producer stored the text (logical vs. already-visual order); this backend
is included specifically so ``scripts/benchmark_parsers.py`` can measure
that empirically. See ``docs/parser_comparison.md`` for results.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from .types import Block, BlockKind, ParsedDocument, TextStyle

_LINE_GROUP_TOLERANCE = 2.0  # px: words within this many px of 'top' are the same line
_PARAGRAPH_GAP_FACTOR = 1.6  # a vertical gap larger than (line height * this) starts a new block


def _point_in_bbox(x: float, y: float, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    return x0 <= x <= x1 and top <= y <= bottom


class PDFPlumberBackend:
    name = "pdfplumber"

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        blocks: list[Block] = []
        order = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.find_tables()
                table_bboxes = [tuple(t.bbox) for t in tables]

                words = page.extract_words(extra_attrs=["fontname", "size"], use_text_flow=False)
                words = [
                    w
                    for w in words
                    if not any(
                        _point_in_bbox((w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2, tb)
                        for tb in table_bboxes
                    )
                ]

                lines_by_top: dict[float, list[dict]] = {}
                for w in words:
                    key = round(w["top"] / _LINE_GROUP_TOLERANCE) * _LINE_GROUP_TOLERANCE
                    lines_by_top.setdefault(key, []).append(w)

                lines: list[tuple[float, float, str, float | None, bool]] = []
                for top, ws in lines_by_top.items():
                    ws.sort(key=lambda w: w["x0"])
                    text = " ".join(w["text"] for w in ws)
                    sizes = [w["size"] for w in ws if w.get("size")]
                    avg_size = sum(sizes) / len(sizes) if sizes else None
                    bold = any("bold" in (w.get("fontname") or "").lower() for w in ws)
                    height = max((w["bottom"] - w["top"]) for w in ws)
                    lines.append((top, height, text, avg_size, bold))
                lines.sort(key=lambda item: item[0])

                page_items: list[tuple[float, str, object]] = []
                para_lines: list[str] = []
                para_sizes: list[float] = []
                para_bold: list[bool] = []
                para_top = 0.0
                prev_bottom: float | None = None
                prev_height = 0.0

                def flush_paragraph() -> None:
                    if not para_lines:
                        return
                    text = "\n".join(para_lines).strip()
                    if text:
                        style = TextStyle(
                            font_size=(sum(para_sizes) / len(para_sizes)) if para_sizes else None,
                            bold=any(para_bold),
                        )
                        page_items.append((para_top, "text", (text, style)))

                for top, height, text, size, bold in lines:
                    gap = (top - prev_bottom) if prev_bottom is not None else 0.0
                    if prev_bottom is not None and gap > prev_height * _PARAGRAPH_GAP_FACTOR:
                        flush_paragraph()
                        para_lines, para_sizes, para_bold = [], [], []
                    if not para_lines:
                        para_top = top
                    para_lines.append(text)
                    if size:
                        para_sizes.append(size)
                    para_bold.append(bold)
                    prev_bottom = top + height
                    prev_height = height
                flush_paragraph()

                for table in tables:
                    rows = table.extract()
                    rows = [["" if c is None else str(c) for c in row] for row in rows]
                    page_items.append((table.bbox[1], "table", rows))

                page_items.sort(key=lambda it: it[0])
                for _top, kind, payload in page_items:
                    if kind == "text":
                        text, style = payload
                        blocks.append(
                            Block(kind=BlockKind.TEXT, page=page_num, order=order, text=text, style=style)
                        )
                    else:
                        blocks.append(
                            Block(kind=BlockKind.TABLE, page=page_num, order=order, table_rows=payload)
                        )
                    order += 1

        return ParsedDocument(source_path=str(pdf_path), backend=self.name, blocks=blocks)
