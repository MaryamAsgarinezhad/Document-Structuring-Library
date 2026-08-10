"""PDF parsing backend built on PyMuPDF (``pymupdf``, formerly ``fitz``).

PyMuPDF's text extraction applies its own bidi/shaping handling when it
builds line text from spans, and ``Page.find_tables`` gives structured table
cells directly (no separate OCR/heuristic layer needed) — see
``docs/parser_comparison.md`` for how this fared against the other backends.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from .types import Block, BlockKind, ParsedDocument, TextStyle

_BOLD_FLAG = 1 << 4  # per PyMuPDF span "flags" bitfield: 1=superscript, 2=italic, 4=serifed, 8=monospaced, 16=bold


def _bbox_contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1


class PyMuPDFBackend:
    name = "pymupdf"

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        blocks: list[Block] = []
        order = 0
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                found = page.find_tables()
                table_infos = list(found.tables) if found else []
                table_bboxes = [tuple(t.bbox) for t in table_infos]

                items: list[tuple[float, float, str, object]] = []

                text_dict = page.get_text("dict")
                for text_block in text_dict.get("blocks", []):
                    if text_block.get("type") != 0:
                        continue
                    bbox = tuple(text_block["bbox"])
                    if any(_bbox_contains(tb, bbox) for tb in table_bboxes):
                        continue
                    line_texts: list[str] = []
                    font_sizes: list[float] = []
                    bold_flags: list[bool] = []
                    for line in text_block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(span["text"] for span in spans)
                        if line_text.strip():
                            line_texts.append(line_text)
                        for span in spans:
                            font_sizes.append(span["size"])
                            bold_flags.append(bool(span["flags"] & _BOLD_FLAG))
                    text = "\n".join(line_texts).strip()
                    if not text:
                        continue
                    style = TextStyle(
                        font_size=(sum(font_sizes) / len(font_sizes)) if font_sizes else None,
                        bold=any(bold_flags),
                    )
                    items.append((bbox[1], bbox[0], "text", (text, style)))

                for table in table_infos:
                    rows = table.extract()
                    rows = [["" if c is None else str(c) for c in row] for row in rows]
                    items.append((table.bbox[1], table.bbox[0], "table", rows))

                items.sort(key=lambda it: (it[0], it[1]))
                for _y, _x, kind, payload in items:
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
