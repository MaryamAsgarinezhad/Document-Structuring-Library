"""PDF parsing backend built on ``unstructured``.

Only the ``fast`` (pdfminer-based) strategy is used: ``hi_res``/``ocr_only``
require system binaries (Tesseract, poppler) this project doesn't install,
and OCR/scanned documents are out of scope anyway (see project README).
``fast`` gives useful element categories (Title/Header/Footer/...) for
heading detection, but — as ``docs/parser_comparison.md`` documents — it
does **not** apply bidi reordering, so Persian text comes out
character-reversed per line. This backend deliberately does not "fix" that
so the benchmark reflects the library's real out-of-the-box behavior.
"""

from __future__ import annotations

from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from .types import Block, BlockKind, ParsedDocument, TextStyle

_SKIP_CATEGORIES = {"Header", "Footer"}
_TITLE_CATEGORIES = {"Title"}


def _html_table_to_rows(html: str) -> list[list[str]]:
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.rows: list[list[str]] = []
            self._row: list[str] | None = None
            self._cell: list[str] | None = None

        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []

        def handle_endtag(self, tag: str) -> None:
            if tag == "tr" and self._row is not None:
                self.rows.append(self._row)
                self._row = None
            elif tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append("".join(self._cell).strip())
                self._cell = None

        def handle_data(self, data: str) -> None:
            if self._cell is not None:
                self._cell.append(data)

    parser = _TableParser()
    parser.feed(html)
    return parser.rows


class UnstructuredBackend:
    name = "unstructured"

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            languages=["fas"],
            infer_table_structure=True,
        )

        blocks: list[Block] = []
        for order, el in enumerate(elements):
            category = getattr(el, "category", "")
            if category in _SKIP_CATEGORIES:
                continue
            page_number = getattr(el.metadata, "page_number", None) or 1
            page = page_number - 1

            if category == "Table":
                html = getattr(el.metadata, "text_as_html", None)
                rows = _html_table_to_rows(html) if html else [[str(el)]]
                blocks.append(Block(kind=BlockKind.TABLE, page=page, order=order, table_rows=rows))
                continue

            text = str(el).strip()
            if not text:
                continue
            style = TextStyle(bold=category in _TITLE_CATEGORIES)
            blocks.append(Block(kind=BlockKind.TEXT, page=page, order=order, text=text, style=style))

        return ParsedDocument(source_path=str(pdf_path), backend=self.name, blocks=blocks)
