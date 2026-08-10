"""PDF parsing backend built on ``docling``.

Docling runs a learned layout model (downloaded from Hugging Face on first
use) to segment pages and a TableFormer model for table structure. Neither
is trained specifically for Arabic-script documents, so
``docs/parser_comparison.md`` evaluates how well its layout/heading and
table detection generalize to Persian text rather than assuming parity
with its (strong) results on Latin-script papers.
"""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel

from .types import Block, BlockKind, ParsedDocument, TextStyle

_HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
_SKIP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE}


class DoclingBackend:
    name = "docling"

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        blocks: list[Block] = []
        for order, (item, _level) in enumerate(doc.iterate_items()):
            label = getattr(item, "label", None)
            if label in _SKIP_LABELS:
                continue

            prov = getattr(item, "prov", None)
            page = (prov[0].page_no - 1) if prov else 0

            if label == DocItemLabel.TABLE:
                try:
                    df = item.export_to_dataframe()
                    rows = [list(df.columns)] + df.astype(str).values.tolist()
                except Exception:
                    rows = [[item.export_to_markdown(doc)]]
                blocks.append(Block(kind=BlockKind.TABLE, page=page, order=order, table_rows=rows))
                continue

            text = getattr(item, "text", None)
            if not text or not text.strip():
                continue
            style = TextStyle(bold=label in _HEADING_LABELS)
            blocks.append(Block(kind=BlockKind.TEXT, page=page, order=order, text=text.strip(), style=style))

        return ParsedDocument(source_path=str(pdf_path), backend=self.name, blocks=blocks)
