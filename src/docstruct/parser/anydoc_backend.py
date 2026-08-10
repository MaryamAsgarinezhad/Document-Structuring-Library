"""PDF parsing backend built on ``anydoc`` (PyPI: ``firecrawl-anydoc``), a
Rust-based multi-format-to-Markdown converter with Python bindings.

Unlike the other three backends, anydoc's structured ``to_document()`` API
explicitly does **not** support PDF — calling it raises
``UnsupportedError: PDF converts directly to Markdown; use to_markdown or
to_markdown_bytes``. So there is no native per-block page number or
font-size/bold metadata available for PDFs the way there is for PyMuPDF,
pdfplumber, or docling. This backend instead takes anydoc's Markdown output
and parses *that* back into our ``Block`` schema: ``#``-heading lines become
heading-hinted TEXT blocks, contiguous ``|...|`` lines become TABLE blocks,
everything else is paragraph text. ``Block.page`` is always ``0`` (anydoc's
PDF path doesn't report page numbers), and ``TextStyle.bold`` reflects only
whether anydoc *itself* decided a line was a heading — not real font
weight. See ``docs/parser_comparison.md`` for how reliable that heading
detection turned out to be on real Persian PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path

import anydoc

from .types import Block, BlockKind, ParsedDocument, TextStyle

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|\s*$")
_INLINE_MARKUP_RE = re.compile(r"\*\*|<u>|</u>")


def _clean_text(text: str) -> str:
    return _INLINE_MARKUP_RE.sub("", text).strip()


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


class AnyDocBackend:
    name = "anydoc"

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        markdown = anydoc.to_markdown(str(pdf_path))
        blocks: list[Block] = []
        order = 0
        lines = markdown.split("\n")
        paragraph_lines: list[str] = []

        def flush_paragraph() -> None:
            nonlocal order
            text = _clean_text(" ".join(paragraph_lines))
            paragraph_lines.clear()
            if text:
                blocks.append(
                    Block(kind=BlockKind.TEXT, page=0, order=order, text=text, style=TextStyle(bold=False))
                )
                order += 1

        i = 0
        while i < len(lines):
            line = lines[i]

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                flush_paragraph()
                text = _clean_text(heading_match.group(2))
                if text:
                    blocks.append(
                        Block(kind=BlockKind.TEXT, page=0, order=order, text=text, style=TextStyle(bold=True))
                    )
                    order += 1
                i += 1
                continue

            if _TABLE_ROW_RE.match(line):
                flush_paragraph()
                table_lines: list[str] = []
                while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                rows = [_split_table_row(l) for l in table_lines if not _TABLE_SEPARATOR_RE.match(l)]
                if rows:
                    blocks.append(Block(kind=BlockKind.TABLE, page=0, order=order, table_rows=rows))
                    order += 1
                continue

            if line.strip() == "":
                flush_paragraph()
            else:
                paragraph_lines.append(line.strip())
            i += 1

        flush_paragraph()
        return ParsedDocument(source_path=str(pdf_path), backend=self.name, blocks=blocks)
