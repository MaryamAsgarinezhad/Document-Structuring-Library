"""Pure StructuredDocument -> Markdown conversion.

Depends only on :mod:`docstruct.schema` — never on the parser or agent
modules — so it can be used standalone on any JSON that validates against
:class:`~docstruct.schema.StructuredDocument`.
"""

from __future__ import annotations

from .schema import Section, StructuredDocument, TableData

_MAX_HEADING_LEVEL = 6


def _heading_line(text: str, level: int) -> str:
    hashes = "#" * min(level, _MAX_HEADING_LEVEL)
    return f"{hashes} {text}".rstrip()


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _table_to_markdown(table: TableData) -> list[str]:
    lines: list[str] = []
    if table.caption:
        lines.append(f"*{table.caption}*")
        lines.append("")

    headers = table.headers
    rows = table.rows
    if not headers and rows:
        headers = [""] * len(rows[0])

    if not headers:
        return lines

    lines.append("| " + " | ".join(_escape_cell(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(_escape_cell(c) for c in cells[: len(headers)]) + " |")

    return lines


def _section_to_markdown(section: Section) -> list[str]:
    # +1: the document title (if any) owns H1, so top-level sections start at H2.
    lines = [_heading_line(section.heading, section.level + 1), ""]

    for paragraph in section.paragraphs:
        lines.append(paragraph.strip())
        lines.append("")

    for table in section.tables:
        lines.extend(_table_to_markdown(table))
        lines.append("")

    for subsection in section.subsections:
        lines.extend(_section_to_markdown(subsection))

    return lines


def structured_document_to_markdown(document: StructuredDocument) -> str:
    """Render a :class:`StructuredDocument` as well-formed Markdown.

    Reading order is preserved by construction: title, then preamble
    content, then each top-level section (recursing into subsections)
    in the order they appear on the document.
    """
    lines: list[str] = []

    if document.title:
        lines.append(_heading_line(document.title, 1))
        lines.append("")

    for paragraph in document.preamble_paragraphs:
        lines.append(paragraph.strip())
        lines.append("")

    for table in document.preamble_tables:
        lines.extend(_table_to_markdown(table))
        lines.append("")

    for section in document.sections:
        lines.extend(_section_to_markdown(section))

    markdown = "\n".join(lines)
    while "\n\n\n" in markdown:
        markdown = markdown.replace("\n\n\n", "\n\n")
    markdown = markdown.strip()
    return markdown + "\n" if markdown else ""
