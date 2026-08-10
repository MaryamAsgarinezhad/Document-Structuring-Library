"""Deterministic, pure-Python merge of per-chunk operations into a running
:class:`~docstruct.schema.StructuredDocument` tree.

This is the "running document state" the chunked agent maintains: the LLM
only has to classify each chunk's content (see :mod:`docstruct.agent.ops`),
and this builder is what actually grows the tree, in reading order, one
chunk at a time.
"""

from __future__ import annotations

from ..schema import Section, StructuredDocument, TableData
from .ops import ChunkOp, NewHeading, NewParagraph, NewTable


class DocumentBuilder:
    def __init__(self) -> None:
        self.document = StructuredDocument()
        self._open_sections: list[Section] = []

    def set_title(self, title: str) -> None:
        if not self.document.title and title.strip():
            self.document.title = title.strip()

    def apply(self, op: ChunkOp) -> None:
        if isinstance(op, NewHeading):
            self._apply_heading(op)
        elif isinstance(op, NewParagraph):
            self._apply_paragraph(op)
        elif isinstance(op, NewTable):
            self._apply_table(op)

    def _apply_heading(self, op: NewHeading) -> None:
        while self._open_sections and self._open_sections[-1].level >= op.level:
            self._open_sections.pop()

        section = Section(heading=op.text, level=op.level)
        parent = self._open_sections[-1] if self._open_sections else None
        if parent is not None:
            parent.subsections.append(section)
        else:
            self.document.sections.append(section)
        self._open_sections.append(section)

    def _apply_paragraph(self, op: NewParagraph) -> None:
        if not op.text.strip():
            return
        if self._open_sections:
            self._open_sections[-1].paragraphs.append(op.text)
        else:
            self.document.preamble_paragraphs.append(op.text)

    def _apply_table(self, op: NewTable) -> None:
        table = TableData(caption=op.caption, headers=op.headers, rows=op.rows)
        if self._open_sections:
            self._open_sections[-1].tables.append(table)
        else:
            self.document.preamble_tables.append(table)

    def breadcrumb(self, max_items: int = 3) -> list[tuple[int, str]]:
        """The currently-open heading path, most-recently-opened last."""
        return [(s.level, s.heading) for s in self._open_sections[-max_items:]]

    def build(self) -> StructuredDocument:
        return self.document
