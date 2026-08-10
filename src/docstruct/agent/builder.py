"""Deterministic, pure-Python merge of per-chunk operations into a running
:class:`~docstruct.schema.StructuredDocument` tree.

This is the "running document state" the chunked agent maintains: the LLM
only has to classify each chunk's content (see :mod:`docstruct.agent.ops`),
and this builder is what actually grows the tree, in reading order, one
chunk at a time.
"""

from __future__ import annotations

from collections import deque

from ..schema import Section, StructuredDocument, TableData
from .ops import ChunkOp, NewHeading, NewParagraph, NewTable

# How many recently-applied ops to remember for duplicate detection. This is
# a rolling window, not a whole-document set, on purpose: it exists to catch
# the case where chunk overlap (see chunking.chunk_blocks) causes the model
# to re-emit an op for content it already saw and reported on in the
# previous chunk, which — because each chunk call is otherwise stateless —
# it has no way to know it already did. A small window catches that
# boundary-adjacent duplication without suppressing genuine, far-apart
# repeated content (e.g. the same boilerplate phrase reused across the
# document), which a global/whole-document dedup set would incorrectly eat.
_DEDUP_WINDOW = 12


def _op_signature(op: ChunkOp) -> str:
    if isinstance(op, NewHeading):
        return f"heading:{op.level}:{op.text.strip()}"
    if isinstance(op, NewParagraph):
        return f"paragraph:{op.text.strip()}"
    return f"table:{op.caption}:{tuple(op.headers)}:{tuple(tuple(row) for row in op.rows)}"


class DocumentBuilder:
    def __init__(self) -> None:
        self.document = StructuredDocument()
        self._open_sections: list[Section] = []
        self._recent_signatures: deque[str] = deque(maxlen=_DEDUP_WINDOW)

    def set_title(self, title: str) -> None:
        if not self.document.title and title.strip():
            self.document.title = title.strip()

    def apply(self, op: ChunkOp) -> None:
        signature = _op_signature(op)
        if signature in self._recent_signatures:
            return
        self._recent_signatures.append(signature)

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
