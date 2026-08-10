"""Generic, domain-agnostic structured-document schema.

This module has no dependencies beyond pydantic, and is imported by both
``docstruct.agent`` (which produces a :class:`StructuredDocument`) and
``docstruct.converter`` (which consumes one) — never the other way around.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TableData(BaseModel):
    """A single table extracted/inferred from the document."""

    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class Section(BaseModel):
    """A heading and everything nested under it, in reading order."""

    heading: str
    level: int = Field(ge=1)
    paragraphs: list[str] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    subsections: list[Section] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    """Root of a document tree: an optional title plus top-level sections.

    ``preamble`` holds any paragraphs/tables that appear before the first
    heading (e.g. an abstract or introduction with no heading of its own).
    """

    title: str | None = None
    preamble_paragraphs: list[str] = Field(default_factory=list)
    preamble_tables: list[TableData] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)


Section.model_rebuild()
