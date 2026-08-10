"""Backend-agnostic output of a PDF parser: a flat, order-preserving list of
text/table blocks plus whatever style hints (font size, bold) a backend can
provide. These hints are generic layout signals, not domain assumptions —
the structuring agent uses them to help infer headings.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BlockKind(str, Enum):
    TEXT = "text"
    TABLE = "table"


class TextStyle(BaseModel):
    font_size: float | None = None
    bold: bool = False


class Block(BaseModel):
    kind: BlockKind
    page: int
    order: int
    text: str | None = None
    style: TextStyle | None = None
    table_rows: list[list[str]] | None = None


class ParsedDocument(BaseModel):
    source_path: str
    backend: str
    blocks: list[Block] = Field(default_factory=list)
