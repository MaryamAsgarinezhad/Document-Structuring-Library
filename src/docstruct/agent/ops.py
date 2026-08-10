"""Per-chunk output schema for the structuring agent.

Rather than re-emitting the whole document tree on every chunk, the model
only classifies the *current* chunk's content into small operations. A
:class:`~docstruct.agent.builder.DocumentBuilder` applies these to the
running tree — see that module for why.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NewHeading(BaseModel):
    op: Literal["heading"] = "heading"
    level: int = Field(ge=1, le=6, description="1 = top-level heading")
    text: str


class NewParagraph(BaseModel):
    op: Literal["paragraph"] = "paragraph"
    text: str


class NewTable(BaseModel):
    op: Literal["table"] = "table"
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


ChunkOp = Annotated[NewHeading | NewParagraph | NewTable, Field(discriminator="op")]


class ChunkResult(BaseModel):
    """What the agent returns for a single chunk."""

    title: str | None = Field(
        default=None,
        description="The document's overall title, if it becomes apparent in this chunk (usually only the first).",
    )
    operations: list[ChunkOp] = Field(default_factory=list)
