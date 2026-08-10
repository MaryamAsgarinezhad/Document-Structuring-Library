"""Group a :class:`~docstruct.parser.types.ParsedDocument`'s blocks into
LLM-sized chunks, in order, without ever splitting a single block (a table
always stays whole in one chunk) across chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..parser.types import Block, BlockKind, ParsedDocument


@dataclass
class Chunk:
    blocks: list[Block]


def _block_char_len(block: Block) -> int:
    if block.kind == BlockKind.TEXT:
        return len(block.text or "")
    return sum(len(cell) for row in (block.table_rows or []) for cell in row) + 20


def chunk_blocks(document: ParsedDocument, max_chars: int = 4000) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Block] = []
    current_len = 0

    for block in document.blocks:
        block_len = _block_char_len(block)
        if current and current_len + block_len > max_chars:
            chunks.append(Chunk(blocks=current))
            current = []
            current_len = 0
        current.append(block)
        current_len += block_len

    if current:
        chunks.append(Chunk(blocks=current))

    return chunks
