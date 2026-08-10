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
    context_prefix_len: int = 0
    """Number of leading entries in `blocks` that are repeated from the tail
    of the previous chunk as read-only context, not new content — see
    `overlap_blocks` on `chunk_blocks`."""

    @property
    def new_blocks(self) -> list[Block]:
        return self.blocks[self.context_prefix_len :]


def _block_char_len(block: Block) -> int:
    if block.kind == BlockKind.TEXT:
        return len(block.text or "")
    return sum(len(cell) for row in (block.table_rows or []) for cell in row) + 20


def chunk_blocks(document: ParsedDocument, max_chars: int = 4000, overlap_blocks: int = 0) -> list[Chunk]:
    """Group `document`'s blocks into chunks of at most `max_chars`, never
    splitting a single block across chunks.

    If `overlap_blocks` > 0, each chunk after the first is prefixed with the
    last `overlap_blocks` blocks of the *previous* chunk, marked via
    `context_prefix_len`. This is pure context for the model (so it can see
    what immediately precedes the new content, e.g. to correctly continue a
    paragraph or judge nesting) — the caller is responsible for telling the
    model not to re-emit operations for it, and for not double-counting it
    if the model does anyway (see `DocumentBuilder`'s dedup window).
    """
    core_chunks: list[list[Block]] = []
    current: list[Block] = []
    current_len = 0

    for block in document.blocks:
        block_len = _block_char_len(block)
        if current and current_len + block_len > max_chars:
            core_chunks.append(current)
            current = []
            current_len = 0
        current.append(block)
        current_len += block_len

    if current:
        core_chunks.append(current)

    chunks: list[Chunk] = []
    for i, core in enumerate(core_chunks):
        if i == 0 or overlap_blocks <= 0:
            chunks.append(Chunk(blocks=core, context_prefix_len=0))
            continue
        context = core_chunks[i - 1][-overlap_blocks:]
        chunks.append(Chunk(blocks=context + core, context_prefix_len=len(context)))

    return chunks
