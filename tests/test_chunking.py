from docstruct.agent.chunking import chunk_blocks
from docstruct.parser.types import Block, BlockKind, ParsedDocument


def _text_block(order: int, text: str) -> Block:
    return Block(kind=BlockKind.TEXT, page=0, order=order, text=text)


def _table_block(order: int, rows: list[list[str]]) -> Block:
    return Block(kind=BlockKind.TABLE, page=0, order=order, table_rows=rows)


def test_small_document_fits_in_one_chunk():
    doc = ParsedDocument(
        source_path="x.pdf", backend="pymupdf", blocks=[_text_block(0, "کوتاه"), _text_block(1, "باز هم کوتاه")]
    )

    chunks = chunk_blocks(doc, max_chars=4000)

    assert len(chunks) == 1
    assert len(chunks[0].blocks) == 2


def test_splits_once_char_budget_exceeded():
    blocks = [_text_block(i, "س" * 60) for i in range(5)]
    doc = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    chunks = chunk_blocks(doc, max_chars=100)

    assert len(chunks) > 1
    # every original block must appear in exactly one chunk, in order, none dropped or duplicated
    flattened = [b for c in chunks for b in c.blocks]
    assert flattened == blocks


def test_table_block_is_never_split_across_chunks():
    rows = [["a" * 30, "b" * 30] for _ in range(10)]
    blocks = [_text_block(0, "متن قبل از جدول"), _table_block(1, rows), _text_block(2, "متن بعد از جدول")]
    doc = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    chunks = chunk_blocks(doc, max_chars=50)

    table_chunks = [c for c in chunks if any(b.kind == BlockKind.TABLE for b in c.blocks)]
    assert len(table_chunks) == 1
    (table_containing_chunk,) = table_chunks
    table_blocks_in_chunk = [b for b in table_containing_chunk.blocks if b.kind == BlockKind.TABLE]
    assert len(table_blocks_in_chunk) == 1
    assert table_blocks_in_chunk[0].table_rows == rows


def test_empty_document_produces_no_chunks():
    doc = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=[])

    assert chunk_blocks(doc) == []
