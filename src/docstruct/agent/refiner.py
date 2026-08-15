"""The chunked structuring agent: turns a :class:`ParsedDocument` into a
:class:`StructuredDocument` by walking it chunk by chunk with a pydantic-ai
agent, merging each chunk's operations into a running
:class:`DocumentBuilder` rather than asking the model for the whole tree at
once.

Model access is via any OpenAI-compatible endpoint (e.g. a self-hosted
LiteLLM gateway), configured through environment variables so it is never
hardcoded to a specific vendor:

- ``DOCSTRUCT_BASE_URL``: OpenAI-compatible base URL (default: the
  project's LiteLLM gateway).
- ``DOCSTRUCT_API_KEY``: API key for that gateway.
- ``DOCSTRUCT_MODEL``: model id as listed by the gateway's ``/v1/models``.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..parser.types import Block, BlockKind, ParsedDocument
from ..schema import StructuredDocument
from .builder import DocumentBuilder
from .chunking import Chunk, chunk_blocks
from .ops import ChunkResult

_DEFAULT_BASE_URL = "https://ai-gateway.mohaymen.ir/v1"
_DEFAULT_MODEL = "openai/gpt-4o"

_SYSTEM_PROMPT = """\
You are converting a Persian (Farsi) document into a structured outline, one chunk of the document at a
time. You do not see the whole document at once, only the current chunk of raw text/table blocks below,
plus a short breadcrumb of the section headings currently open above this point in the document.

Each TEXT block below is ONE LINE as segmented by the PDF's line-by-line text extraction — NOT one
paragraph and NOT one heading. A single heading or paragraph is very often split across several
consecutive TEXT blocks purely because of how the page was laid out (line wrapping, columns, justified
text). You must reassemble these before classifying: read the blocks in order and merge any run of
consecutive blocks that clearly continue one sentence/label into a SINGLE operation whose text is the
full combined wording (join the fragments with a single space, in order) — do NOT emit one operation per
line fragment. For example, "بخشنامه های مطالعات و" immediately followed by "مقررات بانکی" is one heading,
"بخشنامه های مطالعات و مقررات بانکی", not two; two paragraph fragments that together form one sentence are
one paragraph operation, not two.

The signal for "these blocks belong together" is semantic and grammatical, not just short length: if a
block ends with no closing punctuation and grammatically requires what follows to complete the thought (a
genitive/ezafe construction trailing off, a dangling connector, a sentence with no verb yet), merge it
with the next block(s) until you reach one that actually completes the sentence or label. Do NOT merge
blocks that are each already complete on their own, even if short and unpunctuated — most importantly, a
table of contents or any list/index section (many short entries, each ending in a page number, often
after a run of dots) is a sequence of SEPARATE complete entries: each one is its own paragraph operation,
never merged with its neighbors just because none of them end in punctuation.

Ignore, and never emit any operation for:
- A block whose entire content is just a page number (a bare number, optionally with surrounding dashes
  or dots, e.g. "5", "- 12 -") — this is not document content.
- A running header or footer: a short phrase that recurs verbatim at the top (or bottom) of page after
  page throughout the document — typically the document's own title/letterhead repeated everywhere, not
  a fresh paragraph or heading each time. A chunk long enough to span multiple pages will often show this
  same exact phrase reappearing more than once; treat every occurrence as page furniture to skip, not as
  new content — including a single occurrence you recognize as this kind of running letterhead text even
  if it only appears once in the current chunk.

Treat a (possibly merged) span as a "heading" ONLY when it plausibly functions as a title/section label on
its own: a standalone label, PLUS at least one supporting signal — larger font_size than surrounding
blocks, bold, or a structural numbering/legal-article pattern (see below). Something that reads like a
list/table-of-contents entry, or a mid-sentence continuation, is body text, not a heading, regardless of
length.

Nesting from explicit numbering and legal structure — these are stronger signals than font size/boldness
alone, use them whenever present:
- Explicit numbering at the start of a heading, like "1-", "1.2-", or "2.4.1", tells you the nesting level
  directly: count the number of dot/dash-separated numeric segments (e.g. "2.4.1" has 3 segments, so it's
  3 levels deep) and nest it that many levels under whichever unnumbered section (e.g. a "فصل"/"بخش") is
  currently open in the breadcrumb.
- "ماده" marks a legal Article — it normally nests directly under the currently open بخش/فصل, at whatever
  level comes after that in the breadcrumb.
- "تبصره" marks a Note/clause that belongs to the ماده immediately preceding it — nest it ONE LEVEL DEEPER
  than that ماده, as its child, not as its sibling.

For the CURRENT CHUNK ONLY, emit a list of operations, in the same order the content appears:
- "heading": as described above. Infer its nesting level (1 = top-level) from the signals above and from
  the currently-open breadcrumb — there is no explicit table of contents to rely on, so structure must be
  inferred conservatively; when in doubt, prefer "paragraph" over "heading".
- "paragraph": a paragraph (after merging any wrapped-line fragments per above) of body text belonging to
  whichever heading is currently open (or to the document's preamble if no heading has appeared yet
  anywhere in the document).
- "table": a table's caption/headers/rows, preserved as given, attached to the currently open heading.

Rules:
- Only emit operations for content that is actually in the current chunk. Never repeat content already
  covered by earlier chunks (you only see their headings, as a breadcrumb, not their full text).
- Some blocks at the start of a chunk are marked "[CONTEXT - already covered by the previous chunk, do
  NOT emit an operation for this]". They are shown only so you can see what immediately precedes the new
  content, to correctly judge whether the new content continues the same paragraph/section or starts a
  new one. Never emit a heading/paragraph/table operation for a block marked this way.
- Never invent headings, structure, or content that are not present in this chunk's text, and never
  paraphrase or "clean up" the source text — reproduce it as given, aside from joining wrapped-line
  fragments with a single space as instructed above.
- If the whole document's title becomes apparent in this chunk (usually only possible in the first
  chunk), set `title`; otherwise leave it null.
"""


def default_model() -> OpenAIChatModel:
    base_url = os.environ.get("DOCSTRUCT_BASE_URL", _DEFAULT_BASE_URL)
    api_key = os.environ.get("DOCSTRUCT_API_KEY")
    model_name = os.environ.get("DOCSTRUCT_MODEL", _DEFAULT_MODEL)
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name, provider=provider)


_ARABIC_TO_PERSIAN_LETTERS = str.maketrans({"ي": "ی", "ك": "ک"})


def _normalize_persian(text: str) -> str:
    """NFKC-fold presentation-form glyphs and map Arabic Yeh/Kaf to their
    Persian-specific forms — see docs/parser_comparison.md for why this
    matters for text coming out of a PDF's text layer."""
    return unicodedata.normalize("NFKC", text).translate(_ARABIC_TO_PERSIAN_LETTERS)


def _render_block(block: Block) -> str:
    if block.kind == BlockKind.TEXT:
        hints = []
        if block.style:
            if block.style.font_size:
                hints.append(f"font_size={block.style.font_size:.0f}")
            if block.style.bold:
                hints.append("bold")
        hint = f" [{', '.join(hints)}]" if hints else ""
        # Collapse the block's internal line-wrap newlines to spaces: they're
        # PDF layout artifacts, not paragraph/heading boundaries, and left in
        # they make single paragraphs look like several short standalone lines.
        text = " ".join(_normalize_persian(block.text or "").split())
        return f"TEXT{hint}: {text}"

    rows_preview = "; ".join(" | ".join(row) for row in (block.table_rows or []))
    return f"TABLE: {rows_preview}"


def _render_chunk(chunk: Chunk) -> str:
    rendered = []
    for i, block in enumerate(chunk.blocks):
        text = _render_block(block)
        if i < chunk.context_prefix_len:
            text = f"[CONTEXT - already covered by the previous chunk, do NOT emit an operation for this] {text}"
        rendered.append(text)
    return "\n\n".join(rendered)


def _render_breadcrumb(breadcrumb: list[tuple[int, str]]) -> str:
    if not breadcrumb:
        return "(none open yet)"
    return " > ".join(f"H{level}:{text}" for level, text in breadcrumb)


@dataclass
class ChunkTrace:
    """One chunk's prompt and outcome, handed to ``on_chunk`` for
    troubleshooting ``structure_document`` — e.g. to see exactly what was
    sent to the model for a chunk that produced unexpected or missing
    output. ``result`` is set on success, ``error`` on failure; exactly one
    of the two is non-None. See ``docstruct.agent.tracing.file_trace_writer``
    for a ready-made callback that writes these to disk."""

    index: int
    chunk: Chunk
    prompt: str
    result: ChunkResult | None
    error: BaseException | None


def structure_document(
    parsed: ParsedDocument,
    *,
    model: Model | str | None = None,
    max_chars_per_chunk: int = 8000,
    overlap_blocks: int = 2,
    on_chunk: Callable[[ChunkTrace], None] | None = None,
    skip_failed_chunks: bool = False,
) -> StructuredDocument:
    """Build a :class:`StructuredDocument` from ``parsed``, chunk by chunk.

    ``overlap_blocks`` repeats the last N blocks of each chunk as read-only
    context at the start of the next one (see ``chunking.chunk_blocks``) —
    this helps the model correctly continue a paragraph/section that was
    cut mid-way by a chunk boundary. Each chunk call is still independent
    (no shared conversation history), so the model has no memory of what it
    already emitted for those overlapping blocks; the prompt instructs it
    not to re-emit them, and ``DocumentBuilder`` additionally dedupes
    identical ops within a small rolling window as a safety net in case it
    does anyway.

    ``on_chunk``, if given, is called once per chunk (success or failure)
    with a :class:`ChunkTrace` — for troubleshooting which chunk produced a
    given piece of output, or none at all.

    ``skip_failed_chunks``: by default, a chunk that raises (a model output
    validation failure, a transient network error, ...) propagates and
    aborts the whole run. On a long document a single bad chunk out of many
    is disproportionately costly to lose everything over, so set this to
    ``True`` to log the failure via ``on_chunk`` and continue with the next
    chunk instead — that chunk's content is then simply missing from the
    result rather than the whole run failing.
    """
    resolved_model = model if model is not None else default_model()
    agent: Agent[None, ChunkResult] = Agent(
        resolved_model, output_type=ChunkResult, system_prompt=_SYSTEM_PROMPT
    )

    builder = DocumentBuilder()
    chunks = chunk_blocks(parsed, max_chars=max_chars_per_chunk, overlap_blocks=overlap_blocks)
    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            f"Currently open headings: {_render_breadcrumb(builder.breadcrumb(max_items=5))}\n\n"
            f"Current chunk:\n{_render_chunk(chunk)}"
        )
        try:
            result = agent.run_sync(prompt)
        except Exception as exc:
            if on_chunk is not None:
                on_chunk(ChunkTrace(index=index, chunk=chunk, prompt=prompt, result=None, error=exc))
            if skip_failed_chunks:
                continue
            raise

        chunk_result = result.output
        if on_chunk is not None:
            on_chunk(ChunkTrace(index=index, chunk=chunk, prompt=prompt, result=chunk_result, error=None))
        if chunk_result.title:
            builder.set_title(chunk_result.title)
        for op in chunk_result.operations:
            builder.apply(op)

    return builder.build()
