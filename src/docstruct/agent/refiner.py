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
- ``DOCSTRUCT_REASONING_EFFORT``: optional. For a reasoning-capable model, ``"none"`` disables its hidden
  chain-of-thought tokens (which otherwise multiply per-chunk latency for no benefit on this small-output,
  structured-extraction task). Sent as both ``reasoning_effort`` and OpenRouter-style
  ``reasoning: {"enabled": false}`` in the request body, since gateways differ on which field they honor.
  Unset by default, so a model's own default reasoning behavior applies.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ..parser.types import Block, BlockKind, ParsedDocument
from ..schema import StructuredDocument
from .builder import DocumentBuilder
from .chunking import Chunk, chunk_blocks
from .ops import ChunkOp, ChunkResult, NewHeading, NewParagraph

_DEFAULT_BASE_URL = "https://ai-gateway.mohaymen.ir/v1"
_DEFAULT_MODEL = "openai/gpt-4o"

# The document's own table of contents (فهرست مطالب, PDF pages 5-9), transcribed once and reused as a
# fixed reference in every chunk's prompt — see _SYSTEM_PROMPT's "TABLE OF CONTENTS" section for how it's
# used. It only lists the document's outermost two levels (بخش, بخشنامه); everything deeper is not in it.
_TABLE_OF_CONTENTS = """\
بخش اول: بخشنامه های مطالعات و مقررات بانکی
1. 21270 - ابلاغ «دستورالعمل نحوه تشخیص و حذف مطالبات غیرقابل وصول از دفاتر مؤسسه اعتباری و افشای آن در
   صورت های مالی» و اصلاح «دستورالعمل نحوه محاسبه ذخیره مطالبات موسسات اعتباری»
2. 29506 - ابلاغ دستورالعمل حسابداری کارت اعتباری بر پایه عقد مرابحه و کارت میزان
3. 38094 - تعیین نرخ سود جدید تسهیلات مرابحه از طریق کارت اعتباری میزان
4. 44341 - ابلاغ مصوبه کمیسیون اعتباری بانک مرکزی در خصوص اجرایی نمودن تسهیلات بیمه
5. 59912 - ابلاغ دستورالعمل حساب جاری
6. 62668 - ابلاغ ممنوعیت بلوکه نمودن بخشی از تسهیلات اعطایی به عنوان سپرده
7. 66711 - اطلاع رسانی بانک ها در خصوص تشکیل کمیسیون مقررات و نظارت مؤسسات اعتباری
8. 78002 - تاکید بر عندالمطالبه بودن موعد پرداخت مبلغ ضمانت نامه بانکی به ذینفع
9. 77969 - ابلاغ ضوابط اجرایی تبصره 1 ماده 186 قانون مالیات های مستقیم
10. 85331 - ابلاغ نرخ های جدید کارمزد خدمات بانکی (ریالی) به شبکه بانکی کشور
11. 88608 - نحوه محاسبه سود و اقساط در تسهیلات دارای یارانه سود سهم دولت
12. 91728 - ابلاغ مفاد بند 7-7 ماده واحده قانون بودجه سال 1391 کل کشور
13. 99844 - لزوم اخذ شماره اقتصادی توسط اشخاص حقیقی و حقوقی
14. 108312 - تعیین حداقل میزان بازپرداخت بدهی ناشی از خرید کالا از سوی دارنده کارت اعتباری میزان
15. 108313 - ابلاغ تکالیف بانک های غیردولتی و موسسات اعتباری غیربانکی در قانون بودجه سال 1391
16. 108314 - ابلاغ تکالیف بانک های دولتی در قانون بودجه سال 1391 کل کشور
17. 110078 - ابلاغ نرخ های جدید کارمزد خدمات بانکی (ریالی) - پیرو بخشنامه شماره 85331/91
18. 122342 - ابلاغ بخشنامه ناظر بر نسخه آیین نامه جدید تسهیلات و تعهدات کلان
19. 130995 - اعلام مصوبه شورای پول و اعتبار درخصوص پذیرش سهام بورسی به عنوان وثیقه
20. 131164 - نحوه استفاده از خدمات بانکی توسط نابینایان کشور
21. 143259 - ابلاغ تصویب نامه هیأت وزیران در خصوص اساسنامه صندوق ضمانت سپرده ها
22. 147874 - ابلاغ حداکثر سقف کارت اعتباری مرابحه در سال 91
23. 166503 - ابلاغ دستورالعمل ناظر بر تسهیلات سندیکایی
24. 178901 - ابلاغ نرخ سود تسهیلات کارت اعتباری میزان
25. 186342 - ابلاغ دستورالعمل حسابداری عقد استصناع و دستورالعمل حسابداری عقد خرید دین
26. 210843 - ابلاغ دستورالعمل نگاهداری انواع حساب برای دستگاه های اجرایی و مؤسسات دولتی
27. 212746 - اصلاح ماده 16 دستورالعمل اجرایی کارت اعتباری بر پایه عقد مرابحه
28. 243110 - ابلاغ دستورالعمل تعیین نسبت تعهدات و بدهی های ارزی به دارایی های ارزی
29. 244700 - ابلاغ بند (1) صورتجلسه شورای پول و اعتبار در خصوص اعتبار اسنادی داخلی-ریالی
30. 252693 - ابلاغ اصلاحیه شرایط و ضوابط افتتاح حساب قرض الحسنه ویژه
31. 253004 - ابلاغ ممنوعیت اخذ هرگونه کارمزد برای دریافت حضوری قبوض از مشتریان
32. 277102 - ابلاغ ضوابط ناظر بر تعرفه های بانکی اعتبارات اسنادی داخلی-ریالی
33. 292087 - ابلاغ نسخه نهایی دستورالعمل اجرایی تأسیس و نظارت بر صندوق های قرض الحسنه
34. 294711 - ابلاغ بند (2) مصوبه شورای پول و اعتبار در خصوص نرخ سود تسهیلات ریالی صادراتی
35. 294844 - ابلاغ مصوبه کمیسیون مقررات در خصوص الزام درج شرط وصایت در قراردادهای سپرده گیری
36. 298263 - هشدار به بانک ها و مؤسسات برای رعایت دقیق قوانین و مقررات نظارتی
37. 306208 - ارسال جدول تسهیلات و تعهدات کلان
38. 311201 - ابلاغ حداقل ضوابط مشتری معتبر، موضوع تبصره ذیل ماده 12 دستورالعمل حساب جاری
39. 325334 - ابلاغ مصوبه کمیسیون مقررات در خصوص نحوه صدور کارت هدیه و کارت بن
40. 332502 - ابلاغ آئین نامه ایجاد یا تعطیل شعبه یا باجه مؤسسات اعتباری در داخل کشور
41. 343185 - ابلاغ مستثنی شدن طرحهای فولادی سرمایه گذاری شده توسط بخش خصوصی از آیین نامه وصول مطالبات
42. 352010 - ابلاغ آیین نامه ایجاد و تأسیس شعب بانک های قرض الحسنه مهر ایران و رسالت
43. 353546 - ابلاغ آیین نامه اجرایی بند (102) قانون بودجه سال 1391 کل کشور

بخش دوم: بخشنامه های مبارزه با پولشویی
1. 2979 - درخصوص اخذ شماره فراگیر مشتری
2. 36978 - درخصوص اخذ شماره فراگیر مشتریان
3. 55597 - درخصوص ایجاد قابلیت در سامانه های بانکی
4. 75324 - درخصوص لغو استثنای برخی اشخاص در ارائه شناسه ملی
5. 111131 - درخصوص تاکید بر لغو استثنای برخی اشخاص در ارائه شناسه ملی
6. 141997 - درخصوص ایجاد قابلیت در سامانه های بانکی
7. 161337 - درخصوص برنامه نرم افزاری مبارزه با پولشویی
8. 177911 - درخصوص تاکید بر فراگیری شناسه ملی در کلیه فعالیت های بانک
9. 202825 - درخصوص شاخص معاملات مشکوک
10. 213103 - موضوع تاکید بر عدم پرداخت وجه نقد بیش از سقف مقرر
11. 257835 - موضوع تاکید بر اعمال تغییر در سامانه های بانکی و تهیه نرم افزار مبارزه با پولشویی
12. 266254 - موضوع معرفی پایگاه اطلاع رسانی مرتبط با شناسه ملی
13. 282772 - موضوع تاکید بر ایجاد اداره مستقل مبارزه با پولشویی
14. 340549 - موضوع پرسشنامه اصلاح فرایندهای بانکی و تولید نرم افزارهای مبارزه با پولشویی

بخش سوم: بخشنامه های مجوزهای بانکی
1. 2123 - اعلام نظر بانک مرکزی درخصوص برنامه تأسیس واحدهای بانکی بانک ها
2. 160017 - صدور مجوز فعالیت بانک قوامین
"""

_SYSTEM_PROMPT = f"""\
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
- A block that is the document's running header, printed on nearly every page: "بخشنامه های مدیریت کل
  مقررات، مجوزهای بانکی و مبارزه با پولشویی سال 1391". PDF text extraction sometimes reorders or garbles
  its characters (e.g. digits/words out of order, stray isolated letters), so match it by recognizing the
  same words in roughly the same combination, not by requiring an exact character-for-character match.
  Skip it in every form it appears in, including inside a table-of-contents list — never emit a paragraph
  (or any other) operation for it, and never merge it into a neighboring paragraph.

Treat a (possibly merged) span as a "heading" ONLY when it plausibly functions as a title/section label on
its own: a standalone label, PLUS at least one supporting signal — larger font_size than surrounding
blocks, bold, or a structural numbering/legal-article pattern (see below). Something that reads like a
list/table-of-contents entry, or a mid-sentence continuation, is body text, not a heading, regardless of
length.

TABLE OF CONTENTS — the document's own table of contents, reproduced below as a fixed reference (the same
text on every chunk, independent of what you're currently looking at). It only lists the two outermost
structural levels: 3 top-level "بخش" sections (L1), and the numbered "بخشنامه" entries within each (L2,
identified by their unique number). Nothing deeper (دستورالعمل/فصل/ماده/بند/تبصره titles) is listed here.

{_TABLE_OF_CONTENTS}
Before applying the signal-based nesting rules below, check whether the current chunk contains a heading
whose text matches one of the entries above — match primarily by the بخشنامه NUMBER embedded in the
heading text (numbers survive PDF extraction far more reliably than the surrounding words), falling back
to matching by title wording only if no number is present. If it matches a "بخش" line, that heading is
level 1. If it matches a numbered بخشنامه entry, that heading is level 2. Use that level regardless of
what a local numbering/font/bold signal alone would otherwise suggest. If a heading does NOT match
anything above — which is normal and expected for everything below a بخشنامه (its internal
دستورالعمل/فصل titles, ماده, بند, تبصره) — keep using the signal-based rules below to place it, nested
under whichever بخش/بخشنامه is currently open in the breadcrumb.

EXCEPTION, and it takes priority over the table-matching rule above: if the matching line is itself
formatted as a table-of-contents/list entry — i.e. it ends in a run of dots/leader characters followed by
a page number (the same pattern the earlier "table of contents ... is a sequence of SEPARATE complete
entries" rule already identifies) — do NOT assign it a heading level from the table at all. That pattern
means you are still reading the فهرست مطالب listing itself (which can span several chunks), not a real
occurrence of that section later in the document's body, so the earlier rule applies instead: it is body
text, emit it as a paragraph (including its dot-leaders and page number, unmodified), never as a heading —
even though its wording matches an entry in the table above. Only apply the table's heading level to a
matching line that does NOT end in a dot-leader/page-number pattern, since that is what marks a real
section start in the body rather than a listing of one.

Nesting from explicit numbering and legal structure — these are stronger signals than font size/boldness
alone, use them whenever present:
- Explicit numbering at the start of a heading, like "1-", "1.2-", or "2.4.1", tells you the nesting level
  directly: count the number of dot/dash-separated numeric segments (e.g. "2.4.1" has 3 segments, so it's
  3 levels deep) and nest it that many levels under whichever unnumbered section (e.g. a "فصل"/"بخش") is
  currently open in the breadcrumb.
- "ماده" marks a legal Article — it normally nests directly under the currently open بخش/فصل, at whatever
  level comes after that in the breadcrumb. A new ماده is a SIBLING of the most recent other ماده, always
  — scan the whole breadcrumb (not just its innermost/last entry) for the most recent heading that is
  itself a ماده, and reuse that exact level, no matter how many بند/زیربند/تبصره levels have opened and
  closed since then. Do NOT derive a new ماده's level from "one level shallower than whatever is currently
  innermost" — that default is only a coincidence when the innermost open heading happens to be a direct
  sibling, and gives the wrong (too-deep) level whenever the previous ماده's own بند/زیربند/تبصره content
  is still open. If no ماده appears anywhere in the breadcrumb, fall back to nesting it directly under the
  currently open بخش/فصل as before.
- Numbered بند/زیربند items read their parent from their OWN numeric prefix, not from whichever heading
  happens to be currently open. The hierarchy is ماده → بند → زیربند: for numbering like "2-2", the first
  segment (2) is the ماده number and the second segment (2) is the بند number within that ماده — nest it as
  a child (بند) of ماده 2. For numbering like "2-2-1", "2-2-2", "2-2-3", ..., the first two segments ("2-2")
  identify the parent بند, and the last segment (1, 2, 3, ...) is just the sequential زیربند number within
  that بند — nest each of these as a child (زیربند) of بند 2-2, i.e. two levels under ماده 2. Every entry
  sharing the same leading segments (e.g. everything starting "2-2-") is a sibling at that same زیربند
  level under that one بند, regardless of what else appeared in the text between them.
- "تبصره" marks a Note/clause. It is its own structural element, NOT the parent of any numbered بند/زیربند
  items that happen to follow it in the text — those still nest strictly by their own numeric prefix as
  described above, never as children of a تبصره. Nest the تبصره itself ONE LEVEL DEEPER than, and as a
  child of, whichever ماده or بند it is actually attached to by its position in the document (normally the
  ماده, or the specific numbered بند, whose text it immediately follows) — not automatically the outermost
  or most-recently-opened ماده if a more specific بند is the one it's really commenting on.

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
  NOT emit an operation for this]". Normally, do NOT emit a heading/paragraph/table operation for a block
  marked this way — they're shown only so you can see what immediately precedes the new content, to
  correctly judge whether the new content continues the same paragraph/section or starts a new one.
  EXCEPTION: the previous chunk can occasionally fail to actually cover a block despite it being shown to
  you as context — check CONTEXT blocks against "Currently open headings" above. If a CONTEXT block
  plausibly functions as a heading (by the same signals as elsewhere: standalone label, larger font/bold,
  numbering/ماده/تبصره) but is NOT the innermost (last) entry in "Currently open headings", the previous
  chunk did not actually capture it as a heading — emit it yourself, as a new heading at the correct
  nesting level, instead of silently skipping it, since otherwise everything nested under it in this
  chunk would attach one level too shallow. Likewise, if a non-heading CONTEXT block is an unfinished
  sentence that your chunk's first new block grammatically continues, merge it into your first paragraph
  the same way you would merge any other wrapped fragment, instead of leaving it stranded.
- Never invent headings, structure, or content that are not present in this chunk's text, and never
  paraphrase or "clean up" the source text — reproduce it as given, aside from joining wrapped-line
  fragments with a single space as instructed above.
- If two headings are consecutive — one heading immediately followed by another heading, with no
  paragraph or table between them — their levels can never be equal, and must differ by exactly 1 (never
  by 2 or more). A heading directly followed by another heading is normally that heading's own title/
  caption for what follows, one level deeper; assign levels accordingly instead of leaving them as
  siblings or skipping a level.
- If the whole document's title becomes apparent in this chunk (usually only possible in the first
  chunk), set `title`; otherwise leave it null.
"""


def default_model() -> OpenAIChatModel:
    base_url = os.environ.get("DOCSTRUCT_BASE_URL", _DEFAULT_BASE_URL)
    api_key = os.environ.get("DOCSTRUCT_API_KEY")
    model_name = os.environ.get("DOCSTRUCT_MODEL", _DEFAULT_MODEL)
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)

    reasoning_effort = os.environ.get("DOCSTRUCT_REASONING_EFFORT")
    settings = None
    if reasoning_effort:
        # Different OpenAI-compatible gateways honor different fields for this ("reasoning_effort" vs
        # OpenRouter-style "reasoning": {"enabled": ...}) — send both, unrecognized fields are ignored.
        extra_body: dict = {"reasoning_effort": reasoning_effort}
        if reasoning_effort == "none":
            extra_body["reasoning"] = {"enabled": False}
        settings = OpenAIChatModelSettings(extra_body=extra_body)

    return OpenAIChatModel(model_name, provider=provider, settings=settings)


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


def _coverage_ratio(chunk: Chunk, operations: list[ChunkOp]) -> float:
    """Rough heuristic for how much of a chunk's new (non-CONTEXT) content
    made it into `operations`. Used only to compare repeated attempts at
    the *same* chunk against each other — not an exact completeness proof:
    legitimately-skipped page numbers/running headers, and whitespace
    differences introduced by merging wrapped lines, both push a fully
    correct response below 1.0. A higher ratio means a more complete
    response relative to other attempts at the same prompt."""
    source_chars = sum(len(b.text or "") for b in chunk.new_blocks if b.kind == BlockKind.TEXT)
    if source_chars == 0:
        return 1.0
    covered_chars = sum(len(op.text) for op in operations if isinstance(op, (NewHeading, NewParagraph)))
    return min(covered_chars / source_chars, 1.0)


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
    max_attempts_per_chunk: int = 2,
    completeness_threshold: float = 0.7,
    align_maddeh_siblings: bool = True,
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

    ``max_attempts_per_chunk`` / ``completeness_threshold``: a chunk's
    response can silently under-cover its content even though it parses
    fine — verified directly against the live gateway that a single chunk
    with several varied blocks (a paragraph, a heading, another paragraph)
    can come back with just one operation covering the first few blocks and
    nothing else, no error raised. This is a model reliability gap, not
    reliably fixed by prompt wording alone (multiple prompt-only attempts
    still showed it — see README's "Known limitations"). After each
    attempt, ``_coverage_ratio`` compares the resolved operations' text
    against the chunk's source blocks; if the ratio is below
    ``completeness_threshold``, the chunk is retried — a fresh,
    independently-sampled call, not a continuation — up to
    ``max_attempts_per_chunk`` times, keeping whichever attempt scored
    highest. An attempt that raises (e.g. a malformed-output validation
    failure) also counts as a reason to retry rather than an immediate
    failure, so this doubles as resilience against that failure class too.
    This is a heuristic that improves the odds of a complete result by
    resampling — it does not verify completeness exactly, and doesn't
    eliminate the failure mode, only reduces its frequency.

    ``on_chunk``, if given, is called once per chunk (after all attempts
    for it are done) with a :class:`ChunkTrace` — for troubleshooting which
    chunk produced a given piece of output, or none at all.

    ``skip_failed_chunks``: by default, a chunk where every attempt raises
    propagates and aborts the whole run. On a long document a single bad
    chunk out of many is disproportionately costly to lose everything over,
    so set this to ``True`` to log the failure via ``on_chunk`` and
    continue with the next chunk instead — that chunk's content is then
    simply missing from the result rather than the whole run failing.

    ``align_maddeh_siblings``: see :class:`~docstruct.agent.builder.DocumentBuilder`. Only relevant to
    documents that use ماده-numbered legal articles; set to ``False`` for documents that don't, or if
    you'd rather trust the model's own heading level for ماده-like text unmodified.
    """
    resolved_model = model if model is not None else default_model()
    agent: Agent[None, ChunkResult] = Agent(
        resolved_model, output_type=ChunkResult, system_prompt=_SYSTEM_PROMPT
    )

    builder = DocumentBuilder(align_maddeh_siblings=align_maddeh_siblings)
    chunks = chunk_blocks(parsed, max_chars=max_chars_per_chunk, overlap_blocks=overlap_blocks)
    for index, chunk in enumerate(chunks, start=1):
        prompt = (
            f"Currently open headings: {_render_breadcrumb(builder.breadcrumb(max_items=5))}\n\n"
            f"Current chunk:\n{_render_chunk(chunk)}"
        )

        best_result: ChunkResult | None = None
        best_ratio = -1.0
        last_error: Exception | None = None
        for _attempt in range(max_attempts_per_chunk):
            try:
                result = agent.run_sync(prompt)
            except Exception as exc:  # noqa: BLE001 - retried below; re-raised only if every attempt fails
                last_error = exc
                continue
            ratio = _coverage_ratio(chunk, result.output.operations)
            if ratio > best_ratio:
                best_result, best_ratio = result.output, ratio
            if ratio >= completeness_threshold:
                break

        if best_result is None:
            if on_chunk is not None:
                on_chunk(ChunkTrace(index=index, chunk=chunk, prompt=prompt, result=None, error=last_error))
            if skip_failed_chunks:
                continue
            raise last_error

        chunk_result = best_result
        if on_chunk is not None:
            on_chunk(ChunkTrace(index=index, chunk=chunk, prompt=prompt, result=chunk_result, error=None))
        if chunk_result.title:
            builder.set_title(chunk_result.title)
        for op in chunk_result.operations:
            builder.apply(op)

    return builder.build()
