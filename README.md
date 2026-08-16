# docstruct

Turn a long, unstructured Persian (Farsi) PDF into a structured JSON document
(title, nested headings, sections, tables) and clean Markdown.

```
PDF  ──parse──▶  ParsedDocument  ──structure_document (LLM, chunk by chunk)──▶  StructuredDocument (JSON)  ──convert──▶  Markdown
```

Three independent modules, each usable on its own:

- **`docstruct.parser`** — PDF → `ParsedDocument` (flat, order-preserving text/table blocks with style hints). Five swappable backends (PyMuPDF, pdfplumber, `unstructured`, `docling`, `anydoc`); see [`docs/parser_comparison.md`](docs/parser_comparison.md) for how they were benchmarked on real Persian PDFs and why PyMuPDF is the default.
- **`docstruct.agent`** — `ParsedDocument` → `StructuredDocument`, via a [pydantic-ai](https://ai.pydantic.dev/) agent that processes the document **chunk by chunk** (never the whole text in one prompt), maintaining a running document tree that each chunk's classified operations are merged into. Model access is any OpenAI-compatible endpoint (e.g. a self-hosted LiteLLM gateway), configured entirely through environment variables.
- **`docstruct.converter`** — `StructuredDocument` → Markdown. Pure function, depends only on `docstruct.schema` (not on the parser or agent modules).

`docstruct.schema` defines the generic, domain-agnostic document tree (`StructuredDocument` / `Section` / `TableData`) shared by the agent and converter.

## Install

Requires Python 3.12+. This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                     # core install: pymupdf + pydantic-ai + the framework
uv sync --extra benchmark   # + pdfplumber, unstructured, docling, anydoc — needed to reproduce docs/parser_comparison.md
```

The other four parser backends are optional extras on their own too:
`uv sync --extra pdfplumber`, `--extra unstructured`, `--extra docling`, `--extra anydoc`.

## Configure

Copy `.env.example` to `.env` and fill in your gateway's values:

```bash
cp .env.example .env
```

```dotenv
DOCSTRUCT_BASE_URL=https://your-litellm-gateway/v1
DOCSTRUCT_API_KEY=sk-...
DOCSTRUCT_MODEL=openai/gpt-4o   # any model id your gateway's /v1/models lists
```

`.env` is gitignored — never commit real keys. Nothing in the codebase hardcodes a model vendor: swap `DOCSTRUCT_MODEL`/`DOCSTRUCT_BASE_URL` to point at any OpenAI-compatible endpoint.

## Usage

```python
from dotenv import load_dotenv
load_dotenv()

from docstruct.parser import parse
from docstruct.agent import structure_document
from docstruct.converter import structured_document_to_markdown

parsed = parse("my_document.pdf")          # ParsedDocument (PyMuPDF backend by default)
document = structure_document(parsed)       # StructuredDocument — chunked LLM structuring
markdown = structured_document_to_markdown(document)

print(document.model_dump_json(indent=2))   # structured JSON
print(markdown)                             # clean Markdown
```

Swap the parser backend explicitly if you want:

```python
parsed = parse("my_document.pdf", backend="pdfplumber")  # or "unstructured", "docling", "anydoc"
```

To troubleshoot a specific chunk (e.g. a run that produced unexpected or missing output), pass `on_chunk` to
write every chunk's exact prompt and the model's raw response (or error) to disk, and optionally
`skip_failed_chunks=True` so one bad chunk doesn't abort the whole run:

```python
from docstruct.agent import file_trace_writer

document = structure_document(
    parsed,
    on_chunk=file_trace_writer("trace/"),   # trace/chunk_001.prompt.txt, chunk_001.result.json, ...
    skip_failed_chunks=True,                # log a failed chunk's error and continue instead of raising
)
```

Or convert a JSON file you already have, with zero parser/agent dependency:

```python
from docstruct.schema import StructuredDocument
from docstruct.converter import structured_document_to_markdown

document = StructuredDocument.model_validate_json(open("document.json", encoding="utf-8").read())
print(structured_document_to_markdown(document))
```

Runnable end-to-end examples, including the exact JSON/Markdown a real run produced, are in
[`tests/fixtures/`](tests/fixtures/) (`*.pdf` in, `*.expected.json` / `*.expected.md` out) — see
[`tests/fixtures/README.md`](tests/fixtures/README.md) for what each fixture is.

## Testing

```bash
uv run pytest
```

Unit tests (schema, converter, chunking, builder, and the agent's wiring — the latter via pydantic-ai's `FunctionModel`, so no live model calls) run offline and fast. `tests/test_integration.py` additionally runs the real PDF → JSON → Markdown pipeline against your configured gateway on the two fixtures; it's automatically skipped if `DOCSTRUCT_API_KEY` isn't set. Skip it explicitly with `uv run pytest -k "not integration"`.

**Troubleshooting a `Connection error` from the integration test / any live gateway call:** some sandboxes route outbound traffic through a local proxy (`HTTPS_PROXY`/`HTTP_PROXY`) that can't reach a self-hosted gateway. If `curl`/PowerShell can reach your gateway directly but Python can't, try clearing those env vars for the call (`HTTPS_PROXY= HTTP_PROXY= uv run pytest`) or add the gateway's host to `NO_PROXY`.

To reproduce the parser benchmark:

```bash
uv sync --extra benchmark
uv run python scripts/benchmark_parsers.py
```

## Known limitations

- **Chunked LLM structuring can lose content in two distinct ways.** (1) Content near a chunk's edges can be under- or over-attributed — the model has no memory between calls, so a paragraph/heading split by a chunk boundary is occasionally misjudged (a model-fidelity limitation of a single-pass-per-chunk design, not a data-pipeline bug; visible in `tests/fixtures/*.expected.md`). Mitigated by `overlap_blocks` (default 2, see `structure_document`), which repeats the last few blocks of each chunk as read-only context at the start of the next one, so boundary-adjacent content is judged with its immediate neighbor in view — `DocumentBuilder` also dedupes identical ops within a small rolling window in case the model re-emits an overlapping block anyway. Running with `overlap_blocks=0` reintroduces this directly: a sentence split exactly across a chunk boundary can lose its opening half entirely, since the earlier chunk correctly declines to emit an incomplete fragment and the later chunk has no visibility into what preceded it. (2) A chunk's response can silently under-cover its content even though it parses fine and raises no error — verified directly against the live gateway, and *not* limited to long repetitive lists like a table of contents (a single response covering only the first of several varied blocks — a paragraph, a heading, another paragraph — was also observed). Multiple prompt-only mitigations (an explicit completeness instruction, a smaller `max_chars_per_chunk`) were tried and found unreliable — they helped on some chunks, not others. `max_attempts_per_chunk` / `completeness_threshold` (see `structure_document`) address this at the code level instead: a rough character-coverage heuristic (`_coverage_ratio` in `refiner.py`) compares each attempt's resolved text against the chunk's source blocks, and an under-covered or outright-failed attempt is retried — a fresh, independently-sampled call — keeping the best attempt seen. This reduces the frequency of both failure modes; it doesn't formally guarantee zero loss for either (an exact guarantee would need a completeness-verification pass keyed to source block indices, reasonable future work beyond this project's scope).
- **`docling` did not finish converting even a single page within a 240s timeout** on the CPU-only sandbox this was built on (no GPU) — see `docs/parser_comparison.md`. The backend code is implemented and correct against the library's API, just unverified against real Persian content here.
- **OCR / scanned documents are out of scope** (per the project brief) — all five parser backends operate on the PDF's text layer only.
- **`anydoc` is by far the fastest backend (~1–2s on a ~280-page document) but the noisiest** — its Markdown-based heading/table detection produces significantly more false positives than PyMuPDF on these real documents (e.g. wrapped lines of one paragraph emitted as several separate headings). See `docs/parser_comparison.md`.
- Extracted Persian text can use Arabic Presentation Forms and/or Arabic (rather than Persian-specific) Yeh/Kaf letterforms — cosmetically identical but breaks naive string comparisons. `docstruct.agent.refiner` normalizes text before prompting the model; see `docs/parser_comparison.md` for the specifics if you're processing extracted text elsewhere.

## Project layout

```
src/docstruct/
  schema.py            StructuredDocument / Section / TableData — no dependencies beyond pydantic
  parser/               PDF -> ParsedDocument, 5 swappable backends
  agent/                ParsedDocument -> StructuredDocument, chunk by chunk (pydantic-ai)
  converter.py          StructuredDocument -> Markdown, pure function
scripts/
  benchmark_parsers.py  reproduces docs/parser_comparison.md
tests/
  fixtures/              real Persian PDF excerpts + example JSON/Markdown outputs
data/                   the two full real source PDFs the fixtures were excerpted from
docs/
  parser_comparison.md  parser benchmark write-up + reasoning for the PyMuPDF pick
```
