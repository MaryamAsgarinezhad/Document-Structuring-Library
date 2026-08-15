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

- **Chunked LLM structuring can occasionally under- or over-attribute content near a chunk's edges**, most visibly as an omitted heading or list item from a long chunk rather than fabricated content (we verified the raw parsed text reaches the model in these cases — this is a model-fidelity limitation of a single-pass-per-chunk design, not a data-pipeline bug). Visible in `tests/fixtures/*.expected.md`: e.g. a short run of numbered sub-articles occasionally goes missing between two chunks. Two mitigations are applied: a larger `max_chars_per_chunk` (fewer boundaries — tuned up from an initial 4000 to 8000 during development), and `overlap_blocks` (default 2, see `structure_document`), which repeats the last few blocks of each chunk as read-only context at the start of the next one so boundary-adjacent content is judged with its immediate neighbor in view. Since each chunk call is independent (no shared conversation history), the model has no memory of having already reported on that overlapping content, so it could in principle re-emit it — `DocumentBuilder` guards against that with a small rolling dedup window that drops an op if an identical one was applied recently. This reduces the frequency of the problem; it doesn't formally guarantee zero loss (that would need a completeness-verification pass keyed to source block indices, reasonable future work beyond this project's scope).
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
