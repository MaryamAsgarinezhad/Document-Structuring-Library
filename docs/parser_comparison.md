# PDF Parser Evaluation

Five backends were implemented behind a common interface
(`docstruct.parser.ParserBackend`, one file per backend under
`src/docstruct/parser/`) and benchmarked with
[`scripts/benchmark_parsers.py`](../scripts/benchmark_parsers.py) — the code
for every backend and the benchmark harness itself is kept in the repo, this
document only summarizes what running it produced.

Fixtures: two real Persian PDFs supplied for this project (`data/1391.pdf`,
295 pages, and `data/380889.pdf`, 277 pages — both real Central Bank of Iran
regulatory circular collections), plus two short excerpts of them
(`tests/fixtures/circular_1391_excerpt.pdf`,
`tests/fixtures/governance_380889_excerpt.pdf`) carved out of clean,
well-extracting page ranges for fast, tractable, hand-checkable testing —
see `tests/fixtures/README.md` for exactly which pages and why.

Reproduce with:

```bash
uv sync --extra benchmark
uv run python scripts/benchmark_parsers.py
```

Raw output: [`parser_benchmark_results.json`](parser_benchmark_results.json).

## Results

| Backend | RTL order correct | Tables found (1391 / 380889, full docs) | Heading-candidates (bold/title hint) | Speed (full doc, ~280 pages) |
|---|---|---|---|---|
| **PyMuPDF** | ✅ all 4 fixtures | 14 / 15 | 5400 / 4063 (raw; noisy — see below) | ~13–32s |
| pdfplumber | ❌ all 4 fixtures | 16 / 72 | 677 / 548 | ~20s |
| unstructured (`fast`) | ❌ both fixtures tested | 0 / — (not attempted on full docs) | 289 / 985 | ~7–8s (excerpts only) |
| docling | *(not measured — see below)* | *(not measured)* | *(not measured)* | **timed out** |
| anydoc | ✅ all 4 fixtures (macro-level — see below) | 46 / 146 (heavily inflated — see below) | 3272 / 580 (heavily inflated — see below) | **~1–2s** (fastest by far) |

### RTL / reading-order correctness

This was the deciding criterion, and the difference was stark and easy to
verify by hand. PyMuPDF's `get_text("dict")` applies bidi reordering when it
assembles line text from spans, so Persian comes out in correct logical
order:

```
PyMuPDF:      بخشنامه‌های مدیریت کل مقررات، مجوزهای بانکی و مبارزه با پولشویی سال ۱۳۹۱
pdfplumber:   ۱۳۹۱ سال پولشویی با مبارزه و بانکی مجوزهای مقررات، کل مدیریت بخشنامه‌های
unstructured: (same reversed order as pdfplumber — it uses pdfminer.six under
               the `fast` strategy, with no bidi post-processing)
```

pdfplumber and `unstructured`'s `fast` strategy both report words/glyphs in
raw content-stream order, which for RTL text is the *visual* order, i.e. the
reverse of how the text should be read. Neither applies bidi correction.
`unstructured`'s `hi_res`/`ocr_only` strategies render pages to images and
OCR them, which sidesteps this specific defect — but they require system
Tesseract/poppler binaries not installed in this environment, and OCR is
explicitly out of scope for this project anyway, so they weren't evaluated.

anydoc also passes the phrase-level check, on all four fixtures — its PDF
path applies its own reordering, unlike pdfplumber/unstructured. It isn't
flawless at a finer grain, though: spot-checking its Markdown output against
the source, some sentences have individual clauses transposed relative to
each other (correct words, occasionally wrong clause order within a
paragraph) — plausibly from how it merges multi-line justified text. See the
dedicated anydoc section below for the more significant issues it has.

The automated check in the benchmark script (`REFERENCE_PHRASE` search) is a
simple spot-check, not exhaustive — see `normalize_persian()` in
`scripts/benchmark_parsers.py` for two real normalization gotchas it has to
account for that are worth knowing about when working with Persian PDF text
generally, regardless of backend:

1. **Presentation-form glyphs.** Raw extracted text often uses Arabic
   Presentation Forms-B codepoints (e.g. `U+FEE3`) rather than standard
   Arabic-block letters (`U+0645`) — cosmetically identical, but `==`/`in`
   comparisons against normal Persian text silently fail until you
   `unicodedata.normalize("NFKC", text)`.
2. **Arabic vs. Persian letterforms.** Even after NFKC, extracted text
   commonly uses the Arabic Yeh/Kaf (`ي` `ك`) instead of the Persian-specific
   Farsi Yeh/Keheh (`ی` `ک`) that a Persian keyboard/typist would produce —
   a one-line `str.translate()` fix, but silent data-quality trap if unknown.

### anydoc: fast, but structurally noisy

[`anydoc`](https://github.com/firecrawl/anydoc) (PyPI: `firecrawl-anydoc`) is
a Rust-based multi-format-to-Markdown converter with Python bindings. It's
the one backend here that doesn't have a native structured API for PDF at
all: calling `anydoc.to_document()` on a PDF raises
`UnsupportedError: PDF converts directly to Markdown; use to_markdown or
to_markdown_bytes`. So `AnyDocBackend` ([`src/docstruct/parser/anydoc_backend.py`](../src/docstruct/parser/anydoc_backend.py))
gets a Markdown string back and parses *that* into our `Block` schema
(`#`-heading lines → heading-hinted blocks, `|...|` runs → table blocks,
everything else → paragraph text) — there's no page number or font-size
metadata available for this backend, unlike the other four.

It's dramatically the fastest backend measured — **1–2 seconds on the full
~280-page documents**, roughly 10–20x faster than PyMuPDF/pdfplumber and
without needing a subprocess timeout at all. But its Markdown output is
structurally noisy on these real documents in a way that shows up directly
in the inflated table/heading counts above:

- **Over-eager heading detection.** On the justified, multi-line body
  paragraphs typical of these documents, anydoc frequently emits *every
  wrapped line of a single running paragraph* as its own `###` heading,
  rather than recognizing it as one paragraph. A single sentence in the
  source PDF came out as eight separate H3 "headings" in our spot check —
  this is most of where the 3272/580 heading-candidate counts come from,
  and it's considerably worse than PyMuPDF's already-noisy bold-hint count.
- **False-positive tables.** Its Markdown table syntax (`|cell|cell|`)
  gets triggered on content that isn't tabular — e.g. a single unrelated
  text fragment wrapped as a malformed one-row table in our spot check.
  This inflates its table counts (46/146) well past even pdfplumber's
  already-inflated ones, without a corresponding increase in genuine tables
  found.
- **Spurious bold spans and stray `<u>...</u>` fragments** wrap large,
  arbitrary runs of body text and isolated repeated header words
  respectively — cosmetic noise for a Markdown viewer, but it means
  `AnyDocBackend` has to regex-strip `**`/`<u>`/`</u>` out of extracted text
  before it's usable (see `_clean_text` in the backend file).

None of this is a dealbreaker for its intended use case — quickly getting
*a* readable Markdown rendition of many formats — but for this project's
purpose (feeding reliable heading/paragraph/table structure into the
agent) its structural signal is currently less trustworthy than PyMuPDF's,
despite winning decisively on raw speed and getting phrase-level RTL order
right.

### Table detection

PyMuPDF's `Page.find_tables()` gives structured cells directly with no extra
heuristics needed on our part. pdfplumber's `find_tables()`/`extract_tables()`
also work, but on `380889.pdf` (a dense, multi-column legal document) it
reported **72** tables versus PyMuPDF's 15 — a spot check of a few suggests
pdfplumber's line/rect-based heuristic is over-triggering on bordered
boxes and multi-column body text that aren't really tables; a full
false-positive audit was out of scope here, but this is a real,
reproducible difference worth knowing about if you pick pdfplumber.
`unstructured`'s `fast` strategy detected no tables on either excerpt (table
structure inference is tied to its `hi_res` layout model, which wasn't
usable here — see above). anydoc's counts (46/146) are the least trustworthy
of all — see the anydoc section above.

### Heading detection

All five backends' output is normalized into our shared `Block` schema with
a `style.bold` hint (see `src/docstruct/parser/types.py`), computed from
font flags (PyMuPDF/pdfplumber), `category == "Title"` (unstructured),
`TITLE`/`SECTION_HEADER` labels (docling), or a Markdown `#`-line (anydoc —
the only backend inferring this from its own heuristic rather than font
metadata, since none is available for its PDF path). Raw counts aren't directly
comparable across backends because they segment text into very different
numbers of blocks (PyMuPDF alone produced 6147 blocks for `1391.pdf` versus
pdfplumber's 1259, from finer per-run/per-line segmentation), and on closer
inspection PyMuPDF's bold hint fires on a implausibly high fraction of those
blocks (~88%) — much of the document apparently uses a bold body font, or
bold is applied to short fragments/running headers. **This is precisely why
structure inference is delegated to the LLM agent (`docstruct.agent`) rather
than trusting a font-weight heuristic directly**: these hints are a useful
*signal* fed into the agent's prompt, not a reliable standalone heading
detector at real-document scale.

### Speed

anydoc is in a different class entirely — **1–2 seconds** for a ~280-page
document, native Rust doing simple text extraction with no per-page Python
overhead. PyMuPDF and pdfplumber are both still fast pure-extraction
libraries (13–32s for the same documents); `unstructured`'s `fast` strategy
is slower per-page but still practical. **docling never completed a run** — its `DocumentConverter`
downloaded and loaded its layout model successfully in ~1s, but converting
even a *single page* of `circular_1391_excerpt.pdf` did not finish within a
240-second timeout on this CPU-only Windows sandbox (no GPU); the benchmark
harness forcibly killed the subprocess (see
`scripts/benchmark_parsers.py::run_one`, which runs every backend in an
isolated, kill-able `multiprocessing.Process` specifically because of this).
Docling's transformer-based layout + TableFormer pipeline is presumably far
more capable on Latin-script scientific PDFs on GPU hardware than this
environment could exercise; we can't respectably claim its Persian layout
quality here since it never actually ran on Persian content.

## Pick: PyMuPDF

PyMuPDF is the default backend (`docstruct.parser.DEFAULT_BACKEND`):

- Correct Persian reading order out of the box, with no extra bidi handling
  on our part — and, unlike anydoc, correct at both the phrase and clause
  level, not just the phrase level.
- Native structured table extraction (`find_tables()`), no OCR or heavy
  model dependency, and no evidence of the severe false-positive rate
  anydoc's Markdown-table detection showed on these documents.
- Fast and it actually finishes — unlike docling, and without anydoc's
  heading/table noise problem.
- Lightweight dependency (no torch/transformers), which keeps `docstruct`'s
  default install small — the other four backends are optional extras
  (`docstruct[pdfplumber]`, `docstruct[unstructured]`, `docstruct[docling]`,
  `docstruct[anydoc]`, or `docstruct[benchmark]` for all of them) so this
  benchmark stays reproducible without forcing every user to install them.

**anydoc was the closest competitor** — it matched PyMuPDF on phrase-level
RTL correctness and is 10–20x faster — but its heading/table detection was
the noisiest of all five backends on these real documents (see the anydoc
section above), and structural signal quality matters more here than raw
speed: `docstruct.agent` already has to compensate for noisy hints from
*every* backend (see "Heading detection" above), and starting from
PyMuPDF's comparatively cleaner signal gives it less to correct. If anydoc's
Markdown-generation heuristics improve on documents like these, it's worth
re-benchmarking — the speed difference is large enough to matter for bulk
processing.

The parser abstraction (`docstruct.parser.get_backend`) keeps this
swappable — `parse(path, backend="anydoc")` etc. still works — but
`pymupdf` is what `docstruct.parser.parse()` uses by default and what the
agent pipeline is tuned against.
