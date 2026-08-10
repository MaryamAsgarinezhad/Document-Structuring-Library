# Test fixtures

Both PDFs here are short excerpts of the two real Persian documents supplied
for this project (`data/1391.pdf` and `data/380889.pdf`, both real Central
Bank of Iran regulatory circular collections — 295 and 277 pages
respectively). The full originals are used directly by
`scripts/benchmark_parsers.py`; these excerpts exist because hand-verifying
expected JSON/Markdown output against a 295-page document isn't tractable,
while an excerpt of a real document still is.

## `circular_1391_excerpt.pdf`

Pages 10–16 (1-indexed) of `data/1391.pdf`: a "Part One" section heading,
followed by one full circular's letter-style body text, followed by the
start of a second circular (`دستورالعمل نحوه محاسبه ذخیره مطالبات مؤسسات
اعتباری`) that includes numbered articles (`ماده ۱`, `ماده ۲`) and a small
percentage-rate table. Chosen because it extracts cleanly with PyMuPDF and
exercises heading + paragraph + table structure without an explicit table
of contents on these particular pages.

## `governance_380889_excerpt.pdf`

Pages 12–20 (1-indexed) of `data/380889.pdf`: a complete, self-contained
regulatory decree (`دستورالعمل تأیید صلاحیت و عزل یا هرگونه تغییر مدیران
ارشد...`) reproduced in full within the larger circular collection — title
page through a closing "لازم‌الاجرا است" line. It has real multi-level
structure with no explicit TOC of its own: چصل (chapter, level 1) →
lettered subsection ب/الف (level 2) → ماده (numbered article). Chosen as
the richer fixture for exercising the agent's heading-level inference.

## A note on these documents' text encoding

While inspecting `data/380889.pdf` page by page (see
`docs/parser_comparison.md`), some pages showed badly garbled text — e.g.
quoted law/decree names rendered as symbol soup — mixed with otherwise
clean Persian on the same page. This traces to a font embedded in the PDF
with a broken/incomplete ToUnicode CMap for certain glyphs (used for
quoted terms), not to a bug in any extraction library — every text-layer
extractor decodes the same broken mapping the same way. It's a real,
useful thing to know about when sourcing "real" Persian PDFs: always spot
check a few pages before trusting extracted text, since a document can be
clean on one page and corrupted on the next. Both excerpts here were
chosen from page ranges that were manually verified to extract cleanly.
