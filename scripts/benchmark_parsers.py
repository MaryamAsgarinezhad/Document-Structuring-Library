"""Benchmark the five PDF parsing backends on real Persian PDFs.

Usage:
    uv run python scripts/benchmark_parsers.py
    uv run python scripts/benchmark_parsers.py --timeout 60 --skip docling

Measures, per (backend, fixture) pair:
  - speed (wall-clock seconds)
  - RTL/reading-order correctness (a known Persian phrase must appear intact
    and in logical order in the extracted text; see REFERENCE_PHRASE below)
  - table detection (number of table blocks found)
  - heading-candidate detection (number of text blocks the backend flagged
    as bold/title-like — the common "heading hint" signal in our Block
    schema, see docstruct.parser.types)

Each backend runs in its own subprocess with a timeout, so a hang or crash
in one backend (docling in particular needed this during development — see
docs/parser_comparison.md) can't take down the whole benchmark run.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# A phrase that recurs, verbatim and in this word order, on nearly every page
# of both real fixtures ("General Directorate of Regulations, Banking
# Licenses and AML" - part of the banking authority's letterhead). A backend
# that doesn't apply RTL/bidi reordering emits Persian words (and often
# characters) in reverse, so this phrase will NOT appear intact in its output.
REFERENCE_PHRASE = "مدیریت کل مقررات، مجوزهای بانکی"

# PDF text layers commonly store Arabic-script letters using either
# presentation-form glyph codepoints (NFKC folds those back to base letters)
# and/or the plain-Arabic Yeh/Kaf (ي ك) instead of the Persian-specific
# Farsi Yeh/Keheh (ی ک) — normalize both sides of the RTL-order comparison
# the same way so this doesn't get misread as an ordering bug.
_ARABIC_TO_PERSIAN_LETTERS = str.maketrans({"ي": "ی", "ك": "ک"})


def normalize_persian(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ARABIC_TO_PERSIAN_LETTERS)
    # Collapse all whitespace (including the mid-phrase line wraps line-based
    # extraction naturally produces) so the comparison is about word/character
    # order, not incidental line breaks.
    return " ".join(text.split())


ALL_BACKENDS = ["pymupdf", "pdfplumber", "unstructured", "docling", "anydoc"]

FIXTURES = [
    ("circular_1391_excerpt", ROOT / "tests" / "fixtures" / "circular_1391_excerpt.pdf"),
    ("governance_380889_excerpt", ROOT / "tests" / "fixtures" / "governance_380889_excerpt.pdf"),
]

# The two heavy, model-based backends are only run against the small excerpts
# above (a single page took docling several minutes on this CPU-only sandbox
# during development, and it never finished). pymupdf/pdfplumber/anydoc are
# fast pure-extraction backends and are also run against the full real
# documents to get realistic full-scale numbers.
FULL_DOCUMENT_BACKENDS = ["pymupdf", "pdfplumber", "anydoc"]
FULL_DOCUMENT_FIXTURES = [
    ("1391_full", ROOT / "data" / "1391.pdf"),
    ("380889_full", ROOT / "data" / "380889.pdf"),
]


def _run_backend(backend_name: str, pdf_path: str, result_queue: "multiprocessing.Queue") -> None:
    """Runs in its own subprocess so a hang (docling did, on this CPU-only
    sandbox) or crash can be killed/isolated without taking down the run."""
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "src"))
    try:
        from docstruct.parser import get_backend
        from docstruct.parser.types import BlockKind

        backend = get_backend(backend_name)
        start = _time.perf_counter()
        parsed = backend.parse(pdf_path)
        elapsed = _time.perf_counter() - start

        full_text = "\n".join(b.text or "" for b in parsed.blocks if b.kind == BlockKind.TEXT)
        n_tables = sum(1 for b in parsed.blocks if b.kind == BlockKind.TABLE)
        n_heading_candidates = sum(
            1 for b in parsed.blocks if b.kind == BlockKind.TEXT and b.style and b.style.bold
        )

        result_queue.put(
            {
                "elapsed_seconds": round(elapsed, 2),
                "n_blocks": len(parsed.blocks),
                "n_tables": n_tables,
                "n_heading_candidates": n_heading_candidates,
                "rtl_order_correct": normalize_persian(REFERENCE_PHRASE) in normalize_persian(full_text),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report the failure to the parent instead of crashing silently
        result_queue.put({"error": f"{type(exc).__name__}: {exc}"})


def run_one(backend_name: str, fixture_path: Path, timeout: float) -> dict:
    print(f"  {backend_name} on {fixture_path.name} ...", end=" ", flush=True)
    result_queue: "multiprocessing.Queue" = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_backend, args=(backend_name, str(fixture_path), result_queue)
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
        print(f"TIMED OUT (> {timeout}s, process killed)")
        return {"error": f"timed out after {timeout}s"}

    if not result_queue.empty():
        result = result_queue.get()
        if "error" in result:
            print(f"FAILED: {result['error']}")
        else:
            print(f"ok ({result['elapsed_seconds']}s)")
        return result

    print(f"FAILED: process exited with code {process.exitcode} and no result")
    return {"error": f"process exited with code {process.exitcode} and no result"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-run timeout in seconds")
    parser.add_argument("--skip", action="append", default=[], help="Backend(s) to skip, e.g. --skip docling")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "parser_benchmark_results.json", help="Where to save raw JSON"
    )
    args = parser.parse_args()

    backends = [b for b in ALL_BACKENDS if b not in args.skip]
    results: dict[str, dict[str, dict]] = {}

    for backend_name in backends:
        print(f"\n=== {backend_name} ===")
        results[backend_name] = {}
        for fixture_name, fixture_path in FIXTURES:
            results[backend_name][fixture_name] = run_one(backend_name, fixture_path, args.timeout)

        if backend_name in FULL_DOCUMENT_BACKENDS:
            for fixture_name, fixture_path in FULL_DOCUMENT_FIXTURES:
                results[backend_name][fixture_name] = run_one(backend_name, fixture_path, args.timeout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved raw results to {args.out}")


if __name__ == "__main__":
    main()
