"""Run the full parse -> structure -> markdown pipeline on a real PDF, with
per-chunk tracing enabled so every prompt sent to the model (and its raw
response or error) can be inspected afterward for troubleshooting.

Usage:
    uv run python scripts/run_pipeline.py data/1391.pdf
    uv run python scripts/run_pipeline.py data/1391.pdf --trace-dir trace/1391 --out-dir data

Writes:
    <out-dir>/<pdf-stem>.structured.json   the StructuredDocument, as JSON
    <out-dir>/<pdf-stem>.md                the same document, converted to Markdown
    <trace-dir>/chunk_NNN.prompt.txt       exactly what was sent to the model for chunk NNN
    <trace-dir>/chunk_NNN.result.json      the model's parsed ChunkResult (on success)
    <trace-dir>/chunk_NNN.error.txt        the exception (on failure — that chunk is skipped,
                                            not fatal, see skip_failed_chunks in docstruct.agent)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from docstruct.agent import ChunkTrace, chunk_blocks, file_trace_writer, structure_document
from docstruct.converter import structured_document_to_markdown
from docstruct.parser import parse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "--trace-dir", type=Path, default=None, help="Where to write per-chunk trace files (default: trace/<pdf-stem>/)"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data", help="Where to write <stem>.structured.json / .md")
    parser.add_argument("--max-chars-per-chunk", type=int, default=8000)
    parser.add_argument("--overlap-blocks", type=int, default=2)
    parser.add_argument(
        "--max-attempts-per-chunk", type=int, default=2, help="Retry a chunk (fresh sample) if under-covered"
    )
    parser.add_argument("--completeness-threshold", type=float, default=0.7)
    args = parser.parse_args()

    out_stem = args.pdf_path.stem
    trace_dir = args.trace_dir or (ROOT / "trace" / out_stem)

    print(f"Parsing {args.pdf_path} ...", flush=True)
    t0 = time.perf_counter()
    parsed = parse(str(args.pdf_path))
    print(f"  parsed {len(parsed.blocks)} blocks in {time.perf_counter() - t0:.1f}s", flush=True)

    total_chunks = len(chunk_blocks(parsed, max_chars=args.max_chars_per_chunk, overlap_blocks=args.overlap_blocks))
    print(f"  {total_chunks} chunks (max_chars_per_chunk={args.max_chars_per_chunk}, overlap_blocks={args.overlap_blocks})")
    print(f"  tracing every chunk's prompt/result to {trace_dir}/", flush=True)

    write_trace = file_trace_writer(trace_dir)
    t_start = time.perf_counter()

    def on_chunk(trace: ChunkTrace) -> None:
        write_trace(trace)
        elapsed = time.perf_counter() - t_start
        if trace.error is not None:
            print(
                f"  chunk {trace.index}/{total_chunks} FAILED (total {elapsed:.0f}s): "
                f"{type(trace.error).__name__}: {trace.error}",
                flush=True,
            )
        else:
            n_ops = len(trace.result.operations) if trace.result else 0
            print(f"  chunk {trace.index}/{total_chunks} ok ({n_ops} ops, total {elapsed:.0f}s)", flush=True)

    document = structure_document(
        parsed,
        max_chars_per_chunk=args.max_chars_per_chunk,
        overlap_blocks=args.overlap_blocks,
        on_chunk=on_chunk,
        skip_failed_chunks=True,
        max_attempts_per_chunk=args.max_attempts_per_chunk,
        completeness_threshold=args.completeness_threshold,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{out_stem}.structured.json"
    out_md = args.out_dir / f"{out_stem}.md"
    out_json.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    out_md.write_text(structured_document_to_markdown(document), encoding="utf-8")

    print(f"Wrote {out_json} and {out_md}", flush=True)
    print(f"Trace files: {trace_dir}/chunk_NNN.prompt.txt / .result.json / .error.txt", flush=True)


if __name__ == "__main__":
    main()
