"""Assemble a StructuredDocument (and its Markdown) from whatever
chunk_NNN.result.json trace files exist so far in a trace directory — for
inspecting a partial/in-progress run without waiting for it to finish.

Usage:
    uv run python scripts/assemble_partial.py trace/1391_v5 --out-dir data/v5_partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from docstruct.agent.builder import DocumentBuilder
from docstruct.agent.ops import ChunkResult
from docstruct.converter import structured_document_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--stem", default=None, help="Output filename stem (default: trace dir's name)")
    args = parser.parse_args()

    result_files = sorted(args.trace_dir.glob("chunk_*.result.json"))
    if not result_files:
        raise SystemExit(f"No chunk_*.result.json files found in {args.trace_dir}")

    builder = DocumentBuilder()
    for path in result_files:
        chunk_result = ChunkResult.model_validate_json(path.read_text(encoding="utf-8"))
        if chunk_result.title:
            builder.set_title(chunk_result.title)
        for op in chunk_result.operations:
            builder.apply(op)

    document = builder.build()

    stem = args.stem or args.trace_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{stem}.structured.json"
    out_md = args.out_dir / f"{stem}.md"
    out_json.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    out_md.write_text(structured_document_to_markdown(document), encoding="utf-8")

    print(f"Assembled {len(result_files)} chunk results -> {out_json} and {out_md}")


if __name__ == "__main__":
    main()
