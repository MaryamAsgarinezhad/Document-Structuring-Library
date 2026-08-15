"""Optional on-disk tracing for ``structure_document``: write each chunk's
exact prompt and the model's raw response (or error) to numbered files, for
troubleshooting a chunk that produced unexpected or missing output.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .refiner import ChunkTrace


def file_trace_writer(trace_dir: str | Path) -> Callable[[ChunkTrace], None]:
    """Returns an ``on_chunk`` callback for ``structure_document`` that
    writes, per chunk, ``chunk_{i:03d}.prompt.txt`` (exactly what was sent
    to the model) plus either ``chunk_{i:03d}.result.json`` (the parsed
    ``ChunkResult`` on success) or ``chunk_{i:03d}.error.txt`` (the
    exception, on failure).

    Usage::

        structure_document(parsed, on_chunk=file_trace_writer("trace/"))
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    def on_chunk(trace: ChunkTrace) -> None:
        base = f"chunk_{trace.index:03d}"
        (trace_dir / f"{base}.prompt.txt").write_text(trace.prompt, encoding="utf-8")
        if trace.error is not None:
            (trace_dir / f"{base}.error.txt").write_text(
                f"{type(trace.error).__name__}: {trace.error}", encoding="utf-8"
            )
        else:
            assert trace.result is not None
            (trace_dir / f"{base}.result.json").write_text(
                trace.result.model_dump_json(indent=2), encoding="utf-8"
            )

    return on_chunk
