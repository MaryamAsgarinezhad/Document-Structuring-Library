"""PDF parsing: a common ``ParsedDocument`` output produced by one of several
swappable backends (PyMuPDF, pdfplumber, unstructured, docling, anydoc).

Only PyMuPDF is a required dependency; the others are optional extras
(``docstruct[pdfplumber]``, ``docstruct[unstructured]``, ``docstruct[docling]``,
``docstruct[anydoc]``, or ``docstruct[benchmark]`` for all of them) and are
imported lazily so a plain install of docstruct never requires them.
"""

from __future__ import annotations

from pathlib import Path

from .base import ParserBackend
from .types import Block, BlockKind, ParsedDocument, TextStyle

# Picked in docs/parser_comparison.md after benchmarking all four backends.
DEFAULT_BACKEND = "pymupdf"

_BACKEND_IMPORTERS = {
    "pymupdf": lambda: __import__(
        "docstruct.parser.pymupdf_backend", fromlist=["PyMuPDFBackend"]
    ).PyMuPDFBackend,
    "pdfplumber": lambda: __import__(
        "docstruct.parser.pdfplumber_backend", fromlist=["PDFPlumberBackend"]
    ).PDFPlumberBackend,
    "unstructured": lambda: __import__(
        "docstruct.parser.unstructured_backend", fromlist=["UnstructuredBackend"]
    ).UnstructuredBackend,
    "docling": lambda: __import__(
        "docstruct.parser.docling_backend", fromlist=["DoclingBackend"]
    ).DoclingBackend,
    "anydoc": lambda: __import__(
        "docstruct.parser.anydoc_backend", fromlist=["AnyDocBackend"]
    ).AnyDocBackend,
}


def get_backend(name: str) -> ParserBackend:
    try:
        importer = _BACKEND_IMPORTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown parser backend {name!r}. Available: {sorted(_BACKEND_IMPORTERS)}"
        ) from exc
    return importer()()


def parse(pdf_path: str | Path, backend: str = DEFAULT_BACKEND) -> ParsedDocument:
    """Parse ``pdf_path`` into a :class:`ParsedDocument` using the named backend."""
    return get_backend(backend).parse(pdf_path)


__all__ = [
    "Block",
    "BlockKind",
    "ParsedDocument",
    "ParserBackend",
    "TextStyle",
    "DEFAULT_BACKEND",
    "get_backend",
    "parse",
]
