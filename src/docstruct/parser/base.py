from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .types import ParsedDocument


class ParserBackend(Protocol):
    """Common interface every PDF parsing backend implements."""

    name: str

    def parse(self, pdf_path: str | Path) -> ParsedDocument: ...
