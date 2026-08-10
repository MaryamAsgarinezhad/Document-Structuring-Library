"""End-to-end pipeline test against the real LiteLLM gateway: PDF -> parse ->
structure_document() -> Markdown, on the two real-Persian-PDF fixtures.

Opt-in and skipped unless DOCSTRUCT_API_KEY is set (via .env or the
environment) — no live model calls happen in the default test run. LLM
output is not byte-for-byte deterministic (and, as documented in the
README's Known limitations, a single chunk's response can occasionally omit
some of its input content), so this checks the pipeline produces a
well-formed, non-trivial document rather than diffing against a frozen
"golden" file.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from docstruct.agent import structure_document
from docstruct.converter import structured_document_to_markdown
from docstruct.parser import parse
from docstruct.schema import Section, StructuredDocument

FIXTURES_DIR = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DOCSTRUCT_API_KEY"),
    reason="requires a live DOCSTRUCT_API_KEY / gateway (see README)",
)


def _walk_sections(sections: list[Section]) -> Iterator[Section]:
    for section in sections:
        yield section
        yield from _walk_sections(section.subsections)


@pytest.mark.parametrize("fixture_name", ["circular_1391_excerpt", "governance_380889_excerpt"])
def test_pdf_to_structured_document_to_markdown(fixture_name: str) -> None:
    pdf_path = FIXTURES_DIR / f"{fixture_name}.pdf"

    parsed = parse(pdf_path, backend="pymupdf")
    document = structure_document(parsed)

    # Re-validates against the schema defensively (pydantic-ai's output
    # validation already enforces this per chunk, but the assembled tree
    # is worth checking too).
    StructuredDocument.model_validate(document.model_dump())

    assert document.title
    assert document.sections, "expected at least one top-level section to have been inferred"

    all_sections = list(_walk_sections(document.sections))
    total_paragraphs = sum(len(s.paragraphs) for s in all_sections)
    assert total_paragraphs > 0, "expected at least some body paragraphs to have been captured"

    markdown = structured_document_to_markdown(document)
    assert markdown.startswith(f"# {document.title}")
    assert len(markdown) > 200
