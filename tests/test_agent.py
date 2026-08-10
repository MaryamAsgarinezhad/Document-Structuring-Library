"""Agent wiring tests. These never call a real model: pydantic-ai's
FunctionModel lets us script deterministic per-chunk responses so we can
verify structure_document() correctly drives DocumentBuilder chunk by chunk.
"""

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from docstruct.agent.ops import ChunkResult, NewHeading, NewParagraph
from docstruct.agent.refiner import structure_document
from docstruct.parser.types import Block, BlockKind, ParsedDocument


def _text_block(order: int, text: str) -> Block:
    return Block(kind=BlockKind.TEXT, page=0, order=order, text=text)


def _last_user_prompt(messages) -> str:
    for part in reversed(messages[-1].parts):
        if isinstance(part, UserPromptPart):
            return str(part.content)
    return ""


def test_structure_document_merges_chunk_operations_in_order():
    scripted_responses = [
        ChunkResult(
            title="سند آزمایشی",
            operations=[NewHeading(level=1, text="فصل اول"), NewParagraph(text="پاراگراف اول")],
        ),
        ChunkResult(operations=[NewHeading(level=1, text="فصل دوم"), NewParagraph(text="پاراگراف دوم")]),
    ]
    seen_prompts: list[str] = []

    def respond(messages, info):
        chunk_result = scripted_responses[len(seen_prompts)]
        seen_prompts.append(_last_user_prompt(messages))
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, "متن " * 200), _text_block(1, "متن دیگر " * 200)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    doc = structure_document(parsed, model=model, max_chars_per_chunk=500)

    assert len(seen_prompts) == 2
    assert doc.title == "سند آزمایشی"
    assert [s.heading for s in doc.sections] == ["فصل اول", "فصل دوم"]
    assert doc.sections[0].paragraphs == ["پاراگراف اول"]
    assert doc.sections[1].paragraphs == ["پاراگراف دوم"]


def test_breadcrumb_of_prior_chunks_is_passed_to_later_calls():
    scripted_responses = [
        ChunkResult(operations=[NewHeading(level=1, text="فصل اول")]),
        ChunkResult(operations=[NewParagraph(text="ادامه فصل اول")]),
    ]
    seen_prompts: list[str] = []

    def respond(messages, info):
        chunk_result = scripted_responses[len(seen_prompts)]
        seen_prompts.append(_last_user_prompt(messages))
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, "متن " * 200), _text_block(1, "متن دیگر " * 200)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    structure_document(parsed, model=model, max_chars_per_chunk=500)

    assert "none open" in seen_prompts[0]
    assert "فصل اول" in seen_prompts[1]


def test_no_chunks_returns_empty_document():
    def respond(messages, info):
        raise AssertionError("model should not be called when there are no chunks")

    model = FunctionModel(respond)
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=[])

    doc = structure_document(parsed, model=model)

    assert doc.title is None
    assert doc.sections == []
