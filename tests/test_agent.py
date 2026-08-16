"""Agent wiring tests. These never call a real model: pydantic-ai's
FunctionModel lets us script deterministic per-chunk responses so we can
verify structure_document() correctly drives DocumentBuilder chunk by chunk.
"""

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from docstruct.agent.ops import ChunkResult, NewHeading, NewParagraph
from docstruct.agent.refiner import ChunkTrace, structure_document
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

    # max_attempts_per_chunk=1: this test's scripted text is much shorter than the
    # filler blocks, which would otherwise score low on the coverage heuristic and
    # trigger an unwanted retry — see the dedicated retry tests below for that.
    doc = structure_document(parsed, model=model, max_chars_per_chunk=500, max_attempts_per_chunk=1)

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

    structure_document(parsed, model=model, max_chars_per_chunk=500, max_attempts_per_chunk=1)

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


def test_on_chunk_reports_prompt_and_result_for_every_chunk():
    scripted_responses = [
        ChunkResult(operations=[NewHeading(level=1, text="فصل اول")]),
        ChunkResult(operations=[NewParagraph(text="ادامه فصل اول")]),
    ]
    call_count = 0

    def respond(messages, info):
        nonlocal call_count
        chunk_result = scripted_responses[call_count]
        call_count += 1
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, "متن " * 200), _text_block(1, "متن دیگر " * 200)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    traces: list[ChunkTrace] = []
    structure_document(
        parsed, model=model, max_chars_per_chunk=500, on_chunk=traces.append, max_attempts_per_chunk=1
    )

    assert [t.index for t in traces] == [1, 2]
    assert all(t.error is None for t in traces)
    assert traces[0].result == scripted_responses[0]
    assert traces[1].result == scripted_responses[1]
    assert "فصل اول" in traces[1].prompt  # breadcrumb from chunk 1 shows up in chunk 2's prompt


def test_skip_failed_chunks_continues_and_reports_error():
    scripted_responses = [None, ChunkResult(operations=[NewParagraph(text="پاراگراف دوم")])]
    call_count = 0

    def respond(messages, info):
        nonlocal call_count
        chunk_result = scripted_responses[call_count]
        call_count += 1
        if chunk_result is None:
            raise RuntimeError("simulated model failure")
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, "متن " * 200), _text_block(1, "متن دیگر " * 200)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    traces: list[ChunkTrace] = []
    doc = structure_document(
        parsed,
        model=model,
        max_chars_per_chunk=500,
        on_chunk=traces.append,
        skip_failed_chunks=True,
        max_attempts_per_chunk=1,
    )

    assert len(traces) == 2
    assert traces[0].error is not None and traces[0].result is None
    assert traces[1].error is None and traces[1].result is not None
    # the failed chunk contributed nothing, but the run continued past it
    assert doc.preamble_paragraphs == ["پاراگراف دوم"]


def test_without_skip_failed_chunks_a_failure_propagates():
    def respond(messages, info):
        raise RuntimeError("simulated model failure")

    model = FunctionModel(respond)
    blocks = [_text_block(0, "متن " * 200)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    try:
        structure_document(parsed, model=model, max_chars_per_chunk=500, max_attempts_per_chunk=1)
        raise AssertionError("expected the simulated failure to propagate")
    except RuntimeError as exc:
        assert "simulated model failure" in str(exc)


def test_low_coverage_attempt_is_retried_and_the_better_attempt_is_kept():
    full_text = "این یک پاراگراف نسبتاً طولانی است که باید به طور کامل در خروجی پوشش داده شود"
    incomplete = ChunkResult(operations=[NewParagraph(text="این یک")])
    complete = ChunkResult(operations=[NewParagraph(text=full_text)])
    scripted_responses = [incomplete, complete]
    call_count = 0

    def respond(messages, info):
        nonlocal call_count
        chunk_result = scripted_responses[call_count]
        call_count += 1
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, full_text)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    doc = structure_document(parsed, model=model, max_attempts_per_chunk=2, completeness_threshold=0.7)

    assert call_count == 2  # the low-coverage first attempt triggered a retry
    assert doc.preamble_paragraphs == [full_text]  # the more complete attempt was kept


def test_best_of_all_incomplete_attempts_is_kept_when_none_meet_the_threshold():
    full_text = "این یک پاراگراف نسبتاً طولانی است که باید به طور کامل در خروجی پوشش داده شود"
    worse = ChunkResult(operations=[NewParagraph(text="این")])
    better = ChunkResult(operations=[NewParagraph(text="این یک پاراگراف نسبتاً")])
    scripted_responses = [worse, better]
    call_count = 0

    def respond(messages, info):
        nonlocal call_count
        chunk_result = scripted_responses[call_count]
        call_count += 1
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, full_text)]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    # neither attempt reaches completeness_threshold, so both attempts are used up,
    # and the better (higher-coverage) of the two -- not simply the last one -- wins
    doc = structure_document(parsed, model=model, max_attempts_per_chunk=2, completeness_threshold=0.99)

    assert call_count == 2
    assert doc.preamble_paragraphs == ["این یک پاراگراف نسبتاً"]


def test_exception_on_first_attempt_is_retried_and_succeeds():
    scripted_responses = [None, ChunkResult(operations=[NewParagraph(text="متن")])]
    call_count = 0

    def respond(messages, info):
        nonlocal call_count
        chunk_result = scripted_responses[call_count]
        call_count += 1
        if chunk_result is None:
            raise RuntimeError("transient failure")
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=chunk_result.model_dump())])

    model = FunctionModel(respond)
    blocks = [_text_block(0, "متن")]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    doc = structure_document(parsed, model=model, max_attempts_per_chunk=2)

    assert call_count == 2
    assert doc.preamble_paragraphs == ["متن"]


def test_all_attempts_exhausted_then_skip_failed_chunks_continues():
    def respond(messages, info):
        raise RuntimeError("persistent failure")

    model = FunctionModel(respond)
    blocks = [_text_block(0, "بلوک اول"), _text_block(1, "بلوک دوم")]
    parsed = ParsedDocument(source_path="x.pdf", backend="pymupdf", blocks=blocks)

    traces: list[ChunkTrace] = []
    doc = structure_document(
        parsed,
        model=model,
        max_chars_per_chunk=len("بلوک اول"),
        on_chunk=traces.append,
        skip_failed_chunks=True,
        max_attempts_per_chunk=2,
    )

    assert len(traces) == 2
    assert all(t.error is not None and t.result is None for t in traces)
    assert doc.sections == []
    assert doc.preamble_paragraphs == []
