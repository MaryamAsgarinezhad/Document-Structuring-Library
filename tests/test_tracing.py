from docstruct.agent.chunking import Chunk
from docstruct.agent.ops import ChunkResult, NewParagraph
from docstruct.agent.refiner import ChunkTrace
from docstruct.agent.tracing import file_trace_writer


def test_file_trace_writer_writes_prompt_and_result_on_success(tmp_path):
    on_chunk = file_trace_writer(tmp_path)
    trace = ChunkTrace(
        index=1,
        chunk=Chunk(blocks=[]),
        prompt="Current chunk:\nTEXT: متن",
        result=ChunkResult(operations=[NewParagraph(text="متن")]),
        error=None,
    )

    on_chunk(trace)

    assert (tmp_path / "chunk_001.prompt.txt").read_text(encoding="utf-8") == trace.prompt
    result_json = (tmp_path / "chunk_001.result.json").read_text(encoding="utf-8")
    assert ChunkResult.model_validate_json(result_json) == trace.result
    assert not (tmp_path / "chunk_001.error.txt").exists()


def test_file_trace_writer_writes_error_on_failure(tmp_path):
    on_chunk = file_trace_writer(tmp_path)
    trace = ChunkTrace(
        index=2,
        chunk=Chunk(blocks=[]),
        prompt="Current chunk:\nTEXT: متن",
        result=None,
        error=RuntimeError("boom"),
    )

    on_chunk(trace)

    assert (tmp_path / "chunk_002.prompt.txt").exists()
    assert "RuntimeError: boom" in (tmp_path / "chunk_002.error.txt").read_text(encoding="utf-8")
    assert not (tmp_path / "chunk_002.result.json").exists()


def test_file_trace_writer_creates_trace_dir(tmp_path):
    trace_dir = tmp_path / "nested" / "trace"
    file_trace_writer(trace_dir)

    assert trace_dir.is_dir()
