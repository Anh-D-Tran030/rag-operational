"""Integration tests for src/generation/query_handler.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.config import Settings
from src.generation.query_handler import handle_query
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache


def _make_chunk(content: str = "The minimum staffing is 5.", doc_type: str = "compliance") -> dict:
    return {
        "content": content,
        "doc_type": doc_type,
        "venue": "RSL",
        "date": "2026-01-01",
        "shift_id": None,
        "page_num": 1,
        "rerank_score": 0.85,
        "content_hash": "abc123",
        "score": 0.85,
    }


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="",
        openai_api_key="",
        ollama_url="http://localhost:11434",
        db_url="sqlite+aiosqlite:///./operational.db",
        semantic_cache_url="",
    )


@pytest.fixture
def tracer():
    return RAGTracer(public_key="", secret_key="", host="")


@pytest.fixture
def cache(settings):
    return SemanticCache("", settings.embedding_model)


@pytest.mark.asyncio
async def test_vector_query_happy_path(settings, tracer, cache):
    mock_client = AsyncMock()
    mock_chunk = _make_chunk()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([mock_chunk], {}),
        ),
        patch(
            "src.generation.query_handler.generate_local",
            return_value="The answer is 5 staff.",
        ),
    ):
        result = await handle_query(
            "What is minimum staffing?", mock_client, settings, tracer, cache
        )

    assert "answer" in result
    assert "trace_id" in result
    assert result["tool"] == "vector_search"
    assert len(result["citations"]) >= 1
    assert result["citations"][0]["doc_type"] == "compliance"


@pytest.mark.asyncio
async def test_tabular_query_path(settings, tracer, cache):
    mock_client = AsyncMock()
    sql_result = {
        "rows": [{"venue": "RSL", "staff_count": 10}],
        "sql_used": "SELECT venue, staff_count FROM v_shift_summary",
        "view_names": ["v_shift_summary"],
        "trace_id": "t1",
    }

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "text_to_sql", "complexity": "simple"},
        ),
        patch("src.generation.query_handler.embed_query", return_value=[0.1] * 384),
        patch(
            "src.generation.query_handler.find_relevant_views",
            return_value=[{
                "view_name": "v_shift_summary",
                "columns": ["venue", "staff_count"],
                "description": "x",
            }],
        ),
        patch("src.generation.query_handler.text_to_sql_tool", return_value=sql_result),
        patch("src.generation.query_handler.generate_local", return_value="10 staff worked."),
    ):
        result = await handle_query("How many staff?", mock_client, settings, tracer, cache)

    assert result["tool"] == "text_to_sql"
    assert any("sql_used" in c for c in result["citations"])


@pytest.mark.asyncio
async def test_cache_hit_skips_retrieval(settings, tracer):
    cache = SemanticCache("memory://test", settings.embedding_model)
    cached_resp = {
        "answer": "cached answer",
        "citations": [],
        "trace_id": "cached-trace",
        "route": "simple",
        "tool": "vector_search",
        "latency_ms": 10.0,
    }
    mock_client = AsyncMock()

    with patch.object(cache, "get", return_value=cached_resp):
        result = await handle_query("same query again", mock_client, settings, tracer, cache)

    assert result["answer"] == "cached answer"
    assert result["trace_id"] == "cached-trace"


@pytest.mark.asyncio
async def test_empty_results_returns_insufficient_context(settings, tracer, cache):
    mock_client = AsyncMock()

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch("src.generation.query_handler.vector_search_pipeline", return_value=([], {})),
    ):
        result = await handle_query("unknowable question", mock_client, settings, tracer, cache)

    assert result["answer"] == "insufficient context"


@pytest.mark.asyncio
async def test_pii_input_span_logged(settings, cache):
    """Input-path redaction must emit a pii_redaction Langfuse span."""
    mock_client = AsyncMock()
    mock_chunk = _make_chunk()

    mock_tracer = MagicMock()
    mock_tracer.start_trace.return_value = ("trace-pii", MagicMock())

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([mock_chunk], {}),
        ),
        patch("src.generation.query_handler.generate_local", return_value="answer text"),
        patch("src.generation.query_handler.check_faithfulness", return_value=True),
    ):
        await handle_query("hello world", mock_client, settings, mock_tracer, cache)

    span_calls = [c for c in mock_tracer.log_span.call_args_list if c.kwargs.get("name") == "pii_redaction"]
    input_spans = [c for c in span_calls if c.kwargs.get("metadata", {}).get("stage") == "input"]
    assert len(input_spans) >= 1, "expected a pii_redaction span with stage=input"


@pytest.mark.asyncio
async def test_pii_output_span_logged(settings, cache):
    """Output-path redaction must emit a pii_redaction Langfuse span."""
    mock_client = AsyncMock()
    mock_chunk = _make_chunk()

    mock_tracer = MagicMock()
    mock_tracer.start_trace.return_value = ("trace-pii", MagicMock())

    with (
        patch(
            "src.generation.query_handler.classify_intent",
            return_value={"tool": "vector_search", "complexity": "simple"},
        ),
        patch(
            "src.generation.query_handler.vector_search_pipeline",
            return_value=([mock_chunk], {}),
        ),
        patch("src.generation.query_handler.generate_local", return_value="answer text"),
        patch("src.generation.query_handler.check_faithfulness", return_value=True),
    ):
        await handle_query("hello world", mock_client, settings, mock_tracer, cache)

    span_calls = [c for c in mock_tracer.log_span.call_args_list if c.kwargs.get("name") == "pii_redaction"]
    output_spans = [c for c in span_calls if c.kwargs.get("metadata", {}).get("stage") == "output"]
    assert len(output_spans) >= 1, "expected a pii_redaction span with stage=output"
