"""Unit tests for src/retrieval/pipeline.py — metric instrumentation."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.config import Settings
from src.observability.tracer import RAGTracer
from src.retrieval.pipeline import vector_search_pipeline


@pytest.fixture
def settings():
    return Settings(semantic_cache_url="", db_url="sqlite+aiosqlite:///./operational.db")


@pytest.fixture
def noop_tracer():
    return RAGTracer(public_key="", secret_key="", host="")


@pytest.mark.asyncio
async def test_rag_agent_iterations_observed_simple(settings, noop_tracer):
    """rag_agent_iterations is observed once for a simple (non-complex) query."""
    mock_client = AsyncMock()

    with (
        patch("src.retrieval.pipeline._expand_and_search", new=AsyncMock(return_value=[])),
        patch("src.retrieval.pipeline.rerank", new=AsyncMock(return_value=[])),
        patch("src.retrieval.pipeline.rag_agent_iterations") as mock_metric,
        patch("src.retrieval.pipeline.rag_chunks_per_query"),
        patch("src.retrieval.pipeline.rag_retrieval_confidence"),
    ):
        await vector_search_pipeline(
            query="test query",
            client=mock_client,
            settings=settings,
            tracer=noop_tracer,
            trace_id="t1",
            complexity="simple",
        )

    mock_metric.observe.assert_called_once_with(1.0)


@pytest.mark.asyncio
async def test_rag_agent_iterations_observed_complex(noop_tracer):
    """rag_agent_iterations reflects the actual iteration count for complex queries."""
    mock_client = AsyncMock()
    mock_chunk = {"content": "x", "content_hash": "h1", "rerank_score": 0.9, "score": 0.9}

    # _grade_chunks returns (accepted, sufficient=True) so the loop exits after 1 iteration
    with (
        patch("src.retrieval.pipeline._expand_and_search", new=AsyncMock(return_value=[mock_chunk])),
        patch("src.retrieval.pipeline.rerank", new=AsyncMock(return_value=[mock_chunk])),
        patch(
            "src.retrieval.pipeline._grade_chunks",
            new=AsyncMock(return_value=([mock_chunk], True)),
        ),
        patch("src.retrieval.pipeline.rag_agent_iterations") as mock_metric,
        patch("src.retrieval.pipeline.rag_chunks_per_query"),
        patch("src.retrieval.pipeline.rag_retrieval_confidence"),
    ):
        settings_complex = Settings(
            semantic_cache_url="",
            db_url="sqlite+aiosqlite:///./operational.db",
            max_retrieval_iterations=3,
        )
        await vector_search_pipeline(
            query="complex query",
            client=mock_client,
            settings=settings_complex,
            tracer=noop_tracer,
            trace_id="t2",
            complexity="complex",
        )

    mock_metric.observe.assert_called_once()
    observed_val = mock_metric.observe.call_args[0][0]
    assert observed_val >= 1.0
