"""Integration tests for the /query route."""
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


def _make_app_state():
    from src.config import Settings
    from src.observability.tracer import RAGTracer
    from src.retrieval.cache import SemanticCache

    settings = Settings(anthropic_api_key="")
    tracer = RAGTracer("", "", "")
    cache = SemanticCache("", settings.embedding_model)
    return settings, tracer, cache


@pytest.fixture
def mock_query_result():
    return {
        "answer": "The procedure requires two signatures.",
        "citations": [
            {
                "doc_type": "procedure",
                "venue": "Fairfield RSL",
                "date": "2026-05-01",
                "shift_id": "N/A",
                "page_num": 1,
                "snippet": "Two signatures are required.",
                "score": 0.92,
            }
        ],
        "trace_id": "trace-abc",
        "route": "simple",
        "tool": "vector_search",
        "latency_ms": 120.0,
    }


@pytest.mark.asyncio
async def test_query_happy_path(mock_query_result):
    settings, tracer, cache = _make_app_state()

    with patch(
        "src.generation.query_handler.handle_query",
        new=AsyncMock(return_value=mock_query_result),
    ):
        from qdrant_client import AsyncQdrantClient
        mock_client = AsyncMock(spec=AsyncQdrantClient)

        app.state.settings = settings
        app.state.tracer = tracer
        app.state.cache = cache
        app.state.qdrant_client = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/query/", json={"query": "What is the procedure?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == mock_query_result["answer"]
    assert body["trace_id"] == "trace-abc"
    assert body["tool"] == "vector_search"
    assert len(body["vector_citations"]) == 1


@pytest.mark.asyncio
async def test_query_empty_string():
    settings, tracer, cache = _make_app_state()

    from qdrant_client import AsyncQdrantClient
    mock_client = AsyncMock(spec=AsyncQdrantClient)

    app.state.settings = settings
    app.state.tracer = tracer
    app.state.cache = cache
    app.state.qdrant_client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/query/", json={"query": ""})

    assert resp.status_code == 422
