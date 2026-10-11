"""Integration tests for the /feedback route."""
from unittest.mock import AsyncMock

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


@pytest.mark.asyncio
async def test_feedback_happy_path():
    settings, tracer, cache = _make_app_state()

    from qdrant_client import AsyncQdrantClient
    mock_client = AsyncMock(spec=AsyncQdrantClient)

    app.state.settings = settings
    app.state.tracer = tracer
    app.state.cache = cache
    app.state.qdrant_client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/feedback/",
            json={"trace_id": "trace-xyz", "score": 1},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "trace-xyz"
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_feedback_score_out_of_range():
    settings, tracer, cache = _make_app_state()

    from qdrant_client import AsyncQdrantClient
    mock_client = AsyncMock(spec=AsyncQdrantClient)

    app.state.settings = settings
    app.state.tracer = tracer
    app.state.cache = cache
    app.state.qdrant_client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/feedback/",
            json={"trace_id": "trace-xyz", "score": 5},
        )

    assert resp.status_code == 422
