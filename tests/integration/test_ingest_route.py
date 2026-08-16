"""Integration tests for the /ingest route."""
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app

FIXTURE_DIR = "tests/fixtures"


def _make_app_state():
    from src.config import Settings
    from src.observability.tracer import RAGTracer
    from src.retrieval.cache import SemanticCache

    settings = Settings(anthropic_api_key="")
    tracer = RAGTracer("", "", "")
    cache = SemanticCache("", settings.embedding_model)
    return settings, tracer, cache


@pytest.mark.asyncio
async def test_ingest_happy_path(qdrant_client, test_collection_name):
    settings, tracer, cache = _make_app_state()
    settings_with_collection = settings.model_copy(
        update={"qdrant_collection": test_collection_name}
    )

    app.state.settings = settings_with_collection
    app.state.tracer = tracer
    app.state.cache = cache
    app.state.qdrant_client = qdrant_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/ingest/sync",
            json={
                "file_path": f"{FIXTURE_DIR}/sample_shift_log.txt",
                "doc_type": "shift_log",
                "venue": "Fairfield RSL",
                "date": "2026-06-08",
                "shift_id": "SH-TEST",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_ingested"] > 0
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_ingest_missing_field():
    settings, tracer, cache = _make_app_state()

    from qdrant_client import AsyncQdrantClient
    mock_client = AsyncMock(spec=AsyncQdrantClient)

    app.state.settings = settings
    app.state.tracer = tracer
    app.state.cache = cache
    app.state.qdrant_client = mock_client

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Missing doc_type → 422
        resp = await client.post(
            "/ingest/sync",
            json={
                "file_path": "some/path.pdf",
                "venue": "Fairfield RSL",
                "date": "2026-06-08",
                "shift_id": "SH-001",
            },
        )

    assert resp.status_code == 422
