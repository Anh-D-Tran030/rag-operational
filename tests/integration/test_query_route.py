"""Integration tests for the /query route."""
import json
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


async def _post(handle_query_mock, headers=None):
    settings, tracer, cache = _make_app_state()
    from qdrant_client import AsyncQdrantClient

    app.state.settings, app.state.tracer, app.state.cache = settings, tracer, cache
    app.state.qdrant_client = AsyncMock(spec=AsyncQdrantClient)
    with patch("src.generation.query_handler.handle_query", new=handle_query_mock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/query/", json={"query": "q"}, headers=headers or {})


def _server_lines(capsys) -> list[dict]:
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    return [line for line in lines if line.get("event") == "query_request"]


@pytest.mark.asyncio
async def test_valid_request_id_is_echoed_and_logged_with_timings(mock_query_result, capsys):
    result = {**mock_query_result, "latency_components": {
        "router_ms": 1.0, "unattributed_ms": 2.5, "sql": {"generation_ms": 3.0, "attempts": ["x"]},
    }}
    resp = await _post(AsyncMock(return_value=result), {"X-Request-ID": "load-r0001"})
    assert resp.headers["X-Request-ID"] == "load-r0001"
    assert resp.json()["stage_timings"] == {
        "router_ms": 1.0, "unattributed_ms": 2.5, "sql_generation_ms": 3.0}
    (line,) = _server_lines(capsys)
    assert line["request_id"] == "load-r0001" and line["status"] == 200
    assert line["route"] == "simple" and line["stage_timings"]["unattributed_ms"] == 2.5
    assert "exception_class" not in line


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "has space", "x" * 81, "a/b"])
async def test_unsafe_request_id_is_replaced(mock_query_result, bad):
    resp = await _post(AsyncMock(return_value=mock_query_result), {"X-Request-ID": bad})
    assert resp.headers["X-Request-ID"] != bad
    assert len(resp.headers["X-Request-ID"]) == 32


@pytest.mark.asyncio
async def test_500_body_is_sanitized_and_log_names_exception_class_only(capsys):
    boom = AsyncMock(side_effect=RuntimeError("db password=hunter2 at /srv/secret"))
    resp = await _post(boom, {"X-Request-ID": "load-r0002"})
    assert resp.status_code == 500
    assert resp.json() == {"error": "query_failed", "request_id": "load-r0002"}
    assert resp.headers["X-Request-ID"] == "load-r0002"
    (line,) = _server_lines(capsys)
    assert line["status"] == 500 and line["exception_class"] == "RuntimeError"
    assert "hunter2" not in json.dumps(line)

