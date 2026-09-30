from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.config import Settings


class _Connection:
    async def __aenter__(self) -> _Connection:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, statement: object) -> None:
        return None


class _Database:
    def connect(self) -> _Connection:
        return _Connection()


class _UnavailableDatabase:
    def connect(self) -> _Connection:
        raise OSError("database unavailable")


def _set_state(*, collections: list[str], point_counts: dict[str, int], database: object) -> AsyncMock:
    qdrant = AsyncMock()
    qdrant.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=name) for name in collections]
    )
    qdrant.get_collection.side_effect = lambda name: SimpleNamespace(
        points_count=point_counts[name]
    )
    app.state.settings = Settings(
        _env_file=None,
        qdrant_collection="active_documents",
        qdrant_schema_collection="active_schema",
    )
    app.state.qdrant_client = qdrant
    app.state.db_engine = database
    return qdrant


@pytest.mark.asyncio
async def test_ready_reports_configured_non_empty_corpus() -> None:
    qdrant = _set_state(
        collections=["active_documents", "active_schema"],
        point_counts={"active_documents": 12, "active_schema": 5},
        database=_Database(),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "database": {"status": "ok"},
            "document_collection": {
                "status": "ok", "collection": "active_documents", "points": 12,
            },
            "schema_collection": {
                "status": "ok", "collection": "active_schema", "points": 5,
            },
        },
    }
    qdrant.get_collection.assert_any_await("active_documents")
    qdrant.get_collection.assert_any_await("active_schema")


@pytest.mark.asyncio
async def test_ready_reports_missing_or_empty_configured_collections() -> None:
    _set_state(
        collections=["active_schema"],
        point_counts={"active_schema": 0},
        database=_Database(),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "database": {"status": "ok"},
            "document_collection": {
                "status": "collection_missing", "collection": "active_documents",
            },
            "schema_collection": {
                "status": "collection_empty", "collection": "active_schema",
            },
        },
    }


@pytest.mark.asyncio
async def test_ready_reports_unavailable_database_without_calling_llms() -> None:
    qdrant = _set_state(
        collections=["active_documents", "active_schema"],
        point_counts={"active_documents": 1, "active_schema": 1},
        database=_UnavailableDatabase(),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["database"] == {"status": "unavailable"}
    assert qdrant.get_collections.await_count == 1


@pytest.mark.asyncio
async def test_ready_reports_unavailable_qdrant() -> None:
    qdrant = _set_state(
        collections=[],
        point_counts={},
        database=_Database(),
    )
    qdrant.get_collections.side_effect = OSError("qdrant unavailable")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "database": {"status": "ok"},
            "document_collection": {"status": "unavailable"},
            "schema_collection": {"status": "unavailable"},
        },
    }
