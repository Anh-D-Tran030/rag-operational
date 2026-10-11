"""Fixtures requiring live integration dependencies."""
from __future__ import annotations

import os
from asyncio import wait_for

import pytest
import pytest_asyncio

from src.config import settings

QDRANT_TEARDOWN_TIMEOUT_SECONDS = 5.0


async def _bounded_qdrant_call(awaitable):
    """Bound an external Qdrant teardown operation so pytest can exit."""
    return await wait_for(awaitable, timeout=QDRANT_TEARDOWN_TIMEOUT_SECONDS)


@pytest_asyncio.fixture(scope="session")
async def qdrant_client():
    """AsyncQdrantClient connected to the configured Qdrant instance."""
    from qdrant_client import AsyncQdrantClient as QdrantClient

    client = QdrantClient(
        url=settings.qdrant_url,
        timeout=QDRANT_TEARDOWN_TIMEOUT_SECONDS,
        check_compatibility=False,
    )
    try:
        yield client
    finally:
        try:
            collections = await _bounded_qdrant_call(client.get_collections())
            for col in collections.collections:
                if col.name.startswith("test_"):
                    await _bounded_qdrant_call(client.delete_collection(col.name))
        except TimeoutError:
            pass
        except Exception:
            pass
        finally:
            try:
                await _bounded_qdrant_call(client.close())
            except TimeoutError:
                pass
            if os.path.exists("./test_operational.db"):
                os.remove("./test_operational.db")


@pytest.fixture
def test_collection_name():
    """Qdrant collection name used in integration tests."""
    return "test_rag_operational"


@pytest.fixture
def test_schema_collection():
    """Qdrant schema collection name used in integration tests."""
    return "test_rag_schema_meta"
