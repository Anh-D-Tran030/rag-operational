"""Shared pytest fixtures for unit and integration tests."""
import os

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import settings


@pytest_asyncio.fixture(scope="session")
async def qdrant_client():
    """AsyncQdrantClient connected to the configured Qdrant instance."""
    from qdrant_client import AsyncQdrantClient as QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    yield client
    await client.close()


@pytest.fixture
def test_collection_name():
    """Qdrant collection name used in tests."""
    return "test_rag_operational"


@pytest.fixture
def test_schema_collection():
    """Qdrant schema collection name used in tests."""
    return "test_rag_schema_meta"


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """SQLAlchemy async engine for test SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///./test_operational.db", echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True, scope="session")
async def cleanup(qdrant_client):
    """Delete test Qdrant collections and test SQLite DB after test session."""
    yield
    try:
        collections = await qdrant_client.get_collections()
        for col in collections.collections:
            if col.name.startswith("test_"):
                await qdrant_client.delete_collection(col.name)
    except Exception:
        pass
    if os.path.exists("./test_operational.db"):
        os.remove("./test_operational.db")
