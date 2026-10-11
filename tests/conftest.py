"""Shared fixtures that do not require live integration dependencies."""
import sys

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine


def pytest_sessionfinish(session, exitstatus):
    """Release reranker workers after all tests have finished using them."""
    reranker = sys.modules.get("src.retrieval.reranker")
    if reranker is not None:
        reranker.shutdown_reranker_executor()


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """SQLAlchemy async engine for test SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite:///./test_operational.db", echo=False)
    yield engine
    await engine.dispose()
