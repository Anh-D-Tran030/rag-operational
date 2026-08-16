"""Unit tests for src/ingestion/structured_loader.py."""
import os
import tempfile

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.ingestion.structured_loader import seed_database

FIXTURE_DIR = "tests/fixtures"


@pytest.mark.asyncio
async def test_seed_database():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        url = f"sqlite+aiosqlite:///{db_path}"
        counts = await seed_database(url, FIXTURE_DIR)
        assert counts["shift_records"] > 0
        assert counts["incident_logs"] > 0
        assert counts["staffing_history"] > 0
    finally:
        os.unlink(db_path)


@pytest.mark.asyncio
async def test_create_views():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        url = f"sqlite+aiosqlite:///{db_path}"
        await seed_database(url, FIXTURE_DIR)

        engine = create_async_engine(url, echo=False)
        expected_views = [
            "v_shift_summary",
            "v_incident_by_venue",
            "v_staffing_gap",
            "v_incident_severity",
            "v_peak_staffing",
        ]
        async with engine.connect() as conn:
            for view in expected_views:
                result = await conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='view' AND name=:name"),
                    {"name": view},
                )
                row = result.fetchone()
                assert row is not None, f"View {view} not found"
        await engine.dispose()
    finally:
        os.unlink(db_path)
