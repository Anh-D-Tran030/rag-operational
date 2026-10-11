"""Unit tests for src/ingestion/structured_loader.py."""
import os
import tempfile

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.ingestion.structured_loader import seed_database, seed_database_sources

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
            "v_shift_details",
            "v_incident_by_shift",
            "v_shift_staffing_evidence",
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


@pytest.mark.asyncio
async def test_evidence_views_expose_only_fixture_backed_shift_facts(tmp_path):
    """The evidence join must not turn absent staffing data into a positive finding."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'evidence.db'}"
    await seed_database(url, FIXTURE_DIR)
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        supervisor = (await conn.execute(text(
            "SELECT supervisor FROM v_shift_details WHERE shift_id = 'SH-010'"
        ))).scalar_one()
        incident_count = (await conn.execute(text(
            "SELECT incident_count FROM v_incident_by_shift WHERE shift_id = 'SH-006'"
        ))).scalar_one()
        staffed_shift = (await conn.execute(text(
            "SELECT gap FROM v_shift_staffing_evidence "
            "WHERE shift_id = 'SH-006' AND role = 'floor_staff'"
        ))).scalar_one()
        missing_shift = (await conn.execute(text(
            "SELECT COUNT(*) FROM v_shift_staffing_evidence WHERE shift_id = 'SH-010'"
        ))).scalar_one()
    await engine.dispose()

    assert supervisor == "S. Lee"
    assert incident_count == 2
    assert staffed_shift == 2
    assert missing_shift == 0


@pytest.mark.asyncio
async def test_seed_database_sources_is_explicit_and_repeat_safe(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'explicit.db'}"
    sources = [{"table": "incident_logs", "path": f"{FIXTURE_DIR}/incident_logs.csv"}]

    first = await seed_database_sources(url, sources)
    second = await seed_database_sources(url, sources)

    assert first == {"incident_logs": 10}
    assert second == first
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        count = (await conn.execute(text("SELECT COUNT(*) FROM incident_logs"))).scalar_one()
    await engine.dispose()
    assert count == 10
