"""Seed relational database and create curated views from CSV fixtures."""
from __future__ import annotations

import csv
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


async def seed_database(db_url: str, csv_dir: str) -> dict:
    """Load CSVs into SQLite tables. Returns row counts per table.

    Tables created: shift_records, incident_logs, staffing_history.
    """
    sources = [
        {"table": table, "path": os.path.join(csv_dir, filename)}
        for table, filename in [
            ("shift_records", "shift_records.csv"),
            ("incident_logs", "incident_logs.csv"),
            ("staffing_history", "staffing_history.csv"),
        ]
    ]
    return await seed_database_sources(db_url, sources)


async def seed_database_sources(db_url: str, sources: list[dict[str, str]]) -> dict:
    """Seed the known SQLite demo tables from explicit manifest sources."""
    supported = {"shift_records", "incident_logs", "staffing_history"}
    if any(source.get("table") not in supported for source in sources):
        raise ValueError("structured source names an unsupported table")

    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS shift_records (
                shift_id TEXT PRIMARY KEY, venue TEXT, date TEXT,
                staff_count INTEGER, shift_type TEXT, supervisor TEXT, notes TEXT
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS incident_logs (
                incident_id TEXT PRIMARY KEY, shift_id TEXT, venue TEXT, date TEXT,
                incident_type TEXT, description TEXT, severity TEXT
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS staffing_history (
                record_id TEXT PRIMARY KEY, venue TEXT, date TEXT, role TEXT,
                count INTEGER, required_count INTEGER
            )
        """))

    counts: dict[str, int] = {}
    for source in sources:
        table = source["table"]
        path = source["path"]
        rows = _read_csv(path)
        if rows:
            async with engine.begin() as conn:
                for row in rows:
                    cols = ", ".join(row.keys())
                    placeholders = ", ".join(f":{k}" for k in row)
                    await conn.execute(
                        text(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"),
                        row,
                    )
        counts[table] = len(rows)

    await create_views(engine)
    await engine.dispose()
    return counts


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def create_views(engine: AsyncEngine) -> None:
    """Create curated SQL views for safe, predictable SQL generation."""
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_shift_summary AS
            SELECT venue, date, shift_type, SUM(staff_count) AS total_staff
            FROM shift_records GROUP BY venue, date, shift_type
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_incident_by_venue AS
            SELECT venue, date, incident_type, COUNT(*) AS incident_count
            FROM incident_logs GROUP BY venue, date, incident_type
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_staffing_gap AS
            SELECT venue, date, role, count, required_count,
                   count - required_count AS gap
            FROM staffing_history
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_incident_severity AS
            SELECT incident_id, shift_id, venue, date, incident_type, description, severity
            FROM incident_logs WHERE severity = 'high'
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_peak_staffing AS
            SELECT s.shift_id, s.venue, s.date, s.staff_count, s.shift_type,
                   avg_s.avg_staff
            FROM shift_records s
            JOIN (
                SELECT venue, AVG(staff_count) AS avg_staff
                FROM shift_records GROUP BY venue
            ) avg_s ON s.venue = avg_s.venue
            WHERE s.staff_count > avg_s.avg_staff
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_shift_details AS
            SELECT shift_id, venue, date, staff_count, shift_type, supervisor
            FROM shift_records
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_incident_by_shift AS
            SELECT shift_id, venue, date, COUNT(*) AS incident_count
            FROM incident_logs GROUP BY shift_id, venue, date
        """))
        await conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_shift_staffing_evidence AS
            SELECT s.shift_id, s.venue, s.date, h.role, h.count, h.required_count,
                   h.count - h.required_count AS gap
            FROM shift_records AS s
            JOIN staffing_history AS h ON h.venue = s.venue AND h.date = s.date
        """))
