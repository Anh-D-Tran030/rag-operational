"""SQL validation and read-only execution against the curated SQLite views."""
from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.retrieval.sql_candidate_parser import _JOIN_PATTERN, _SQL_START

_DML_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE)\b",
    re.IGNORECASE,
)


async def validate_sql(sql: str, db_url: str) -> tuple[bool, str]:
    """Validate a SQL statement before execution.

    Rejects DML statements and queries with more than 3 JOINs.
    Runs EXPLAIN to catch syntax errors. Returns (is_valid, error_message).
    """
    if _DML_PATTERN.match(sql):
        return False, "DML statements are not permitted; only SELECT is allowed."
    if not _SQL_START.match(sql):
        return False, "Only a single SELECT or WITH statement is allowed."
    join_count = len(_JOIN_PATTERN.findall(sql))
    if join_count > 3:
        return False, f"Query join depth {join_count} exceeds maximum of 3."
    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(f"EXPLAIN {sql}"))
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        await engine.dispose()


async def execute_sql(sql: str, db_url: str) -> list[dict]:
    """Execute a validated SELECT statement and return rows as list[dict].

    Raises RuntimeError on execution failure (self-correction at caller level).
    """
    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            keys = list(result.keys())
            rows = [dict(zip(keys, row)) for row in result.fetchall()]
        return rows
    except Exception as exc:
        raise RuntimeError(f"SQL execution failed: {exc}") from exc
    finally:
        await engine.dispose()
