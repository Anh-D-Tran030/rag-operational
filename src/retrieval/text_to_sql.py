"""Text-to-SQL: generate, validate, execute SQL against curated views."""
from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import Settings
from src.llm_client import chat
from src.observability.metrics import rag_sql_execution_errors

_DML_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE)\b",
    re.IGNORECASE,
)
_JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_sql(raw: str) -> str:
    match = _SQL_FENCE.search(raw)
    return match.group(1).strip() if match else raw.strip()


async def generate_sql(
    query: str,
    relevant_views: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    n_candidates: int = 3,
) -> list[str]:
    """Call the cloud LLM to generate n_candidates SQL SELECT statements.

    Prompt includes the query and relevant view definitions.
    Returns a list of candidate SQL strings.
    """
    view_defs = "\n".join(
        f"View: {v['view_name']}\n"
        f"Columns: {', '.join(v.get('columns', []))}\n"
        f"Description: {v.get('description', '')}"
        for v in relevant_views
    )
    prompt = (
        f"Generate {n_candidates} different SQL SELECT statements that answer the question below.\n"
        f"Use ONLY the views listed. Do NOT use raw tables.\n\n"
        f"Available views:\n{view_defs}\n\n"
        f"Question: {query}\n\n"
        f"Return each SQL in a separate ```sql ... ``` code block."
    )
    raw = await chat(
        prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=1024,
    )
    candidates = [_extract_sql(block) for block in _SQL_FENCE.findall(raw)]
    if not candidates:
        candidates = [_extract_sql(raw)]
    return [c for c in candidates if c][:n_candidates]


async def validate_sql(sql: str, db_url: str) -> tuple[bool, str]:
    """Validate a SQL statement before execution.

    Rejects DML statements and queries with more than 3 JOINs.
    Runs EXPLAIN to catch syntax errors. Returns (is_valid, error_message).
    """
    if _DML_PATTERN.match(sql):
        return False, "DML statements are not permitted; only SELECT is allowed."
    join_count = len(_JOIN_PATTERN.findall(sql))
    if join_count > 3:
        return False, f"Query join depth {join_count} exceeds maximum of 3."
    try:
        engine = create_async_engine(db_url)
        async with engine.connect() as conn:
            await conn.execute(text(f"EXPLAIN {sql}"))
        await engine.dispose()
        return True, ""
    except Exception as exc:
        return False, str(exc)


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


async def text_to_sql_tool(
    query: str,
    relevant_views: list[dict],
    settings: Settings,
    trace_id: str,
    max_attempts: int = 3,
) -> dict:
    """Orchestrate generate → validate → execute with self-correction.

    Tries up to max_attempts candidates. Returns:
      {rows, sql_used, view_names, trace_id} on success.
      {rows: [], error: str} after max_attempts exhausted.
    """
    view_names = [v.get("view_name", "") for v in relevant_views]
    candidates = await generate_sql(
        query=query,
        relevant_views=relevant_views,
        model=settings.cloud_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        n_candidates=max_attempts,
    )

    last_error = "no candidates generated"
    for sql in candidates:
        valid, err = await validate_sql(sql, settings.db_url)
        if not valid:
            last_error = err
            continue
        try:
            rows = await execute_sql(sql, settings.db_url)
            return {"rows": rows, "sql_used": sql, "view_names": view_names, "trace_id": trace_id}
        except RuntimeError as exc:
            rag_sql_execution_errors.inc()
            last_error = str(exc)

    return {"rows": [], "error": last_error}
