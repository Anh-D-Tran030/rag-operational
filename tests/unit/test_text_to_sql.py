"""Unit tests for src/retrieval/text_to_sql.py."""
from unittest.mock import AsyncMock, patch

import pytest

from src.retrieval.text_to_sql import execute_sql, text_to_sql_tool, validate_sql

_DB = "sqlite+aiosqlite:///./operational.db"


@pytest.mark.asyncio
async def test_generate_sql_returns_list():
    raw = (
        "```sql\nSELECT venue FROM v_shift_summary\n```\n"
        "```sql\nSELECT * FROM v_shift_summary\n```"
    )
    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=raw)):
        from src.retrieval.text_to_sql import generate_sql
        result = await generate_sql(
            query="How many venues?",
            relevant_views=[{
                "view_name": "v_shift_summary",
                "columns": ["venue"],
                "description": "shift data",
            }],
            model="openai/gpt-oss-120b",
            api_key="fake-key",
            base_url="https://openrouter.ai/api/v1",
        )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(s, str) for s in result)


@pytest.mark.asyncio
async def test_validate_sql_rejects_dml():
    valid, err = await validate_sql("INSERT INTO t VALUES (1)", _DB)
    assert valid is False
    assert "DML" in err or "not permitted" in err


@pytest.mark.asyncio
async def test_validate_sql_rejects_deep_join():
    sql = (
        "SELECT a.x FROM t1 a "
        "JOIN t2 b ON a.id=b.id "
        "JOIN t3 c ON b.id=c.id "
        "JOIN t4 d ON c.id=d.id "
        "JOIN t5 e ON d.id=e.id"
    )
    valid, err = await validate_sql(sql, _DB)
    assert valid is False
    assert "join" in err.lower() or "depth" in err.lower()


@pytest.mark.asyncio
async def test_execute_sql_happy_path():
    rows = await execute_sql("SELECT 1 AS val", _DB)
    assert isinstance(rows, list)
    assert rows[0]["val"] == 1


@pytest.mark.asyncio
async def test_sql_execution_error_increments_counter():
    """rag_sql_execution_errors is incremented once per failed execute_sql call."""
    raw = "```sql\nSELECT 1\n```"

    from src.config import Settings
    settings = Settings(openrouter_api_key="fake", db_url=_DB)

    with (
        patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=raw)),
        patch("src.retrieval.text_to_sql.validate_sql", AsyncMock(return_value=(True, ""))),
        patch(
            "src.retrieval.text_to_sql.execute_sql",
            AsyncMock(side_effect=RuntimeError("table not found")),
        ),
        patch("src.retrieval.text_to_sql.rag_sql_execution_errors") as mock_counter,
    ):
        result = await text_to_sql_tool(
            query="get data",
            relevant_views=[{
                "view_name": "v_shift_summary",
                "columns": ["venue"],
                "description": "shift data",
            }],
            settings=settings,
            trace_id="trace-err",
        )

    mock_counter.inc.assert_called_once()
    assert "error" in result or result.get("rows") == []


@pytest.mark.asyncio
async def test_self_correction_tries_next_candidate():
    raw = "```sql\nINSERT INTO t VALUES (1)\n```\n```sql\nSELECT 1 AS ok\n```"

    from src.config import Settings
    settings = Settings(openrouter_api_key="fake", db_url=_DB)

    with patch("src.retrieval.text_to_sql.chat", AsyncMock(return_value=raw)):
        result = await text_to_sql_tool(
            query="get data",
            relevant_views=[{
                "view_name": "v_shift_summary",
                "columns": ["venue"],
                "description": "x",
            }],
            settings=settings,
            trace_id="trace-abc",
        )
    assert "rows" in result
    if "error" not in result:
        assert isinstance(result["rows"], list)
