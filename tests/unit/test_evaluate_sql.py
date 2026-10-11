"""Focused CLI tests for the SQL evaluator's case selection."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts import evaluate_sql
from src.evaluation.sql_case_eval import aggregate_sql_metrics


def _args(tmp_path, case_id: int | None) -> Namespace:
    return Namespace(
        case_id=case_id,
        sql_json=str(tmp_path / "sql.json"),
        sql_markdown=str(tmp_path / "sql.md"),
        unanswerable_json=str(tmp_path / "unanswerable.json"),
        startup_timeout_s=1.0,
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        openrouter_api_key="test-key",
        qdrant_url="http://qdrant.test",
        qdrant_api_key=None,
        embedding_model="test-model",
        rerank_top_n=5,
        rerank_score_threshold=0.1,
        db_url="sqlite+aiosqlite:///test.db",
    )


def _case_result(case_id: int) -> dict:
    return {
        "query_id": case_id,
        "schema_support": "full",
        "predicted_route": "text_to_sql",
        "route_correct": True,
        "execution_succeeded": True,
        "result_correct": True,
        "wrong_but_runnable": False,
        "sql_error_observed": False,
        "self_corrected": False,
        "latency": {"total_ms": 1.0},
    }


@pytest.mark.asyncio
async def test_case_id_19_runs_alone_and_writes_sql_artifacts(tmp_path, monkeypatch) -> None:
    evaluated: list[int] = []
    fake_client = SimpleNamespace(close=AsyncMock())

    class FakeQdrantClient:
        def __new__(cls, **_kwargs):
            return fake_client

    class FakeCache:
        def __init__(self, **_kwargs):
            pass

    async def evaluate(case_id, *_args):
        evaluated.append(case_id)
        return _case_result(case_id)

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(AsyncQdrantClient=FakeQdrantClient))
    monkeypatch.setattr(evaluate_sql, "Settings", lambda **_kwargs: _settings())
    monkeypatch.setattr(evaluate_sql, "_tracer", lambda _settings: object())
    monkeypatch.setattr(evaluate_sql, "seed_database", AsyncMock())
    monkeypatch.setattr(evaluate_sql, "evaluate_sql_case", evaluate)
    monkeypatch.setattr(evaluate_sql, "aggregate_sql_metrics", aggregate_sql_metrics)
    monkeypatch.setattr(evaluate_sql, "summarize_latencies", lambda *_args: {})
    monkeypatch.setattr("src.retrieval.cache.SemanticCache", FakeCache)
    monkeypatch.setattr(
        evaluate_sql, "load_golden_cases", lambda _path: {case_id: {} for case_id in range(1, 31)}
    )
    monkeypatch.setattr(
        evaluate_sql, "load_sql_expectations", lambda _path: {case_id: {} for case_id in evaluate_sql.SQL_CASE_IDS}
    )

    args = _args(tmp_path, 19)
    assert await evaluate_sql.run(args) == 0
    assert evaluated == [19]
    assert fake_client.close.await_count == 1
    assert json.loads((tmp_path / "sql.json").read_text())[
        "cases"
    ][0]["query_id"] == 19
    assert (tmp_path / "sql.md").read_text().startswith("# SQL evaluation\n")
    assert json.loads((tmp_path / "unanswerable.json").read_text())["case_count"] == 0


def test_default_case_selection_is_full_set() -> None:
    assert evaluate_sql._selected_sql_case_ids(None) == evaluate_sql.SQL_CASE_IDS


def test_invalid_case_id_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        evaluate_sql._parser().parse_args(["--case-id", "999"])
    assert exc_info.value.code == 2
