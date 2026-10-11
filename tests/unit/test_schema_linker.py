"""Unit tests for schema-linking completion of combined SQL intents."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.retrieval.schema_linker import find_relevant_views
from src.retrieval.text_to_sql import generate_sql

_CASE_19_QUERY = (
    "Which shift in May 2026 had the largest staff count, and how many incidents "
    "were logged on it?"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "required_view"),
    [
        ("Who supervised shift SH-003?", "v_shift_details"),
        ("How many incidents were logged on shift SH-006?", "v_incident_by_shift"),
        (
            "Was the supervisor on the day of the medical-event incident the same person "
            "who supervised shift SH-001?",
            "v_shift_details",
        ),
    ],
)
async def test_explicit_schema_requirements_complete_misleading_retrieval(
    query: str, required_view: str,
):
    client = AsyncMock()
    client.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(payload={
            "view_name": "v_peak_staffing", "columns": [], "description": "", "example_query": "",
        }),
    ])

    views = await find_relevant_views(query, [0.1], client, "schema")

    assert required_view in {view["view_name"] for view in views}


@pytest.mark.asyncio
async def test_explicit_schema_requirement_survives_schema_collection_failure():
    client = AsyncMock()
    client.query_points.side_effect = RuntimeError("schema collection unavailable")

    views = await find_relevant_views("Who supervised shift SH-003?", [0.1], client, "schema")

    assert [view["view_name"] for view in views] == ["v_shift_details"]


@pytest.mark.asyncio
async def test_case_19_schema_selection_includes_peak_staffing_and_incidents():
    client = AsyncMock()
    client.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(payload={
            "view_name": "v_shift_summary", "columns": [], "description": "", "example_query": "",
        }),
        SimpleNamespace(payload={
            "view_name": "v_staffing_gap", "columns": [], "description": "", "example_query": "",
        }),
        SimpleNamespace(payload={
            "view_name": "v_peak_staffing", "columns": [], "description": "", "example_query": "",
        }),
    ])

    views = await find_relevant_views(_CASE_19_QUERY, [0.1], client, "schema")

    assert {view["view_name"] for view in views} >= {
        "v_peak_staffing", "v_incident_by_venue",
    }


@pytest.mark.asyncio
async def test_case_19_completed_schema_context_generates_valid_candidate():
    client = AsyncMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    views = await find_relevant_views(_CASE_19_QUERY, [0.1], client, "schema")
    candidate = (
        "SELECT p.shift_id, p.staff_count, SUM(i.incident_count) AS incident_count "
        "FROM v_peak_staffing AS p JOIN v_incident_by_venue AS i "
        "ON i.venue = p.venue AND i.date = p.date "
        "GROUP BY p.shift_id, p.staff_count ORDER BY p.staff_count DESC LIMIT 1"
    )
    with patch(
        "src.retrieval.text_to_sql.chat",
        AsyncMock(return_value=json.dumps({"sql": candidate})),
    ):
        generated = await generate_sql(
            _CASE_19_QUERY, views, "test-model", "fake-key", "https://example.test"
        )

    assert generated == [candidate]
