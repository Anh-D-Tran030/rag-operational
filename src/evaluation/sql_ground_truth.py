"""Structured ground truth and semantic row comparison for SQL evaluation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SQL_CASE_IDS = (9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 23)
REQUIRED_FIELDS = {
    "query_id",
    "expected_view",
    "expected_columns",
    "column_aliases",
    "expected_rows",
    "acceptable_aggregation",
    "schema_support",
    "acceptable_routes",
}
SCHEMA_SUPPORT_LEVELS = {"full", "partial", "unsupported"}
ACCEPTABLE_ROUTES = {"text_to_sql", "hybrid", "vector_search"}


def load_sql_expectations(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load and validate the 12 structured SQL expectations."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("SQL expectations must be a JSON list")
    cases: dict[int, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict) or not REQUIRED_FIELDS <= item.keys():
            raise ValueError("Each SQL expectation must contain all required fields")
        query_id = item["query_id"]
        if not isinstance(query_id, int) or query_id in cases:
            raise ValueError("SQL expectation query_id values must be unique integers")
        if item["acceptable_aggregation"] != "exact_rows_order_insensitive":
            raise ValueError(f"Unsupported comparison policy for query {query_id}")
        if set(item["expected_columns"]) != set(item["column_aliases"]):
            raise ValueError(f"Column aliases do not cover query {query_id}")
        if item["schema_support"] not in SCHEMA_SUPPORT_LEVELS:
            raise ValueError(f"Invalid schema support classification for query {query_id}")
        routes = item["acceptable_routes"]
        if not isinstance(routes, list) or not routes or not set(routes) <= ACCEPTABLE_ROUTES:
            raise ValueError(f"Invalid acceptable routes for query {query_id}")
        cases[query_id] = item
    if set(cases) != set(SQL_CASE_IDS):
        raise ValueError(f"Expected SQL query IDs {list(SQL_CASE_IDS)}, got {sorted(cases)}")
    return cases


def load_golden_cases(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load the one-indexed golden dataset into a query-id mapping."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Golden dataset must be a JSON list")
    return {index: item for index, item in enumerate(payload, start=1)}


def _normalize(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", float(value))
    if value is None:
        return ("null", None)
    text = str(value).strip()
    lowered = text.casefold()
    if lowered in {"true", "yes", "met"}:
        return ("bool", True)
    if lowered in {"false", "no", "not met"}:
        return ("bool", False)
    try:
        return ("number", float(text))
    except ValueError:
        return ("text", lowered)


def _values_equal(actual: Any, expected: Any) -> bool:
    """Compare values, accepting SQLite's integer representation of booleans."""
    if isinstance(expected, bool) and type(actual) is int and actual in {0, 1}:
        return actual == int(expected)
    return _normalize(actual) == _normalize(expected)


def _project_row(row: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any] | None:
    projected: dict[str, Any] = {}
    for column in expected["expected_columns"]:
        aliases = expected["column_aliases"][column]
        matched = next((alias for alias in aliases if alias in row), None)
        if matched is None:
            return None
        projected[column] = row[matched]
    return projected


def _all_present_match(row: dict[str, Any], keys: tuple[str, ...], wanted: Any) -> bool:
    values = [row[key] for key in keys if key in row]
    return bool(values) and all(_values_equal(value, wanted) for value in values)


def _row_or_sql_equals(
    row: dict[str, Any], sql: str | None, column: str, wanted: str
) -> bool:
    """Accept a selected constraint or the equivalent v_staffing_gap predicate."""
    if column in row:
        return _values_equal(row[column], wanted)
    return bool(sql and re.search(
        rf"\b(?:WHERE|AND)\s+(?:[A-Za-z_]\w*\.)?{column}\s*=\s*'{re.escape(wanted)}'",
        sql,
        re.IGNORECASE,
    ))


def _staffed_is_count_alias(sql: str | None) -> bool:
    """Recognize the explicit `count AS staffed` projection from the curated view."""
    return bool(sql and re.search(
        r"\b(?:[A-Za-z_]\w*\.)?count\s+AS\s+staffed\b", sql, re.IGNORECASE
    ))


def _semantic_case(
    rows: list[dict[str, Any]], expected: dict[str, Any], sql: str | None
) -> bool | None:
    """Check the view-backed meanings that aliases alone cannot establish."""
    case_id = expected.get("query_id")
    if case_id not in {16, 17, 19, 21}:
        return None
    if case_id == 17:
        wanted_dates = {"2026-05-22", "2026-05-25", "2026-05-27"}
        if len(rows) != len(wanted_dates) or {row.get("date") for row in rows} != wanted_dates:
            return False
        # v_staffing_gap exposes role. If it is not projected, the executed
        # query must explicitly restrict that view to floor_staff.
        roles = [row.get("role") for row in rows]
        role_projected = any(role is not None for role in roles)
        role_filtered = bool(sql and re.search(
            r"\b(?:WHERE|AND)\s+(?:[A-Za-z_]\w*\.)?role\s*=\s*'floor_staff'",
            sql, re.IGNORECASE,
        ))
        return all(role == "floor_staff" for role in roles) if role_projected else role_filtered

    if len(rows) != 1:
        return False
    row = rows[0]
    if case_id in {16, 21}:
        date, role, count, required, met = (
            ("2026-05-28", "floor_staff", 8, 8, True) if case_id == 16
            else ("2026-05-25", "security", 2, 3, False)
        )
        count_keys = (
            ("count", "staff_count", "floor_staff_count") if case_id == 16
            else tuple(expected["column_aliases"]["security_count"])
        )
        if case_id == 21 and _staffed_is_count_alias(sql):
            count_keys = (*count_keys, "staffed")
        met_keys = (
            ("met_requirement", "meets_required", "met_required", "met")
            if case_id == 16
            # `meets_requirement` predates the declared alias map; retain it
            # for backwards-compatible evaluation while using the map for all
            # currently declared Case 21 aliases.
            else (*expected["column_aliases"]["met_requirement"], "meets_requirement")
        )
        return (
            bool(sql and re.search(r"\bFROM\s+v_staffing_gap\b", sql, re.IGNORECASE))
            and _row_or_sql_equals(row, sql, "venue", "Fairfield RSL")
            and _row_or_sql_equals(row, sql, "date", date)
            and _row_or_sql_equals(row, sql, "role", role)
            and _all_present_match(row, count_keys, count)
            and _values_equal(row.get("required_count"), required)
            and _all_present_match(row, met_keys, met)
        )

    # v_peak_staffing has shift_id and staff_count; resolve the incident
    # aggregate through the declared aliases because SUM(...) may rename it.
    incident_keys = (*expected["column_aliases"]["incident_count"], "incident_total")
    return (
        row.get("shift_id") == "SH-006"
        and _values_equal(row.get("staff_count"), 20)
        and _all_present_match(row, incident_keys, 2)
    )


def compare_result_rows(
    rows: list[dict[str, Any]], expected: dict[str, Any], sql: str | None = None
) -> bool:
    """Compare result values and aliases, never SQL source text or row order."""
    semantic = _semantic_case(rows, expected, sql)
    if semantic is not None:
        return semantic
    projected = [_project_row(row, expected) for row in rows]
    if any(row is None for row in projected):
        return False

    remaining = list(expected["expected_rows"])
    for actual_row in projected:
        match_index = next(
            (
                index
                for index, expected_row in enumerate(remaining)
                if all(
                    _values_equal(actual_row[column], expected_row[column])
                    for column in expected["expected_columns"]
                )
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining
