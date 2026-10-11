"""Deterministic SQL construction, required-view augmentation and the SQL prompt.

Sole owner of ``_with_required_sql_views``, built from the canonical incident
aggregate metadata in ``sql_schema``. No provider call, retry or database work.
"""
from __future__ import annotations

import re

from src.retrieval.sql_candidate_filters import (
    _FLOOR_STAFF_REFERENCE,
    _is_floor_staff_shortfall_listing,
)
from src.retrieval.sql_schema import _INCIDENT_BY_SHIFT_VIEW, _INCIDENT_BY_VENUE_VIEW

SQL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "sql_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}
_SUPERVISOR_COMPARISON_QUERY = re.compile(
    r"\bday of the (?P<incident_type>[a-z]+(?:[- ][a-z]+)*) incident\b.*"
    r"\bsame person who supervised shift (?P<shift_id>SH-\d+)\b",
    re.IGNORECASE,
)


def _supervisor_comparison_sql(query: str) -> str | None:
    """Compile an explicit incident-supervisor comparison without LLM ambiguity."""
    match = _SUPERVISOR_COMPARISON_QUERY.search(query)
    if not match:
        return None
    incident_type = match.group("incident_type").replace("-", " ")
    shift_id = match.group("shift_id")
    return (
        "SELECT incident_shift.supervisor AS incident_supervisor, "
        "comparison_shift.supervisor AS comparison_supervisor, "
        "CASE WHEN incident_shift.supervisor = comparison_shift.supervisor "
        "THEN 1 ELSE 0 END AS same_supervisor, "
        # The identities that establish the comparison, so the evidence cites them.
        "incident.incident_id, incident.incident_type, "
        "incident.shift_id AS incident_shift_id, "
        "comparison_shift.shift_id AS comparison_shift_id "
        "FROM v_incident_severity AS incident "
        "JOIN v_shift_details AS incident_shift ON incident.shift_id = incident_shift.shift_id "
        "CROSS JOIN v_shift_details AS comparison_shift "
        f"WHERE incident.incident_type = '{incident_type}' COLLATE NOCASE "
        f"AND comparison_shift.shift_id = '{shift_id}' LIMIT 1"
    )


def _with_required_sql_views(query: str, relevant_views: list[dict]) -> list[dict]:
    """Add the curated incident aggregates only when the question explicitly needs them.

    A shift-level incident question also gets v_incident_by_shift, whose count
    already spans every incident type for one shift.
    """
    normalized = query.casefold()
    requires_incidents = "incident" in normalized or "rsa-intervention" in normalized
    if not requires_incidents:
        return relevant_views
    present = {view.get("view_name") for view in relevant_views}
    required = [_INCIDENT_BY_VENUE_VIEW]
    if "shift" in normalized:
        required.append(_INCIDENT_BY_SHIFT_VIEW)
    missing = [view for view in required if view["view_name"] not in present]
    return [*relevant_views, *missing] if missing else relevant_views


def build_sql_prompt(query: str, relevant_views: list[dict]) -> str:
    """Render the exact provider-facing SQL prompt for one question."""
    views = _with_required_sql_views(query, relevant_views)
    role_constraint = ""
    if _FLOOR_STAFF_REFERENCE.search(query):
        role_constraint = (
            "The question is specifically about floor staff: include role = 'floor_staff'.\n"
        )
    shortfall_listing_constraint = ""
    if _is_floor_staff_shortfall_listing(query):
        shortfall_listing_constraint = (
            "For a floor-staff shortage or shortfall listing, query v_staffing_gap directly; "
            "do not join v_peak_staffing. Filter role = 'floor_staff' and gap < 0. If the "
            "question specifies May 2026, use date LIKE '2026-05-%'. Return date, count, and "
            "required_count.\n"
        )
    view_defs = "\n".join(
        f"View: {v['view_name']}\n"
        f"Columns: {', '.join(v.get('columns', []))}\n"
        f"Description: {v.get('description', '')}"
        for v in views
    )
    return (
        "Generate one SQL SELECT statement that answers the question below.\n"
        f"Use ONLY the views listed. Do NOT use raw tables. Use SQLite syntax: quote dates as "
        f"'YYYY-MM-DD' and use LIMIT, never DATE literals or FETCH FIRST. Each statement must "
        f"return every requested base metric and preserve every identity or detail field "
        f"explicitly "
        f"requested by the user; select views that expose those fields (for example, 'which "
        f"shift' requires shift_id). When returning a derived comparison, boolean, or status, also "
        f"return the available input metrics used to establish it. Prefer one self-contained view "
        f"when it contains all requested outputs and filters. Join views only when needed for "
        f"requested information, and only on semantically compatible entity keys; never equate "
        f"unrelated categorical domains such as shift type and staffing role. "
        f"Never substitute NULL "
        f"as a requested metric.\n\n"
        f"Available views:\n{view_defs}\n\n"
        f"{role_constraint}"
        f"{shortfall_listing_constraint}"
        f"Question: {query}\n\n"
        "Return exactly one JSON object with this shape: "
        '{"sql":"<one SQL SELECT statement>"}. The sql value must contain one statement '
        "only, with no Markdown, explanation, or prose."
    )
