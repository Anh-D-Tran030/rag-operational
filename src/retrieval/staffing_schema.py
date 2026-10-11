"""Trusted staffing-requirement schema completion for v_staffing_gap.

A staffing-requirement or shortfall question needs per-role ``count`` against
``required_count``. Schema retrieval can miss that view or return incomplete or
misleading metadata, so the trusted curated record is merged in. Pure and typed:
no SQL, rows, provider call or answer lookup, and input records are never mutated.
"""
from __future__ import annotations

import re

STAFFING_VIEW_NAME = "v_staffing_gap"
_COLUMNS = ("venue", "date", "role", "count", "required_count", "gap")
_DESCRIPTION = (
    "Staffing gap analysis: actual count minus required count per role."
    " Negative gap means understaffed."
)
_EXAMPLE_QUERY = "SELECT venue, role, gap FROM v_staffing_gap WHERE gap < 0"

# Explicit staffing-requirement phrases, matched on word boundaries so
# "overstaffing" or "staffroom" never match.
_DIRECT_CONCEPT = re.compile(
    r"\bstaff(?:ing)?\s+(?:requirements?|shortfalls?|gaps?|deficits?)\b"
    r"|\bunderstaff(?:ed|ing)\b"
    r"|\brequired\s+staff(?:ing)?\b",
    re.IGNORECASE,
)
# A bare requirement word counts only alongside an explicit staffing word.
_REQUIREMENT = re.compile(
    r"\brequired\s+(?:count|levels?|numbers?)\b|\brequirements?\b", re.IGNORECASE
)
_STAFFING_CONTEXT = re.compile(r"\bstaff(?:ing|ed)?\b", re.IGNORECASE)


def _canonical_columns() -> list[str]:
    """The curated projection, checked against the canonical SQL inventory."""
    from src.retrieval.sql_schema import CURATED_VIEW_COLUMNS

    if frozenset(_COLUMNS) != CURATED_VIEW_COLUMNS[STAFFING_VIEW_NAME]:
        raise RuntimeError("v_staffing_gap metadata diverged from the curated view")
    return list(_COLUMNS)


def requires_staffing_requirement(query: str) -> bool:
    """Whether the question asks about a staffing requirement or shortfall."""
    if _DIRECT_CONCEPT.search(query):
        return True
    return bool(_REQUIREMENT.search(query) and _STAFFING_CONTEXT.search(query))


def staffing_view() -> dict:
    """A fresh trusted metadata record for v_staffing_gap."""
    return {
        "view_name": STAFFING_VIEW_NAME,
        "columns": _canonical_columns(),
        "description": _DESCRIPTION,
        "example_query": _EXAMPLE_QUERY,
    }


def _repaired(view: dict) -> dict:
    """Return the record unchanged when complete, else a copy with trusted columns."""
    columns = _canonical_columns()
    if list(view.get("columns") or []) == columns and view.get("description"):
        return view
    return {
        **view,
        "columns": columns,
        "description": view.get("description") or _DESCRIPTION,
        "example_query": view.get("example_query") or _EXAMPLE_QUERY,
    }


def complete_staffing_view(query: str, views: list[dict]) -> list[dict]:
    """Merge the trusted staffing view for staffing-requirement questions.

    Deduplicates only v_staffing_gap records (first kept, repaired if needed),
    preserves every other retrieved view in order, and appends the trusted view
    when retrieval missed it. Other questions get the views back unchanged.
    """
    if not requires_staffing_requirement(query):
        return list(views)
    completed: list[dict] = []
    seen = False
    for view in views:
        if view.get("view_name") != STAFFING_VIEW_NAME:
            completed.append(view)
        elif not seen:
            seen = True
            completed.append(_repaired(view))
    if not seen:
        completed.append(staffing_view())
    return completed
