"""Canonical curated SQL view inventory and schema-coverage assessment."""
from __future__ import annotations

import re

_INCIDENT_BY_VENUE_VIEW = {
    "view_name": "v_incident_by_venue",
    "columns": ["venue", "date", "incident_type", "incident_count"],
    "description": "Count of incidents grouped by venue, date, and incident type.",
}
_INCIDENT_BY_SHIFT_VIEW = {
    "view_name": "v_incident_by_shift",
    "columns": ["shift_id", "venue", "date", "incident_count"],
    "description": "Count of all logged incidents per shift, keyed by shift_id.",
}

# Mirrors the curated CREATE VIEW projections in structured_loader.py.
# Coverage checks use these canonical projections, not the subset retrieved by
# schema linking, so a missed Qdrant hit cannot change a full-support verdict.
CURATED_VIEW_COLUMNS = {
    "v_shift_summary": frozenset({"venue", "date", "shift_type", "total_staff"}),
    "v_incident_by_venue": frozenset({
        "venue", "date", "incident_type", "incident_count",
    }),
    "v_staffing_gap": frozenset({
        "venue", "date", "role", "count", "required_count", "gap",
    }),
    "v_incident_severity": frozenset({
        "incident_id", "shift_id", "venue", "date", "incident_type", "description",
        "severity",
    }),
    "v_peak_staffing": frozenset({
        "shift_id", "venue", "date", "staff_count", "shift_type", "avg_staff",
    }),
    "v_shift_details": frozenset({
        "shift_id", "venue", "date", "staff_count", "shift_type", "supervisor",
    }),
    "v_incident_by_shift": frozenset({"shift_id", "venue", "date", "incident_count"}),
    "v_shift_staffing_evidence": frozenset({
        "shift_id", "venue", "date", "role", "count", "required_count", "gap",
    }),
}


def assess_schema_coverage(query: str) -> dict:
    """Identify requested facts or joins that the curated views cannot prove."""
    normalized = query.casefold()
    missing: list[dict[str, str]] = []
    status = "full"

    if re.search(r"\bsupervis(?:or|ed|ing)\b", normalized) and not any(
        "supervisor" in columns for columns in CURATED_VIEW_COLUMNS.values()
    ):
        missing.append({
            "kind": "field", "name": "supervisor",
            "detail": "No curated view exposes the supervisor field for a shift or incident.",
        })
        status = "unsupported_by_schema"

    incident_request = bool(re.search(r"\bincidents?\b|rsa[- ]interventions?", normalized))
    if incident_request and re.search(r"\b(?:rates?|percentages?)\b", normalized):
        missing.append({
            "kind": "field", "name": "intervention_rate_denominator",
            "detail": (
                "The curated views can count RSA interventions by evening shift but define "
                "no denominator for an RSA-intervention rate."
            ),
        })
        status = "unsupported_by_schema"

    high_severity = bool(re.search(r"high[- ]severity", normalized))
    understaffed = bool(re.search(
        r"understaff|low staffing|staffing shortfall|below (?:the )?required staffing",
        normalized,
    ))
    if incident_request and high_severity and understaffed:
        incident_columns = CURATED_VIEW_COLUMNS["v_incident_severity"]
        staffing_columns = CURATED_VIEW_COLUMNS["v_staffing_gap"]
        if "shift_id" in incident_columns and "shift_id" not in staffing_columns:
            missing.append({
                "kind": "relationship", "name": "complete_incident_to_staffing_shift_evidence",
                "detail": (
                    "v_shift_staffing_evidence links a shift only when a staffing record exists; "
                    "it cannot establish staffing for every incident shift."
                ),
            })
            if status == "full":
                status = "partial_schema_support"

    shift_incident_count = incident_request and bool(
        re.search(r"\bincidents?\b.*\b(?:on|per|by)\s+(?:the\s+)?shift\b", normalized)
        or ("how many" in normalized and re.search(r"\bshift\s+sh-\d+\b", normalized))
    )
    if shift_incident_count:
        aggregate = CURATED_VIEW_COLUMNS["v_incident_by_venue"]
        severity = CURATED_VIEW_COLUMNS["v_incident_severity"]
        by_shift = CURATED_VIEW_COLUMNS["v_incident_by_shift"]
        if "shift_id" not in aggregate and "incident_count" not in severity and not {
            "shift_id", "incident_count",
        }.issubset(by_shift):
            missing.append({
                "kind": "relationship", "name": "all_incidents_by_shift_id",
                "detail": (
                    "v_incident_by_venue counts all incidents but has no shift_id; "
                    "v_incident_severity has shift_id but contains only high-severity incidents."
                ),
            })
            status = "unsupported_by_schema"

    return {"status": status, "missing_schema": missing}
