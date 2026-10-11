"""Production-safety semantic checks on one parsed SQL candidate.

Sole owner of the floor-staff/shortfall intent helpers that the prompt builder
imports. Shared parser regexes and the alias rejection enum are imported from
their owning modules, never redefined here.
"""
from __future__ import annotations

import re

from src.retrieval.sql_candidate_aliases import _SCHEMA_CAPABILITY_REJECTION, _invents_metric_alias
from src.retrieval.sql_candidate_parser import _JOIN_PATTERN

_FLOOR_STAFF_ROLE_PREDICATE = re.compile(
    r"\b(?:\w+\.)?role\s*=\s*'floor_staff'", re.IGNORECASE
)
_FLOOR_STAFF_REFERENCE = re.compile(r"\bfloor[ -]staff\b", re.IGNORECASE)
_SUPERVISOR_REFERENCE = re.compile(r"\bsupervis(?:or|ed|ing)\b", re.IGNORECASE)
_SHIFT_INCIDENT_TOTAL_REFERENCE = re.compile(
    r"\b(?:how many|count|total)\b.*\bincidents?\b.*\bshift\s+sh-\d+\b",
    re.IGNORECASE,
)
_INCOMPATIBLE_SHIFT_ID_TYPE_JOIN = re.compile(
    r"\b(?:[A-Za-z_]\w*\.)?shift_id\s*=\s*(?:[A-Za-z_]\w*\.)?shift_type\b|"
    r"\b(?:[A-Za-z_]\w*\.)?shift_type\s*=\s*(?:[A-Za-z_]\w*\.)?shift_id\b",
    re.IGNORECASE,
)
_RETRYABLE_SEMANTIC_REJECTIONS = frozenset({
    "missing_requested_supervisor_field",
    "shift_incident_total_requires_v_incident_by_shift",
    "incompatible_shift_id_shift_type_join",
    "incident_count_grain_mismatch",
    "missing_requested_shift_id",
})
_WHICH_SHIFT_REFERENCE = re.compile(r"\bwhich\s+shift\b", re.IGNORECASE)
_SHIFT_KEYED_VIEWS = (
    "v_peak_staffing", "v_shift_details", "v_incident_severity", "v_shift_staffing_evidence",
)
_SUMMED_INCIDENT_COUNT = re.compile(
    r"\bSUM\s*\(\s*(?:[A-Za-z_]\w*\.)?incident_count\b", re.IGNORECASE
)
_SHORTFALL_REFERENCE = re.compile(
    r"\b(?:understaff(?:ed|ing)?|fall\s+short|staffing\s+(?:shortfall|deficits?)|"
    r"below\s+(?:the\s+)?required(?:\s+(?:staffing|count))?)\b",
    re.IGNORECASE,
)
_LISTING_REFERENCE = re.compile(r"\b(?:which|list|show|when|what\s+dates?)\b", re.IGNORECASE)


def _answers_requested_metrics(query: str, sql: str) -> bool:
    """Reject placeholders for a metric the question explicitly requests."""
    if "incident" not in query.lower():
        return True
    return not re.search(r"\bNULL\s+AS\s+(?:incident_)?count\b", sql, re.IGNORECASE)


def _constrains_requested_role(query: str, sql: str) -> bool:
    """Require the explicit floor-staff predicate when the question names that role."""
    if not _FLOOR_STAFF_REFERENCE.search(query):
        return True
    return bool(_FLOOR_STAFF_ROLE_PREDICATE.search(sql))


def _returns_requested_supervisor(query: str, sql: str) -> bool:
    """Require supervisor evidence when the user explicitly requests it."""
    return not _SUPERVISOR_REFERENCE.search(query) or bool(
        re.search(r"\bsupervisor\b", sql, re.IGNORECASE)
    )


def _uses_shift_incident_aggregate(query: str, sql: str) -> bool:
    """Prevent high-severity-only counts from answering an all-incidents shift total."""
    return not _SHIFT_INCIDENT_TOTAL_REFERENCE.search(query) or bool(
        re.search(r"\bFROM\s+v_incident_by_shift\b", sql, re.IGNORECASE)
    )


def _mixes_incident_grain(sql: str) -> bool:
    """Detect per-incident-type counts joined to one shift without summing them."""
    normalized = sql.casefold()
    return bool(
        _JOIN_PATTERN.search(sql)
        and "v_incident_by_venue" in normalized
        and any(view in normalized for view in _SHIFT_KEYED_VIEWS)
        and not _SUMMED_INCIDENT_COUNT.search(sql)
    )


def _returns_requested_shift_id(query: str, sql: str) -> bool:
    """Require shift_id in the SELECT list when the user asks which shift."""
    if not _WHICH_SHIFT_REFERENCE.search(query):
        return True
    select_list = re.split(r"\bFROM\b", sql, maxsplit=1, flags=re.IGNORECASE)[0]
    return bool(re.search(r"\bshift_id\b", select_list, re.IGNORECASE))


def _is_floor_staff_shortfall_listing(query: str) -> bool:
    """Identify the narrow intent served completely by ``v_staffing_gap``."""
    return bool(
        _FLOOR_STAFF_REFERENCE.search(query)
        and _SHORTFALL_REFERENCE.search(query)
        and _LISTING_REFERENCE.search(query)
    )


def _joins_peak_staffing_to_gap(sql: str) -> bool:
    """Detect the two-view join that cannot improve a shortfall listing."""
    normalized = sql.casefold()
    return bool(
        _JOIN_PATTERN.search(sql)
        and "v_peak_staffing" in normalized
        and "v_staffing_gap" in normalized
    )


def _filter_rejection_reasons(query: str, sql: str) -> list[str]:
    """Return production-safety reasons for rejecting a generated candidate."""
    reasons: list[str] = []
    if not _answers_requested_metrics(query, sql):
        reasons.append("requested_metric_null_placeholder")
    if not _constrains_requested_role(query, sql):
        reasons.append("missing_requested_floor_staff_predicate")
    if not _returns_requested_supervisor(query, sql):
        reasons.append("missing_requested_supervisor_field")
    if not _uses_shift_incident_aggregate(query, sql):
        reasons.append("shift_incident_total_requires_v_incident_by_shift")
    if _INCOMPATIBLE_SHIFT_ID_TYPE_JOIN.search(sql):
        reasons.append("incompatible_shift_id_shift_type_join")
    if _is_floor_staff_shortfall_listing(query) and _joins_peak_staffing_to_gap(sql):
        reasons.append("unnecessary_peak_staffing_join_for_shortfall_listing")
    if _invents_metric_alias(query, sql):
        reasons.append(_SCHEMA_CAPABILITY_REJECTION)
    if _mixes_incident_grain(sql):
        reasons.append("incident_count_grain_mismatch")
    if not _returns_requested_shift_id(query, sql):
        reasons.append("missing_requested_shift_id")
    return reasons
