"""Pure builders for text_to_sql_tool result dictionaries.

``failure_kind`` is the authoritative failure classification. ``fallback_reason``
is kept on alias-only exhaustion solely for backward compatibility: it is not
proof of missing schema and must never authorize document fallback on its own.
"""
from __future__ import annotations

import re

_MISSING_SCHEMA_IDENTIFIER = re.compile(r"\bno such (?:column|table)\b", re.IGNORECASE)
SCHEMA_CAPABILITY_GAP = "schema_capability_gap"
CANDIDATE_ALIAS_REJECTED = "candidate_alias_rejected"


def latency(
    generation_ms: float, validation_ms: float, execution_ms: float, retries: int,
    candidate_calls: list[dict], attempts: list[dict],
) -> dict:
    return {
        "generation_ms": round(generation_ms, 2),
        "validation_ms": round(validation_ms, 2),
        "execution_ms": round(execution_ms, 2),
        "retries": retries,
        "candidate_llm_calls": candidate_calls,
        "attempts": attempts,
    }


def schema_guard_result(trace_id: str, coverage: dict) -> dict:
    """Canonical schema gap: early abstention with an explanation, never fallback."""
    return {
        "rows": [], "sql_used": None, "view_names": [], "trace_id": trace_id,
        "schema_status": coverage["status"],
        "missing_schema": coverage["missing_schema"],
        "explanation": " ".join(item["detail"] for item in coverage["missing_schema"]),
        "failure_kind": SCHEMA_CAPABILITY_GAP,
        "latency": latency(0.0, 0.0, 0.0, 0, [], []),
    }


def generation_unavailable_result(
    view_names: list[str], trace_id: str, reason: str | None, timings: dict
) -> dict:
    return {
        "rows": [], "sql_used": None, "view_names": view_names,
        "trace_id": trace_id, "error": "sql_generation_unavailable",
        "generation_unavailable_reason": reason,
        "latency": timings,
    }


def success_result(
    rows: list[dict], sql: str, view_names: list[str], trace_id: str, timings: dict
) -> dict:
    return {
        "rows": rows, "sql_used": sql, "view_names": view_names, "trace_id": trace_id,
        "latency": timings,
    }


def is_missing_schema_error(error: str) -> bool:
    return bool(_MISSING_SCHEMA_IDENTIFIER.search(error))


def failure_classification(generation_status: dict, failure_errors: list[str]) -> dict:
    """Classify a terminal failure; empty for every non-schema, non-alias failure.

    Alias-only exhaustion requires that no candidate reached validation, so a
    mixed run (an earlier syntax/validation failure) never becomes alias-only.
    A genuine gap requires every validation/execution error to name a missing
    column or table. A valid query with zero rows never reaches this point.
    """
    if not failure_errors and generation_status.get("failure_kind") == CANDIDATE_ALIAS_REJECTED:
        return {
            "fallback_reason": SCHEMA_CAPABILITY_GAP,
            "failure_kind": CANDIDATE_ALIAS_REJECTED,
            "candidate_rejection_reason": generation_status["candidate_rejection_reason"],
        }
    if failure_errors and all(is_missing_schema_error(e) for e in failure_errors):
        return {"fallback_reason": SCHEMA_CAPABILITY_GAP, "failure_kind": SCHEMA_CAPABILITY_GAP}
    return {}


def failure_result(
    last_error: str, generation_status: dict, failure_errors: list[str], timings: dict
) -> dict:
    return {
        "rows": [], "error": last_error,
        **failure_classification(generation_status, failure_errors),
        "latency": timings,
    }
