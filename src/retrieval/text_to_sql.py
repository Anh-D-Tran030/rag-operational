"""Text-to-SQL: generate, validate, execute SQL against curated views."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time

import openai
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.config import Settings
from src.llm_client import ChatCompletionResult, chat
from src.observability.metrics import (
    rag_latency_by_component,
    rag_sql_execution_errors,
    rag_sql_fallback_invocations_total,
    rag_sql_generation_attempts_total,
    rag_sql_generation_unavailable_total,
    rag_sql_partial_schema_support_total,
    rag_sql_schema_guard_abstention_total,
    rag_sql_unsupported_by_schema_total,
)

SQL_LLM_TIMEOUT_S = 120.0
SIMPLE_SQL_LLM_TIMEOUT_S = 45.0
SQL_VALIDATION_TIMEOUT_S = 5.0
SQL_EXECUTION_TIMEOUT_S = 5.0
SQL_FALLBACK_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
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

_DML_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|MERGE)\b",
    re.IGNORECASE,
)
_JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_CANDIDATE_SEPARATOR = re.compile(r"^\s*--\s*candidate\s*$", re.IGNORECASE | re.MULTILINE)
_SQL_START = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_POSTGRES_DATE_LITERAL = re.compile(r"\bDATE\s+'(\d{4}-\d{2}-\d{2})'", re.IGNORECASE)
_FETCH_FIRST = re.compile(r"\bFETCH\s+FIRST\s+(\d+)\s+ROW(?:S)?\s+ONLY\b", re.IGNORECASE)
_LABELED_SQL = re.compile(
    r"^\s*(?:here is (?:the )?)?sql\s*:\s*((?:SELECT|WITH)\b.*)$",
    re.IGNORECASE | re.DOTALL,
)
_DANGLING_LIMIT = re.compile(r"\s+LIMIT\s*;?\s*$", re.IGNORECASE)
_SELECT_ONLY = re.compile(r"^\s*SELECT\s*;?\s*$", re.IGNORECASE)
_DANGLING_AND = re.compile(r"\bAND\s*;?\s*$", re.IGNORECASE)
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
_SUPERVISOR_COMPARISON_QUERY = re.compile(
    r"\bday of the (?P<incident_type>[a-z]+(?:[- ][a-z]+)*) incident\b.*"
    r"\bsame person who supervised shift (?P<shift_id>SH-\d+)\b",
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
_OUTPUT_ALIAS = re.compile(r"\bAS\s+([A-Za-z_]\w*)", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'([^']*)'")
_WORD = re.compile(r"[a-z0-9]+")
# Words that name a computation over curated columns rather than a stored fact.
_DERIVED_MEASURE_WORDS = frozenset({
    "total", "count", "number", "average", "avg", "maximum", "max", "minimum", "min",
    "same", "difference", "rate", "met", "meets", "required", "requirement", "linked",
})
# A retry cannot create a column that no curated view has, so this is terminal.
_SCHEMA_CAPABILITY_REJECTION = "unsupported_metric_alias"
_MISSING_SCHEMA_IDENTIFIER = re.compile(r"\bno such (?:column|table)\b", re.IGNORECASE)
_SHORTFALL_REFERENCE = re.compile(
    r"\b(?:understaff(?:ed|ing)?|fall\s+short|staffing\s+(?:shortfall|deficits?)|"
    r"below\s+(?:the\s+)?required(?:\s+(?:staffing|count))?)\b",
    re.IGNORECASE,
)
_LISTING_REFERENCE = re.compile(r"\b(?:which|list|show|when|what\s+dates?)\b", re.IGNORECASE)
_ROLE_LITERAL_NORMALIZATION = (
    (re.compile(r"'floor'", re.IGNORECASE), "'floor_staff'"),
    (re.compile(r"'floor[ -]staff'", re.IGNORECASE), "'floor_staff'"),
    (re.compile(r"'bar[ -]staff'", re.IGNORECASE), "'bar_staff'"),
    (re.compile(r"'security'", re.IGNORECASE), "'security'"),
)
_INCIDENT_TYPE_LITERAL = re.compile(
    r"(\b(?:[A-Za-z_]\w*\.)?incident_type\s*=\s*')([^']*)(')", re.IGNORECASE
)
_DIAGNOSTIC_RAW_RESPONSE_LIMIT = 512
_DIAGNOSTIC_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|authorization)\s*[:=]\s*\S+"),
)

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


def _parse_sql(raw: str) -> tuple[str, str | None]:
    """Return a SQLite SELECT statement and a diagnostic rejection reason."""
    match = _SQL_FENCE.fullmatch(raw.strip())
    sql = match.group(1).strip() if match else raw.strip()
    if not sql:
        return "", "empty_response"
    labeled = _LABELED_SQL.fullmatch(sql)
    if labeled:
        sql = labeled.group(1).strip()
    if not _SQL_START.match(sql):
        return "", "not_select_or_with"
    sql = _POSTGRES_DATE_LITERAL.sub(r"'\1'", sql)
    sql = _FETCH_FIRST.sub(r"LIMIT \1", sql)
    for pattern, replacement in _ROLE_LITERAL_NORMALIZATION:
        sql = pattern.sub(replacement, sql)
    sql = _INCIDENT_TYPE_LITERAL.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2).replace('-', ' ')}{match.group(3)} COLLATE NOCASE"
        ),
        sql,
    )
    # A model occasionally emits LIMIT without a count. Removing that incomplete
    # optional clause preserves the complete SELECT rather than sending invalid SQL.
    sql = _DANGLING_LIMIT.sub("", sql)
    if _SELECT_ONLY.fullmatch(sql):
        return "", "bare_select"
    if _DANGLING_AND.search(sql):
        return "", "dangling_and"
    if sql.rstrip().endswith(","):
        return "", "trailing_comma"
    if sql.count("'") % 2:
        return "", "unbalanced_quotes"
    if not _has_balanced_parentheses(sql):
        return "", "unbalanced_parentheses"
    return sql, None


def _extract_sql(raw: str) -> str:
    """Return one SQLite SELECT statement, rejecting surrounding prose."""
    return _parse_sql(raw)[0]


def _has_balanced_parentheses(sql: str) -> bool:
    """Reject truncated SQL while respecting parentheses inside string literals."""
    depth = 0
    in_string = False
    index = 0
    while index < len(sql):
        character = sql[index]
        if character == "'":
            if in_string and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 1
            else:
                in_string = not in_string
        elif not in_string and character == "(":
            depth += 1
        elif not in_string and character == ")":
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return not in_string and depth == 0


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


def _stem(word: str) -> str:
    """Fold a simple plural so 'incidents' matches the curated 'incident'."""
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def _words(text: str) -> set[str]:
    return {_stem(word) for word in _WORD.findall(text.casefold())}


_CURATED_VOCABULARY = frozenset(
    word
    for view, columns in CURATED_VIEW_COLUMNS.items()
    for name in (view, *columns)
    for word in _words(name)
)


def _invents_metric_alias(query: str, sql: str) -> bool:
    """Detect a curated column relabelled as a requested fact no curated view stores.

    ``staff_count AS peak_patron_count`` answers a patron question with staff data.
    Alias words that name a curated column, a derived measure, or a filter value
    in the SQL's own literals (``role = 'security'`` for ``security_count``) are
    legitimate; any other question word in an alias is an invented metric.
    """
    question = _words(query)
    literals = set().union(*(_words(lit) for lit in _STRING_LITERAL.findall(sql)))
    allowed = _CURATED_VOCABULARY | _DERIVED_MEASURE_WORDS | literals
    return any(
        len(word) >= 4 and word in question and word not in allowed
        for alias in _OUTPUT_ALIAS.findall(sql)
        for word in _words(alias)
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


def _safe_diagnostic_response(raw: str) -> dict[str, str | int]:
    """Provide bounded, credential-redacted model output for opt-in diagnostics."""
    redacted = raw
    for pattern in _DIAGNOSTIC_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    excerpt = redacted[:_DIAGNOSTIC_RAW_RESPONSE_LIMIT]
    if len(redacted) > _DIAGNOSTIC_RAW_RESPONSE_LIMIT:
        excerpt += "…[truncated]"
    return {
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "char_count": len(raw),
        "excerpt": excerpt,
    }


def _provider_error_details(exc: Exception) -> tuple[str, dict[str, str | int | None]]:
    """Classify provider errors without retaining potentially sensitive messages."""
    if isinstance(exc, openai.APITimeoutError):
        outcome = "timeout"
        error_class = "api_timeout"
    elif isinstance(exc, openai.APIConnectionError):
        outcome = "provider_error"
        error_class = "api_connection_error"
    elif isinstance(exc, openai.RateLimitError):
        outcome = "provider_error"
        error_class = "rate_limit"
    elif isinstance(exc, openai.APIStatusError):
        outcome = "provider_error"
        error_class = "api_status_error"
    else:
        outcome = "provider_error"
        error_class = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    return outcome, {"class": error_class, "status": status if isinstance(status, int) else None}


def _completion_text_and_evidence(
    completion: ChatCompletionResult | str,
    model: str,
    max_tokens: int,
) -> tuple[str, dict[str, object]]:
    """Normalize real and mocked completions into bounded diagnostic evidence."""
    if isinstance(completion, ChatCompletionResult):
        return completion.text, {
            "response_id": completion.response_id,
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "configured_output_token_limit": completion.configured_output_token_limit,
            "model": completion.model or model,
            "provider": completion.provider,
        }
    return str(completion or ""), {
        "response_id": None,
        "finish_reason": None,
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "configured_output_token_limit": max_tokens,
        "model": model,
        "provider": None,
    }


def _extract_sql_envelope(raw: str) -> tuple[str, str | None]:
    """Extract one SQL statement from the strict structured response envelope."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "", "malformed_structured_output"
    if not isinstance(payload, dict) or set(payload) != {"sql"}:
        return "", "incomplete_structured_output"
    sql = payload["sql"]
    if not isinstance(sql, str) or not sql.strip():
        return "", "incomplete_structured_output"
    return sql, None


async def generate_sql(
    query: str,
    relevant_views: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    n_candidates: int = 3,
    call_timings: list[dict] | None = None,
    diagnostics: list[dict] | None = None,
    diagnostic_case_id: int | None = None,
    generation_status: dict | None = None,
) -> list[str]:
    """Call the cloud LLM to generate one structured SQL SELECT candidate.

    Prompt includes the query and relevant view definitions.
    Returns a list containing at most one parsed SQL string.
    """
    if generation_status is not None:
        generation_status.clear()
    if assess_schema_coverage(query)["status"] != "full":
        if generation_status is not None:
            generation_status.update(status="schema_guard", reason=None)
        return []
    deterministic_sql = _supervisor_comparison_sql(query)
    if deterministic_sql:
        if generation_status is not None:
            generation_status.update(status="available", reason=None)
        return [deterministic_sql]
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
    prompt = (
        "Generate one SQL SELECT statement that answers the question below.\n"
        f"Use ONLY the views listed. Do NOT use raw tables. Use SQLite syntax: quote dates as "
        f"'YYYY-MM-DD' and use LIMIT, never DATE literals or FETCH FIRST. Each statement must "
        f"return every requested base metric and preserve every identity or detail field explicitly "
        f"requested by the user; select views that expose those fields (for example, 'which "
        f"shift' requires shift_id). When returning a derived comparison, boolean, or status, also "
        f"return the available input metrics used to establish it. Prefer one self-contained view "
        f"when it contains all requested outputs and filters. Join views only when needed for "
        f"requested information, and only on semantically compatible entity keys; never equate "
        f"unrelated categorical domains such as shift type and staffing role. Never substitute NULL "
        f"as a requested metric.\n\n"
        f"Available views:\n{view_defs}\n\n"
        f"{role_constraint}"
        f"{shortfall_listing_constraint}"
        f"Question: {query}\n\n"
        "Return exactly one JSON object with this shape: "
        '{"sql":"<one SQL SELECT statement>"}. The sql value must contain one statement '
        "only, with no Markdown, explanation, or prose."
    )
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    # Rejection reasons of every generated candidate across attempts; a parser
    # rejection counts as a reason list without the schema-capability marker.
    rejections: list[list[str]] = []

    def exhausted_reason() -> str | None:
        """Name a schema-capability gap only when every candidate hit it."""
        if rejections and all(_SCHEMA_CAPABILITY_REJECTION in r for r in rejections):
            return "schema_capability_gap"
        return None

    async def attempt_model(
        attempt_model_name: str,
        phase: str,
        model_attempt: int,
        fallback_trigger_reason: str | None = None,
    ) -> tuple[list[str], str]:
        call_started = time.monotonic()
        timed_out = False
        provider_outcome = "provider_error"
        provider_error: dict[str, str | int | None] | None = None
        timeout_error_class: str | None = None
        completion_evidence: dict[str, object] = {
            "response_id": None,
            "finish_reason": None,
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "configured_output_token_limit": 512 if n_candidates == 1 else 1024,
            "model": attempt_model_name,
            "provider": None,
        }
        timing: dict | None = None
        timeout_s = SIMPLE_SQL_LLM_TIMEOUT_S if n_candidates == 1 else SQL_LLM_TIMEOUT_S
        output_token_limit = 512 if n_candidates == 1 else 1024
        try:
            completion = await asyncio.wait_for(chat(
                prompt=prompt,
                model=attempt_model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=output_token_limit,
                return_metadata=True,
                response_format=SQL_RESPONSE_FORMAT,
            ), timeout=timeout_s)
            raw, completion_evidence = _completion_text_and_evidence(
                completion, attempt_model_name, output_token_limit
            )
        except TimeoutError:
            timed_out = True
            timeout_error_class = "TimeoutError"
            raw = ""
        except Exception as exc:
            raw = ""
            provider_outcome, provider_error = _provider_error_details(exc)
            if provider_outcome == "timeout":
                timed_out = True
                timeout_error_class = type(exc).__name__
        finally:
            elapsed_ms = round((time.monotonic() - call_started) * 1000, 2)
            rag_latency_by_component.labels(component="sql_candidate_llm").observe(
                elapsed_ms / 1000
            )
            if call_timings is not None:
                timing = {
                    "latency_ms": elapsed_ms,
                    "timed_out": timed_out,
                    "timeout_s": timeout_s,
                    "configured_output_token_limit": output_token_limit,
                    "requested_candidates": n_candidates,
                    "phase": phase,
                    "model": attempt_model_name,
                }
                call_timings.append(timing)
        source = "structured_json"
        sql_text, structured_output_error = _extract_sql_envelope(raw) if raw else ("", None)
        blocks = [sql_text] if structured_output_error is None and sql_text else []
        parsed_candidates = (
            []
            if timed_out or provider_error or completion_evidence["finish_reason"] == "length"
            else [_parse_sql(block) for block in blocks]
        )
        candidates = [candidate for candidate, _ in parsed_candidates]
        if not candidates:
            candidates = []
        candidate_diagnostics = []
        valid_candidates = []
        for candidate_index, (candidate, parser_rejection) in enumerate(parsed_candidates, start=1):
            filter_rejections = [] if parser_rejection else _filter_rejection_reasons(query, candidate)
            candidate_diagnostics.append({
                "candidate_index": candidate_index,
                "parser_rejection": parser_rejection,
                "filter_rejections": filter_rejections,
            })
            if candidate and not filter_rejections:
                valid_candidates.append(candidate)
            else:
                rejections.append([parser_rejection] if parser_rejection else filter_rejections)
        semantic_rejection = any(
            set(item["filter_rejections"]) & _RETRYABLE_SEMANTIC_REJECTIONS
            for item in candidate_diagnostics
        )
        outcome = (
            "timeout" if timed_out else
            provider_outcome if provider_error else
            "truncated_completion" if completion_evidence["finish_reason"] == "length" else
            "empty_completion" if not raw else
            structured_output_error if structured_output_error else
            "accepted_candidates" if valid_candidates else
            "semantic_candidate_rejected" if semantic_rejection else
            "no_valid_candidates"
        )
        rag_sql_generation_attempts_total.labels(
            phase=phase, model=attempt_model_name, outcome=outcome
        ).inc()
        if timing is not None:
            timing.update(
                outcome=outcome,
                fallback_trigger_reason=fallback_trigger_reason,
                provider_error=provider_error,
            )
        if diagnostics is not None:
            diagnostics.append({
                "case_id": diagnostic_case_id,
                "query_sha256": query_sha256,
                "model_attempt": model_attempt,
                "phase": phase,
                "model": attempt_model_name,
                "fallback_trigger_reason": fallback_trigger_reason,
                "elapsed_ms": elapsed_ms,
                "timeout_s": timeout_s,
                "timed_out": timed_out,
                "timeout_error_class": timeout_error_class,
                "completion_evidence": completion_evidence,
                "raw_response": _safe_diagnostic_response(raw),
                "provider_error": provider_error,
                "structured_output_error": structured_output_error,
                "parse_source": source,
                "candidate_diagnostics": candidate_diagnostics,
                "outcome": outcome,
                "will_retry": phase == "primary" and outcome in {
                    "empty_completion", "semantic_candidate_rejected",
                } and model_attempt == 1,
            })
        if valid_candidates:
            return valid_candidates[:n_candidates], outcome
        return [], outcome

    for model_attempt in range(1, 3):
        valid_candidates, outcome = await attempt_model(model, "primary", model_attempt)
        if valid_candidates:
            if generation_status is not None:
                generation_status.update(status="available", reason=None)
            return valid_candidates
        if outcome in {"empty_completion", "semantic_candidate_rejected"} and model_attempt == 1:
            continue
        if outcome in {"empty_completion", "timeout", "provider_error", "truncated_completion"}:
            fallback_trigger_reason = outcome
            break
        if generation_status is not None:
            generation_status.update(status="no_valid_candidates", reason=exhausted_reason())
        return []

    rag_sql_fallback_invocations_total.labels(trigger=fallback_trigger_reason).inc()
    valid_candidates, outcome = await attempt_model(
        SQL_FALLBACK_MODEL, "fallback", 1, fallback_trigger_reason
    )
    if valid_candidates:
        if generation_status is not None:
            generation_status.update(status="available", reason=None)
        return valid_candidates
    if outcome in {"empty_completion", "timeout", "provider_error", "truncated_completion"}:
        if generation_status is not None:
            generation_status.update(status="sql_generation_unavailable", reason=outcome)
        rag_sql_generation_unavailable_total.labels(reason=outcome).inc()
    elif generation_status is not None:
        generation_status.update(status="no_valid_candidates", reason=exhausted_reason())
    return []


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


async def validate_sql(sql: str, db_url: str) -> tuple[bool, str]:
    """Validate a SQL statement before execution.

    Rejects DML statements and queries with more than 3 JOINs.
    Runs EXPLAIN to catch syntax errors. Returns (is_valid, error_message).
    """
    if _DML_PATTERN.match(sql):
        return False, "DML statements are not permitted; only SELECT is allowed."
    if not _SQL_START.match(sql):
        return False, "Only a single SELECT or WITH statement is allowed."
    join_count = len(_JOIN_PATTERN.findall(sql))
    if join_count > 3:
        return False, f"Query join depth {join_count} exceeds maximum of 3."
    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(f"EXPLAIN {sql}"))
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        await engine.dispose()


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
    simple_sql: bool = False,
) -> dict:
    """Orchestrate generate → validate → execute with self-correction.

    Tries up to max_attempts candidates. Returns:
      {rows, sql_used, view_names, trace_id} on success.
      {rows: [], error: str} after max_attempts exhausted.
    """
    coverage = assess_schema_coverage(query)
    if coverage["status"] != "full":
        rag_sql_schema_guard_abstention_total.inc()
        if coverage["status"] == "partial_schema_support":
            rag_sql_partial_schema_support_total.inc()
        else:
            rag_sql_unsupported_by_schema_total.inc()
        explanation = " ".join(item["detail"] for item in coverage["missing_schema"])
        return {
            "rows": [], "sql_used": None, "view_names": [], "trace_id": trace_id,
            "schema_status": coverage["status"],
            "missing_schema": coverage["missing_schema"],
            "explanation": explanation,
            "latency": {
                "generation_ms": 0.0, "validation_ms": 0.0,
                "execution_ms": 0.0, "retries": 0,
                "candidate_llm_calls": [], "attempts": [],
            },
        }
    view_names = [v.get("view_name", "") for v in relevant_views]
    generation_ms = 0.0
    last_error = "no candidates generated"
    validation_ms = 0.0
    execution_ms = 0.0
    candidate_calls: list[dict] = []
    generation_status: dict = {}
    attempts: list[dict] = []
    failure_errors: list[str] = []
    tried = 0
    # A retry after a schema-gap error can produce a schema-valid query on the
    # wrong view (golden 13: SELECT * FROM v_peak_staffing for a sign-off
    # question) whose rows would silently override the gap signal. Stop instead.
    schema_gap_seen = False
    while tried < max_attempts:
        execution_failed = False
        generation_started = time.monotonic()
        candidates = await generate_sql(
            query=query,
            relevant_views=relevant_views,
            model=settings.cloud_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            n_candidates=1 if simple_sql else max_attempts,
            call_timings=candidate_calls,
            generation_status=generation_status,
        )
        generation_ms += (time.monotonic() - generation_started) * 1000
        if not candidates:
            if generation_status.get("status") == "sql_generation_unavailable":
                return {
                    "rows": [], "sql_used": None, "view_names": view_names,
                    "trace_id": trace_id, "error": "sql_generation_unavailable",
                    "generation_unavailable_reason": generation_status["reason"],
                    "latency": {
                        "generation_ms": round(generation_ms, 2),
                        "validation_ms": round(validation_ms, 2),
                        "execution_ms": round(execution_ms, 2),
                        "retries": max(0, tried - 1),
                        "candidate_llm_calls": candidate_calls,
                        "attempts": attempts,
                    },
                }
            break
        for sql in candidates[:max_attempts - tried]:
            tried += 1
            validation_started = time.monotonic()
            validation_timed_out = False
            try:
                valid, err = await asyncio.wait_for(
                    validate_sql(sql, settings.db_url), timeout=SQL_VALIDATION_TIMEOUT_S
                )
            except TimeoutError:
                valid, err = False, "SQL validation timed out"
                validation_timed_out = True
            elapsed = (time.monotonic() - validation_started) * 1000
            validation_ms += elapsed
            rag_latency_by_component.labels(component="sql_validation").observe(elapsed / 1000)
            attempt = {
                "validation_ms": round(elapsed, 2),
                "validation_timed_out": validation_timed_out,
            }
            if not valid:
                last_error = err
                failure_errors.append(err)
                attempts.append(attempt)
                if _MISSING_SCHEMA_IDENTIFIER.search(err):
                    schema_gap_seen = True
                    break
                continue
            execution_started = time.monotonic()
            try:
                rows = await asyncio.wait_for(
                    execute_sql(sql, settings.db_url), timeout=SQL_EXECUTION_TIMEOUT_S
                )
                elapsed = (time.monotonic() - execution_started) * 1000
                execution_ms += elapsed
                rag_latency_by_component.labels(component="sql_execution").observe(elapsed / 1000)
                attempt["execution_ms"] = round(elapsed, 2)
                attempt["execution_timed_out"] = False
                attempts.append(attempt)
                return {
                    "rows": rows, "sql_used": sql, "view_names": view_names, "trace_id": trace_id,
                    "latency": {
                        "generation_ms": round(generation_ms, 2),
                        "validation_ms": round(validation_ms, 2),
                        "execution_ms": round(execution_ms, 2),
                        "retries": tried - 1,
                        "candidate_llm_calls": candidate_calls,
                        "attempts": attempts,
                    },
                }
            except (RuntimeError, TimeoutError) as exc:
                elapsed = (time.monotonic() - execution_started) * 1000
                execution_ms += elapsed
                rag_latency_by_component.labels(component="sql_execution").observe(elapsed / 1000)
                attempt["execution_ms"] = round(elapsed, 2)
                attempt["execution_timed_out"] = isinstance(exc, TimeoutError)
                attempts.append(attempt)
                rag_sql_execution_errors.inc()
                last_error = (
                    "SQL execution timed out" if isinstance(exc, TimeoutError) else str(exc)
                )
                failure_errors.append(last_error)
                execution_failed = True
                if _MISSING_SCHEMA_IDENTIFIER.search(last_error):
                    schema_gap_seen = True
                    break
        if execution_failed or schema_gap_seen:
            break
        if not simple_sql:
            break

    failure: dict = {"rows": [], "error": last_error}
    # Only a schema-capability signal may send the caller to document retrieval:
    # the SQL needed a fact no curated view stores. A valid query with zero rows
    # returned above as a success and never reaches this point.
    schema_gap = generation_status.get("reason") == "schema_capability_gap" or bool(
        failure_errors and all(_MISSING_SCHEMA_IDENTIFIER.search(e) for e in failure_errors)
    )
    if schema_gap:
        failure["fallback_reason"] = "schema_capability_gap"
    return {
        **failure,
        "latency": {
            "generation_ms": round(generation_ms, 2),
            "validation_ms": round(validation_ms, 2),
            "execution_ms": round(execution_ms, 2),
            "retries": max(0, tried - 1),
            "candidate_llm_calls": candidate_calls,
            "attempts": attempts,
        },
    }
