"""SQL candidate generation: schema guard, deterministic SQL, primary retry, fallback.

Dependencies come from the ``text_to_sql`` facade at call time so existing patches
of its ``chat``, ``SQL_FALLBACK_MODEL`` and metrics control the real behaviour.
"""
from __future__ import annotations

import hashlib
from types import ModuleType

from src.retrieval.sql_candidate_attempt import (
    RETRYABLE_PRIMARY_OUTCOMES,
    CandidateRequest,
    attempt_candidates,
)
from src.retrieval.sql_candidate_diagnostics import exhaustion_status
from src.retrieval.sql_candidate_prompt import _supervisor_comparison_sql, build_sql_prompt
from src.retrieval.sql_schema import assess_schema_coverage

FALLBACK_TRIGGERS = frozenset({
    "empty_completion", "timeout", "provider_error", "truncated_completion",
})
SCHEMA_CAPABILITY_GAP = "schema_capability_gap"


def _set_status(generation_status: dict | None, **status: object) -> None:
    if generation_status is not None:
        generation_status.update(status)


async def generate_sql(
    deps: ModuleType,
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
    """Generate at most one accepted SQL candidate list for the question."""
    if generation_status is not None:
        generation_status.clear()
    if assess_schema_coverage(query)["status"] != "full":
        # A canonical schema gap: authoritative, but early abstention, not fallback.
        _set_status(
            generation_status, status="schema_guard", reason=None,
            failure_kind=SCHEMA_CAPABILITY_GAP,
        )
        return []
    deterministic_sql = _supervisor_comparison_sql(query)
    if deterministic_sql:
        _set_status(generation_status, status="available", reason=None)
        return [deterministic_sql]
    request = CandidateRequest(
        query=query, prompt=build_sql_prompt(query, relevant_views), api_key=api_key,
        base_url=base_url, n_candidates=n_candidates,
        query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        call_timings=call_timings, diagnostics=diagnostics, case_id=diagnostic_case_id,
    )
    # Rejection reasons of every generated candidate across attempts; a parser
    # rejection counts as a reason list without the schema-capability marker.
    rejections: list[list[str]] = []

    for model_attempt in range(1, 3):
        attempt = await attempt_candidates(deps, request, model, "primary", model_attempt)
        rejections.extend(attempt.rejections)
        if attempt.candidates:
            _set_status(generation_status, status="available", reason=None)
            return attempt.candidates
        if attempt.outcome in RETRYABLE_PRIMARY_OUTCOMES and model_attempt == 1:
            continue
        if attempt.outcome in FALLBACK_TRIGGERS:
            fallback_trigger_reason = attempt.outcome
            break
        _set_status(generation_status, **exhaustion_status(rejections))
        return []

    deps.rag_sql_fallback_invocations_total.labels(trigger=fallback_trigger_reason).inc()
    attempt = await attempt_candidates(
        deps, request, deps.SQL_FALLBACK_MODEL, "fallback", 1, fallback_trigger_reason
    )
    rejections.extend(attempt.rejections)
    if attempt.candidates:
        _set_status(generation_status, status="available", reason=None)
        return attempt.candidates
    if attempt.outcome in FALLBACK_TRIGGERS:
        _set_status(generation_status, status="sql_generation_unavailable", reason=attempt.outcome)
        deps.rag_sql_generation_unavailable_total.labels(reason=attempt.outcome).inc()
    else:
        _set_status(generation_status, **exhaustion_status(rejections))
    return []
