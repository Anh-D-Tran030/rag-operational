"""One bounded SQL-provider call, parsed and filtered into candidates.

Dependencies (``chat``, timeouts, response format and metrics) come from the
``text_to_sql`` facade at call time, so patches on the facade control the call.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from types import ModuleType

from src.retrieval.sql_candidate_diagnostics import (
    _completion_text_and_evidence,
    _provider_error_details,
    diagnostic_record,
    empty_completion_evidence,
    timing_record,
)
from src.retrieval.sql_candidate_filters import (
    _RETRYABLE_SEMANTIC_REJECTIONS,
    _filter_rejection_reasons,
)
from src.retrieval.sql_candidate_parser import _extract_sql_envelope, _parse_sql

RETRYABLE_PRIMARY_OUTCOMES = frozenset({"empty_completion", "semantic_candidate_rejected"})


@dataclass(frozen=True)
class CandidateRequest:
    """Inputs shared by every provider attempt for one question."""

    query: str
    prompt: str
    api_key: str
    base_url: str
    n_candidates: int
    query_sha256: str
    call_timings: list[dict] | None
    diagnostics: list[dict] | None
    case_id: int | None


@dataclass(frozen=True)
class AttemptResult:
    candidates: list[str]
    outcome: str
    rejections: list[list[str]]


async def attempt_candidates(
    deps: ModuleType, request: CandidateRequest, model: str, phase: str,
    model_attempt: int, fallback_trigger_reason: str | None = None,
) -> AttemptResult:
    """Run one provider call and return its accepted candidates and rejections."""
    n_candidates = request.n_candidates
    output_token_limit = 512 if n_candidates == 1 else 1024
    timeout_s = deps.SIMPLE_SQL_LLM_TIMEOUT_S if n_candidates == 1 else deps.SQL_LLM_TIMEOUT_S
    call_started = time.monotonic()
    timed_out = False
    provider_outcome = "provider_error"
    provider_error: dict[str, str | int | None] | None = None
    timeout_error_class: str | None = None
    completion_evidence = empty_completion_evidence(model, output_token_limit)
    timing: dict | None = None
    raw = ""
    try:
        completion = await asyncio.wait_for(deps.chat(
            prompt=request.prompt, model=model, api_key=request.api_key,
            base_url=request.base_url, max_tokens=output_token_limit,
            return_metadata=True, response_format=deps.SQL_RESPONSE_FORMAT,
        ), timeout=timeout_s)
        raw, completion_evidence = _completion_text_and_evidence(
            completion, model, output_token_limit
        )
    except TimeoutError:
        timed_out = True
        timeout_error_class = "TimeoutError"
    except Exception as exc:
        provider_outcome, provider_error = _provider_error_details(exc)
        if provider_outcome == "timeout":
            timed_out = True
            timeout_error_class = type(exc).__name__
    finally:
        elapsed_ms = round((time.monotonic() - call_started) * 1000, 2)
        deps.rag_latency_by_component.labels(component="sql_candidate_llm").observe(
            elapsed_ms / 1000
        )
        if request.call_timings is not None:
            timing = timing_record(
                latency_ms=elapsed_ms, timed_out=timed_out, timeout_s=timeout_s,
                output_token_limit=output_token_limit, n_candidates=n_candidates,
                phase=phase, model=model,
            )
            request.call_timings.append(timing)
    sql_text, structured_output_error = _extract_sql_envelope(raw) if raw else ("", None)
    blocks = [sql_text] if structured_output_error is None and sql_text else []
    truncated = completion_evidence["finish_reason"] == "length"
    parsed = [] if timed_out or provider_error or truncated else [_parse_sql(b) for b in blocks]
    candidate_diagnostics: list[dict] = []
    valid_candidates: list[str] = []
    rejections: list[list[str]] = []
    for candidate_index, (candidate, parser_rejection) in enumerate(parsed, start=1):
        filter_rejections = (
            [] if parser_rejection else _filter_rejection_reasons(request.query, candidate)
        )
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
        "truncated_completion" if truncated else
        "empty_completion" if not raw else
        structured_output_error if structured_output_error else
        "accepted_candidates" if valid_candidates else
        "semantic_candidate_rejected" if semantic_rejection else
        "no_valid_candidates"
    )
    deps.rag_sql_generation_attempts_total.labels(
        phase=phase, model=model, outcome=outcome
    ).inc()
    if timing is not None:
        timing.update(
            outcome=outcome, fallback_trigger_reason=fallback_trigger_reason,
            provider_error=provider_error,
        )
    if request.diagnostics is not None:
        request.diagnostics.append(diagnostic_record(
            raw=raw, case_id=request.case_id, query_sha256=request.query_sha256,
            model_attempt=model_attempt, phase=phase, model=model,
            fallback_trigger_reason=fallback_trigger_reason, elapsed_ms=elapsed_ms,
            timeout_s=timeout_s, timed_out=timed_out, timeout_error_class=timeout_error_class,
            completion_evidence=completion_evidence, provider_error=provider_error,
            structured_output_error=structured_output_error, parse_source="structured_json",
            candidate_diagnostics=candidate_diagnostics, outcome=outcome,
            will_retry=phase == "primary" and outcome in RETRYABLE_PRIMARY_OUTCOMES
            and model_attempt == 1,
        ))
    return AttemptResult(valid_candidates[:n_candidates], outcome, rejections)
