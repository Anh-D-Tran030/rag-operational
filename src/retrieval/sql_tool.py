"""Generate -> validate -> execute orchestration with bounded self-correction.

``generate_sql``, ``validate_sql``, ``execute_sql``, timeouts and metrics come
from the ``text_to_sql`` facade at call time, so patches on the facade apply.
"""
from __future__ import annotations

import asyncio
import time
from types import ModuleType

from src.config import Settings
from src.retrieval.sql_schema import assess_schema_coverage
from src.retrieval.sql_tool_results import (
    failure_result,
    generation_unavailable_result,
    is_missing_schema_error,
    latency,
    schema_guard_result,
    success_result,
)


def _observe(deps: ModuleType, component: str, elapsed_ms: float) -> None:
    deps.rag_latency_by_component.labels(component=component).observe(elapsed_ms / 1000)


async def text_to_sql_tool(
    deps: ModuleType,
    query: str,
    relevant_views: list[dict],
    settings: Settings,
    trace_id: str,
    max_attempts: int = 3,
    simple_sql: bool = False,
) -> dict:
    """Try up to max_attempts candidates; return rows on success, else a classified failure."""
    coverage = assess_schema_coverage(query)
    if coverage["status"] != "full":
        deps.rag_sql_schema_guard_abstention_total.inc()
        if coverage["status"] == "partial_schema_support":
            deps.rag_sql_partial_schema_support_total.inc()
        else:
            deps.rag_sql_unsupported_by_schema_total.inc()
        return schema_guard_result(trace_id, coverage)
    view_names = [v.get("view_name", "") for v in relevant_views]
    generation_ms = validation_ms = execution_ms = 0.0
    last_error = "no candidates generated"
    candidate_calls: list[dict] = []
    generation_status: dict = {}
    attempts: list[dict] = []
    failure_errors: list[str] = []
    tried = 0

    def timings(retries: int) -> dict:
        return latency(
            generation_ms, validation_ms, execution_ms, retries, candidate_calls, attempts
        )

    # A retry after a schema-gap error can produce a schema-valid query on the
    # wrong view (golden 13: SELECT * FROM v_peak_staffing for a sign-off
    # question) whose rows would silently override the gap signal. Stop instead.
    schema_gap_seen = False
    while tried < max_attempts:
        execution_failed = False
        generation_started = time.monotonic()
        candidates = await deps.generate_sql(
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
                return generation_unavailable_result(
                    view_names, trace_id, generation_status["reason"], timings(max(0, tried - 1))
                )
            break
        for sql in candidates[:max_attempts - tried]:
            tried += 1
            validation_started = time.monotonic()
            validation_timed_out = False
            try:
                valid, err = await asyncio.wait_for(
                    deps.validate_sql(sql, settings.db_url), timeout=deps.SQL_VALIDATION_TIMEOUT_S
                )
            except TimeoutError:
                valid, err = False, "SQL validation timed out"
                validation_timed_out = True
            elapsed = (time.monotonic() - validation_started) * 1000
            validation_ms += elapsed
            _observe(deps, "sql_validation", elapsed)
            attempt = {
                "validation_ms": round(elapsed, 2),
                "validation_timed_out": validation_timed_out,
            }
            if not valid:
                last_error = err
                failure_errors.append(err)
                attempts.append(attempt)
                if is_missing_schema_error(err):
                    schema_gap_seen = True
                    break
                continue
            execution_started = time.monotonic()
            try:
                rows = await asyncio.wait_for(
                    deps.execute_sql(sql, settings.db_url), timeout=deps.SQL_EXECUTION_TIMEOUT_S
                )
                elapsed = (time.monotonic() - execution_started) * 1000
                execution_ms += elapsed
                _observe(deps, "sql_execution", elapsed)
                attempt["execution_ms"] = round(elapsed, 2)
                attempt["execution_timed_out"] = False
                attempts.append(attempt)
                return success_result(rows, sql, view_names, trace_id, timings(tried - 1))
            except (RuntimeError, TimeoutError) as exc:
                elapsed = (time.monotonic() - execution_started) * 1000
                execution_ms += elapsed
                _observe(deps, "sql_execution", elapsed)
                attempt["execution_ms"] = round(elapsed, 2)
                attempt["execution_timed_out"] = isinstance(exc, TimeoutError)
                attempts.append(attempt)
                deps.rag_sql_execution_errors.inc()
                last_error = (
                    "SQL execution timed out" if isinstance(exc, TimeoutError) else str(exc)
                )
                failure_errors.append(last_error)
                execution_failed = True
                if is_missing_schema_error(last_error):
                    schema_gap_seen = True
                    break
        if execution_failed or schema_gap_seen:
            break
        if not simple_sql:
            break

    # Only a genuine schema gap may send the caller to document retrieval: the SQL
    # needed a fact no curated view stores. A valid query with zero rows returned
    # above as a success and never reaches this point.
    return failure_result(
        last_error, generation_status, failure_errors, timings(max(0, tried - 1))
    )
