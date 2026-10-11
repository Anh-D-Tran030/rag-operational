"""Text-to-SQL: generate, validate, execute SQL against curated views.

Compatibility facade over the ``sql_*`` modules. Public names, private helpers
used by tests and every module-level patch point (``chat``, the timeouts,
``SQL_FALLBACK_MODEL``, ``generate_sql``/``validate_sql``/``execute_sql`` and
the metric objects) live here. The wrappers pass this module to the
implementations, which read those attributes at call time, so a patch on
``src.retrieval.text_to_sql`` still controls the real call.
"""
from __future__ import annotations

import sys

from src.config import Settings
from src.llm_client import chat  # noqa: F401 - patch point read by sql_candidate_attempt
from src.observability.metrics import (  # noqa: F401 - patch points read at call time
    rag_latency_by_component,
    rag_sql_execution_errors,
    rag_sql_fallback_invocations_total,
    rag_sql_generation_attempts_total,
    rag_sql_generation_unavailable_total,
    rag_sql_partial_schema_support_total,
    rag_sql_schema_guard_abstention_total,
    rag_sql_unsupported_by_schema_total,
)
from src.retrieval import sql_generation, sql_tool
from src.retrieval.sql_candidate_filters import _filter_rejection_reasons
from src.retrieval.sql_candidate_prompt import (
    SQL_RESPONSE_FORMAT,
    _supervisor_comparison_sql,
    _with_required_sql_views,
)
from src.retrieval.sql_database import execute_sql, validate_sql
from src.retrieval.sql_schema import CURATED_VIEW_COLUMNS, assess_schema_coverage
from src.retrieval.sql_tool_results import _MISSING_SCHEMA_IDENTIFIER

__all__ = [
    "CURATED_VIEW_COLUMNS",
    "SQL_FALLBACK_MODEL",
    "SQL_RESPONSE_FORMAT",
    "_MISSING_SCHEMA_IDENTIFIER",
    "_filter_rejection_reasons",
    "_supervisor_comparison_sql",
    "_with_required_sql_views",
    "assess_schema_coverage",
    "execute_sql",
    "generate_sql",
    "text_to_sql_tool",
    "validate_sql",
]

SQL_LLM_TIMEOUT_S = 120.0
SIMPLE_SQL_LLM_TIMEOUT_S = 45.0
SQL_VALIDATION_TIMEOUT_S = 5.0
SQL_EXECUTION_TIMEOUT_S = 5.0
SQL_FALLBACK_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

_FACADE = sys.modules[__name__]


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
    return await sql_generation.generate_sql(
        _FACADE, query, relevant_views, model, api_key, base_url,
        n_candidates=n_candidates, call_timings=call_timings, diagnostics=diagnostics,
        diagnostic_case_id=diagnostic_case_id, generation_status=generation_status,
    )


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
      {rows: [], error: str} after max_attempts exhausted, with an authoritative
      failure_kind when the failure is a schema gap or alias-only rejection.
    """
    return await sql_tool.text_to_sql_tool(
        _FACADE, query, relevant_views, settings, trace_id,
        max_attempts=max_attempts, simple_sql=simple_sql,
    )
