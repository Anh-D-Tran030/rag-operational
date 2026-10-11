"""Execute and score document-independent SQL golden cases."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.config import Settings
from src.evaluation.sql_ground_truth import compare_result_rows

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient


async def evaluate_sql_case(
    query_id: int,
    golden: dict[str, Any],
    expected: dict[str, Any],
    settings: Settings,
    client: AsyncQdrantClient,
) -> dict[str, Any]:
    """Evaluate route, candidate attempts, execution and semantic result values."""
    from src.generation.router import classify_intent
    from src.retrieval.embedder import embed_query
    from src.retrieval.schema_linker import find_relevant_views
    from src.retrieval.text_to_sql import (
        assess_schema_coverage,
        execute_sql,
        generate_sql,
        validate_sql,
    )

    started = time.monotonic()
    route_started = time.monotonic()
    intent = await classify_intent(golden["question"], settings.local_llm(settings.router_model))
    route_ms = (time.monotonic() - route_started) * 1000

    schema_started = time.monotonic()
    embedding = await embed_query(golden["question"], settings.embedding_model)
    views = await find_relevant_views(
        golden["question"], embedding, client, settings.qdrant_schema_collection
    )
    schema_ms = (time.monotonic() - schema_started) * 1000

    coverage = assess_schema_coverage(golden["question"])
    if coverage["status"] != "full":
        return {
            "query_id": query_id,
            "query": golden["question"],
            "predicted_route": intent["tool"],
            "acceptable_routes": expected["acceptable_routes"],
            "route_correct": intent["tool"] in expected["acceptable_routes"],
            "schema_support": expected["schema_support"],
            "schema_guard_status": coverage["status"],
            "missing_schema": coverage["missing_schema"],
            "generated_sql": None,
            "generated_candidates": [],
            "validated_sql": None,
            "execution_succeeded": False,
            "returned_rows": [],
            "expected_result": expected,
            "result_correct": False,
            "wrong_but_runnable": False,
            "sql_error": None,
            "sql_error_observed": False,
            "self_corrected": False,
            "attempts": [],
            "relevant_views": [view.get("view_name", "") for view in views],
            "latency": {
                "route_ms": round(route_ms, 2), "schema_linking_ms": round(schema_ms, 2),
                "generation_ms": 0.0,
                "total_ms": round((time.monotonic() - started) * 1000, 2),
            },
        }

    generation_started = time.monotonic()
    generation_diagnostics: list[dict[str, Any]] = []
    generation_status: dict[str, Any] = {}
    candidates = await generate_sql(
        golden["question"], views, settings.cloud_model,
        settings.openrouter_api_key, settings.openrouter_base_url,
        diagnostics=generation_diagnostics,
        diagnostic_case_id=query_id,
        generation_status=generation_status,
    )
    generation_ms = (time.monotonic() - generation_started) * 1000
    attempts: list[dict[str, Any]] = []
    selected_sql: str | None = None
    returned_rows: list[dict[str, Any]] = []
    last_error: str | None = None
    if generation_status.get("status") == "sql_generation_unavailable":
        last_error = "sql_generation_unavailable"

    for index, sql in enumerate(candidates, start=1):
        attempt_started = time.monotonic()
        valid, error = await validate_sql(sql, settings.db_url)
        attempt: dict[str, Any] = {
            "attempt": index, "generated_sql": sql,
            "validated_sql": sql if valid else None,
            "execution_succeeded": False, "returned_rows": [],
            "sql_error": error or None,
        }
        if valid:
            try:
                returned_rows = await execute_sql(sql, settings.db_url)
                selected_sql = sql
                attempt["execution_succeeded"] = True
                attempt["returned_rows"] = returned_rows
                last_error = None
            except RuntimeError as exc:
                last_error = str(exc)
                attempt["sql_error"] = last_error
        else:
            last_error = error
        attempt["latency_ms"] = round((time.monotonic() - attempt_started) * 1000, 2)
        attempts.append(attempt)
        if selected_sql is not None:
            break

    succeeded = selected_sql is not None
    result_correct = succeeded and compare_result_rows(returned_rows, expected, selected_sql)
    return {
        "query_id": query_id,
        "query": golden["question"],
        "predicted_route": intent["tool"],
        "acceptable_routes": expected["acceptable_routes"],
        "route_correct": intent["tool"] in expected["acceptable_routes"],
        "schema_support": expected["schema_support"],
        "schema_guard_status": "full",
        "missing_schema": [],
        "generated_sql": selected_sql or (candidates[0] if candidates else None),
        "generated_candidates": candidates,
        "generation_diagnostics": generation_diagnostics,
        "generation_status": generation_status.get("status", "available"),
        "generation_unavailable_reason": generation_status.get("reason"),
        "validated_sql": selected_sql,
        "execution_succeeded": succeeded,
        "returned_rows": returned_rows,
        "expected_result": expected,
        "result_correct": bool(result_correct),
        "wrong_but_runnable": bool(succeeded and not result_correct),
        "sql_error": last_error,
        "sql_error_observed": any(item["sql_error"] for item in attempts),
        "self_corrected": bool(succeeded and len(attempts) > 1),
        "attempts": attempts,
        "relevant_views": [view.get("view_name", "") for view in views],
        "latency": {
            "route_ms": round(route_ms, 2), "schema_linking_ms": round(schema_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round((time.monotonic() - started) * 1000, 2),
        },
    }


def aggregate_sql_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate independent model metrics and schema-coverage gaps."""
    total = len(cases)
    if total == 0:
        raise ValueError("Cannot aggregate an empty SQL case list")

    def rate(field: str) -> float:
        return round(sum(bool(case[field]) for case in cases) / total, 4)

    supported = [case for case in cases if case["schema_support"] == "full"]
    unsupported = [case for case in cases if case["schema_support"] == "unsupported"]

    def supported_rate(field: str) -> float | None:
        if not supported:
            return None
        return round(sum(bool(case[field]) for case in supported) / len(supported), 4)

    supported_result_correctness = supported_rate("result_correct")
    supported_wrong_but_runnable_rate = supported_rate("wrong_but_runnable")
    supported_execution_success_rate = supported_rate("execution_succeeded")
    unsupported_by_schema_rate = round(len(unsupported) / total, 4)
    guarded_unsupported = sum(
        case.get("schema_guard_status") == "unsupported_by_schema" for case in cases
    )
    guarded_partial = sum(
        case.get("schema_guard_status") == "partial_schema_support" for case in cases
    )

    return {
        "case_count": total,
        "route_accuracy": rate("route_correct"),
        "sql_execution_success_rate": rate("execution_succeeded"),
        "result_correctness": rate("result_correct"),
        "wrong_but_runnable_rate": rate("wrong_but_runnable"),
        "sql_error_rate": rate("sql_error_observed"),
        "self_correction_rate": rate("self_corrected"),
        # Keep the legacy name for historical report compatibility, while
        # publishing names that distinguish supported cases from all cases.
        "model_accuracy_on_supported_cases": supported_result_correctness,
        "supported_case_result_correctness": supported_result_correctness,
        "supported_wrong_but_runnable_rate": supported_wrong_but_runnable_rate,
        "supported_execution_success_rate": supported_execution_success_rate,
        "supported_case_count": len(supported),
        "schema_coverage_gaps": {
            "partial_case_count": sum(case["schema_support"] == "partial" for case in cases),
            "unsupported_case_count": len(unsupported),
            "unsupported_by_schema_rate": unsupported_by_schema_rate,
        },
        "unsupported_by_schema_rate": unsupported_by_schema_rate,
        "unsupported_by_schema_count": guarded_unsupported,
        "partial_schema_support_count": guarded_partial,
        "schema_guard_abstention_count": guarded_unsupported + guarded_partial,
    }
