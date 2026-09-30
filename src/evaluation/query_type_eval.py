"""Case-study evaluation for hybrid and unanswerable golden questions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.abstention import ABSTENTION_MARKERS, is_abstention_answer
from src.config import Settings
from src.evaluation.sql_ground_truth import compare_result_rows

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from src.observability.tracer import RAGTracer
    from src.retrieval.cache import SemanticCache

UNCERTAINTY_MARKERS = ABSTENTION_MARKERS + (
    "may",
    "might",
    "uncertain",
    "consult the source",
)
HYBRID_DOCUMENT_IDS = {
    "procedure_pdf:section_1.4",
    "procedure_pdf:section_2.4",
}
HYBRID_EXPECTED_SQL = {
    "expected_columns": ["incident_count"],
    "column_aliases": {
        "incident_count": ["incident_count", "total_incidents", "count"]
    },
    "expected_rows": [{"incident_count": 2}],
}


def classify_unanswerable_answer(answer: str) -> dict[str, bool]:
    """Classify abstention, unsupported answer and confident unsupported answer."""
    lowered = answer.casefold().strip()
    abstained = is_abstention_answer(answer)
    unsupported = bool(lowered) and not abstained
    uncertain = any(marker in lowered for marker in UNCERTAINTY_MARKERS)
    return {
        "abstained": abstained,
        "unsupported_answer": unsupported,
        "false_confidence": unsupported and not uncertain,
    }


async def evaluate_hybrid_case(
    query_id: int,
    golden: dict[str, Any],
    settings: Settings,
    client: AsyncQdrantClient,
    tracer: RAGTracer,
    cache: SemanticCache,
) -> dict[str, Any]:
    """Report case 22 as four checks; do not aggregate it with SQL cases."""
    from src.generation.query_handler import handle_query
    from src.retrieval.text_to_sql import execute_sql

    response = await handle_query(golden["question"], client, settings, tracer, cache)
    citations = response.get("citations", []) or []
    retrieved_ids = {
        str(item)
        for citation in citations
        for item in (citation.get("section_ids", []) or [])
    }
    sql_citation = next((item for item in citations if item.get("sql_used")), None)
    sql_rows: list[dict[str, Any]] = []
    sql_error: str | None = None
    if sql_citation:
        try:
            sql_rows = await execute_sql(sql_citation["sql_used"], settings.db_url)
        except RuntimeError as exc:
            sql_error = str(exc)
    sql_correct = bool(sql_citation) and compare_result_rows(sql_rows, HYBRID_EXPECTED_SQL)
    answer = str(response.get("answer", ""))
    normalized = answer.casefold().replace("-", "")
    final_correct = all(
        marker in normalized for marker in ("yes", "30", "inc001", "inc002")
    )
    return {
        "query_id": query_id,
        "query": golden["question"],
        "report_type": "single_end_to_end_case_study_not_aggregate",
        "router_selected_hybrid": response.get("tool") == "hybrid",
        "predicted_route": response.get("tool"),
        "document_evidence_retrieved": HYBRID_DOCUMENT_IDS <= retrieved_ids,
        "gold_document_evidence": sorted(HYBRID_DOCUMENT_IDS),
        "retrieved_evidence": sorted(retrieved_ids),
        "sql_result_correct": bool(sql_correct),
        "generated_sql": sql_citation.get("sql_used") if sql_citation else None,
        "returned_rows": sql_rows,
        "expected_sql_result": HYBRID_EXPECTED_SQL["expected_rows"],
        "sql_error": sql_error,
        "final_answer_correct": final_correct,
        "final_answer_check": "requires yes, 30, INC-001 and INC-002",
        "answer": answer,
        "citations": citations,
        "latency_ms": response.get("latency_ms"),
    }


async def evaluate_unanswerable_cases(
    golden_cases: dict[int, dict[str, Any]],
    settings: Settings,
    client: AsyncQdrantClient,
    tracer: RAGTracer,
    cache: SemanticCache,
) -> dict[str, Any]:
    """Run IDs 24-30 end-to-end and report three independent safety rates."""
    from src.generation.query_handler import handle_query

    results: list[dict[str, Any]] = []
    for query_id in range(24, 31):
        golden = golden_cases[query_id]
        response = await handle_query(golden["question"], client, settings, tracer, cache)
        classification = classify_unanswerable_answer(str(response.get("answer", "")))
        results.append({
            "query_id": query_id,
            "query": golden["question"],
            "predicted_route": response.get("tool"),
            "answer": response.get("answer", ""),
            "citations": response.get("citations", []),
            "latency_ms": response.get("latency_ms"),
            **classification,
        })
    total = len(results)

    def metric(field: str) -> float:
        return round(sum(item[field] for item in results) / total, 4)

    return {
        "case_count": total,
        "metrics": {
            "abstention_rate": metric("abstained"),
            "unsupported_answer_rate": metric("unsupported_answer"),
            "false_confidence_rate": metric("false_confidence"),
        },
        "cases": results,
    }
