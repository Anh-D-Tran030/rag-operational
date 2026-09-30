"""Deterministic tests for SQL evaluation semantics and reporting."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.config import Settings
from src.evaluation.eval_reporting import render_sql_markdown, summarize_latencies
from src.evaluation.query_type_eval import classify_unanswerable_answer
from src.evaluation.sql_case_eval import aggregate_sql_metrics, evaluate_sql_case
from src.evaluation.sql_ground_truth import (
    SQL_CASE_IDS,
    compare_result_rows,
    load_golden_cases,
    load_sql_expectations,
)


def _expected() -> dict:
    return {
        "expected_columns": ["date", "staff_count"],
        "column_aliases": {
            "date": ["date"],
            "staff_count": ["staff_count", "total_staff"],
        },
        "expected_rows": [
            {"date": "2026-05-28", "staff_count": 15},
            {"date": "2026-05-27", "staff_count": 6},
        ],
    }


def test_sql_ground_truth_has_exact_12_cases() -> None:
    cases = load_sql_expectations("eval_data/sql_expected_results.json")
    assert tuple(sorted(cases)) == SQL_CASE_IDS
    assert all(case["expected_rows"] for case in cases.values())
    assert cases[9]["column_aliases"]["staff_count"][-1] == "staff_rostered"
    assert compare_result_rows([{"rostered_staff": 15}], cases[9])
    assert compare_result_rows([{"staff_rostered": 15}], cases[9])
    assert compare_result_rows([{
        "venue": "Fairfield RSL", "date": "2026-05-25", "role": "security",
        "count": 2, "required_count": 3, "status": "Not Met",
    }], cases[21], "SELECT count FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
       "AND date = '2026-05-25' AND role = 'security'")
    assert cases[10]["schema_support"] == "full"
    assert cases[11]["schema_support"] == "full"
    assert cases[20]["schema_support"] == "full"
    assert cases[23]["schema_support"] == "unsupported"
    assert "denominator" in cases[23]["note"]
    assert all(
        case["schema_support"] in {"full", "partial", "unsupported"}
        for case in cases.values()
    )
    assert all(case["acceptable_routes"] for case in cases.values())


def test_result_comparison_accepts_aliases_and_ignores_order() -> None:
    rows = [
        {"date": "2026-05-27", "total_staff": "6"},
        {"date": "2026-05-28", "total_staff": 15.0},
    ]
    assert compare_result_rows(rows, _expected())


def test_result_comparison_rejects_wrong_runnable_values() -> None:
    rows = [
        {"date": "2026-05-27", "total_staff": 7},
        {"date": "2026-05-28", "total_staff": 15},
    ]
    assert not compare_result_rows(rows, _expected())


@pytest.mark.parametrize(("expected_value", "sqlite_value"), [
    (True, 1),
    (False, 0),
])
def test_result_comparison_accepts_sqlite_boolean_integers(
    expected_value: bool, sqlite_value: int
) -> None:
    expected = {
        "expected_columns": ["met_requirement"],
        "column_aliases": {"met_requirement": ["met_requirement"]},
        "expected_rows": [{"met_requirement": expected_value}],
    }

    assert compare_result_rows([{"met_requirement": sqlite_value}], expected)


def test_result_comparison_keeps_numeric_values_distinct_from_booleans() -> None:
    expected = {
        "expected_columns": ["count"],
        "column_aliases": {"count": ["count"]},
        "expected_rows": [{"count": 1}],
    }

    assert compare_result_rows([{"count": 1}], expected)
    assert not compare_result_rows([{"count": True}], expected)


def test_case_16_accepts_sqlite_boolean_alias_from_live_evaluation() -> None:
    expected = load_sql_expectations("eval_data/sql_expected_results.json")[16]
    sql = (
        "SELECT count, required_count, CASE WHEN gap >= 0 THEN 1 ELSE 0 END AS meets_required "
        "FROM v_staffing_gap WHERE venue = 'Fairfield RSL' AND date = '2026-05-28' "
        "AND role = 'floor_staff'"
    )
    row = {
        "venue": "Fairfield RSL",
        "date": "2026-05-28",
        "role": "floor_staff",
        "count": 8,
        "required_count": 8,
        "gap": 0,
        "meets_required": 1,
    }

    assert compare_result_rows([row], expected, sql)


@pytest.mark.parametrize("case_id,row", [
    (16, {"venue": "Fairfield RSL", "date": "2026-05-28", "role": "floor_staff",
          "staff_count": 8, "required_count": 8, "met_required": "Yes"}),
    (21, {"venue": "Fairfield RSL", "date": "2026-05-25", "role": "security",
          "count": 2, "required_count": 3, "meets_requirement": "NO"}),
    (19, {"shift_id": "SH-006", "staff_count": 20, "incident_count": 2}),
    (19, {"shift_id": "SH-006", "staff_count": 20, "total_incidents": 2}),
    (19, {"shift_id": "SH-006", "staff_count": 20, "incident_total": 2}),
    (19, {"shift_id": "SH-006", "staff_count": 20, "incidents": 2}),
])
def test_view_backed_semantic_cases_accept_exact_results(case_id: int, row: dict) -> None:
    expected = load_sql_expectations("eval_data/sql_expected_results.json")[case_id]
    sql = {
        16: "SELECT count FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
            "AND date = '2026-05-28' AND role = 'floor_staff'",
        21: "SELECT count FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
            "AND date = '2026-05-25' AND role = 'security'",
        19: "SELECT staff_count, incident_count FROM v_peak_staffing",
    }[case_id]
    assert compare_result_rows([row], expected, sql)
    for key in row:
        assert not compare_result_rows([{**row, key: "wrong"}], expected, sql), key


def test_case_19_semantics_reject_aggregate_only_result_without_shift_id() -> None:
    expected = load_sql_expectations("eval_data/sql_expected_results.json")[19]
    aggregate_only_row = {
        "venue": "Fairfield RSL",
        "date": "2026-05-23",
        "shift_type": "Evening",
        "staff_count": 20,
        "incident_count": 2,
    }

    assert not compare_result_rows(
        [aggregate_only_row], expected, "SELECT staff_count FROM v_shift_summary"
    )


def test_staffing_semantic_comparison_accepts_constrained_projection_only() -> None:
    cases = load_sql_expectations("eval_data/sql_expected_results.json")
    case_16_sql = (
        "SELECT count, required_count, CASE WHEN gap >= 0 THEN 'Yes' ELSE 'No' END AS met "
        "FROM v_staffing_gap WHERE venue = 'Fairfield RSL' AND date = '2026-05-28' "
        "AND role = 'floor_staff'"
    )
    assert compare_result_rows([{
        "count": 8, "required_count": 8, "gap": 0, "met": "Yes",
    }], cases[16], case_16_sql)
    assert not compare_result_rows([{
        "count": 8, "required_count": 8, "gap": 0, "met": "Yes",
    }], cases[16], case_16_sql.replace("floor_staff", "security"))

    case_21_sql = (
        "SELECT count AS staffed, required_count, CASE WHEN gap >= 0 THEN 'Yes' ELSE 'No' "
        "END AS met_requirement FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
        "AND date = '2026-05-25' AND role = 'security'"
    )
    assert compare_result_rows([{
        "staffed": 2, "required_count": 3, "met_requirement": "No",
    }], cases[21], case_21_sql)
    assert not compare_result_rows([{
        "staffed": 2, "required_count": 3, "met_requirement": "No",
    }], cases[21], case_21_sql.replace("count AS staffed", "required_count AS staffed"))


@pytest.mark.parametrize("status_key,status_value", [
    ("met_security_requirement", 0),
    ("requirement_met", 0),
    ("status", "No"),
    ("status", "Not Met"),
])
def test_case_21_semantics_accept_declared_status_aliases(
    status_key: str, status_value: int | str,
) -> None:
    expected = load_sql_expectations("eval_data/sql_expected_results.json")[21]
    sql = (
        "SELECT count, required_count FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
        "AND date = '2026-05-25' AND role = 'security'"
    )
    row = {"count": 2, "required_count": 3, status_key: status_value}

    assert compare_result_rows([row], expected, sql)


@pytest.mark.parametrize("row", [
    {
        "count": 2,
        "security_count": 3,
        "required_count": 3,
        "requirement_met": 0,
    },
    {
        "count": 2,
        "required_count": 3,
        "requirement_met": 0,
        "met_requirement": True,
    },
])
def test_case_21_semantics_reject_conflicting_declared_aliases(row: dict) -> None:
    expected = load_sql_expectations("eval_data/sql_expected_results.json")[21]
    sql = (
        "SELECT count, required_count FROM v_staffing_gap WHERE venue = 'Fairfield RSL' "
        "AND date = '2026-05-25' AND role = 'security'"
    )

    assert not compare_result_rows([row], expected, sql)


def test_case_17_requires_exact_dates_and_floor_staff_role() -> None:
    expected = load_sql_expectations("eval_data/sql_expected_results.json")[17]
    rows = [{"date": day, "shift_type": "Morning", "gap": -1} for day in (
        "2026-05-22", "2026-05-25", "2026-05-27"
    )]
    sql = "SELECT date, gap FROM v_staffing_gap WHERE role = 'floor_staff' AND gap < 0"
    assert compare_result_rows(rows, expected, sql)
    assert not compare_result_rows(rows, expected)
    assert not compare_result_rows(rows, expected, sql.replace("floor_staff", "security"))
    assert not compare_result_rows(rows[:-1], expected, sql)
    assert not compare_result_rows([*rows[:-1], {"date": "2026-05-28"}], expected, sql)
    assert not compare_result_rows([{**row, "role": "security"} for row in rows], expected, sql)
    assert not compare_result_rows([{**rows[0], "role": "security"}, *rows[1:]], expected, sql)


def test_semantic_cases_reject_conflicting_alias_values() -> None:
    cases = load_sql_expectations("eval_data/sql_expected_results.json")
    assert not compare_result_rows([{
        "venue": "Fairfield RSL", "date": "2026-05-28", "role": "floor_staff",
        "count": 8, "staff_count": 7, "required_count": 8, "met_required": "Yes",
    }], cases[16])
    assert not compare_result_rows([{
        "shift_id": "SH-006", "staff_count": 20,
        "incident_count": 2, "incident_total": 3,
    }], cases[19])
    assert not compare_result_rows([{
        "shift_id": "SH-006", "staff_count": 20,
        "incident_count": 2, "total_incidents": 3,
    }], cases[19])


def test_case_23_remains_unsupported_by_schema() -> None:
    cases = load_sql_expectations("eval_data/sql_expected_results.json")
    assert cases[23]["schema_support"] == "unsupported"


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", [18, 23])
async def test_evaluator_records_schema_guard_without_sql(case_id: int) -> None:
    golden = load_golden_cases("eval_data/golden_dataset.json")
    expected = load_sql_expectations("eval_data/sql_expected_results.json")
    with (
        patch("src.generation.router.classify_intent", new_callable=AsyncMock) as route,
        patch("src.retrieval.embedder.embed_query", new_callable=AsyncMock) as embed,
        patch("src.retrieval.schema_linker.find_relevant_views", new_callable=AsyncMock) as views,
        patch("src.retrieval.text_to_sql.generate_sql", new_callable=AsyncMock) as generate,
        patch("src.retrieval.text_to_sql.validate_sql", new_callable=AsyncMock) as validate,
        patch("src.retrieval.text_to_sql.execute_sql", new_callable=AsyncMock) as execute,
    ):
        route.return_value = {"tool": "text_to_sql", "complexity": "simple"}
        embed.return_value = [0.1] * 384
        views.return_value = []
        result = await evaluate_sql_case(
            case_id, golden[case_id], expected[case_id], Settings(_env_file=None), AsyncMock()
        )
    assert result["schema_guard_status"] in {
        "unsupported_by_schema", "partial_schema_support",
    }
    assert result["returned_rows"] == []
    assert result["generated_candidates"] == []
    assert result["generated_sql"] is None
    assert result["latency"]["generation_ms"] == 0
    generate.assert_not_awaited()
    validate.assert_not_awaited()
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluator_records_sql_generation_diagnostics() -> None:
    golden = load_golden_cases("eval_data/golden_dataset.json")
    expected = load_sql_expectations("eval_data/sql_expected_results.json")

    async def generate(*_args, diagnostics, diagnostic_case_id, **_kwargs):
        diagnostics.append({
            "case_id": diagnostic_case_id,
            "outcome": "no_valid_candidates",
            "raw_response": {"excerpt": "", "char_count": 0, "sha256": "hash"},
        })
        return []

    with (
        patch("src.generation.router.classify_intent", new_callable=AsyncMock) as route,
        patch("src.retrieval.embedder.embed_query", new_callable=AsyncMock) as embed,
        patch("src.retrieval.schema_linker.find_relevant_views", new_callable=AsyncMock) as views,
        patch("src.retrieval.text_to_sql.generate_sql", side_effect=generate),
    ):
        route.return_value = {"tool": "text_to_sql", "complexity": "complex"}
        embed.return_value = [0.1] * 384
        views.return_value = []
        result = await evaluate_sql_case(
            19, golden[19], expected[19], Settings(_env_file=None), AsyncMock()
        )

    assert result["generated_candidates"] == []
    assert result["generation_diagnostics"] == [{
        "case_id": 19,
        "outcome": "no_valid_candidates",
        "raw_response": {"excerpt": "", "char_count": 0, "sha256": "hash"},
    }]


@pytest.mark.asyncio
async def test_evaluator_records_sql_generation_unavailable_without_sql() -> None:
    golden = load_golden_cases("eval_data/golden_dataset.json")
    expected = load_sql_expectations("eval_data/sql_expected_results.json")

    async def generate(*_args, generation_status, **_kwargs):
        generation_status.update(
            status="sql_generation_unavailable", reason="timeout"
        )
        return []

    with (
        patch("src.generation.router.classify_intent", new_callable=AsyncMock) as route,
        patch("src.retrieval.embedder.embed_query", new_callable=AsyncMock) as embed,
        patch("src.retrieval.schema_linker.find_relevant_views", new_callable=AsyncMock) as views,
        patch("src.retrieval.text_to_sql.generate_sql", side_effect=generate),
    ):
        route.return_value = {"tool": "text_to_sql", "complexity": "complex"}
        embed.return_value = [0.1] * 384
        views.return_value = []
        result = await evaluate_sql_case(
            19, golden[19], expected[19], Settings(_env_file=None), AsyncMock()
        )

    assert result["generation_status"] == "sql_generation_unavailable"
    assert result["generation_unavailable_reason"] == "timeout"
    assert result["generated_sql"] is None
    assert result["returned_rows"] == []
    assert result["execution_succeeded"] is False


def test_sql_metrics_separate_execution_from_result_correctness() -> None:
    cases = [
        {
            "route_correct": True,
            "execution_succeeded": True,
            "result_correct": False,
            "wrong_but_runnable": True,
            "sql_error_observed": False,
            "self_corrected": False,
            "schema_support": "full",
        },
        {
            "route_correct": False,
            "execution_succeeded": True,
            "result_correct": True,
            "wrong_but_runnable": False,
            "sql_error_observed": True,
            "self_corrected": True,
            "schema_support": "unsupported",
        },
    ]
    metrics = aggregate_sql_metrics(cases)
    assert metrics["sql_execution_success_rate"] == 1.0
    assert metrics["result_correctness"] == 0.5
    assert metrics["wrong_but_runnable_rate"] == 0.5
    assert metrics["sql_error_rate"] == 0.5
    assert metrics["self_correction_rate"] == 0.5
    assert metrics["model_accuracy_on_supported_cases"] == 0.0
    assert metrics["supported_case_result_correctness"] == 0.0
    assert metrics["supported_wrong_but_runnable_rate"] == 1.0
    assert metrics["supported_execution_success_rate"] == 1.0
    assert metrics["unsupported_by_schema_rate"] == 0.5
    assert metrics["schema_coverage_gaps"]["unsupported_by_schema_rate"] == 0.5


def test_sql_metrics_count_schema_guard_abstentions() -> None:
    cases = [{
        "route_correct": True, "execution_succeeded": False,
        "result_correct": False, "wrong_but_runnable": False,
        "sql_error_observed": False, "self_corrected": False,
        "schema_support": support, "schema_guard_status": status,
    } for support, status in (
        ("full", "full"),
        ("partial", "partial_schema_support"),
        ("unsupported", "unsupported_by_schema"),
    )]
    metrics = aggregate_sql_metrics(cases)
    assert metrics["unsupported_by_schema_count"] == 1
    assert metrics["partial_schema_support_count"] == 1
    assert metrics["schema_guard_abstention_count"] == 2


def test_unanswerable_classification_separates_false_confidence() -> None:
    assert classify_unanswerable_answer("insufficient context") == {
        "abstained": True,
        "unsupported_answer": False,
        "false_confidence": False,
    }
    assert classify_unanswerable_answer("The room costs $500.") == {
        "abstained": False,
        "unsupported_answer": True,
        "false_confidence": True,
    }


@pytest.mark.parametrize(
    "answer",
    [
        (
            "I cannot find any information about the price of a schooner of beer at "
            "Fairfield RSL in the provided context."
        ),
        (
            "I'm unable to provide the price of a schooner of beer at Fairfield RSL as the "
            "information is not included in the provided context."
        ),
        "The price is not provided in the given context.",
        "No information about the venue dress code is provided in the given context.",
        "I cannot find information about function-room cost in the provided context.",
        "The provided context does not contain information about the venue's founding.",
        "The provided context does not include information about the venue's founding.",
        "No information provided in the given context about the dress code.",
        (
            "No information provided about a dress code for entry to Fairfield RSL in the "
            "given context."
        ),
        "There is no information about the function-room cost.",
        (
            "I'm unable to provide the cost to book a function room at Fairfield RSL as the "
            "information is not present in the given context."
        ),
        (
            "The provided context does not contain any information about a member loyalty "
            "program at Fairfield RSL."
        ),
        (
        "The context provided does not include any information about a member loyalty "
            "program at Fairfield RSL. Therefore, I cannot provide an answer to this "
            "question based on the given context."
        ),
        (
            "I cannot provide the cost to book a function room at Fairfield RSL as it is "
            "not mentioned in the provided context."
        ),
        (
            "I'm unable to provide information about Fairfield RSL's member loyalty program "
            "as it is not mentioned in the provided context."
        ),
    ],
)
def test_unanswerable_classification_recognizes_live_abstention_wording(answer: str) -> None:
    assert classify_unanswerable_answer(answer) == {
        "abstained": True,
        "unsupported_answer": False,
        "false_confidence": False,
    }


@pytest.mark.parametrize("answer", [
    "The price of a schooner of beer at Fairfield RSL is not provided in the given context.",
    "insufficient context",
    "The provided context does not contain information about when Fairfield RSL was founded.",
    ("No information about a dress code for entry to Fairfield RSL was found in the "
     "provided context."),
    ("The provided context does not contain information about the cost to book a "
     "function room at Fairfield RSL."),
    ("Based on the provided context, there is no information about a member loyalty "
     "program at Fairfield RSL."),
])
def test_latest_artifact_unanswerable_wordings(answer: str) -> None:
    assert classify_unanswerable_answer(answer) == {
        "abstained": True, "unsupported_answer": False, "false_confidence": False,
    }


def test_abstention_wording_does_not_match_supported_fact() -> None:
    result = classify_unanswerable_answer("The venue dress code is smart casual.")
    assert result["abstained"] is False


def test_latency_summary_reports_exact_percentiles_and_targets() -> None:
    sql_cases = [
        {"query_id": 9, "latency": {"route_ms": 100, "total_ms": 800},
         "attempts": [{"latency_ms": 20}]},
        {"query_id": 12, "latency": {"route_ms": 300, "total_ms": 1200},
         "attempts": [{"latency_ms": 40}]},
        {"query_id": 19, "latency": {"total_ms": 3500}, "attempts": []},
    ]
    summary = summarize_latencies(sql_cases, {"latency_ms": 5000}, {
        "cases": [{"latency_ms": 2000}],
    })
    assert summary["simple_sql"]["total_ms"] == {
        "count": 2, "p50_ms": 1000.0, "p95_ms": 1180.0,
        "target_ms": 1000, "p50_meets_target": False, "p95_meets_target": False,
    }
    assert summary["simple_sql"]["route_ms"]["p50_ms"] == 200.0
    assert summary["simple_sql"]["sql_attempts_ms"]["p95_ms"] == 39.0
    assert summary["complex_sql"]["total_ms"]["p95_meets_target"] is True
    assert summary["hybrid"]["total_ms"]["target_ms"] is None
    assert summary["unanswerable"]["total_ms"]["count"] == 1


def test_markdown_labels_hybrid_as_non_aggregate() -> None:
    metrics = {
        "route_accuracy": 1.0,
        "sql_execution_success_rate": 1.0,
        "result_correctness": 0.5,
        "wrong_but_runnable_rate": 0.5,
        "sql_error_rate": 0.0,
        "self_correction_rate": 0.0,
        "model_accuracy_on_supported_cases": 0.5,
        "supported_case_result_correctness": 0.5,
        "supported_wrong_but_runnable_rate": 0.0,
        "supported_execution_success_rate": 1.0,
        "supported_case_count": 1,
        "schema_coverage_gaps": {
            "partial_case_count": 0,
            "unsupported_case_count": 0,
            "unsupported_by_schema_rate": 0.0,
        },
        "unsupported_by_schema_rate": 0.0,
    }
    case = {
        "query_id": 9,
        "predicted_route": "text_to_sql",
        "schema_support": "full",
        "route_correct": True,
        "execution_succeeded": True,
        "result_correct": True,
        "wrong_but_runnable": False,
        "self_corrected": False,
        "latency": {"total_ms": 1.0},
    }
    hybrid = {
        "router_selected_hybrid": True,
        "document_evidence_retrieved": True,
        "sql_result_correct": False,
        "final_answer_correct": False,
    }
    report = render_sql_markdown({"metrics": metrics, "cases": [case], "hybrid_case_study": hybrid})
    assert "single end-to-end case study" in report
    assert "not reported as an aggregate metric" in report
    assert "Hybrid and adversarial evaluation" in report
