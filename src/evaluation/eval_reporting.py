"""Stable JSON and Markdown output for SQL evaluation evidence."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median, quantiles
from typing import Any


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 JSON report with deterministic formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


SIMPLE_SQL_IDS = frozenset({9, 10, 11, 12, 13})
LATENCY_TARGET_MS = {"simple_sql": 1000, "complex_sql": 4000}


def summarize_latencies(
    sql_cases: list[dict[str, Any]],
    hybrid: dict[str, Any] | None,
    unanswerable: dict[str, Any],
) -> dict[str, Any]:
    """Summarize observed component timings by query type, in milliseconds."""
    samples: dict[tuple[str, str], list[float]] = {}

    def add(query_type: str, component: str, value: Any) -> None:
        if isinstance(value, (int, float)) and value >= 0:
            samples.setdefault((query_type, component), []).append(float(value))

    for case in sql_cases:
        query_type = "simple_sql" if case["query_id"] in SIMPLE_SQL_IDS else "complex_sql"
        for component, value in case.get("latency", {}).items():
            add(query_type, component, value)
        if case.get("attempts"):
            add(query_type, "sql_attempts_ms", sum(
                attempt["latency_ms"] for attempt in case["attempts"]
            ))
    if hybrid is not None:
        add("hybrid", "total_ms", hybrid.get("latency_ms"))
    for case in unanswerable.get("cases", []):
        add("unanswerable", "total_ms", case.get("latency_ms"))

    summary: dict[str, Any] = {}
    for (query_type, component), values in sorted(samples.items()):
        p50 = median(values)
        p95 = quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0]
        target = LATENCY_TARGET_MS.get(query_type) if component == "total_ms" else None
        summary.setdefault(query_type, {})[component] = {
            "count": len(values), "p50_ms": round(p50, 2), "p95_ms": round(p95, 2),
            "target_ms": target,
            "p50_meets_target": p50 < target if target is not None else None,
            "p95_meets_target": p95 < target if target is not None else None,
        }
    return summary


def render_sql_markdown(payload: dict[str, Any]) -> str:
    """Render SQL aggregates, per-case evidence and the separate hybrid study."""
    metrics = payload["metrics"]
    execution_scope = payload.get(
        "evaluation_scope", "12 SQL cases; hybrid case 22 reported separately"
    )
    execution_heading = (
        "all 12 SQL cases"
        if execution_scope == "12 SQL cases; hybrid case 22 reported separately"
        else execution_scope
    )
    lines = [
        "# SQL evaluation",
        "",
        "Result correctness compares returned values and view-backed semantic assertions; "
        "it never compares raw SQL strings.",
        "",
        f"## Execution metrics ({execution_heading})",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for name in (
        "route_accuracy",
        "sql_execution_success_rate",
        "result_correctness",
        "wrong_but_runnable_rate",
        "sql_error_rate",
        "self_correction_rate",
    ):
        lines.append(f"| {name} | {metrics[name]:.4f} |")
    lines.extend([
        "",
        "## Model accuracy on schema-supported cases",
        "",
        "| metric | value |",
        "|---|---:|",
        "| model_accuracy_on_supported_cases | "
        f"{metrics['model_accuracy_on_supported_cases']:.4f} |",
        "| supported_case_result_correctness | "
        f"{metrics['supported_case_result_correctness']:.4f} |",
        "| supported_wrong_but_runnable_rate | "
        f"{metrics['supported_wrong_but_runnable_rate']:.4f} |",
        f"| supported_execution_success_rate | {metrics['supported_execution_success_rate']:.4f} |",
        f"| supported_case_count | {metrics['supported_case_count']} |",
        "",
        "## Schema coverage gaps",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| partial_case_count | {metrics['schema_coverage_gaps']['partial_case_count']} |",
        f"| unsupported_case_count | {metrics['schema_coverage_gaps']['unsupported_case_count']} |",
        "| unsupported_by_schema_rate | "
        f"{metrics['schema_coverage_gaps']['unsupported_by_schema_rate']:.4f} |",
    ])
    if "schema_guard_abstention_count" in metrics:
        lines.extend([
            f"| unsupported_by_schema_count | {metrics['unsupported_by_schema_count']} |",
            f"| partial_schema_support_count | {metrics['partial_schema_support_count']} |",
            f"| schema_guard_abstention_count | {metrics['schema_guard_abstention_count']} |",
        ])
    lines.extend([
        "", "## Per-case outcomes", "",
        "| id | schema support | route | route accepted | executed | result correct | "
        "wrong-but-runnable | self-corrected | total ms |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for case in payload["cases"]:
        lines.append(
            f"| {case['query_id']} | {_cell(case['schema_support'])} | "
            f"{_cell(case['predicted_route'])} | "
            f"{case['route_correct']} | {case['execution_succeeded']} | {case['result_correct']} | "
            f"{case['wrong_but_runnable']} | {case['self_corrected']} | "
            f"{case['latency']['total_ms']:.2f} |"
        )
    if "latency_by_query_type" in payload:
        lines.extend([
            "", "## Latency by component and query type", "",
            "Targets apply to end-to-end SQL latency: simple < 1000 ms; complex < 4000 ms.",
            "Simple SQL cases: 9–13. Complex SQL cases: 16–21 and 23.",
            "", "| query type | component | n | p50 ms | p95 ms | "
            "target ms | p50 pass | p95 pass |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ])
        for query_type, components in payload["latency_by_query_type"].items():
            for component, stats in components.items():
                target = stats["target_ms"] if stats["target_ms"] is not None else "—"
                lines.append(
                    f"| {query_type} | {component} | {stats['count']} | "
                    f"{stats['p50_ms']:.2f} | {stats['p95_ms']:.2f} | {target} | "
                    f"{stats['p50_meets_target']} | {stats['p95_meets_target']} |"
                )
    hybrid = payload.get("hybrid_case_study")
    if hybrid is not None:
        lines.extend([
            "",
            "## Hybrid case 22 — single end-to-end case study",
            "",
            "This is one case and is not reported as an aggregate metric.",
            "",
            "| component | passed |",
            "|---|---:|",
            f"| Router selected hybrid | {hybrid['router_selected_hybrid']} |",
            f"| Document evidence retrieved | {hybrid['document_evidence_retrieved']} |",
            f"| SQL/result correct | {hybrid['sql_result_correct']} |",
            f"| Final answer correct | {hybrid['final_answer_correct']} |",
            "",
            "## Hybrid and adversarial evaluation",
            "",
            "Defined cases and any blocked human decisions are recorded in the JSON evidence pack.",
            "",
            "Production reranker configuration was observed but not changed by this run.",
        ])
    return "\n".join(lines) + "\n"


def write_sql_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    """Write the human-readable SQL evidence report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_sql_markdown(payload), encoding="utf-8")
