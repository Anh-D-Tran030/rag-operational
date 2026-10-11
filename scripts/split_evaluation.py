"""Create a source-aware evaluation split for the golden dataset.

This is an evaluation taxonomy, not a routing-model prediction. The assignment
uses each case's gold evidence source plus the task it asks the system to solve,
so document retrieval, SQL execution, and hybrid composition can be measured
without conflating their failures.

Usage:
    python3 scripts/split_evaluation.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

GOLDEN_PATH = Path("eval_data/golden_dataset.json")
OUTPUT_JSON = Path(".logs/evaluation_split.json")
OUTPUT_MD = Path(".logs/evaluation_split.md")

DOCUMENT_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 14, 15}
SQL_IDS = {9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 23}
HYBRID_IDS = {22}
UNANSWERABLE_IDS = {24, 25, 26, 27, 28, 29, 30}

METRIC_FAMILIES = {
    "document": ["recall@k", "MRR", "nDCG", "citation correctness"],
    "sql": ["route accuracy", "SQL execution accuracy", "result correctness"],
    "hybrid": [
        "subtask success",
        "SQL correctness",
        "document recall",
        "final answer correctness",
    ],
    "unanswerable": ["abstention precision", "abstention recall"],
    "adversarial": ["version conflict handling", "unsupported-claim rejection"],
}


def _query_type(query_id: int) -> str:
    if query_id in DOCUMENT_IDS:
        return "document"
    if query_id in SQL_IDS:
        return "sql"
    if query_id in HYBRID_IDS:
        return "hybrid"
    if query_id in UNANSWERABLE_IDS:
        return "unanswerable"
    raise ValueError(f"No evaluation type assigned for golden query {query_id}")


def _gold_source(ids: list[str]) -> list[str]:
    if not ids:
        return ["none"]
    sources: list[str] = []
    for evidence_id in ids:
        source = evidence_id.split(":", 1)[0]
        if source not in sources:
            sources.append(source)
    return sources


def _expected_tool(query_type: str) -> str:
    return {
        "document": "vector_search",
        "sql": "text_to_sql",
        "hybrid": "hybrid",
        "unanswerable": "abstain",
        "adversarial": "abstain_or_supported_response",
    }[query_type]


def _markdown(rows: list[dict], counts: dict[str, int]) -> str:
    lines = [
        "# Evaluation split",
        "",
        "This split is source-aware: it is the expected evaluation surface, not a measurement of the router's current prediction.",
        "",
        "| Query type | Cases | Metric family |",
        "|---|---:|---|",
    ]
    for query_type in ("document", "sql", "hybrid", "unanswerable", "adversarial"):
        lines.append(f"| {query_type} | {counts.get(query_type, 0)} | {', '.join(METRIC_FAMILIES[query_type])} |")
    lines.extend(
        [
            "",
            "| query_id | query_type | expected_tool | gold_source | metric_family |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['query_id']} | {row['query_type']} | {row['expected_tool']} | "
            f"{', '.join(row['gold_source'])} | {', '.join(row['metric_family'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    golden = json.loads(GOLDEN_PATH.read_text())
    rows: list[dict] = []
    for query_id, case in enumerate(golden, start=1):
        query_type = _query_type(query_id)
        evidence_ids = list(case.get("evidence_chunk_ids") or [])
        rows.append(
            {
                "query_id": query_id,
                "query": case["question"],
                "dataset_category": case["category"],
                "query_type": query_type,
                "expected_tool": _expected_tool(query_type),
                "gold_source": _gold_source(evidence_ids),
                "gold_evidence_chunk_ids": evidence_ids,
                "metric_family": METRIC_FAMILIES[query_type],
            }
        )
    counts = dict(Counter(row["query_type"] for row in rows))
    if len(rows) != 30:
        raise ValueError(f"Expected 30 golden cases, found {len(rows)}")
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({"summary": counts, "questions": rows}, indent=2))
    OUTPUT_MD.write_text(_markdown(rows, counts))
    print(json.dumps(counts, sort_keys=True))
    print(f"written: {OUTPUT_JSON}")
    print(f"written: {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
