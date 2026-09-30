"""Evaluate SQL, hybrid case 22, and unanswerable golden questions live."""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Awaitable, TypeVar

from src.config import Settings
from src.evaluation.eval_reporting import summarize_latencies, write_json_report, write_sql_markdown
from src.evaluation.query_type_eval import evaluate_hybrid_case, evaluate_unanswerable_cases
from src.evaluation.sql_case_eval import aggregate_sql_metrics, evaluate_sql_case
from src.evaluation.sql_ground_truth import (
    SQL_CASE_IDS,
    load_golden_cases,
    load_sql_expectations,
)
from src.ingestion.structured_loader import seed_database

GOLDEN_PATH = Path("eval_data/golden_dataset.json")
EXPECTED_PATH = Path("eval_data/sql_expected_results.json")
HYBRID_ADVERSARIAL_PATH = Path("eval_data/hybrid_adversarial_cases.json")
FIXTURE_DIR = Path("tests/fixtures")

_T = TypeVar("_T")


def _stage(name: str, started_at: float) -> None:
    """Emit a flushed, monotonic stage marker for live-evaluation diagnosis."""
    elapsed_ms = (time.perf_counter() - started_at) * 1_000
    print(f"[sql-eval] stage={name} elapsed_ms={elapsed_ms:.1f}", flush=True)


async def _bounded_stage(
    stage: str, operation: Awaitable[_T], timeout_s: float, started_at: float
) -> _T:
    """Await one setup operation with a diagnostic timeout, never a silent hang."""
    try:
        return await asyncio.wait_for(operation, timeout=timeout_s)
    except TimeoutError as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1_000
        message = (
            f"stage={stage} status=timeout elapsed_ms={elapsed_ms:.1f} "
            f"timeout_s={timeout_s:.1f}"
        )
        print(f"[sql-eval] {message}", file=sys.stderr, flush=True)
        raise RuntimeError(message) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        type=int,
        choices=SQL_CASE_IDS,
        help="Evaluate only one SQL case (for example, 19). Omit for the full evaluation.",
    )
    parser.add_argument("--sql-json", default=".logs/sql_evaluation.json")
    parser.add_argument("--sql-markdown", default=".logs/sql_evaluation.md")
    parser.add_argument("--unanswerable-json", default=".logs/unanswerable_evaluation.json")
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=30.0,
        help="Maximum seconds for isolated evaluation database initialization.",
    )
    return parser


def _selected_sql_case_ids(case_id: int | None) -> tuple[int, ...]:
    """Return the requested SQL scope, preserving the full default scope."""
    return (case_id,) if case_id is not None else SQL_CASE_IDS


def _unanswerable_not_evaluated() -> dict:
    """Create a valid ancillary report for a SQL-only evaluation."""
    return {
        "evaluation_scope": "not evaluated for single SQL case",
        "case_count": 0,
        "metrics": {
            "abstention_rate": None,
            "unsupported_answer_rate": None,
            "false_confidence_rate": None,
        },
        "cases": [],
    }


def _tracer(settings: Settings):
    from src.observability.tracer import RAGTracer

    return RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def _hybrid_adversarial_manifest() -> dict:
    """Load the defined hybrid/adversarial scope without inventing outcomes."""
    import json

    cases = json.loads(HYBRID_ADVERSARIAL_PATH.read_text(encoding="utf-8"))
    if len(cases) < 5:
        raise ValueError("Hybrid/adversarial evaluation requires at least five defined cases")
    blocked = [case for case in cases if case["status"] == "blocked_human_decision"]
    return {
        "case_count": len(cases),
        "ready_case_count": len(cases) - len(blocked),
        "blocked_human_decision_count": len(blocked),
        "cases": cases,
    }


async def run(args: argparse.Namespace) -> int:
    """Seed isolated data, run live evaluations and write the evidence pack."""
    started_at = time.perf_counter()
    _stage("startup", started_at)
    golden = load_golden_cases(GOLDEN_PATH)
    expected = load_sql_expectations(EXPECTED_PATH)
    _stage("fixtures_loaded", started_at)
    with tempfile.TemporaryDirectory(prefix="rag-sql-eval-", dir="/tmp") as temp_dir:
        db_path = Path(temp_dir) / "evaluation.sqlite"
        settings = Settings(
            db_url=f"sqlite+aiosqlite:///{db_path}",
            semantic_cache_url="",
        )
        _stage("config_loaded", started_at)
        if not settings.openrouter_api_key:
            print("OPENROUTER_API_KEY is not set; live SQL evaluation cannot run.", file=sys.stderr)
            return 2
        _stage("qdrant_import_started", started_at)
        from qdrant_client import AsyncQdrantClient
        _stage("qdrant_import_completed", started_at)

        _stage("semantic_cache_import_started", started_at)
        from src.retrieval.cache import SemanticCache
        _stage("semantic_cache_import_completed", started_at)

        _stage("database_seed_started", started_at)
        await _bounded_stage(
            "database_seed",
            seed_database(settings.db_url, str(FIXTURE_DIR)),
            args.startup_timeout_s,
            started_at,
        )
        _stage("database_seed_completed", started_at)
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        _stage("vector_client_initialized", started_at)
        tracer = _tracer(settings)
        _stage("tracer_initialized", started_at)
        cache = SemanticCache(cache_url="", embedding_model=settings.embedding_model)
        _stage("semantic_cache_initialized", started_at)
        try:
            sql_cases = []
            for query_id in _selected_sql_case_ids(args.case_id):
                print(f"[sql-eval] SQL case {query_id}/30", flush=True)
                sql_cases.append(await evaluate_sql_case(
                    query_id, golden[query_id], expected[query_id], settings, client
                ))
            if args.case_id is None:
                print("[sql-eval] hybrid case 22/30", flush=True)
                hybrid = await evaluate_hybrid_case(
                    22, golden[22], settings, client, tracer, cache
                )
                print("[sql-eval] unanswerable cases 24-30", flush=True)
                unanswerable = await evaluate_unanswerable_cases(
                    golden, settings, client, tracer, cache
                )
            else:
                hybrid = None
                unanswerable = _unanswerable_not_evaluated()
        finally:
            await client.close()

    timestamp = datetime.now(UTC).isoformat()
    sql_report = {
        "generated_at": timestamp,
        "evaluation_scope": (
            "12 SQL cases; hybrid case 22 reported separately"
            if args.case_id is None else f"SQL case {args.case_id}"
        ),
        "result_comparison": "view-backed semantic assertions and exact rows",
        "production_reranker_config_changed": False,
        "observed_reranker_config": {
            "top_n": settings.rerank_top_n,
            "score_threshold": settings.rerank_score_threshold,
        },
        "metrics": aggregate_sql_metrics(sql_cases),
        "latency_by_query_type": summarize_latencies(sql_cases, hybrid, unanswerable),
        "cases": sql_cases,
        "hybrid_case_study": hybrid,
        "hybrid_adversarial_evaluation": _hybrid_adversarial_manifest(),
    }
    unanswerable["generated_at"] = timestamp
    write_json_report(args.sql_json, sql_report)
    write_sql_markdown(args.sql_markdown, sql_report)
    write_json_report(args.unanswerable_json, unanswerable)
    print(f"[sql-eval] metrics={sql_report['metrics']}", flush=True)
    print(f"[sql-eval] unanswerable={unanswerable['metrics']}", flush=True)
    return 0


def main() -> int:
    """Parse CLI arguments and run the asynchronous evaluator."""
    return asyncio.run(run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
