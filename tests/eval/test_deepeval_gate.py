"""DeepEval CI gate.

Loads the golden dataset, runs `handle_query` for each QA pair, and asserts
faithfulness, answer relevancy and contextual precision; contextual recall is
scored and logged only (PRD.md:438 blocks merges on faithfulness and context
precision; PRD.md:302-306 sets contextual precision >= 0.8 and no recall
threshold).
Unanswerable cases get faithfulness plus an abstention check only.
The judge is nondeterministic even at temperature 0, so each metric is scored
JUDGE_RUNS times and thresholds apply to the median (see _judge_utils). Faithfulness
below 0.9 hard-blocks the PR. Per-case medians are pushed to W&B when
WANDB_API_KEY is set.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

from ._case_log import capture_gate_trace, capture_sql_attempts, time_judge_calls, write_case_log
from ._judge_utils import JUDGE_RUNS, below_threshold, median_scores

GOLDEN_PATH = Path(__file__).parents[2] / "eval_data" / "golden_dataset.json"

# PLAN-003 §2.4: RAG_EVAL_REQUIRED=1 must hard-fail on any missing eval package,
# not silently skip. This is the full eval-extra set, not just what this file
# imports directly, per the amendment's explicit list.
EVAL_MODULES = ["deepeval", "ragas", "wandb", "datasets", "langchain_community"]
INSTALL_HINT = "install with `pip install -e .[eval]`"
TRACKED_ONLY = frozenset({"contextual_recall"})  # logged, never asserted (no PRD threshold)


def _load_golden() -> list[dict]:
    with GOLDEN_PATH.open() as fh:
        return json.load(fh)


def _ids(cases: list[dict]) -> list[str]:
    return [f"{i:02d}-{c.get('category', 'unknown')}" for i, c in enumerate(cases)]


GOLDEN = _load_golden()


def _check_eval_extra() -> None:
    # PLAN-003 §2.4: this gate is opt-in only — it always skips for a plain
    # developer run, live infra or not. Only RAG_EVAL_REQUIRED=1 (TEST sets
    # it) may run it live, and then a missing package is a hard failure.
    if os.environ.get("RAG_EVAL_REQUIRED") != "1":
        pytest.skip("developer run — set RAG_EVAL_REQUIRED=1 to run the live DeepEval gate")
    missing = [m for m in EVAL_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        pytest.fail(f"missing eval packages {missing} ({INSTALL_HINT})")


def _wandb_run():
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    if wandb.run is not None:
        return wandb.run
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "rag-operational"),
        job_type="deepeval_gate",
        reinit=True,
    )


async def _run_query(question: str) -> dict:
    """Run handle_query end-to-end against the live dependencies."""
    from qdrant_client import AsyncQdrantClient

    from src.config import Settings
    from src.generation.query_handler import handle_query
    from src.observability.tracer import RAGTracer
    from src.retrieval.cache import SemanticCache

    settings = Settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    tracer = RAGTracer(
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
    )
    cache = SemanticCache(settings.semantic_cache_url, settings.embedding_model)
    try:
        return await handle_query(question, client, settings, tracer, cache)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", GOLDEN, ids=_ids(GOLDEN))
async def test_deepeval_gate(case: dict, request: pytest.FixtureRequest) -> None:
    _check_eval_extra()
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models.llms.deepseek_model import DeepSeekModel
    from deepeval.test_case import LLMTestCase

    judge_model = DeepSeekModel(model="deepseek-chat")
    case_id = request.node.callspec.id

    case_started = time.time()
    with capture_sql_attempts() as sql_attempts, capture_gate_trace() as trace:
        result = await _run_query(case["question"])
    query_ended = time.time()
    answer = str(result.get("answer", ""))
    retrieval_context = [  # SQL citations contribute their row values, then the query text
        text for c in result.get("citations", [])
        for text in ([str(c["snippet"])] if "snippet" in c else
                     [str(r) for r in c.get("rows") or []] + [str(c.get("sql_used", ""))])
    ] or [""]

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=answer,
        expected_output=case["ground_truth"],
        retrieval_context=retrieval_context,
    )

    unanswerable = case.get("category") == "unanswerable"
    metrics = {"faithfulness": FaithfulnessMetric(threshold=0.9, model=judge_model)}
    if not unanswerable:
        metrics |= {
            "answer_relevancy": AnswerRelevancyMetric(threshold=0.75, model=judge_model),
            "contextual_precision": ContextualPrecisionMetric(
                threshold=0.8, model=judge_model
            ),  # PRD.md:302-306: context_precision >= 0.8
            "contextual_recall": ContextualRecallMetric(model=judge_model),
        }
    judge_calls = time_judge_calls(metrics)
    runs, medians, invalid = median_scores(metrics, test_case)
    print("DEEPEVAL_MEDIAN " + json.dumps({"case": case_id, "runs": runs, "median": medians}))
    if trace is not None:  # RAG_EVAL_GATE_TRACE=1 only; observability, asserts nothing
        trace |= {
            "case_started_at": case_started, "query_ended_at": query_ended,
            "judging_ended_at": time.time(), "deepeval_judge_calls": judge_calls,
            "latency_components": result.get("latency_components"),
            "retrieval_context_items": len(retrieval_context),
        }
    write_case_log(
        case_id, case, result, retrieval_context, sql_attempts, metrics, runs, medians, trace
    )
    blocking = {name: m for name, m in metrics.items() if name not in TRACKED_ONLY}
    assert not set(invalid) & blocking.keys(), f"judge returned missing/NaN scores: {runs}"

    run = _wandb_run()
    if run is not None:
        run.log({"category": case.get("category", "unknown"), **medians})

    if unanswerable:
        from src.abstention import is_abstention_answer

        assert is_abstention_answer(answer), f"unanswerable case did not abstain: {answer!r}"
    failed = below_threshold(blocking, medians)
    assert not failed, f"median-of-{JUDGE_RUNS} below threshold (median, threshold): {failed}"
