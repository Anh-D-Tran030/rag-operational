"""DeepEval CI gate.

Loads the golden dataset, runs `handle_query` for each QA pair, and asserts
faithfulness, answer relevancy, contextual precision and contextual recall.
Faithfulness below 0.9 hard-blocks the PR. Per-case scores are pushed to W&B
when WANDB_API_KEY is set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parents[2] / "eval_data" / "golden_dataset.json"


def _load_golden() -> list[dict]:
    with GOLDEN_PATH.open() as fh:
        return json.load(fh)


def _ids(cases: list[dict]) -> list[str]:
    return [f"{i:02d}-{c.get('category', 'unknown')}" for i, c in enumerate(cases)]


GOLDEN = _load_golden()


def _require(module: str):
    try:
        return __import__(module)
    except ImportError:
        pytest.skip(f"{module} not installed (install with `pip install -e .[eval]`)")


def _wandb_run():
    if not os.environ.get("WANDB_API_KEY"):
        return None
    wandb = _require("wandb")
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
async def test_deepeval_gate(case: dict) -> None:
    deepeval = _require("deepeval")
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    result = await _run_query(case["question"])
    answer = str(result.get("answer", ""))
    retrieval_context = [
        str(c.get("snippet", c.get("sql_used", "")))
        for c in result.get("citations", [])
    ] or [""]

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=answer,
        expected_output=case["ground_truth"],
        retrieval_context=retrieval_context,
    )

    metrics = [
        FaithfulnessMetric(threshold=0.9),
        AnswerRelevancyMetric(threshold=0.75),
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
    ]
    for metric in metrics:
        metric.measure(test_case)

    run = _wandb_run()
    if run is not None:
        run.log(
            {
                "category": case.get("category", "unknown"),
                "faithfulness": metrics[0].score,
                "answer_relevancy": metrics[1].score,
                "contextual_precision": metrics[2].score,
                "contextual_recall": metrics[3].score,
            }
        )

    deepeval.assert_test(test_case, metrics)
