"""Run RAGAS + retrieval eval against the live stack and print a score table.

Requires QDRANT_URL, OLLAMA_URL and OPENROUTER_API_KEY. Both the pipeline under
test and the RAGAS judge are routed through OpenRouter; answer_relevancy needs an
embedder, which runs locally because OpenRouter serves chat only.

Exits 1 if a PLAN threshold (0.9/0.8/0.8/0.8) fails or a metric is missing/NaN.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import sys
import tempfile

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.evaluation.ragas_runner import run_ragas_eval
from src.generation.query_handler import handle_query
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache

INSTALL_HINT = "install with `pip install -e .[eval]`"
# ragas_runner/_build_judge import these lazily inside function bodies, so a
# module-level try/except here would never see the missing extra (PLAN-003 §2.4).
EVAL_MODULES = ("ragas", "langchain_community", "datasets")


def _missing_eval_extra() -> list[str]:
    return [m for m in EVAL_MODULES if importlib.util.find_spec(m) is None]

GOLDEN_PATH = "eval_data/golden_dataset.json"
# PLAN thresholds keyed by ragas' real score-dict names (it names its
# precision metric llm_context_precision_with_reference, not context_precision).
THRESHOLDS = {
    "faithfulness": 0.9,
    "answer_relevancy": 0.8,
    "llm_context_precision_with_reference": 0.8,
    "context_recall": 0.8,
}
METRIC_LABELS = {"llm_context_precision_with_reference": "context_precision"}


def _build_handle_fn(
    settings: Settings,
    qdrant_client: AsyncQdrantClient,
    tracer: RAGTracer,
    cache: SemanticCache,
):
    async def _fn(question: str) -> dict:
        return await handle_query(
            query=question,
            qdrant_client=qdrant_client,
            settings=settings,
            tracer=tracer,
            cache=cache,
        )

    return _fn


def _build_judge(settings: Settings):
    """Return (llm, embeddings) for RAGAS, both independent of OpenAI.

    Uses the RAGAS 0.4 llm_factory API (InstructorLLM over openai.OpenAI) so that
    the base_url is honoured correctly when routing through OpenRouter.

    Embeddings are wrapped with LangchainEmbeddingsWrapper so that embed_query() is
    available — AnswerRelevancy calls embed_query() internally and the native RAGAS
    HuggingFaceEmbeddings only exposes embed_text().
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmbeddings
    from openai import OpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        max_retries=0,  # RAGAS handles retries at a higher level; SDK retries cause loops
        timeout=60.0,   # hard per-request timeout — the SDK default is 10 min, and
                        # one hung OpenRouter call otherwise deadlocks the eval
    )
    llm = llm_factory(model=settings.judge_model, client=client)
    device = os.environ.get("RAG_EMBEDDER_DEVICE")
    if not device:
        try:
            import torch  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    embeddings = LangchainEmbeddingsWrapper(
        LCHFEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
        )
    )
    print(f"[eval] RAGAS embedder device: {device}", flush=True)
    return llm, embeddings


def _subset_path(limit: int) -> str:
    """Write the first `limit` golden entries to a temp file and return its path."""
    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(golden[:limit], tmp)
    tmp.close()
    return tmp.name


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only the first N golden entries (0 = all). Use for a cheap smoke run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples per RAGAS evaluate() call; smaller = less peak RAM. Default 10.",
    )
    args = parser.parse_args()

    missing = _missing_eval_extra()
    if missing:
        print(f"missing eval packages {missing} ({INSTALL_HINT})", file=sys.stderr)
        return 1

    settings = Settings(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        semantic_cache_url="",  # disabled for eval
    )
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY is not set; nothing to evaluate.", file=sys.stderr)
        return 1

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    tracer = RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    cache = SemanticCache(cache_url="", embedding_model=settings.embedding_model)

    handle_fn = _build_handle_fn(settings, qdrant_client, tracer, cache)
    judge_llm, judge_embeddings = _build_judge(settings)
    dataset_path = _subset_path(args.limit) if args.limit else GOLDEN_PATH

    scores = await run_ragas_eval(
        golden_dataset_path=dataset_path,
        handle_query_fn=handle_fn,
        wandb_run=None,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        batch_size=args.batch_size,
    )

    header = f"{'Metric':<20} | {'Score':>6} | {'n':>4}"
    print(header)
    print("-" * len(header))
    retrieval_n = scores.get("retrieval_cases", 0)
    denominators = {"mrr": retrieval_n, "ndcg_at_10": retrieval_n}
    # PLAN-003 §2.4/§8: a missing score, a NaN, or a zero denominator is a
    # failed acceptance check, not a silently averaged-away row.
    failures = []
    for key in (*THRESHOLDS, "mrr", "ndcg_at_10"):
        label = METRIC_LABELS.get(key, key)
        value = scores.get(key)
        n = scores.get(f"{key}_n", denominators.get(key, 0))
        print(f"{label:<20} | {value if value is not None else float('nan'):>6.2f} | {n:>4.0f}")
        threshold = THRESHOLDS.get(key)
        if threshold is None:
            continue
        if value is None or math.isnan(value) or n <= 0:
            failures.append(f"{label}: missing/NaN score or zero denominator (n={n})")
        elif value < threshold:
            failures.append(f"{label}: {value:.2f} < {threshold} threshold")
    print(
        f"\nscored {scores.get('generation_cases', 0):.0f} questions "
        f"({retrieval_n:.0f} with evidence ids)"
    )
    for line in failures:
        print(f"FAIL: {line}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
