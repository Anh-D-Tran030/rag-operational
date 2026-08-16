"""FlashRank-based reranker running in a thread pool (sync/CPU)."""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flashrank import Ranker, RerankRequest

_executor = ThreadPoolExecutor(max_workers=2)
_ranker: Ranker | None = None

# FlashRank caches its model under cache_dir, which defaults to /tmp. Anything that
# sweeps /tmp — a container restart, a CI runner, a desktop tmpfiles job — silently
# deletes the model and the next rerank raises FileNotFoundError mid-request.
_CACHE_DIR = os.environ.get("FLASHRANK_CACHE_DIR", ".model_cache")


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        _ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir=_CACHE_DIR,
            max_length=512,
        )
    return _ranker


def _sync_rerank(
    query: str, chunks: list[dict], top_n: int, score_threshold: float, trace_id: str
) -> list[dict]:
    """Run FlashRank synchronously. trace_id is passed explicitly — not via contextvars."""
    ranker = _get_ranker()
    passages = [{"id": i, "text": c.get("content", "")} for i, c in enumerate(chunks)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    if not results or max(r["score"] for r in results) < score_threshold:
        return []

    scored = sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]
    output = []
    for r in scored:
        chunk = dict(chunks[r["id"]])
        chunk["rerank_score"] = float(r["score"])
        output.append(chunk)
    return output


async def rerank(
    query: str,
    chunks: list[dict],
    top_n: int,
    score_threshold: float,
    trace_id: str,
) -> list[dict]:
    """Rerank chunks using FlashRank and return top_n above score_threshold.

    Runs the CPU-bound ranker in a ThreadPoolExecutor.
    trace_id MUST be passed explicitly into the thread — do not rely on contextvars.
    Returns [] if the maximum rerank score is below score_threshold.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _sync_rerank,
        query,
        chunks,
        top_n,
        score_threshold,
        trace_id,
    )
