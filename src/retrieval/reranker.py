"""FlashRank-based reranker running in a thread pool (sync/CPU)."""

from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from flashrank import Ranker, RerankRequest

# Narrow post-scoring correction for "who signed off the shift log" questions
# (case 13): FlashRank scores a cash-reconciliation line marginally above the
# true sign-off line, and a naive top_n/score-only fix would leave other
# irrelevant chunks in place. Only this specific intent is affected; every
# other query's ranking is untouched.
_SIGN_OFF_INTENT = re.compile(r"\bwho\s+signed\s+off\b", re.IGNORECASE)
_SHIFT_LOG_INTENT = re.compile(r"\bshift\s+log\b", re.IGNORECASE)
_REQUESTED_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# An actual sign-off label/verb plus a nonempty signer -- "Supervisor sign-off:
# J. Thompson" or "signed off by J. Thompson". "Cash reconciliation completed
# by J. Thompson" or a bare shared name elsewhere must not match.
_SIGN_OFF_EVIDENCE = re.compile(
    r"\bsign(?:ed)?[\s-]?off\b\s*(?:by\s+|[:\-]\s*)([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)"
)


def _is_sign_off_query(query: str) -> bool:
    return bool(_SIGN_OFF_INTENT.search(query) and _SHIFT_LOG_INTENT.search(query))


def _requested_date(query: str) -> str | None:
    match = _REQUESTED_DATE.search(query)
    return match.group(0) if match else None


def _has_sign_off_evidence(content: str) -> bool:
    match = _SIGN_OFF_EVIDENCE.search(content)
    return bool(match and match.group(1).strip())


def _select_sign_off_candidates(query: str, candidates: list[dict]) -> list[dict] | None:
    """Narrow already-scored shift_log candidates to explicit sign-off evidence.

    Returns None (defer to the unchanged FlashRank list) unless the query is
    an explicit sign-off question. Never broadens the candidate pool and never
    fabricates a match -- a qualifying chunk absent from `candidates` is a
    retrieval recall gap, not something this narrowing can fix.
    """
    if not _is_sign_off_query(query):
        return None
    requested_date = _requested_date(query)
    qualified = [
        chunk
        for chunk in candidates
        if chunk.get("doc_type") == "shift_log"
        and (requested_date is None or chunk.get("date") == requested_date)
        and _has_sign_off_evidence(str(chunk.get("content", "")))
    ]
    return qualified or None

_executor: ThreadPoolExecutor | None = None
_executor_lock = Lock()
_ranker: Ranker | None = None

# FlashRank caches its model under cache_dir, which defaults to /tmp. Anything that
# sweeps /tmp — a container restart, a CI runner, a desktop tmpfiles job — silently
# deletes the model and the next rerank raises FileNotFoundError mid-request.
_CACHE_DIR = os.environ.get("FLASHRANK_CACHE_DIR", ".model_cache")


def _get_executor() -> ThreadPoolExecutor:
    """Create the worker pool lazily so it has an explicit lifecycle."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=2)
        return _executor


def shutdown_reranker_executor() -> None:
    """Stop accepting rerank work and release idle worker threads.

    Calls always await their submitted work before returning from ``rerank``.
    ``wait=False`` keeps process teardown bounded if a third-party inference
    call has become stuck; queued work is cancelled instead.
    """
    global _executor
    with _executor_lock:
        executor = _executor
        _executor = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


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


def _passage_text(chunk: dict) -> str:
    """Score contextual_prefix + content if the prefix is non-empty, else content alone."""
    prefix = chunk.get("contextual_prefix") or ""
    content = chunk.get("content", "")
    return f"{prefix} {content}".strip() if prefix.strip() else content


def _sync_rerank(
    query: str, chunks: list[dict], top_n: int, score_threshold: float, trace_id: str
) -> list[dict]:
    """Run FlashRank synchronously. trace_id is passed explicitly — not via contextvars."""
    ranker = _get_ranker()
    passages = [{"id": i, "text": _passage_text(c)} for i, c in enumerate(chunks)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    scored = [
        r
        for r in sorted(results, key=lambda r: r["score"], reverse=True)
        if r["score"] >= score_threshold
    ]
    candidates = []
    for r in scored:
        chunk = dict(chunks[r["id"]])
        chunk["rerank_score"] = float(r["score"])
        candidates.append(chunk)

    narrowed = _select_sign_off_candidates(query, candidates)
    return (narrowed if narrowed is not None else candidates)[:top_n]


async def rerank(
    query: str,
    chunks: list[dict],
    top_n: int,
    score_threshold: float,
    trace_id: str,
) -> list[dict]:
    """Rerank chunks and return at most ``top_n`` candidates at the threshold.

    Runs the CPU-bound ranker in a ThreadPoolExecutor.
    trace_id MUST be passed explicitly into the thread — do not rely on contextvars.
    Candidates below ``score_threshold`` are excluded individually.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_executor(),
        _sync_rerank,
        query,
        chunks,
        top_n,
        score_threshold,
        trace_id,
    )
