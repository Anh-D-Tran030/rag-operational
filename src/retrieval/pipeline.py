"""Vector search pipeline orchestrator for the RAG system.

Coordinates query expansion, embedding, hybrid search, reranking, and an
optional self-correcting grader loop for complex queries.
"""

from __future__ import annotations

import asyncio
import time

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.observability.metrics import rag_chunks_per_query, rag_retrieval_confidence
from src.observability.tracer import RAGTracer
from src.retrieval.embedder import embed_query
from src.retrieval.query_expander import expand_query
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import hybrid_search

_GRADER_PASS_RATIO = 0.5  # fraction of "yes" verdicts required to accept context


async def _search_one(
    query: str, client: AsyncQdrantClient, settings: Settings, filters: dict | None = None
) -> list[dict]:
    embedding = await embed_query(query, settings.embedding_model)
    return await hybrid_search(
        query=query,
        query_embedding=embedding,
        client=client,
        collection=settings.qdrant_collection,
        top_k=settings.retrieval_top_k,
        filters=filters,
    )


async def _expand_and_search(
    query: str, client: AsyncQdrantClient, settings: Settings, filters: dict | None = None
) -> list[dict]:
    """Expand query, search for each expansion, deduplicate by content_hash."""
    expansions = await expand_query(query, settings.local_llm())
    all_results = await asyncio.gather(
        *[_search_one(q, client, settings, filters) for q in expansions]
    )
    seen: set[str] = set()
    merged: list[dict] = []
    for results in all_results:
        for chunk in results:
            key = chunk.get("content_hash") or chunk.get("content", "")[:64]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
    return merged


async def _grade_chunks(
    query: str, chunks: list[dict], settings: Settings
) -> tuple[list[dict], bool]:
    """Grade each chunk for relevance; return (accepted_chunks, context_sufficient)."""
    from src.generation.local_llm import generate_local  # noqa: PLC0415
    from src.generation.prompts import build_retrieval_grader_prompt  # noqa: PLC0415

    yes_count = 0
    accepted: list[dict] = []
    for chunk in chunks:
        try:
            verdict = await generate_local(
                build_retrieval_grader_prompt(query, chunk),
                settings.local_llm(),
            )
        except Exception:
            verdict = "yes"
        if "yes" in verdict.strip().lower():
            yes_count += 1
            accepted.append(chunk)
    return accepted, (yes_count / max(len(chunks), 1)) >= _GRADER_PASS_RATIO


async def vector_search_pipeline(
    query: str,
    client: AsyncQdrantClient,
    settings: Settings,
    tracer: RAGTracer,
    trace_id: str,
    complexity: str = "simple",
    filters: dict | None = None,
) -> tuple[list[dict], dict]:
    """Run the full retrieval pipeline for a single query.

    Steps:
      1. Expand the query and hybrid-search each expansion; deduplicate by content_hash.
      2. Rerank the merged candidate set.
      3. For complex queries: run a grader loop (max settings.max_retrieval_iterations).
         Reformulate the query and re-retrieve when context is insufficient, stopping
         when context is accepted or the token budget is exhausted.
      4. Record Prometheus metrics and log a Langfuse span.

    Returns (chunks, metadata_dict).
    """
    t0 = time.monotonic()
    current_query = query
    chunks: list[dict] = []
    tokens_used = 0
    iterations = 0

    for iteration in range(max(1, settings.max_retrieval_iterations)):
        iterations = iteration + 1
        candidates = await _expand_and_search(current_query, client, settings, filters)
        chunks = await rerank(
            query=current_query,
            chunks=candidates,
            top_n=settings.rerank_top_n,
            score_threshold=settings.rerank_score_threshold,
            trace_id=trace_id,
        )

        if complexity != "complex" or not chunks:
            break

        accepted, sufficient = await _grade_chunks(current_query, chunks, settings)
        tokens_used += len(chunks) * 250  # ~250 tokens per grader prompt

        if tokens_used >= settings.token_budget_per_query or sufficient:
            if accepted:
                chunks = accepted
            break

        # reformulate: take the last expansion as the new query; stop if unchanged
        expansions = await expand_query(current_query, settings.local_llm(), n_expansions=1)
        new_query = expansions[-1] if len(expansions) > 1 else current_query
        if new_query == current_query:
            break
        current_query = new_query

    latency_s = time.monotonic() - t0

    if chunks:
        score = float(chunks[0].get("rerank_score", chunks[0].get("score", 0.0)))
        rag_retrieval_confidence.observe(score)
    rag_chunks_per_query.observe(float(len(chunks)))

    tracer.log_span(
        trace_obj=None,
        name="retrieval",
        input={"query": query, "complexity": complexity},
        output={"n_chunks": len(chunks)},
        latency_ms=latency_s * 1000,
    )

    return chunks, {
        "retrieval_latency_s": latency_s,
        "n_chunks": len(chunks),
        "iterations": iterations,
        "tokens_used": tokens_used,
        "final_query": current_query,
    }
