"""Demo: retrieval thrash — grader loop reformulates without converging.

Sends a deliberately vague multi-part query with contradictory sub-questions
straight into the complex grader loop. Each iteration reformulates the query
via expand_query and re-retrieves; because the sub-questions pull toward
different document types, no single reformulation satisfies the grader.

The script traces the (iteration, reformulated_query, n_chunks, accepted_count)
sequence and reports which safety mechanism actually terminates the loop —
this is *observation*, not assumption. Possible terminators:

  * iterations == settings.max_retrieval_iterations
  * tokens_used  >= settings.token_budget_per_query
  * reformulation returns identical query (stagnation break)
  * grader ratio >= 0.5 on some iteration (loop converges after all)

Usage:
    .venv/bin/python scripts/demo_retrieval_thrash.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.observability.tracer import RAGTracer
from src.retrieval.embedder import embed_query
from src.retrieval.query_expander import expand_query
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import hybrid_search

# Deliberately vague and multi-part: mixes 10-Q financial content, venue ops
# procedures, and a made-up compliance term. Expected to make each expand_query
# pass drift in a different direction.
QUERY = (
    "Summarise the interplay between quarterly revenue variance, floor "
    "supervisor rota deviation, and the venue's XR-27 compliance clause "
    "for evening shifts, with emphasis on discrepancies."
)


async def _grade(query: str, chunks: list[dict], settings: Settings) -> tuple[int, list[dict]]:
    """Yes/no grader — same logic as pipeline._grade_chunks but instrumented."""
    from src.generation.local_llm import generate_local
    from src.generation.prompts import build_retrieval_grader_prompt

    accepted: list[dict] = []
    yes_count = 0
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
    return yes_count, accepted


async def main() -> int:
    settings = Settings(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        semantic_cache_url="",
    )
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY not set; cannot run demo.", file=sys.stderr)
        return 1

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    _tracer = RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    print("=" * 72)
    print("DEMO: retrieval thrash — reformulation without convergence")
    print("=" * 72)
    print(f"query: {QUERY}")
    print(f"max_retrieval_iterations:  {settings.max_retrieval_iterations}")
    print(f"token_budget_per_query:    {settings.token_budget_per_query}")
    print(f"grader_pass_ratio:         0.5")
    print()

    # Instrumented reimplementation of the grader loop so we can capture the
    # per-iteration reformulated query and grader ratio.
    current = QUERY
    seen_queries: list[str] = [current]
    tokens_used = 0
    trace: list[dict] = []
    terminator = "iteration_cap"
    accepted_final: list[dict] = []

    t0 = time.monotonic()
    for i in range(1, max(1, settings.max_retrieval_iterations) + 1):
        # 1 primary + N expansions, then merge (mirrors _expand_and_search).
        expansions = await expand_query(current, settings.local_llm())
        all_hits: list[dict] = []
        seen: set[str] = set()
        for q in expansions:
            emb = await embed_query(q, settings.embedding_model)
            hits = await hybrid_search(
                query=q,
                query_embedding=emb,
                client=qdrant,
                collection=settings.qdrant_collection,
                top_k=settings.retrieval_top_k,
                filters=None,
            )
            for h in hits:
                k = h.get("content_hash") or h.get("content", "")[:64]
                if k not in seen:
                    seen.add(k)
                    all_hits.append(h)

        reranked = await rerank(
            query=current,
            chunks=all_hits,
            top_n=settings.rerank_top_n,
            score_threshold=settings.rerank_score_threshold,
            trace_id="demo-thrash",
        )
        yes_count, accepted = await _grade(current, reranked, settings)
        ratio = yes_count / max(len(reranked), 1)
        tokens_used += len(reranked) * 250
        trace.append(
            {
                "iter": i,
                "query": current,
                "n_reranked": len(reranked),
                "n_accepted": yes_count,
                "grader_ratio": round(ratio, 3),
                "tokens_used_running": tokens_used,
            }
        )

        if not reranked:
            terminator = "empty_reranked"
            accepted_final = accepted
            break
        if ratio >= 0.5:
            terminator = "grader_converged"
            accepted_final = accepted
            break
        if tokens_used >= settings.token_budget_per_query:
            terminator = "token_budget"
            accepted_final = accepted
            break
        if i >= settings.max_retrieval_iterations:
            terminator = "iteration_cap"
            accepted_final = accepted
            break

        reformulated = await expand_query(current, settings.local_llm(), n_expansions=1)
        new_query = reformulated[-1] if len(reformulated) > 1 else current
        if new_query == current or new_query in seen_queries:
            terminator = "reformulation_stagnated"
            accepted_final = accepted
            break
        current = new_query
        seen_queries.append(current)

    dt = time.monotonic() - t0

    print("-- per-iteration trace --")
    for row in trace:
        print(
            f"  iter {row['iter']}: reranked={row['n_reranked']:2d} "
            f"accepted={row['n_accepted']:2d} ratio={row['grader_ratio']:.2f} "
            f"tokens={row['tokens_used_running']}"
        )
        print(f"    query -> {row['query'][:120]}")
    print()
    print(f"iterations_run:        {len(trace)}")
    print(f"tokens_used:           {tokens_used}")
    print(f"loop terminator:       {terminator}")
    print(f"final_accepted_chunks: {len(accepted_final)}")
    print(f"wall_time_s:           {dt:.2f}")

    Path(".logs").mkdir(exist_ok=True)
    out = Path(".logs/demo_retrieval_thrash.json")
    out.write_text(
        json.dumps(
            {
                "query": QUERY,
                "settings": {
                    "max_retrieval_iterations": settings.max_retrieval_iterations,
                    "token_budget_per_query": settings.token_budget_per_query,
                },
                "trace": trace,
                "iterations_run": len(trace),
                "tokens_used": tokens_used,
                "terminator": terminator,
                "final_accepted": len(accepted_final),
                "wall_time_s": dt,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nevidence written to: {out}")

    await qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
