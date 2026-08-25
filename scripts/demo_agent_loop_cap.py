"""Demo: agentic grader loop hits a safety cap and returns "insufficient context".

Runs vector_search_pipeline with complexity="complex" on a query that is
off-topic relative to the ingested corpus (venue operational logs + SEC 10-Qs).
The retrieval grader repeatedly rejects the returned chunks, and the loop
terminates on the first cap it reaches — either settings.max_retrieval_iterations
or settings.token_budget_per_query, whichever binds first.

Then hands the resulting empty context to handle_query so the caller sees the
final "insufficient context" outcome the API would return in production.

Usage:
    .venv/bin/python scripts/demo_agent_loop_cap.py
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
from src.generation.query_handler import handle_query
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache
from src.retrieval.pipeline import vector_search_pipeline

# Off-topic for both the venue-ops and SEC-10Q corpora. Retrieval will still
# return the closest 7 chunks, but the grader should reject them all, forcing
# the loop to iterate until a cap fires.
QUERY = (
    "Explain the venue's cryptocurrency payment reconciliation policy and "
    "the KYC procedure for on-chain settlements exceeding 10 BTC per shift."
)


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
    tracer = RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    cache = SemanticCache(cache_url="", embedding_model=settings.embedding_model)

    print("=" * 72)
    print("DEMO: agentic grader loop cap")
    print("=" * 72)
    print(f"query:                       {QUERY}")
    print(f"max_retrieval_iterations:    {settings.max_retrieval_iterations}")
    print(f"token_budget_per_query:      {settings.token_budget_per_query}")
    print(f"rerank_top_n:                {settings.rerank_top_n}")
    print()

    # (1) call the pipeline directly with complexity="complex" so the grader
    #     loop is guaranteed to engage.
    print("-- vector_search_pipeline(complexity='complex') --")
    t0 = time.monotonic()
    chunks, meta = await vector_search_pipeline(
        query=QUERY,
        client=qdrant,
        settings=settings,
        tracer=tracer,
        trace_id="demo-loop-cap",
        complexity="complex",
    )
    dt = time.monotonic() - t0
    print(json.dumps(meta, indent=2, default=str))
    print(f"n_chunks_returned:           {len(chunks)}")
    print(f"latency_s:                   {dt:.2f}")
    hit_iter_cap = meta["iterations"] >= settings.max_retrieval_iterations
    hit_token_cap = meta["tokens_used"] >= settings.token_budget_per_query
    print(f"hit max_retrieval_iter cap:  {hit_iter_cap}")
    print(f"hit token_budget cap:        {hit_token_cap}")
    print()

    # (2) end-to-end via handle_query so we see the caller-visible outcome.
    print("-- handle_query end-to-end --")
    t0 = time.monotonic()
    response = await handle_query(
        query=QUERY,
        qdrant_client=qdrant,
        settings=settings,
        tracer=tracer,
        cache=cache,
    )
    dt = time.monotonic() - t0
    print(f"answer:                      {response.get('answer')}")
    print(f"tool:                        {response.get('tool')}")
    print(f"route:                       {response.get('route')}")
    print(f"n_citations:                 {len(response.get('citations', []))}")
    print(f"latency_ms:                  {response.get('latency_ms', 0):.0f}")
    print(f"end-to-end wall-time:        {dt:.2f}s")

    Path(".logs").mkdir(exist_ok=True)
    out = Path(".logs/demo_agent_loop_cap.json")
    out.write_text(
        json.dumps(
            {
                "query": QUERY,
                "settings": {
                    "max_retrieval_iterations": settings.max_retrieval_iterations,
                    "token_budget_per_query": settings.token_budget_per_query,
                    "rerank_top_n": settings.rerank_top_n,
                },
                "pipeline_meta": meta,
                "hit_iter_cap": hit_iter_cap,
                "hit_token_cap": hit_token_cap,
                "handle_query_response": {
                    k: v for k, v in response.items() if k != "citations"
                },
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
