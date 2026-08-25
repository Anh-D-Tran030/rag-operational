"""30-question retrieval-only eval (MRR + nDCG@10).

Runs the full pipeline via handle_query for every entry in the golden dataset
and computes MRR and nDCG@10 directly from the returned citations vs the golden
evidence_chunk_ids. No LLM judge involved — the RAGAS judge-dependent metrics
(faithfulness / answer_relevancy / context_precision / context_recall) deadlock
under the current gpt-oss-120b + Instructor retry combination when a completion
hits max_tokens, so this script deliberately skips them.

Prints a per-question row and an aggregate summary. Writes .logs/retrieval_eval_30q.json.

Usage:
    .venv/bin/python scripts/run_retrieval_eval.py
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from src.config import Settings
from src.evaluation.ragas_runner import _citation_ids, _ndcg_at_k, _reciprocal_rank
from src.generation.query_handler import handle_query
from src.observability.tracer import RAGTracer
from src.retrieval.cache import SemanticCache

GOLDEN_PATH = "eval_data/golden_dataset.json"


async def main() -> int:
    settings = Settings(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        semantic_cache_url="",
    )
    if not settings.openrouter_api_key:
        print("OPENROUTER_API_KEY not set.", file=sys.stderr)
        return 1

    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    tracer = RAGTracer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    cache = SemanticCache(cache_url="", embedding_model=settings.embedding_model)

    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)

    CONCURRENCY = 5  # 5 parallel queries; bounded by OpenRouter rate limits
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: list[dict | None] = [None] * len(golden)
    t_start = time.monotonic()
    completed = 0

    async def _run_one(i: int, case: dict) -> None:
        nonlocal completed
        async with sem:
            t0 = time.monotonic()
            try:
                response = await handle_query(
                    query=case["question"],
                    qdrant_client=qdrant,
                    settings=settings,
                    tracer=tracer,
                    cache=cache,
                )
            except Exception as exc:
                completed += 1
                print(f"[{completed:02d}/{len(golden)}] ERROR q{i}: {exc}", flush=True)
                rows[i] = {"i": i + 1, "question": case["question"], "error": str(exc)}
                return
            dt = time.monotonic() - t0

            citations = response.get("citations", []) or []
            predicted = _citation_ids(citations)
            relevant = set(case.get("evidence_chunk_ids", []) or [])
            rr = _reciprocal_rank(predicted, relevant) if relevant else None
            ndcg = _ndcg_at_k(predicted, relevant, k=10) if relevant else None

            rows[i] = {
                "i": i + 1,
                "question": case["question"],
                "route": response.get("route"),
                "tool": response.get("tool"),
                "n_citations": len(citations),
                "predicted_top5": predicted[:5],
                "relevant_size": len(relevant),
                "mrr": rr,
                "ndcg": ndcg,
                "latency_s": round(dt, 2),
            }
            completed += 1
            rr_disp = f"{rr:.3f}" if rr is not None else "n/a"
            ndcg_disp = f"{ndcg:.3f}" if ndcg is not None else "n/a"
            print(
                f"[{completed:02d}/{len(golden)}] tool={response.get('tool'):<12} "
                f"cites={len(citations):>2}  mrr={rr_disp:>5}  ndcg={ndcg_disp:>5}  "
                f"{dt:5.1f}s",
                flush=True,
            )

    await asyncio.gather(*[_run_one(i, case) for i, case in enumerate(golden)])

    rows_clean = [r for r in rows if r is not None]
    mrr_scores = [r["mrr"] for r in rows_clean if r.get("mrr") is not None]
    ndcg_scores = [r["ndcg"] for r in rows_clean if r.get("ndcg") is not None]

    total_dt = time.monotonic() - t_start
    mrr_mean = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    ndcg_mean = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    print()
    print("=" * 56)
    print(f"scored {len(golden)} questions ({len(mrr_scores)} with evidence_chunk_ids)")
    print(f"wall time: {total_dt:.0f}s")
    print(f"{'Metric':<20} | {'Score':>6}")
    print("-" * 32)
    print(f"{'mrr':<20} | {mrr_mean:>6.2f}")
    print(f"{'ndcg_at_10':<20} | {ndcg_mean:>6.2f}")
    print()
    print("(faithfulness / answer_relevancy / context_precision / context_recall")
    print(" not refreshed — judge deadlock; see .logs/eval_30q_gpu_timeout_*.log)")

    Path(".logs").mkdir(exist_ok=True)
    out = Path(".logs/retrieval_eval_30q.json")
    out.write_text(
        json.dumps(
            {
                "n_questions": len(golden),
                "n_with_evidence": len(mrr_scores),
                "wall_time_s": total_dt,
                "mrr": mrr_mean,
                "ndcg_at_10": ndcg_mean,
                "rows": rows_clean,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwritten: {out}")

    await qdrant.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
