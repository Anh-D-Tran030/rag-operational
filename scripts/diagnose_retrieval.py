"""Stage-level retrieval diagnosis against the golden evidence anchors.

Runs dense search, the production BM25 candidate fetch, RRF, and FlashRank for
each golden case carrying ``evidence_chunk_ids``.  It writes JSON plus a compact
Markdown table so that a recall miss can be assigned to one retrieval stage.

Usage:
    .venv/bin/python scripts/diagnose_retrieval.py
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi

from src.config import Settings
from src.retrieval.embedder import embed_query
from src.retrieval.reranker import rerank
from src.retrieval.vector_search import _hit_to_dict, _rrf_merge, _scroll_all_bm25_points

GOLDEN_PATH = Path("eval_data/golden_dataset.json")
OUTPUT_JSON = Path(".logs/retrieval_diagnosis.json")
OUTPUT_MD = Path(".logs/retrieval_diagnosis.md")


class _FakePoint:
    """Adapt a Qdrant scroll record to the minimal RRF scored-point interface."""

    def __init__(self, score: float, point: object) -> None:
        self.id = point.id  # type: ignore[attr-defined]
        self.payload = point.payload  # type: ignore[attr-defined]
        self.score = score


def _evidence_ids(chunks: list[dict]) -> list[str]:
    """Flatten chunk evidence anchors in retrieval rank order."""
    ids: list[str] = []
    for chunk in chunks:
        section_ids = chunk.get("section_ids") or []
        if section_ids:
            ids.extend(str(item) for item in section_ids)
        elif chunk.get("chunk_id"):
            ids.append(str(chunk["chunk_id"]))
    return ids


def _first_gold_rank(chunks: list[dict], gold: set[str]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if set(_evidence_ids([chunk])) & gold:
            return rank
    return None


async def _stages(
    query: str,
    client: AsyncQdrantClient,
    settings: Settings,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Reproduce one production search pass while retaining intermediate ranks."""
    fetch_k = settings.retrieval_top_k * 4
    embedding = await embed_query(query, settings.embedding_model)
    dense_response = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=embedding,
        using="dense",
        limit=fetch_k,
        with_payload=True,
    )
    dense_points = dense_response.points

    scroll_points = await _scroll_all_bm25_points(
        client, settings.qdrant_collection, qdrant_filter=None
    )
    corpus = [
        point.payload.get("bm25_text", "") if point.payload else ""
        for point in scroll_points
    ]
    bm25 = BM25Okapi([text.lower().split() for text in corpus]) if corpus else None
    bm25_points = (
        [
            _FakePoint(score, point)
            for score, point in sorted(
                zip(bm25.get_scores(query.lower().split()), scroll_points),
                key=lambda pair: pair[0],
                reverse=True,
            )[:fetch_k]
        ]
        if bm25
        else []
    )
    rrf_points = _rrf_merge(dense_points, bm25_points, top_k=settings.retrieval_top_k)
    dense = [_hit_to_dict(point) for point in dense_points]
    bm25_hits = [_hit_to_dict(point) for point in bm25_points]
    rrf = [_hit_to_dict(point) for point in rrf_points]
    # Diagnostic cutoffs do not change production configuration.  Score the
    # fixed RRF candidates once to the largest reported cutoff, then derive
    # recall@5/@7/@10 from that one ranking.
    reranked = await rerank(
        query=query,
        chunks=rrf,
        top_n=10,
        score_threshold=settings.rerank_score_threshold,
        trace_id="retrieval-diagnosis",
    )
    return dense, bm25_hits, rrf, reranked


def _recall_at(chunks: list[dict], gold: set[str], cutoff: int) -> bool:
    return bool(set(_evidence_ids(chunks[:cutoff])) & gold)


def _metrics(rows: list[dict]) -> dict[str, float | int]:
    """Calculate deterministic anchor-based diagnostics by pipeline stage."""
    total = len(rows)
    if not total:
        return {"case_count": 0}

    def rate(values: list[bool]) -> float:
        return round(sum(values) / total, 4)

    precision_values = [row["reranker_context_precision"] for row in rows]
    return {
        "case_count": total,
        "rrf_candidate_recall": rate([row["rrf_hit"] for row in rows]),
        "reranker_recall_at_5": rate([row["reranker_hit_at_5"] for row in rows]),
        "reranker_recall_at_7": rate([row["reranker_hit_at_7"] for row in rows]),
        "reranker_recall_at_10": rate([row["reranker_hit_at_10"] for row in rows]),
        # Precision is the mean share of returned context chunks with a supplied
        # evidence anchor; it is not a model-judged answer-quality metric.
        "context_precision": round(sum(precision_values) / total, 4),
        "empty_result_rate": rate([row["empty_reranked"] for row in rows]),
    }


def _markdown(rows: list[dict], metrics: dict[str, float | int]) -> str:
    lines = [
        "# Retrieval diagnosis",
        "",
        "RRF candidate recall is measured before reranking. Reranker recall is reported "
        "at each cutoff against the same fixed RRF candidates. Context precision is the "
        "mean fraction of returned chunks carrying a declared evidence anchor.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for name in (
        "case_count",
        "rrf_candidate_recall",
        "reranker_recall_at_5",
        "reranker_recall_at_7",
        "reranker_recall_at_10",
        "context_precision",
        "empty_result_rate",
    ):
        value = metrics[name]
        lines.append(
            f"| {name} | {value:.4f} |"
            if isinstance(value, float)
            else f"| {name} | {value} |"
        )
    lines.extend([
        "",
        "| # | Query | Gold evidence_chunk_ids | Retrieved chunk_ids (reranked) | "
        "Gold rank (RRF / rerank) | Dense | BM25 | RRF | Reranker |",
        "|---:|---|---|---|---|:---:|:---:|:---:|:---:|",
    ])
    for row in rows:
        query = row["query"].replace("|", "\\|").replace("\n", " ")
        gold = ", ".join(row["gold_evidence_chunk_ids"])
        retrieved = ", ".join(row["retrieved_chunk_ids"])
        rrf_rank = row["rrf_gold_rank"] or "—"
        rerank_rank = row["reranker_gold_rank"] or "—"
        lines.append(
            f"| {row['case']} | {query} | {gold} | {retrieved} | {rrf_rank} / {rerank_rank} | "
            f"{'Y' if row['dense_hit'] else 'N'} | {'Y' if row['bm25_hit'] else 'N'} | "
            f"{'Y' if row['rrf_hit'] else 'N'} | {'Y' if row['reranker_hit'] else 'N'} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> int:
    settings = Settings(qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    golden = json.loads(GOLDEN_PATH.read_text())
    cases = [case for case in golden if case.get("evidence_chunk_ids")]
    client = AsyncQdrantClient(url=settings.qdrant_url)
    rows: list[dict] = []
    try:
        for number, case in enumerate(cases, start=1):
            gold = set(case["evidence_chunk_ids"])
            dense, bm25, rrf, reranked = await _stages(case["question"], client, settings)
            row = {
                "case": number,
                "query": case["question"],
                "gold_evidence_chunk_ids": sorted(gold),
                "dense_chunk_ids": _evidence_ids(dense),
                "bm25_chunk_ids": _evidence_ids(bm25),
                "rrf_chunk_ids": _evidence_ids(rrf),
                "retrieved_chunk_ids": _evidence_ids(reranked),
                "dense_hit": bool(set(_evidence_ids(dense)) & gold),
                "bm25_hit": bool(set(_evidence_ids(bm25)) & gold),
                "rrf_hit": bool(set(_evidence_ids(rrf)) & gold),
                "reranker_hit": bool(set(_evidence_ids(reranked)) & gold),
                "reranker_hit_at_5": _recall_at(reranked, gold, 5),
                "reranker_hit_at_7": _recall_at(reranked, gold, 7),
                "reranker_hit_at_10": _recall_at(reranked, gold, 10),
                "rrf_gold_rank": _first_gold_rank(rrf, gold),
                "reranker_gold_rank": _first_gold_rank(reranked, gold),
                "empty_reranked": not reranked,
                "rerank_scores": [chunk.get("rerank_score") for chunk in reranked],
                "reranker_context_precision": (
                    round(
                        sum(bool(set(_evidence_ids([chunk])) & gold) for chunk in reranked)
                        / len(reranked),
                        4,
                    )
                    if reranked else 0.0
                ),
            }
            rows.append(row)
            print(
                f"[{number:02d}/{len(cases)}] dense={row['dense_hit']} bm25={row['bm25_hit']} "
                f"rrf={row['rrf_hit']} reranker={row['reranker_hit']}",
                flush=True,
            )
    finally:
        await client.close()

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    metrics = _metrics(rows)
    OUTPUT_JSON.write_text(json.dumps({"metrics": metrics, "cases": rows}, indent=2))
    OUTPUT_MD.write_text(_markdown(rows, metrics))
    print(f"written: {OUTPUT_JSON}")
    print(f"written: {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
