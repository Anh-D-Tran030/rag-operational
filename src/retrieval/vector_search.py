"""Hybrid BM25 + dense vector search with Reciprocal Rank Fusion."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, ScoredPoint
from rank_bm25 import BM25Okapi


def _rrf_merge(
    dense_hits: list[ScoredPoint],
    bm25_hits: list[ScoredPoint],
    top_k: int,
    k: int = 60,
) -> list[ScoredPoint]:
    """Merge two ranked lists via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    point_map: dict[str, ScoredPoint] = {}

    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        point_map[pid] = hit

    for rank, hit in enumerate(bm25_hits):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in point_map:
            point_map[pid] = hit

    ranked = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)
    return [point_map[pid] for pid in ranked[:top_k]]


def _hit_to_dict(hit: ScoredPoint) -> dict:
    payload = hit.payload or {}
    return {
        "content": payload.get("content", ""),
        "page_num": payload.get("page_num"),
        "doc_type": payload.get("doc_type"),
        "date": payload.get("date"),
        "venue": payload.get("venue"),
        "shift_id": payload.get("shift_id"),
        "table_flag": payload.get("table_flag"),
        "contextual_prefix": payload.get("contextual_prefix", ""),
        "bm25_text": payload.get("bm25_text", ""),
        "score": hit.score,
        "content_hash": payload.get("content_hash", ""),
        "chunk_id": payload.get("chunk_id", ""),
        "section_ids": payload.get("section_ids", []) or [],
    }


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    client: AsyncQdrantClient,
    collection: str,
    top_k: int,
    filters: dict | None = None,
) -> list[dict]:
    """Perform dense + BM25 search and merge results via RRF.

    Dense: qdrant vector search using query_embedding.
    BM25: sparse search over bm25_text payload using rank-bm25 BM25Okapi.
    Merges with RRF (k=60), deduplicates by content_hash, returns top_k dicts.
    """
    qdrant_filter = Filter(**filters) if filters else None
    fetch_k = top_k * 4

    # Dense search. The deprecated `search()` is gone from the pinned qdrant-client;
    # `query_points` needs the named vector ("dense") the collection was created with.
    dense_response = await client.query_points(
        collection_name=collection,
        query=query_embedding,
        using="dense",
        limit=fetch_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )
    dense_results = dense_response.points

    # BM25 search: fetch payload texts then re-rank locally
    scroll_result = await client.scroll(
        collection_name=collection,
        scroll_filter=qdrant_filter,
        limit=fetch_k,
        with_payload=True,
        with_vectors=False,
    )
    points, _ = scroll_result

    if points:
        corpus = [p.payload.get("bm25_text", "") if p.payload else "" for p in points]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        scored_points = sorted(
            zip(bm25_scores, points), key=lambda x: x[0], reverse=True
        )[:fetch_k]

        class _FakePoint:
            def __init__(self, score: float, point: object) -> None:
                self.id = point.id  # type: ignore[attr-defined]
                self.payload = point.payload  # type: ignore[attr-defined]
                self.score = score

        bm25_results: list[ScoredPoint] = [_FakePoint(s, p) for s, p in scored_points]  # type: ignore[assignment]
    else:
        bm25_results = []

    merged = _rrf_merge(dense_results, bm25_results, top_k)

    # Deduplicate by content_hash
    seen: set[str] = set()
    output: list[dict] = []
    for hit in merged:
        d = _hit_to_dict(hit)
        h = d["content_hash"]
        if h and h in seen:
            continue
        if h:
            seen.add(h)
        output.append(d)
        if len(output) >= top_k:
            break

    return output
