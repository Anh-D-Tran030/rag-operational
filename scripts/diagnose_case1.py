"""Case-1 diagnostic (PLAN-003 §5): 5 mode-A + 5 mode-B runs, classified by
scripts/case1_classifier.classify. Usage: .venv/bin/python scripts/diagnose_case1.py"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from flashrank import RerankRequest
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi

from scripts.case1_classifier import CLASSES, classify
from scripts.diagnose_retrieval import _evidence_ids, _FakePoint
from src.config import Settings
from src.retrieval.embedder import embed_query
from src.retrieval.query_expander import expand_query
from src.retrieval.reranker import _get_ranker, _passage_text
from src.retrieval.vector_search import _hit_to_dict, _rrf_merge, _scroll_all_bm25_points

QUERY = "What is the RSA intervention protocol for floor staff at Fairfield RSL?"
GOLD = {"procedure_pdf:section_2.2", "procedure_pdf:section_2.3", "procedure_pdf:section_2.4"}
SECTION_MARKERS = ("2.2", "2.3", "2.4")
GOLDEN_SHA = "6dadc61dabc1c082f1b8d896390ece8f5d73eedfe832ce8627fbbb3ec6a8670a"
CORPUS_SHA = "4d1854b0bf94a5ee02e357ee95532d10cae27cd381e68fd1f83e59495731d7a3"
OUT_ROOT = Path(".logs/case1-diagnostic")


def _digest(points: list) -> str:
    def h(p):
        content_sha = hashlib.sha256(p.payload.get("content", "").encode()).hexdigest()
        return p.payload.get("chunk_id", ""), content_sha
    return hashlib.sha256(json.dumps(sorted(h(p) for p in points)).encode()).hexdigest()


def _rank(points: list, gold: set[str]) -> int | None:
    hits = (set(_evidence_ids([_hit_to_dict(p)])) & gold for p in points)
    return next((i for i, hit in enumerate(hits, 1) if hit), None)


async def _search(query: str, client, settings, bm25_index, bm25_points, fetch_k: int):
    embedding = await embed_query(query, settings.embedding_model)
    resp = await client.query_points(collection_name=settings.qdrant_collection, query=embedding,
                                       using="dense", limit=fetch_k, with_payload=True)
    pairs = zip(bm25_index.get_scores(query.lower().split()), bm25_points) if bm25_index else []
    ranked = sorted(pairs, key=lambda x: x[0], reverse=True)[:fetch_k]
    bm25 = [_FakePoint(s, p) for s, p in ranked]
    return resp.points, bm25, _rrf_merge(resp.points, bm25, top_k=settings.retrieval_top_k)


def _score(query: str, merged: list[dict], settings) -> dict:
    """Full FlashRank scoring of every candidate; gold's position/score/decision."""
    passages = [{"id": i, "text": _passage_text(c)} for i, c in enumerate(merged)]
    results = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
    scored = sorted(results, key=lambda r: r["score"], reverse=True)
    n = len(scored)
    is_gold = (set(_evidence_ids([merged[r["id"]]])) & GOLD for r in scored)
    gold_idx = next((i for i, hit in enumerate(is_gold) if hit), None)
    above = [r for r in scored if r["score"] >= settings.rerank_score_threshold]
    excluded = gold_idx is None or gold_idx >= len(above[: settings.rerank_top_n])
    score = None if gold_idx is None else float(scored[gold_idx]["score"])
    pct = None if gold_idx is None or n < 2 else round((1 - gold_idx / (n - 1)) * 100, 2)
    bottom_half = gold_idx is not None and gold_idx >= n / 2
    return {
        "rerank_input_size": n, "gold_rerank_rank": None if gold_idx is None else gold_idx + 1,
        "gold_rerank_score": score, "gold_score_percentile": pct,
        "excluded_by_threshold_or_topn": excluded, "bottom_half": bottom_half,
        "minimal_admitting_threshold": score, "cited": gold_idx is not None and not excluded,
    }


async def _mode_a(client, settings, bm25_index, bm25_points, gold_chunk, fetch_k: int) -> dict:
    expansions = await expand_query(QUERY, settings.local_llm())
    per_expansion, per_expansion_rrf = [], []
    for exp in expansions:
        dense, bm25, rrf = await _search(exp, client, settings, bm25_index, bm25_points, fetch_k)
        per_expansion.append({"expansion": exp, "dense_rank": _rank(dense, GOLD),
                               "bm25_rank": _rank(bm25, GOLD), "rrf_rank": _rank(rrf, GOLD)})
        per_expansion_rrf.append([_hit_to_dict(p) for p in rrf])

    seen: set[str] = set()
    merged: list[dict] = []
    for chunks in per_expansion_rrf:
        for c in chunks:
            key = c.get("content_hash") or c.get("content", "")[:64]
            if key not in seen:
                seen.add(key)
                merged.append(c)

    result = _score(QUERY, merged, settings)
    result["expansions"] = per_expansion
    result["gold_in_rerank_input"] = any(set(_evidence_ids([c])) & GOLD for c in merged)
    result["chunk_found"] = gold_chunk is not None
    text = gold_chunk.get("content", "") if gold_chunk else ""
    result["chunk_has_section_text"] = gold_chunk is not None and all(
        m in text for m in SECTION_MARKERS
    )
    # max_length=512 is FlashRank's tokenizer sequence-length cap, not a character
    # count — a char-length proxy would false-positive on any prefix-heavy chunk.
    result["truncated"] = gold_chunk is not None and len(
        _get_ranker().tokenizer.encode(_passage_text(gold_chunk)).ids
    ) > 512
    return result


async def _mode_b(client, settings, bm25_index, bm25_points, fetch_k: int) -> dict:
    _, _, rrf = await _search(QUERY, client, settings, bm25_index, bm25_points, fetch_k)
    return _score(QUERY, [_hit_to_dict(p) for p in rrf], settings)


def _classifier_rows(mode_a: list[dict], mode_b: list[dict]):
    keys = ("chunk_found", "chunk_has_section_text", "truncated", "gold_in_rerank_input",
            "excluded_by_threshold_or_topn", "bottom_half", "cited")
    names = ("chunk_found", "chunk_has_section_text", "truncated", "in_rerank_input",
             "excluded", "bottom_half", "cited")
    a_rows = [dict(zip(names, (r[k] for k in keys))) for r in mode_a]
    b_rows = [{"excluded": r["excluded_by_threshold_or_topn"]} for r in mode_b]
    return a_rows, b_rows


async def _collect(settings) -> tuple[list[dict], list[dict], str, str, int]:
    client = AsyncQdrantClient(url=settings.qdrant_url)
    fetch_k = settings.retrieval_top_k * 4
    try:
        bm25_points = await _scroll_all_bm25_points(client, settings.qdrant_collection, None)
        corpus = [p.payload.get("bm25_text", "") if p.payload else "" for p in bm25_points]
        bm25_index = BM25Okapi([c.lower().split() for c in corpus]) if corpus else None
        digest_before = _digest(bm25_points)
        gold_chunk = next((dict(p.payload) for p in bm25_points
                            if GOLD & set((p.payload or {}).get("section_ids") or [])), None)
        mode_a = [await _mode_a(client, settings, bm25_index, bm25_points, gold_chunk, fetch_k)
                   for _ in range(5)]
        mode_b = [await _mode_b(client, settings, bm25_index, bm25_points, fetch_k)
                   for _ in range(5)]
        after = await _scroll_all_bm25_points(client, settings.qdrant_collection, None)
        digest_after = _digest(after)
    finally:
        await client.close()
    return mode_a, mode_b, digest_before, digest_after, len(bm25_points)


async def main() -> int:
    settings = Settings(qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    mode_a, mode_b, digest_before, digest_after, point_count = await _collect(settings)

    a_rows, b_rows = _classifier_rows(mode_a, mode_b)
    result_class, evidence = classify(a_rows, b_rows)
    b_scores = [r["gold_rerank_score"] for r in mode_b if r["gold_rerank_score"] is not None]
    b_stable = (max(b_scores) - min(b_scores) <= 1e-6) if len(b_scores) > 1 else True
    digest_stable = digest_before == digest_after

    diag_run_id = f"case1-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
    out_dir = OUT_ROOT / diag_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    golden_sha = hashlib.sha256(Path("eval_data/golden_dataset.json").read_bytes()).hexdigest()
    corpus_sha = hashlib.sha256(Path("corpus/v1.yaml").read_bytes()).hexdigest()
    report = {
        "diag_run_id": diag_run_id, "query": QUERY, "gold_anchors": sorted(GOLD),
        "golden_dataset_sha256_match": golden_sha == GOLDEN_SHA,
        "corpus_sha256_match": corpus_sha == CORPUS_SHA,
        "collection": settings.qdrant_collection, "collection_point_count": point_count,
        "collection_digest_stable": digest_stable,
        "retrieval_params": {
            "retrieval_top_k": settings.retrieval_top_k, "rerank_top_n": settings.rerank_top_n,
            "rerank_score_threshold": settings.rerank_score_threshold,
            "max_retrieval_iterations": settings.max_retrieval_iterations,
        },
        "reranker_params": {"model": "ms-marco-MiniLM-L-12-v2", "max_length": 512},
        "mode_a_runs": mode_a, "mode_b_runs": mode_b, "mode_b_score_stable": b_stable,
        "classification": result_class, "evidence": evidence,
        "diagnostic_passed": digest_stable and b_stable and result_class in CLASSES,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "report.md").write_text(
        f"# Case 1 diagnostic — {diag_run_id}\n\n"
        f"Classification: **{result_class}**\n\n{evidence}\n\n"
        f"Digest stable: {digest_stable}. Mode-B score stable: {b_stable}.\n"
    )
    print(f"classification: {result_class}")
    print(f"written: {out_dir / 'report.json'}")
    print(f"written: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
