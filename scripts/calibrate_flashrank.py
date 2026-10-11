"""Calibrate FlashRank against the fixed RRF candidates for document-backed gold.

The experiment does not modify production settings. It runs dense/BM25/RRF once
per eligible golden case, scores that fixed candidate set once with FlashRank,
then evaluates every (top_n, threshold) combination in memory. Threshold
semantics deliberately match ``src.retrieval.reranker``: it is a global gate on
the maximum score, not a per-candidate cutoff.

Usage:
    .venv/bin/python scripts/calibrate_flashrank.py
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

from flashrank import RerankRequest
from qdrant_client import AsyncQdrantClient

from scripts.diagnose_retrieval import _evidence_ids, _stages
from src.config import Settings
from src.retrieval.reranker import _get_ranker

GOLDEN_PATH = Path("eval_data/golden_dataset.json")
OUTPUT_JSON = Path(".logs/flashrank_calibration.json")
OUTPUT_MD = Path(".logs/flashrank_calibration.md")
TOP_NS = (5, 7, 10, 15)
THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4)
FOCUS_QUERIES = {
    "What is the RSA intervention protocol for floor staff at Fairfield RSL?",
    "Where is the staff assembly point in case of fire?",
    "Were RSA interventions on 2026-05-28 logged within the procedure's required time window?",
}


def _chunk_relevant(chunk: dict, gold: set[str]) -> bool:
    return bool(set(_evidence_ids([chunk])) & gold)


def _score_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Score all fixed RRF candidates once and return them in FlashRank order."""
    passages = [{"id": index, "text": chunk.get("content", "")} for index, chunk in enumerate(candidates)]
    results = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
    scored: list[dict] = []
    for result in sorted(results, key=lambda item: item["score"], reverse=True):
        chunk = dict(candidates[result["id"]])
        chunk["rerank_score"] = float(result["score"])
        scored.append(chunk)
    return scored


def _selected(scored: list[dict], top_n: int, threshold: float) -> list[dict]:
    """Apply the exact threshold behaviour currently used by production rerank."""
    if not scored or max(float(chunk["rerank_score"]) for chunk in scored) < threshold:
        return []
    return scored[:top_n]


def _row_metrics(
    selected: list[dict], gold: set[str], available_relevant_chunks: int, top_n: int
) -> dict:
    """Compute chunk-rank metrics; recall is anchor-level and precision is chunk-level."""
    recalls: dict[str, float | None] = {}
    precisions: dict[str, float | None] = {}
    for k in (5, 7, 10):
        if k > top_n:
            recalls[f"recall_at_{k}"] = None
            precisions[f"precision_at_{k}"] = None
            continue
        top = selected[:k]
        recovered = set(_evidence_ids(top)) & gold
        recalls[f"recall_at_{k}"] = len(recovered) / len(gold)
        precisions[f"precision_at_{k}"] = sum(_chunk_relevant(chunk, gold) for chunk in top) / k

    relevant_ranks = [
        rank for rank, chunk in enumerate(selected[:10], start=1) if _chunk_relevant(chunk, gold)
    ]
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    # Candidate-relative ideal: this answers how much of RRF's available relevant
    # context FlashRank preserves, rather than penalising it for missing index IDs.
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(available_relevant_chunks, 10) + 1)
    )
    return {
        **recalls,
        **precisions,
        "mrr": mrr,
        "ndcg_at_10": (dcg / ideal if ideal else 0.0) if top_n >= 10 else None,
    }


def _aggregate(rows: list[dict], top_n: int, threshold: float) -> dict:
    per_case = [
        _row_metrics(
            _selected(row["scored_candidates"], top_n, threshold),
            set(row["gold"]),
            row["rrf_relevant_chunks"],
            top_n,
        )
        for row in rows
    ]
    count = len(per_case)
    metric_keys = (
        "recall_at_5", "recall_at_7", "recall_at_10", "precision_at_5", "precision_at_7",
        "precision_at_10", "mrr", "ndcg_at_10",
    )
    metrics = {
        key: None if any(case[key] is None for case in per_case) else sum(case[key] for case in per_case) / count
        for key in metric_keys
    }
    metrics.update(
        {
            "top_n": top_n,
            "threshold": threshold,
            "empty_rate": sum(not _selected(row["scored_candidates"], top_n, threshold) for row in rows) / count,
        }
    )
    return metrics


def _markdown(rows: list[dict], configurations: list[dict], rrf_recall: float) -> str:
    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    lines = [
        "# FlashRank calibration",
        "",
        "Fixed candidate set: RRF output before FlashRank. Metrics use the nine golden cases whose evidence anchors appear in that RRF set.",
        "Recall@k is anchor-level over the first k chunks; precision@k is relevant chunks divided by k (missing output counts as non-relevant). Metrics whose cutoff exceeds top_n are N/A, rather than being measured on fewer results. MRR uses chunk rank. Threshold uses production's max-score global-gate semantics.",
        "",
        f"RRF candidate anchor recall: **{rrf_recall:.3f}**. All configurations preserve this candidate set.",
        "",
        "| top_n | threshold | recall@5 | recall@7 | recall@10 | nDCG@10 | MRR | precision@5 | precision@7 | precision@10 | empty rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in configurations:
        lines.append(
            f"| {row['top_n']} | {row['threshold']:.1f} | {metric(row['recall_at_5'])} | {metric(row['recall_at_7'])} | "
            f"{metric(row['recall_at_10'])} | {metric(row['ndcg_at_10'])} | {metric(row['mrr'])} | "
            f"{metric(row['precision_at_5'])} | {metric(row['precision_at_7'])} | {metric(row['precision_at_10'])} | {row['empty_rate']:.3f} |"
        )
    lines.extend(["", "## Focus queries — raw FlashRank scores", ""])
    for row in rows:
        if row["query"] not in FOCUS_QUERIES:
            continue
        lines.extend([f"### {row['query']}", "", "| Rank | Score | Chunk ID | Gold hit |", "|---:|---:|---|:---:|"])
        for candidate in row["scored_candidates"]:
            chunk_id = candidate.get("chunk_id", "")
            lines.append(
                f"| {candidate['rank']} | {candidate['rerank_score']:.5f} | {chunk_id} | "
                f"{'Y' if candidate['gold_hit'] else 'N'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


async def main() -> int:
    settings = Settings(qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    golden = json.loads(GOLDEN_PATH.read_text())
    client = AsyncQdrantClient(url=settings.qdrant_url)
    rows: list[dict] = []
    try:
        for dataset_index, case in enumerate(golden, start=1):
            gold = set(case.get("evidence_chunk_ids") or [])
            if not gold:
                continue
            _, _, rrf, _ = await _stages(case["question"], client, settings)
            if not (set(_evidence_ids(rrf)) & gold):
                continue
            scored = _score_candidates(case["question"], rrf)
            rows.append(
                {
                    "dataset_index": dataset_index,
                    "query": case["question"],
                    "gold": sorted(gold),
                    "rrf_candidate_ids": _evidence_ids(rrf),
                    "rrf_relevant_chunks": sum(_chunk_relevant(chunk, gold) for chunk in rrf),
                    "scored_candidates": [
                        {
                            "rank": rank,
                            "rerank_score": float(chunk["rerank_score"]),
                            "chunk_id": chunk.get("chunk_id", ""),
                            "section_ids": chunk.get("section_ids", []) or [],
                            "gold_hit": _chunk_relevant(chunk, gold),
                        }
                        for rank, chunk in enumerate(scored, start=1)
                    ],
                }
            )
            print(f"[{len(rows):02d}] {case['question']}", flush=True)
    finally:
        await client.close()

    configurations = [
        _aggregate(rows, top_n, threshold) for top_n in TOP_NS for threshold in THRESHOLDS
    ]
    rrf_recall = sum(
        len(set(row["rrf_candidate_ids"]) & set(row["gold"])) / len(row["gold"]) for row in rows
    ) / len(rows)
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({"rrf_anchor_recall": rrf_recall, "cases": rows, "configurations": configurations}, indent=2))
    OUTPUT_MD.write_text(_markdown(rows, configurations, rrf_recall))
    print(f"document_backed_cases: {len(rows)}")
    print(f"written: {OUTPUT_JSON}")
    print(f"written: {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
