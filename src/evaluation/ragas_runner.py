"""Offline RAGAS evaluation runner.

Runs the golden dataset through `handle_query_fn`, collects per-row contexts and
answers, and computes the standard RAG quality metrics
(faithfulness, answer_relevancy, context_precision, context_recall) plus the
retrieval metrics MRR and nDCG@10. Aggregate (mean) scores are returned; when a
W&B run handle is supplied the per-metric means are pushed to that run.

RAGAS 0.4 API notes:
- Metrics are instantiated objects from ragas.metrics._* (still recognised by
  evaluate() via the Metric base class).
- answer_relevancy requires strictness=1 because OpenRouter returns n=1 completions;
  the default strictness=3 would request n=3 and hang.
- Embeddings for answer_relevancy use LangchainEmbeddingsWrapper so that
  embed_query() is available (the native RAGAS HuggingFaceEmbeddings only has
  embed_text(); LangchainEmbeddingsWrapper bridges the two APIs).
- evaluate() now accepts EvaluationDataset of SingleTurnSample objects; result.scores
  is a list[dict] (one per row).
"""
from __future__ import annotations

import json
import math
from typing import Awaitable, Callable

CITATION_LIMIT = 10


def _reciprocal_rank(predicted: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(predicted[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _citation_ids(citations: list[dict]) -> list[str]:
    """Flatten citations into a rank-ordered list of evidence ids.

    A single citation can carry several ids — a chunk spanning numbered clauses, or
    one SQL citation holding every returned row — so its ids occupy consecutive
    ranks in their original order. Citations with no evidence id contribute no rank
    rather than a snippet fallback, which could never match and only depressed MRR.
    """
    ids: list[str] = []
    for c in citations[:CITATION_LIMIT]:
        section_ids = c.get("section_ids") or []
        if section_ids:
            ids.extend(str(s) for s in section_ids)
        elif c.get("chunk_id"):
            ids.append(str(c["chunk_id"]))
    return ids


async def run_ragas_eval(
    golden_dataset_path: str,
    handle_query_fn: Callable[[str], Awaitable[dict]],
    wandb_run=None,
    judge_llm=None,
    judge_embeddings=None,
    batch_size: int = 10,
) -> dict:
    """Run RAGAS metrics over the golden dataset.

    Args:
      golden_dataset_path: path to a JSON list of `{question, ground_truth,
        evidence_chunk_ids}` records.
      handle_query_fn: async callable that takes a question and returns the
        `handle_query` response dict (`answer`, `citations`, ...).
      wandb_run: optional `wandb.Run` to receive `wandb.log({...})` aggregates.
      judge_llm: RAGAS InstructorLLM from llm_factory(). Must be passed explicitly
        when routing through OpenRouter (the default OpenAI backend won't have the
        right base_url).
      judge_embeddings: LangchainEmbeddingsWrapper around a local HuggingFace model.
        Required for answer_relevancy; must expose embed_query().

    Returns:
      `{faithfulness, answer_relevancy, context_precision, context_recall,
        mrr, ndcg_at_10}` — each value the mean across the dataset.
    """
    import warnings

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.run_config import RunConfig

    metrics = [
        Faithfulness(llm=judge_llm),
        # strictness=1: OpenRouter returns n=1 completion; default strictness=3
        # requests n=3 and hangs because OpenRouter ignores the n parameter.
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ]

    with open(golden_dataset_path) as fh:
        golden = json.load(fh)

    samples: list[SingleTurnSample] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for case in golden:
        response = await handle_query_fn(case["question"])
        citations = response.get("citations", []) or []
        # Truncate each context to 600 chars so the faithfulness prompt stays within
        # the model's output token budget.  Full snippets can exceed 2 000 chars each
        # and cause IncompleteOutputException when all citations are batched together.
        raw_contexts = [str(c.get("snippet", c.get("sql_used", ""))) for c in citations]
        contexts = [ctx[:600] for ctx in raw_contexts if ctx] or [""]
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=str(response.get("answer", "")),
                retrieved_contexts=contexts,
                reference=case["ground_truth"],
            )
        )
        # Unanswerable cases carry no evidence ids; scoring them would report 0.0 for
        # correct refusals and silently drag the retrieval means down.
        relevant = set(case.get("evidence_chunk_ids", []) or [])
        if relevant:
            predicted = _citation_ids(citations)
            mrr_scores.append(_reciprocal_rank(predicted, relevant))
            ndcg_scores.append(_ndcg_at_k(predicted, relevant, k=10))

    # Evaluate in batches to bound peak RAM usage (RAGAS holds all samples +
    # intermediate LLM payloads in memory during evaluate()).
    all_row_scores: list[dict] = []
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]
        ds = EvaluationDataset(samples=batch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ragas_result = evaluate(
                ds,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_embeddings,
                # max_retries=0: don't retry on IncompleteOutputException.
                # max_workers=4: default 16 spins up too many concurrent LLM
                # calls and exhausts RAM on memory-constrained machines.
                run_config=RunConfig(max_retries=0, max_workers=4),
            )
        all_row_scores.extend(ragas_result.scores)

    # result.scores is a list[dict] — one per row. Aggregate to means.
    agg: dict[str, list[float]] = {}
    for row_scores in all_row_scores:
        for k, v in row_scores.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                agg.setdefault(k, []).append(float(v))
    scores = {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}
    scores["mrr"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    scores["ndcg_at_10"] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    scores["retrieval_cases"] = float(len(mrr_scores))
    scores["generation_cases"] = float(len(samples))

    if wandb_run is not None:
        wandb_run.log(scores)

    return scores
