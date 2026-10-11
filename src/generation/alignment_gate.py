"""Query-evidence alignment gate: block an answer whose evidence is off-topic.

Faithfulness only asks whether the answer's claims are supported by *some*
evidence; it does not ask whether that evidence answers the *requested*
role/date/venue. An off-scope SQL row (or chunk) can support a fluent, wrong
answer. This gate runs after a non-abstention draft and before faithfulness,
PII redaction of output, citation delivery, and cache.set.

Fails closed: only `passed` may ship a factual answer. Deterministic
mismatches (role/date/venue/filters) are rejected before the LLM judge is
ever called. Diagnostics are internal -- only the outcome, a safe reason
enum and, for a provider failure, the exception class name; never raw judge
output or evidence.
"""

from __future__ import annotations

from src.generation.alignment_constraints import hard_constraint_decision
from src.generation.alignment_evidence import render_bounded_evidence
from src.generation.alignment_types import (
    DETERMINISTIC_REJECT_REASONS,
    AlignmentDecision,
    AlignmentOutcome,
    AlignmentReason,
)
from src.llm_client import chat

__all__ = [
    "DETERMINISTIC_REJECT_REASONS",
    "AlignmentDecision",
    "AlignmentOutcome",
    "AlignmentReason",
    "check_alignment",
    "chat",
]


async def check_alignment(
    query: str,
    answer: str,
    chunks: list[dict],
    sql_result: dict | None,
    filters: dict | None,
    model: str,
    api_key: str,
    base_url: str,
    token_budget: int = 4000,
) -> AlignmentDecision:
    """Ask whether the evidence used to answer concerns the requested entity/date/role.

    Deterministic checks run first and can only reject, defer or return a
    nonpassing decision, never pass on their own -- a query naming no explicit
    role/date/venue defers straight to the semantic judge. The judge's yes
    cannot override a hard mismatch because a hard mismatch already returned
    before the judge is called.
    """
    hard = hard_constraint_decision(query, sql_result, filters)
    if hard is not None:
        return hard

    if not api_key:
        return AlignmentDecision("indeterminate", "unknown")

    evidence, truncated = render_bounded_evidence(chunks, sql_result, token_budget)
    if not evidence.strip():
        return AlignmentDecision("indeterminate", "unknown")
    if truncated:
        # The cut evidence might have dropped the one row/chunk that would
        # prove or disprove alignment; judging an incomplete picture as
        # complete risks a false "yes", so this fails closed instead.
        return AlignmentDecision("indeterminate", "unknown")

    filter_note = (
        f"\n\nCaller-specified filters that must be honored by every "
        f"answer-bearing source: {filters}"
        if filters
        else ""
    )
    prompt = (
        "Question:\n"
        + query
        + filter_note
        + "\n\nCandidate answer:\n"
        + answer
        + "\n\nEvidence used to produce the answer:\n"
        + evidence
        + "\n\nDoes this evidence concern the same requested entity, date, role, "
        "and operation as the question — not a similar-looking but different one? "
        "Reply with exactly one word: yes or no."
    )
    try:
        raw = await chat(
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as exc:
        return AlignmentDecision("judge_error", "unknown", type(exc).__name__)

    verdict = str(raw).strip().lower().lstrip("*_ ").rstrip(".,!*_ ")
    if verdict == "yes":
        return AlignmentDecision("passed")
    if verdict == "no":
        return AlignmentDecision("rejected", "semantic_mismatch")
    return AlignmentDecision("indeterminate", "unknown")
