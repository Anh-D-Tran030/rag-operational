"""Faithfulness gate: verify an answer is grounded in its evidence before delivery.

The gate fails closed. Only the `passed` outcome may deliver a non-abstention
answer; every other outcome is replaced by GATE_ABSTENTION with citations cleared.
Decisions are internal diagnostics: the outcome enum plus, for judge_error and
timeout only, the exception class name. Never the exception text or judge output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.llm_client import chat
from src.observability.metrics import (
    rag_faithfulness_gate_failures,
    rag_faithfulness_gate_total,
)

GateOutcome = Literal[
    "passed", "rejected", "skipped_no_api_key", "skipped_no_context", "judge_error", "timeout",
    "skipped_due_to_alignment", "skipped_generated_abstention",
]
# A refusal from a working judge is deterministic and may be cached; a refusal
# caused by a judge that could not run must be retried on the next request.
# skipped_due_to_alignment/skipped_generated_abstention are never in this set:
# their cacheability depends on the upstream decision, which the caller
# (query_handler) computes and passes as an explicit override.
_CACHEABLE: frozenset[str] = frozenset({"passed", "rejected", "skipped_no_context"})
# Rough English token size, used to bound judge evidence by token_budget_per_query.
_CHARS_PER_TOKEN = 4

# One public message for every non-passing outcome, so it reveals nothing about why.
GATE_ABSTENTION = (
    "The generated answer could not be verified against the retrieved context. "
    "Please rephrase your question or consult the source documents directly."
)


@dataclass(frozen=True)
class GateDecision:
    outcome: GateOutcome
    exception_class: str | None = None

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def cacheable(self) -> bool:
        return self.outcome in _CACHEABLE

    def as_dict(self) -> dict[str, str]:
        out = {"outcome": self.outcome}
        if self.exception_class is not None:
            out["exception_class"] = self.exception_class
        return out


def record_gate_decision(
    decision: GateDecision, tracer: Any, trace_obj: Any, latency_ms: float
) -> None:
    """Count a decision once on the 6-label counter and record it on the trace."""
    rag_faithfulness_gate_total.labels(outcome=decision.outcome).inc()
    tracer.log_span(
        trace_obj=trace_obj,
        name="faithfulness_gate",
        input={},
        output=decision.as_dict(),
        latency_ms=latency_ms,
    )


def render_evidence(chunks: list[dict], sql_result: dict | None, token_budget: int) -> str:
    """Render every chunk and the exact SQL rows the generator saw, within the budget."""
    parts = [str(c.get("content", "")) for c in chunks]
    rows: list[dict] = list((sql_result or {}).get("rows") or [])
    if rows:
        headers = list(rows[0].keys())
        lines = [" | ".join(headers)]
        lines += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows]
        parts.append("Database result:\n" + "\n".join(lines))
    evidence = "\n\n".join(p for p in parts if p.strip())
    return evidence[: max(0, token_budget) * _CHARS_PER_TOKEN]


async def check_faithfulness(
    answer: str,
    evidence: str,
    model: str,
    api_key: str,
    base_url: str,
) -> GateDecision:
    """Ask the judge whether every claim in the answer is supported by the evidence.

    Uses a lightweight LLM call that asks: "Is every claim in the answer directly
    supported by the provided context? Reply yes or no."

    Args:
        answer: The generated answer string.
        evidence: Output of render_evidence (vector chunks and SQL rows).
        model: OpenRouter model id for the judge.
        api_key: OpenRouter API key; if empty the judge cannot run.
        base_url: OpenAI-compatible base URL.

    Returns:
        A GateDecision. Missing key, missing evidence, and judge errors never pass.
        On `rejected`, also increments rag_faithfulness_gate_failures.
    """
    if not evidence.strip():
        return GateDecision("skipped_no_context")
    if not api_key:
        return GateDecision("skipped_no_api_key")
    prompt = (
        "Context:\n"
        + evidence
        + "\n\nAnswer:\n"
        + answer
        + "\n\nIs every claim in the answer directly supported by the context above? "
        "Reply with exactly one word: yes or no."
    )
    try:
        raw = await chat(
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=8,
            # A grounded/not-grounded verdict must not vary between identical
            # requests. Left unpinned, the same answer scored both ways.
            temperature=0.0,
        )
    except Exception as exc:
        return GateDecision("judge_error", type(exc).__name__)
    # Tolerate "Yes." / "yes, ..." — the judgement is the first word, not the format.
    verdict = str(raw).strip().lower().lstrip("*_ ").rstrip(".,!*_ ")
    if verdict != "yes":
        rag_faithfulness_gate_failures.inc()
        return GateDecision("rejected")
    return GateDecision("passed")
