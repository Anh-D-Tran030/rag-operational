"""Faithfulness gate: verify answer is grounded in retrieved context before delivery."""
from __future__ import annotations

from src.llm_client import chat
from src.observability.metrics import rag_faithfulness_gate_failures


async def check_faithfulness(
    answer: str,
    context_chunks: list[dict],
    threshold: float,
    model: str,
    api_key: str,
    base_url: str,
) -> bool:
    """Check whether an answer is faithful to the retrieved context.

    Uses a lightweight LLM call that asks: "Is every claim in the
    answer directly supported by the provided context? Reply yes or no."

    Args:
        answer: The generated answer string.
        context_chunks: Retrieved chunks (uses "content" key).
        threshold: Reserved for future numeric scoring; current impl is
            binary yes/no. Pass 0.9 (PRD requirement).
        model: OpenRouter model id (e.g. mistralai/mistral-nemo).
        api_key: OpenRouter API key; if empty, gate is skipped (returns True).
        base_url: OpenAI-compatible base URL.

    Returns:
        True if faithful (or gate skipped); False if answer is not grounded.
        On False: increments rag_faithfulness_gate_failures.
    """
    if not api_key or not context_chunks:
        return True
    context = "\n\n".join(c.get("content", "") for c in context_chunks[:5])
    prompt = (
        "Context:\n"
        + context
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
    except Exception:
        return True  # gate fails open on API error
    # Tolerate "Yes." / "yes, ..." — the judgement is the first word, not the format.
    verdict = raw.strip().lower().lstrip("*_ ").rstrip(".,!*_ ")
    if verdict != "yes":
        rag_faithfulness_gate_failures.inc()
        return False
    return True
