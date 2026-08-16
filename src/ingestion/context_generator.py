"""Generate contextual prefixes for chunks via an OpenAI-compatible endpoint."""
from __future__ import annotations

from src.llm_client import chat


async def generate_contextual_prefix(
    chunk_content: str,
    doc_type: str,
    venue: str,
    date: str,
    api_key: str,
    model: str,
    base_url: str,
) -> str:
    """Generate a 1–2 sentence prefix summarising where a chunk fits in its document.

    Returns "" if api_key is empty or on any API error.
    """
    if not api_key:
        return ""

    system_text = (
        f"You are a document analyst for {venue} operational records. "
        f"Document type: {doc_type}. Date: {date}. "
        "Write 1–2 sentences explaining where the following chunk fits in its document "
        "and what operational context it provides. Be concise and factual."
    )
    try:
        text = await chat(
            prompt=chunk_content[:1000],
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=128,
            system=system_text,
        )
        return text.strip()
    except Exception:
        return ""
