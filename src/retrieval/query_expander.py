"""Multi-query expansion via the configured small model."""
from __future__ import annotations

from src.llm_client import LocalLLMConfig, complete


async def expand_query(
    query: str,
    config: LocalLLMConfig,
    n_expansions: int = 3,
) -> list[str]:
    """Return the original query plus n_expansions reformulations.

    Falls back to [query] on any provider error so retrieval can continue.
    """
    prompt = (
        f"Generate {n_expansions} alternative reformulations of the following search query. "
        f"Return only the reformulations, one per line, without numbering or extra text.\n\n"
        f"Query: {query}"
    )
    try:
        raw = await complete(prompt, config, max_tokens=256)
        expansions = [line.strip() for line in raw.splitlines() if line.strip()]
        return [query] + expansions[:n_expansions]
    except Exception:
        return [query]
