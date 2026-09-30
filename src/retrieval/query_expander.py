"""Multi-query expansion via the configured small model."""
from __future__ import annotations

from src.llm_client import LocalLLMConfig, complete


def _build_prompt(query: str, n_expansions: int) -> str:
    return (
        f"Generate {n_expansions} distinct alternative reformulations of the following "
        f"search query for retrieving passages from operational documents.\n"
        f"- Expand abbreviations and acronyms into their full terms.\n"
        f"- Use synonyms likely to appear in document section headings "
        f"(for example procedure, policy, requirement, approval).\n"
        f"- Keep every named entity, identifier, and number exactly as written.\n"
        f"Return only the reformulations, one per line, without numbering or extra text.\n\n"
        f"Query: {query}"
    )


async def expand_query(
    query: str,
    config: LocalLLMConfig,
    n_expansions: int = 3,
) -> list[str]:
    """Return the original query plus up to n_expansions distinct reformulations.

    Falls back to [query] on any provider error so retrieval can continue.
    """
    try:
        raw = await complete(_build_prompt(query, n_expansions), config, max_tokens=256)
    except Exception:
        return [query]
    seen = {query.strip().casefold()}
    expansions: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            expansions.append(text)
    return [query] + expansions[:n_expansions]
