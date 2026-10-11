"""Semantic cache backed by embedding similarity (disabled when cache_url is empty).

Entries written under GATE_VERSION require an exact normalized-query and exact
filters match to be reused. Embedding similarity alone is not enough for a
versioned entry: an older or merely similar cached response must not bypass a
gate (alignment, generation-abstention) that did not exist, or ran under
different filters, when that entry was written.
"""
from __future__ import annotations

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from src.observability.metrics import rag_cache_hit_total

logger = logging.getLogger(__name__)

_THRESHOLD = 0.95
# Bump whenever a new gate changes what a cached response is allowed to mean.
# An entry written under an older version is never eligible for reuse (see
# get()) -- it is a miss, not a lower-confidence match, since it may predate a
# gate/contract the current code depends on.
GATE_VERSION = 3


def _normalize(query: str) -> str:
    return " ".join(query.strip().casefold().split())


def _same_filters(stored: object, current: dict | None) -> bool:
    """Exact typed filter equality (None and {} are the same empty filter).

    Plain `==` would treat 1, 1.0 and True as one filter value, so a cached
    answer for one scope could serve another; the types must match too.
    """
    expected = current or {}
    if not isinstance(stored, dict) or stored.keys() != expected.keys():
        return False
    return all(type(stored[k]) is type(expected[k]) and stored[k] == expected[k] for k in stored)


def _usable(entry: object) -> bool:
    """True only for a well-formed entry written under exactly GATE_VERSION.

    Legacy shapes, unversioned entries, older versions and future versions all
    return False, so they miss instead of raising or being served.
    """
    return (
        isinstance(entry, tuple)
        and len(entry) == 5
        and isinstance(entry[0], list)
        and isinstance(entry[1], dict)
        and isinstance(entry[2], str)
        and type(entry[4]) is int
        and entry[4] == GATE_VERSION
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    """Embedding-similarity cache. All methods are no-ops when disabled."""

    def __init__(self, cache_url: str, embedding_model: str) -> None:
        """Initialise cache. Disabled (no-op) when cache_url is empty."""
        self._enabled = bool(cache_url)
        if not self._enabled:
            logger.info("SemanticCache disabled (no cache_url configured)")
            self._model: SentenceTransformer | None = None
            self._entries: list[tuple] = []
            return
        self._model = SentenceTransformer(embedding_model, device="cpu")
        self._entries = []

    def _embed(self, text: str) -> list[float]:
        assert self._model is not None
        return self._model.encode(text, show_progress_bar=False).tolist()

    async def get(self, query: str, filters: dict | None = None) -> dict | None:
        """Return cached response if a semantically similar query was seen before.

        Only an entry written under exactly the current GATE_VERSION can hit:
        older, future, unversioned or malformed entries always miss (they may
        predate or postdate a gate/contract this code relies on, and
        re-evaluating is the only safe option). A current entry additionally
        requires an exact normalized query and exact typed filters before
        embedding similarity is even consulted; a merely similar query misses.
        """
        if not self._enabled:
            return None
        normalized = _normalize(query)
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, self._embed, query)
        for entry in self._entries:
            if not _usable(entry):
                continue
            stored_vec, response, stored_query, stored_filters, _version = entry
            if stored_query != normalized or not _same_filters(stored_filters, filters):
                continue
            if _cosine(vec, stored_vec) >= _THRESHOLD:
                rag_cache_hit_total.labels(result="hit").inc()
                return response
        rag_cache_hit_total.labels(result="miss").inc()
        return None

    async def set(self, query: str, response: dict, filters: dict | None = None) -> None:
        """Store response keyed by query embedding, normalized query, and filters."""
        if not self._enabled:
            return
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, self._embed, query)
        self._entries.append((vec, response, _normalize(query), filters or {}, GATE_VERSION))
