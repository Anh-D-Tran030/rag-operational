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
GATE_VERSION = 1


def _normalize(query: str) -> str:
    return " ".join(query.strip().casefold().split())


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
            self._entries: list[tuple[list[float], dict, str, dict, int]] = []
            return
        self._model = SentenceTransformer(embedding_model, device="cpu")
        self._entries = []

    def _embed(self, text: str) -> list[float]:
        assert self._model is not None
        return self._model.encode(text, show_progress_bar=False).tolist()

    async def get(self, query: str, filters: dict | None = None) -> dict | None:
        """Return cached response if a semantically similar query was seen before.

        An entry written at the current GATE_VERSION additionally requires an
        exact normalized-query and filters match; a merely similar query misses.
        """
        if not self._enabled:
            return None
        normalized = _normalize(query)
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, self._embed, query)
        for stored_vec, response, stored_query, stored_filters, stored_version in self._entries:
            if stored_version >= GATE_VERSION:
                if stored_query != normalized or stored_filters != (filters or {}):
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
