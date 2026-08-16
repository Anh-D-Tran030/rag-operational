"""Semantic cache backed by embedding similarity (disabled when cache_url is empty)."""
from __future__ import annotations

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from src.observability.metrics import rag_cache_hit_total

logger = logging.getLogger(__name__)

_THRESHOLD = 0.95


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
            self._entries: list[tuple[list[float], dict]] = []
            return
        self._model = SentenceTransformer(embedding_model, device="cpu")
        self._entries = []

    def _embed(self, text: str) -> list[float]:
        assert self._model is not None
        return self._model.encode(text, show_progress_bar=False).tolist()

    async def get(self, query: str) -> dict | None:
        """Return cached response if a semantically similar query was seen before."""
        if not self._enabled:
            return None
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, self._embed, query)
        for stored_vec, response in self._entries:
            if _cosine(vec, stored_vec) >= _THRESHOLD:
                rag_cache_hit_total.labels(result="hit").inc()
                return response
        rag_cache_hit_total.labels(result="miss").inc()
        return None

    async def set(self, query: str, response: dict) -> None:
        """Store response keyed by query embedding."""
        if not self._enabled:
            return
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, self._embed, query)
        self._entries.append((vec, response))
