"""Query-time embedding using SentenceTransformers (singleton, GPU when available)."""
from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer

_models: dict[str, SentenceTransformer] = {}


def _resolve_device() -> str:
    """Pick a torch device. RAG_EMBEDDER_DEVICE overrides autodetect."""
    override = os.environ.get("RAG_EMBEDDER_DEVICE")
    if override:
        return override
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance (load once, reuse)."""
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name, device=_resolve_device())
    return _models[model_name]


async def embed_query(query: str, model_name: str) -> list[float]:
    """Embed a single query string and return its vector.

    Uses a module-level singleton so the model is loaded once per process.
    Runs on CUDA when torch.cuda.is_available(); set RAG_EMBEDDER_DEVICE=cpu
    to force CPU (e.g. in a container without a GPU).
    """
    model = _get_model(model_name)
    embedding = model.encode(query, show_progress_bar=False)
    return embedding.tolist()
