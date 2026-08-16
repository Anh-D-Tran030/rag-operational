"""Query-time embedding using SentenceTransformers on CPU (singleton)."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

_models: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance (load once, reuse)."""
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name, device="cpu")
    return _models[model_name]


async def embed_query(query: str, model_name: str) -> list[float]:
    """Embed a single query string and return its vector.

    Uses a module-level singleton so the model is loaded once per process.
    CPU only — never moves tensors to CUDA (Ollama owns the GPU).
    """
    model = _get_model(model_name)
    embedding = model.encode(query, show_progress_bar=False)
    return embedding.tolist()
