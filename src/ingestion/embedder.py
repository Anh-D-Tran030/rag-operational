"""Batch embedding for ingestion using SentenceTransformers on CPU."""
from __future__ import annotations


class Embedder:
    """CPU-only SentenceTransformer wrapper for ingestion-time batch embedding."""

    def __init__(self, model_name: str) -> None:
        """Load the SentenceTransformer model on CPU."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts and return their embedding vectors."""
        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return [e.tolist() for e in embeddings]


async def embed_chunks(chunks: list[dict], embedder: Embedder) -> list[dict]:
    """Add 'embedding' field to each chunk.

    Embeds contextual_prefix + content if prefix is non-empty, else content alone.
    Returns updated chunk list (original dicts are not mutated).
    """
    texts = []
    for chunk in chunks:
        prefix = chunk.get("contextual_prefix", "")
        content = chunk.get("content", "")
        texts.append(f"{prefix} {content}".strip() if prefix else content)

    embeddings = embedder.embed_batch(texts)
    return [{**chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
