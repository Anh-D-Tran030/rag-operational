"""Build BM25 text payload fields for Qdrant upsert."""


def build_bm25_text(chunk: dict) -> str:
    """Return the BM25 text for a chunk: contextual_prefix + content, whitespace-normalised."""
    prefix = chunk.get("contextual_prefix", "")
    content = chunk.get("content", "")
    combined = f"{prefix} {content}" if prefix else content
    return " ".join(combined.split())


def add_bm25_text(chunks: list[dict]) -> list[dict]:
    """Add 'bm25_text' field to each chunk dict. Returns updated list."""
    return [{**chunk, "bm25_text": build_bm25_text(chunk)} for chunk in chunks]
