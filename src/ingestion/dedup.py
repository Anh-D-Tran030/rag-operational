"""MD5-based deduplication for ingestion chunks."""
import hashlib
import re


def _normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[​‌‍﻿ ]", "", text)
    return text


def deduplicate(chunks: list[dict]) -> list[dict]:
    """Return chunks with unique content, adding 'content_hash' to each.

    Normalises text before hashing: strip, lowercase, remove invisible unicode.
    Preserves first occurrence of each unique chunk.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for chunk in chunks:
        normalised = _normalise(chunk["content"])
        h = hashlib.md5(normalised.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append({**chunk, "content_hash": h})
    return result
