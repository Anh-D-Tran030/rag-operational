"""Evidence rendering for the alignment judge.

Owns only rendering and bounded truncation signaling: content order and
provenance metadata are preserved, and the caller is told when the budget cut
anything so it can fail closed instead of judging a partial picture.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4
_CHUNK_META_KEYS = ("doc_type", "venue", "date", "shift_id", "page_num", "chunk_id")


def _render_chunk(chunk: dict) -> str:
    """A chunk's content alone can look on-topic while its metadata (wrong
    date/venue/shift) reveals it isn't; the judge needs both."""
    meta = " | ".join(f"{k}={chunk[k]}" for k in _CHUNK_META_KEYS if chunk.get(k) not in (None, ""))
    content = str(chunk.get("content", ""))
    if meta and content.strip():
        return f"[{meta}]\n{content}"
    return content or (f"[{meta}]" if meta else "")


def render_bounded_evidence(
    chunks: list[dict], sql_result: dict | None, token_budget: int
) -> tuple[str, bool]:
    """Render evidence plus whether it had to be cut to fit the budget.

    A silently truncated render can drop the one row/chunk that would have
    proven or disproven alignment, so the caller must know when that
    happened rather than judging an incomplete picture as if it were whole.
    """
    parts = [_render_chunk(c) for c in chunks]
    rows: list[dict] = list((sql_result or {}).get("rows") or [])
    if rows:
        headers = list(rows[0].keys())
        lines = [" | ".join(headers)]
        lines += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows]
        parts.append("Database result:\n" + "\n".join(lines))
    sql_used = (sql_result or {}).get("sql_used")
    if sql_used:
        parts.append(f"SQL query:\n{sql_used}")
    evidence = "\n\n".join(p for p in parts if p.strip())
    limit = max(0, token_budget) * _CHARS_PER_TOKEN
    return evidence[:limit], len(evidence) > limit
