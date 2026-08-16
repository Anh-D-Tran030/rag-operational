"""Derive stable section anchors for ingested chunks.

Anchors are what let a retrieved chunk be joined back to the golden dataset's
`evidence_chunk_ids` (e.g. `procedure_pdf:section_2.2`). Without them MRR and
nDCG@10 are structurally zero regardless of retrieval quality, because there is
no shared key between a citation and its expected evidence.

A chunk may legitimately cover several anchors (word-window chunking spans
numbered clauses), so anchors are a list, not a scalar.
"""
from __future__ import annotations

import re

# `2.2 Protocol: ...` — a numbered clause followed by prose. The lookbehind keeps
# decimals inside larger numbers (e.g. `$5,500.00`) from matching.
_CLAUSE = re.compile(r"(?<![\d.])(\d+\.\d+)(?=\s+[A-Z])")
_COMPLIANCE_SECTION = re.compile(r"##\s*Section\s+(\d+)", re.IGNORECASE)
_INCIDENT_ID = re.compile(r"Incident ID:\s*(INC-\d+)", re.IGNORECASE)

# Single-fact anchors in the shift log that the golden set references by name.
_SHIFT_LOG_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Total patron count", re.IGNORECASE), "patron_count"),
    (re.compile(r"Supervisor sign-off", re.IGNORECASE), "supervisor_signoff"),
)

# doc_type -> the prefix the golden dataset uses for that source.
DOC_KEYS: dict[str, str] = {
    "procedure": "procedure_pdf",
    "compliance": "compliance_pdf",
    "shift_log": "shift_log.txt",
}


# SQL row-id column -> the golden-dataset key for the table it came from. Ordered:
# the most specific identifier on a row wins, since an incident row also carries a
# shift_id but its evidence id is the incident's.
_ROW_ID_COLUMNS: tuple[tuple[str, str], ...] = (
    ("incident_id", "incident_logs.csv"),
    ("record_id", "staffing_history.csv"),
    ("shift_id", "shift_records.csv"),
)


def row_anchor(row: dict) -> str | None:
    """Return the fully-qualified evidence id for a SQL result row, if it has one."""
    for column, doc_key in _ROW_ID_COLUMNS:
        value = row.get(column)
        if value:
            return f"{doc_key}:{value}"
    return None


def doc_key_for(doc_type: str) -> str:
    """Return the golden-dataset document key for a doc_type."""
    return DOC_KEYS.get(doc_type, doc_type)


def extract_anchors(content: str, doc_type: str, is_first_chunk: bool = False) -> list[str]:
    """Return the ordered, de-duplicated anchor suffixes present in `content`.

    `is_first_chunk` marks the document's leading chunk. The compliance document
    opens with a title block rather than a `## Section 1` heading, but the golden
    set still refers to that block as `section_1`.
    """
    anchors: list[str] = []

    if doc_type == "procedure":
        anchors.extend(f"section_{m}" for m in _CLAUSE.findall(content))

    elif doc_type == "compliance":
        anchors.extend(f"section_{m}" for m in _COMPLIANCE_SECTION.findall(content))
        if is_first_chunk and not _COMPLIANCE_SECTION.search(content):
            anchors.append("section_1")

    elif doc_type == "shift_log":
        anchors.extend(_INCIDENT_ID.findall(content))
        anchors.extend(name for pattern, name in _SHIFT_LOG_MARKERS if pattern.search(content))

    seen: set[str] = set()
    return [a for a in anchors if not (a in seen or seen.add(a))]


def annotate_chunks(chunks: list[dict], doc_type: str) -> list[dict]:
    """Attach `doc_key`, `section_ids` and `chunk_id` to each chunk in place.

    `section_ids` are fully-qualified (`procedure_pdf:section_2.2`) so they can be
    compared directly against `evidence_chunk_ids`. `chunk_id` is the chunk's
    primary anchor, falling back to a positional id when a chunk carries none.
    """
    doc_key = doc_key_for(doc_type)

    for position, chunk in enumerate(chunks):
        anchors = extract_anchors(
            chunk.get("content", ""), doc_type, is_first_chunk=(position == 0)
        )
        section_ids = [f"{doc_key}:{a}" for a in anchors]
        chunk["doc_key"] = doc_key
        chunk["section_ids"] = section_ids
        fallback = f"{doc_key}:chunk_{chunk.get('chunk_index', position)}"
        chunk["chunk_id"] = section_ids[0] if section_ids else fallback

    return chunks
