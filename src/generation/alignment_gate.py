"""Query-evidence alignment gate: block an answer whose evidence is off-topic.

Faithfulness only asks whether the answer's claims are supported by *some*
evidence; it does not ask whether that evidence answers the *requested*
role/date/venue. An off-scope SQL row (or chunk) can support a fluent, wrong
answer. This gate runs after a non-abstention draft and before faithfulness,
PII redaction of output, citation delivery, and cache.set.

Fails closed: only `passed` may ship a factual answer. Deterministic
mismatches (role/date/venue/filters) are rejected before the LLM judge is
ever called. Diagnostics are internal — only the outcome and a safe reason
enum, never raw judge output or evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.llm_client import chat

AlignmentOutcome = Literal["passed", "rejected", "indeterminate", "judge_error", "timeout"]
AlignmentReason = Literal[
    "role_mismatch", "date_mismatch", "venue_mismatch", "filter_mismatch",
    "semantic_mismatch", "unknown",
]
_CHARS_PER_TOKEN = 4

# SQL role literals used by the curated views (src/retrieval/text_to_sql.py).
# Longest alias first so "floor staff" matches before a shorter substring would.
_ROLE_ALIASES: dict[str, str] = {
    "floor staff": "floor_staff",
    "floor-staff": "floor_staff",
    "floor_staff": "floor_staff",
    "bar staff": "bar_staff",
    "bartender": "bar_staff",
    "security": "security",
    "supervisor": "supervisor",
}
_ROLE_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_ROLE_ALIASES, key=len, reverse=True)),
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class AlignmentDecision:
    outcome: AlignmentOutcome
    reason: AlignmentReason | None = None

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    def as_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"outcome": self.outcome}
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def _requested_role(query: str) -> str | None:
    match = _ROLE_PATTERN.search(query)
    return _ROLE_ALIASES[match.group(0).lower()] if match else None


def _requested_date(query: str) -> str | None:
    match = _ISO_DATE.search(query)
    return match.group(0) if match else None


def _column_values(rows: list[dict], key: str) -> set[str]:
    return {str(r[key]) for r in rows if r.get(key) is not None}


def _provable_in_sql(sql_used: str, literal: str) -> bool:
    """True only if the literal appears as a quoted SQL literal, not incidentally."""
    return bool(re.search(rf"['\"]{re.escape(literal)}['\"]", sql_used, re.IGNORECASE))


def _hard_constraint_decision(
    query: str, sql_result: dict | None, filters: dict | None
) -> AlignmentDecision | None:
    """Deterministic role/date/venue/filter checks. None defers to the semantic judge."""
    rows = list((sql_result or {}).get("rows") or [])
    sql_used = str((sql_result or {}).get("sql_used") or "")

    requested_role = _requested_role(query)
    if requested_role is not None and rows:
        row_roles = _column_values(rows, "role")
        if row_roles:
            # Every answer-bearing row must carry the requested role: a mixed
            # set would let an off-scope row hide behind an on-topic one.
            if row_roles != {requested_role}:
                return AlignmentDecision("rejected", "role_mismatch")
        elif not _provable_in_sql(sql_used, requested_role):
            return None  # field absent from rows and not provably filtered; defer

    requested_date = _requested_date(query)
    if requested_date is not None and rows:
        row_dates = _column_values(rows, "date")
        if row_dates:
            if row_dates != {requested_date}:
                return AlignmentDecision("rejected", "date_mismatch")
        elif not _provable_in_sql(sql_used, requested_date):
            return None

    requested_venue = (filters or {}).get("venue")
    if requested_venue and rows:
        row_venues = _column_values(rows, "venue")
        if row_venues and str(requested_venue) not in row_venues:
            return AlignmentDecision("rejected", "venue_mismatch")

    # Caller filters (e.g. venue scoping) must bind every answer-bearing source.
    # A SQL route that returned rows without a checkable venue/date column cannot
    # prove it honored the filter, so it fails closed rather than being laundered.
    if filters and rows:
        row_venues = _column_values(rows, "venue")
        if "venue" in filters and not row_venues and not _provable_in_sql(
            sql_used, str(filters["venue"])
        ):
            return AlignmentDecision("rejected", "filter_mismatch")

    return None


def _render_bounded_evidence(
    chunks: list[dict], sql_result: dict | None, token_budget: int
) -> str:
    parts = [str(c.get("content", "")) for c in chunks]
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
    return evidence[: max(0, token_budget) * _CHARS_PER_TOKEN]


async def check_alignment(
    query: str,
    answer: str,
    chunks: list[dict],
    sql_result: dict | None,
    filters: dict | None,
    model: str,
    api_key: str,
    base_url: str,
    token_budget: int = 4000,
) -> AlignmentDecision:
    """Ask whether the evidence used to answer concerns the requested entity/date/role.

    Deterministic checks run first and can only reject or defer, never pass on
    their own — a query naming no explicit role/date/venue defers straight to
    the semantic judge. The judge's yes cannot override a hard mismatch because
    a hard mismatch already returned before the judge is called.
    """
    hard = _hard_constraint_decision(query, sql_result, filters)
    if hard is not None:
        return hard

    if not api_key:
        return AlignmentDecision("indeterminate", "unknown")

    evidence = _render_bounded_evidence(chunks, sql_result, token_budget)
    if not evidence.strip():
        return AlignmentDecision("indeterminate", "unknown")

    prompt = (
        "Question:\n"
        + query
        + "\n\nCandidate answer:\n"
        + answer
        + "\n\nEvidence used to produce the answer:\n"
        + evidence
        + "\n\nDoes this evidence concern the same requested entity, date, role, "
        "and operation as the question — not a similar-looking but different one? "
        "Reply with exactly one word: yes or no."
    )
    try:
        raw = await chat(
            prompt=prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=8,
            temperature=0.0,
        )
    except Exception:
        return AlignmentDecision("judge_error", "unknown")

    verdict = str(raw).strip().lower().lstrip("*_ ").rstrip(".,!*_ ")
    if verdict == "yes":
        return AlignmentDecision("passed")
    if verdict == "no":
        return AlignmentDecision("rejected", "semantic_mismatch")
    return AlignmentDecision("indeterminate", "unknown")
