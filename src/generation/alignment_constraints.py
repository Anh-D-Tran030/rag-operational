"""Deterministic role/date/venue/filter checks that run before the LLM judge.

Every constraint is evaluated and every violation collected before deciding:
an explicit row mismatch anywhere rejects, whatever an earlier constraint
left unprovable. A caller-supplied SQL filter must be bound to the outer
source predicate -- projected values alone never prove it -- and an
unprovable one rejects. A question-derived constraint may be proven by the
source predicate or by fully projected rows covering exactly the requested
values; anything else is a safe `indeterminate`, never a silent pass.
"""

from __future__ import annotations

from typing import Literal

from src.generation.alignment_literals import (
    norm_venue,
    question_venues,
    requested_dates,
    requested_roles,
)
from src.generation.alignment_types import AlignmentDecision, AlignmentReason
from src.generation.sql_constraint_proof import NO_PROOF, SourceProof, source_proof

Verdict = Literal["ok", "mismatch", "unprovable"]
_SCALARS = (str, int, float)


def _canon(key: str, value: object) -> tuple[str, object]:
    """Typed canonical form: no str() coercion, so 1, 1.0 and '1' stay distinct."""
    if key == "venue" and isinstance(value, str):
        return ("str", norm_venue(value))
    return (type(value).__name__, value)


def _check_column(
    rows: list[dict], key: str, wanted: set, proof: SourceProof, caller: bool
) -> Verdict:
    present = [r[key] for r in rows if r.get(key) is not None]
    seen = {_canon(key, v if caller else str(v)) for v in present}
    if seen - wanted:
        return "mismatch"
    domain = proof.domains.get(key)
    if domain is not None and {_canon(key, v) for _, v in domain} == wanted:
        return "ok"
    if not caller and not proof.lacks(key) and len(present) == len(rows) and seen == wanted:
        return "ok"
    return "unprovable"


def hard_constraint_decision(
    query: str, sql_result: dict | None, filters: dict | None
) -> AlignmentDecision | None:
    """Reject, defer (None) or return a nonpassing `indeterminate` decision.

    Caller venue is authoritative: a conflicting question venue is its own
    deterministic reason and is never unioned in. A zero-row or vector-only
    result has no SQL rows to check and defers to the semantic judge.
    """
    rows = list((sql_result or {}).get("rows") or [])
    proof = source_proof(str((sql_result or {}).get("sql_used") or "")) if rows else NO_PROOF
    violations: list[AlignmentReason] = []
    unresolved = False

    filter_venue = norm_venue(str(filters["venue"])) if filters and filters.get("venue") else None
    q_venues = question_venues(query)
    if filter_venue and q_venues and filter_venue not in q_venues:
        violations.append("caller_question_venue_conflict")

    def run(key: str, wanted: set[str], reason: AlignmentReason, caller: bool = False) -> None:
        nonlocal unresolved
        verdict = _check_column(rows, key, {_canon(key, v) for v in wanted}, proof, caller)
        if verdict == "mismatch":
            violations.append(reason)
        elif verdict == "unprovable" and caller:
            violations.append("filter_mismatch")
        elif verdict == "unprovable":
            unresolved = True

    if rows:
        if roles := requested_roles(query):
            run("role", roles, "role_mismatch")
        if dates := requested_dates(query):
            run("date", dates, "date_mismatch")
        if venue_scope := ({filter_venue} if filter_venue else q_venues):
            run("venue", venue_scope, "venue_mismatch", caller=bool(filter_venue))
        for key, value in (filters or {}).items():
            if key == "venue":
                continue
            if type(value) in _SCALARS:
                run(key, {value}, "filter_mismatch", caller=True)
            else:  # complex/None/bool caller values are never coerced via str()
                violations.append("filter_mismatch")

    if violations:
        return AlignmentDecision("rejected", violations[0])
    return AlignmentDecision("indeterminate", "unknown") if unresolved else None
