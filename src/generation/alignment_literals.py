"""Supported question literals: staffing roles, ISO dates and the venue spelling.

Natural-language extraction beyond this enumerated vocabulary is outside the
alignment guarantee. Matching is case-insensitive and word-bounded, so an
embedded substring ("floor staffing", "insecurity", "bartenders") is not a
role. `supervisor` is deliberately absent: in the curated views it is a
person/field on `v_shift_details`, not a staffing-role literal.
"""

from __future__ import annotations

import re

# SQL role literals used by the curated staffing views (src/retrieval/text_to_sql.py).
_ROLE_ALIASES: dict[str, str] = {
    "floor staff": "floor_staff",
    "floor-staff": "floor_staff",
    "floor_staff": "floor_staff",
    "bar staff": "bar_staff",
    "bartender": "bar_staff",
    "security": "security",
}
_ROLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(k) for k in sorted(_ROLE_ALIASES, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# The corpus covers a single venue under two spellings (DB literal uses an
# underscore); both normalize to the same lowercase, space-separated form so
# a query-text mention and a filters/row value compare equal regardless of
# which spelling each side uses.
_VENUE_ALIASES: dict[str, str] = {
    "fairfield rsl": "fairfield rsl",
    "fairfield_rsl": "fairfield rsl",
}
_VENUE_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_VENUE_ALIASES, key=len, reverse=True)),
    re.IGNORECASE,
)


def requested_roles(query: str) -> set[str]:
    """Every role named in the question, not just the first -- a query can
    legitimately name more than one (e.g. "security or floor staff")."""
    return {_ROLE_ALIASES[m.lower()] for m in _ROLE_PATTERN.findall(query)}


def requested_dates(query: str) -> set[str]:
    return set(_ISO_DATE.findall(query))


def norm_venue(value: str) -> str:
    key = value.strip().casefold()
    return _VENUE_ALIASES.get(key, key)


def question_venues(query: str) -> set[str]:
    return {norm_venue(m) for m in _VENUE_PATTERN.findall(query)}
