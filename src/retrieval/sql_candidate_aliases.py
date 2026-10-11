"""Normalize SQL output aliases and detect invented requested metrics.

Sole owner of ``_SCHEMA_CAPABILITY_REJECTION``; candidate filters and provider
diagnostics import it rather than redefining the safe enum value.
"""
from __future__ import annotations

import re

from src.retrieval.sql_schema import CURATED_VIEW_COLUMNS

_OUTPUT_ALIAS = re.compile(r"\bAS\s+([A-Za-z_]\w*)", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'([^']*)'")
_WORD = re.compile(r"[a-z0-9]+")
# Inflections of one derived concept: whether a count meets its requirement.
# Canonicalized on both sides so meets_requirement, requirement_met and the
# question's "meet" compare as the same derived word.
_MEET_FORMS = frozenset({"meet", "meets", "met"})
_MEET = "meet"
# Words that name a computation over curated columns rather than a stored fact.
_DERIVED_MEASURE_WORDS = frozenset({
    "total", "count", "number", "average", "avg", "maximum", "max", "minimum", "min",
    "same", "difference", "rate", "met", "meets", "required", "requirement", "linked",
})
# A retry cannot create a column that no curated view has, so this is terminal.
_SCHEMA_CAPABILITY_REJECTION = "unsupported_metric_alias"


def _stem(word: str) -> str:
    """Fold a simple plural so 'incidents' matches the curated 'incident'."""
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def _canonical(word: str) -> str:
    """Stem a word, then map every meet/meets/met inflection to one concept."""
    stemmed = _stem(word)
    return _MEET if word in _MEET_FORMS or stemmed in _MEET_FORMS else stemmed


def _words(text: str) -> set[str]:
    return {_canonical(word) for word in _WORD.findall(text.casefold())}


_CURATED_VOCABULARY = frozenset(
    word
    for view, columns in CURATED_VIEW_COLUMNS.items()
    for name in (view, *columns)
    for word in _words(name)
)
_DERIVED_VOCABULARY = frozenset(_canonical(word) for word in _DERIVED_MEASURE_WORDS)


def _invents_metric_alias(query: str, sql: str) -> bool:
    """Detect a curated column relabelled as a requested fact no curated view stores.

    ``staff_count AS peak_patron_count`` answers a patron question with staff data.
    Alias words that name a curated column, a derived measure, or a filter value
    in the SQL's own literals (``role = 'security'`` for ``security_count``) are
    legitimate; any other question word in an alias is an invented metric.
    """
    question = _words(query)
    literals = set().union(*(_words(lit) for lit in _STRING_LITERAL.findall(sql)))
    allowed = _CURATED_VOCABULARY | _DERIVED_VOCABULARY | literals
    return any(
        len(word) >= 4 and word in question and word not in allowed
        for alias in _OUTPUT_ALIAS.findall(sql)
        for word in _words(alias)
    )
