"""Bounded parsing of one structured SQL candidate into SQLite SELECT text.

Sole owner of the shared ``_JOIN_PATTERN`` and ``_SQL_START`` definitions;
candidate filters and SQL validation import them rather than copying literals.
"""
from __future__ import annotations

import json
import re

_JOIN_PATTERN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_CANDIDATE_SEPARATOR = re.compile(r"^\s*--\s*candidate\s*$", re.IGNORECASE | re.MULTILINE)
_SQL_START = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_POSTGRES_DATE_LITERAL = re.compile(r"\bDATE\s+'(\d{4}-\d{2}-\d{2})'", re.IGNORECASE)
_FETCH_FIRST = re.compile(r"\bFETCH\s+FIRST\s+(\d+)\s+ROW(?:S)?\s+ONLY\b", re.IGNORECASE)
_LABELED_SQL = re.compile(
    r"^\s*(?:here is (?:the )?)?sql\s*:\s*((?:SELECT|WITH)\b.*)$",
    re.IGNORECASE | re.DOTALL,
)
_DANGLING_LIMIT = re.compile(r"\s+LIMIT\s*;?\s*$", re.IGNORECASE)
_SELECT_ONLY = re.compile(r"^\s*SELECT\s*;?\s*$", re.IGNORECASE)
_DANGLING_AND = re.compile(r"\bAND\s*;?\s*$", re.IGNORECASE)
_ROLE_LITERAL_NORMALIZATION = (
    (re.compile(r"'floor'", re.IGNORECASE), "'floor_staff'"),
    (re.compile(r"'floor[ -]staff'", re.IGNORECASE), "'floor_staff'"),
    (re.compile(r"'bar[ -]staff'", re.IGNORECASE), "'bar_staff'"),
    (re.compile(r"'security'", re.IGNORECASE), "'security'"),
)
_INCIDENT_TYPE_LITERAL = re.compile(
    r"(\b(?:[A-Za-z_]\w*\.)?incident_type\s*=\s*')([^']*)(')", re.IGNORECASE
)


def _parse_sql(raw: str) -> tuple[str, str | None]:
    """Return a SQLite SELECT statement and a diagnostic rejection reason."""
    match = _SQL_FENCE.fullmatch(raw.strip())
    sql = match.group(1).strip() if match else raw.strip()
    if not sql:
        return "", "empty_response"
    labeled = _LABELED_SQL.fullmatch(sql)
    if labeled:
        sql = labeled.group(1).strip()
    if not _SQL_START.match(sql):
        return "", "not_select_or_with"
    sql = _POSTGRES_DATE_LITERAL.sub(r"'\1'", sql)
    sql = _FETCH_FIRST.sub(r"LIMIT \1", sql)
    for pattern, replacement in _ROLE_LITERAL_NORMALIZATION:
        sql = pattern.sub(replacement, sql)
    sql = _INCIDENT_TYPE_LITERAL.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2).replace('-', ' ')}{match.group(3)} COLLATE NOCASE"
        ),
        sql,
    )
    # A model occasionally emits LIMIT without a count. Removing that incomplete
    # optional clause preserves the complete SELECT rather than sending invalid SQL.
    sql = _DANGLING_LIMIT.sub("", sql)
    if _SELECT_ONLY.fullmatch(sql):
        return "", "bare_select"
    if _DANGLING_AND.search(sql):
        return "", "dangling_and"
    if sql.rstrip().endswith(","):
        return "", "trailing_comma"
    if sql.count("'") % 2:
        return "", "unbalanced_quotes"
    if not _has_balanced_parentheses(sql):
        return "", "unbalanced_parentheses"
    return sql, None


def _extract_sql(raw: str) -> str:
    """Return one SQLite SELECT statement, rejecting surrounding prose."""
    return _parse_sql(raw)[0]


def _has_balanced_parentheses(sql: str) -> bool:
    """Reject truncated SQL while respecting parentheses inside string literals."""
    depth = 0
    in_string = False
    index = 0
    while index < len(sql):
        character = sql[index]
        if character == "'":
            if in_string and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 1
            else:
                in_string = not in_string
        elif not in_string and character == "(":
            depth += 1
        elif not in_string and character == ")":
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return not in_string and depth == 0


def _extract_sql_envelope(raw: str) -> tuple[str, str | None]:
    """Extract one SQL statement from the strict structured response envelope."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "", "malformed_structured_output"
    if not isinstance(payload, dict) or set(payload) != {"sql"}:
        return "", "incomplete_structured_output"
    sql = payload["sql"]
    if not isinstance(sql, str) or not sql.strip():
        return "", "incomplete_structured_output"
    return sql, None
