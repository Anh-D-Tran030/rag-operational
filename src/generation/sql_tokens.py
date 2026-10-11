"""Bounded SQL tokenizer for the alignment proof.

Produces typed tokens and nothing else: comments and whitespace are dropped,
strings and quoted identifiers are decoded, and anything it cannot classify
with certainty raises `SqlSyntaxError` so a caller fails closed. Comment
markers, keywords and parentheses inside a string or quoted identifier are
data, never structure.
"""

from __future__ import annotations

import re
from typing import NamedTuple

MAX_SQL_CHARS = 4000
MAX_TOKENS = 400
MAX_PAREN_DEPTH = 8


class SqlSyntaxError(ValueError):
    """The SQL is malformed, unsupported or over a configured limit."""


class Tok(NamedTuple):
    kind: str  # word | ident | str | num | op
    text: str  # word/num/op as written; ident/str decoded

    @property
    def upper(self) -> str:
        return self.text.upper() if self.kind == "word" else ""


_LEX = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<lc>--[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<bad>/\*)
  | (?P<str>'(?:[^']|'')*')
  | (?P<qid>"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\])
  | (?P<num>\d+(?:\.\d+)?)
  | (?P<word>[A-Za-z_][A-Za-z0-9_$]*)
  | (?P<op><>|!=|<=|>=|==|\|\||[=<>(),;.*+\-/%])
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(sql: str) -> list[Tok]:
    """Typed tokens of `sql`; raises SqlSyntaxError on anything unsafe."""
    if not isinstance(sql, str) or not sql.strip() or len(sql) > MAX_SQL_CHARS:
        raise SqlSyntaxError("empty or oversized sql")
    out: list[Tok] = []
    pos, depth = 0, 0
    while pos < len(sql):
        m = _LEX.match(sql, pos)
        if m is None or m.lastgroup == "bad":
            raise SqlSyntaxError("unterminated or unknown token")
        pos, kind, text = m.end(), m.lastgroup, m.group()
        if kind in ("ws", "lc", "bc"):
            continue
        if kind == "str":
            out.append(Tok("str", text[1:-1].replace("''", "'")))
        elif kind == "qid":
            quote = text[0]
            body = text[1:-1]
            out.append(Tok("ident", body.replace(quote * 2, quote) if quote != "[" else body))
        else:
            out.append(Tok(kind, text))
        if text == "(":
            depth += 1
        elif text == ")":
            depth -= 1
        if depth < 0 or depth > MAX_PAREN_DEPTH or len(out) > MAX_TOKENS:
            raise SqlSyntaxError("paren imbalance or token limit")
    if depth != 0:
        raise SqlSyntaxError("unbalanced parentheses")
    return out
