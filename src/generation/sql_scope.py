"""Isolate the one outer SELECT, its sole FROM source and its WHERE tokens.

Works on tokens from `sql_tokens`, so a quoted `ORDER BY`, `WHERE` or `OR`
is never a clause boundary. It identifies structure only and proves nothing.
Anything it cannot place with certainty returns None, which callers treat as
"no proof": joins, CTEs, nested SELECTs, set operations, multiple statements,
comma sources and unexpected tokens after the source.
"""

from __future__ import annotations

from typing import NamedTuple

from src.generation.sql_tokens import Tok

_REFUSED = frozenset(
    {
        "WITH",
        "JOIN",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "NATURAL",
        "CROSS",
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL",
        "OUTER",
        "VALUES",
    }
)
_CLAUSE_ENDS = frozenset({"GROUP", "ORDER", "LIMIT", "HAVING", "WINDOW", "OFFSET"})


class Scope(NamedTuple):
    source: str  # lower-cased source name
    alias: str | None  # lower-cased unique alias, if any
    where: tuple[Tok, ...]  # tokens of the outer WHERE (empty when absent)

    @property
    def qualifier(self) -> str:
        """The only name a qualified column may use to reach this source."""
        return self.alias or self.source


def _name(tok: Tok) -> str | None:
    return (
        tok.text.casefold()
        if tok.kind == "ident"
        or (tok.kind == "word" and tok.upper not in _CLAUSE_ENDS | {"WHERE", "AS", "ON", "USING"})
        else None
    )


def outer_scope(tokens: list[Tok]) -> Scope | None:
    """The single outer SELECT's source/alias/WHERE, or None when unprovable."""
    toks = list(tokens)
    if toks and toks[-1] == Tok("op", ";"):
        toks.pop()
    words = [t.upper for t in toks]
    if not toks or words[0] != "SELECT" or words.count("SELECT") != 1:
        return None
    if _REFUSED & set(words) or any(t == Tok("op", ";") for t in toks):
        return None
    depth, top = 0, []  # indexes of depth-0 tokens
    for i, t in enumerate(toks):
        depth += (t.text == "(" and t.kind == "op") - (t.text == ")" and t.kind == "op")
        if depth == 0 and not (t.kind == "op" and t.text == ")"):
            top.append(i)
    from_at = next((i for i in top if words[i] == "FROM"), None)
    if from_at is None or from_at + 1 >= len(toks):
        return None
    source = _name(toks[from_at + 1])
    if source is None:
        return None
    rest = from_at + 2
    alias = None
    if rest < len(toks) and words[rest] == "AS":
        rest += 1
        alias = _name(toks[rest]) if rest < len(toks) else None
        if alias is None:
            return None
        rest += 1
    elif rest < len(toks) and _name(toks[rest]) is not None:
        alias, rest = _name(toks[rest]), rest + 1
    if rest < len(toks) and words[rest] not in _CLAUSE_ENDS | {"WHERE"}:
        return None
    where: tuple[Tok, ...] = ()
    if rest < len(toks) and words[rest] == "WHERE":
        end = next((i for i in top if i > rest and words[i] in _CLAUSE_ENDS), len(toks))
        where = tuple(toks[rest + 1 : end])
        if not where or any(words[i] == "WHERE" for i in top if i > rest):
            return None
    return Scope(source, alias, where)
