"""Exact value domains proven by a top-level WHERE conjunction.

Supported proof forms for a column: `col = literal` and `col IN (literal, ...)`
(typed str/int/float literals), optionally qualified by the sole source
qualifier and optionally wrapped in balanced parentheses. A top-level `AND`
splits conjuncts (a `BETWEEN .. AND` or `CASE .. AND .. END` is never split),
and an unrelated conjunct can only narrow the rows, so it is ignored.

Fail-closed rules: an `OR` anywhere voids every proof; a conjunct that
mentions a column but is not an exact supported form (NOT, IS NOT, <>, <,
LIKE, BETWEEN, functions, CASE, reversed or trailing syntax, wrong qualifier)
voids that column; two differing exact predicates on one column void it.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.generation.sql_tokens import Tok

Scalar = tuple[str, str | int | float]  # (type name, value)


def _split(toks: list[Tok], depth: int = 0) -> list[list[Tok]] | None:
    """Top-level AND conjuncts; None when an OR is present or nesting is odd."""
    if depth > 4 or any(t.upper == "OR" for t in toks):
        return None
    parts: list[list[Tok]] = [[]]
    level = case = 0
    between = False
    for t in toks:
        if t.kind == "op" and t.text in "()":
            level += 1 if t.text == "(" else -1
        elif level == 0 and t.upper == "CASE":
            case += 1
        elif level == 0 and t.upper == "END":
            case -= 1
        elif level == 0 and case == 0 and t.upper == "BETWEEN":
            between = True
        elif level == 0 and case == 0 and t.upper == "AND":
            if between:
                between = False
            else:
                parts.append([])
                continue
        parts[-1].append(t)
    out: list[list[Tok]] = []
    for part in parts:
        wrapped = len(part) > 2 and part[0] == Tok("op", "(") and part[-1] == Tok("op", ")")
        if wrapped and _balanced_wrap(part):
            inner = _split(part[1:-1], depth + 1)
            if inner is None:
                return None
            out.extend(inner)
        else:
            out.append(part)
    return out


def _balanced_wrap(part: list[Tok]) -> bool:
    level = 0
    for i, t in enumerate(part):
        if t.kind == "op" and t.text in "()":
            level += 1 if t.text == "(" else -1
            if level == 0 and i < len(part) - 1:
                return False
    return True


def _literal(tok: Tok) -> Scalar | None:
    if tok.kind == "str":
        return ("str", tok.text)
    if tok.kind == "num":
        return ("float", float(tok.text)) if "." in tok.text else ("int", int(tok.text))
    return None


def _column(toks: list[Tok], qualifier: str, columns: frozenset[str]) -> tuple[str, int] | None:
    """(column, tokens consumed) for `[qualifier .] column` at the head, if bound."""

    def ident(t: Tok) -> str | None:
        return t.text.casefold() if t.kind in ("word", "ident") else None

    if len(toks) >= 3 and toks[1] == Tok("op", "."):
        if ident(toks[0]) != qualifier or ident(toks[2]) not in columns:
            return None
        return ident(toks[2]), 3  # type: ignore[return-value]
    name = ident(toks[0]) if toks else None
    return (name, 1) if name in columns else None


def _exact(
    conjunct: list[Tok], qualifier: str, columns: frozenset[str]
) -> tuple[str, set[Scalar]] | None:
    head = _column(conjunct, qualifier, columns)
    if head is None:
        return None
    col, n = head
    rest = conjunct[n:]
    if len(rest) == 2 and rest[0] == Tok("op", "=") and (lit := _literal(rest[1])):
        return col, {lit}
    if (
        len(rest) >= 4
        and rest[0].upper == "IN"
        and rest[1] == Tok("op", "(")
        and rest[-1] == Tok("op", ")")
    ):
        inner = rest[2:-1]
        values = [_literal(t) for t in inner[0::2]]
        commas = all(t == Tok("op", ",") for t in inner[1::2])
        if commas and len(inner) % 2 == 1 and all(values):
            return col, set(values)  # type: ignore[arg-type]
    return None


def predicate_domains(
    where: Iterable[Tok], qualifier: str, columns: frozenset[str]
) -> dict[str, frozenset[Scalar]]:
    """Exact proven value domain per bound column; unprovable columns are absent."""
    conjuncts = _split(list(where))
    if conjuncts is None:
        return {}
    domains: dict[str, frozenset[Scalar]] = {}
    voided: set[str] = set()
    for conjunct in conjuncts:
        mentioned = {t.text.casefold() for t in conjunct if t.kind in ("word", "ident")} & columns
        exact = _exact(conjunct, qualifier, columns)
        if exact is None:
            voided |= mentioned
            continue
        col, values = exact
        if col in domains and domains[col] != frozenset(values):
            voided.add(col)
        domains[col] = frozenset(values)
        voided |= mentioned - {col}
    return {c: d for c, d in domains.items() if c not in voided}
