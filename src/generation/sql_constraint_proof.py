"""Source-bound SQL constraint proof for the alignment gate.

A proof is about a tuple (outer curated source, column, exact value domain),
never about a literal merely appearing in the SQL text. `source_proof` returns
what the outer WHERE of the one curated source provably restricts; anything
unparseable, unsupported or over a limit yields an empty proof, so a caller
fails closed. Projected row values or aliases never count as source scope.
"""

from __future__ import annotations

from typing import NamedTuple

from src.generation.sql_predicates import Scalar, predicate_domains
from src.generation.sql_scope import outer_scope
from src.generation.sql_tokens import SqlSyntaxError, tokenize


class SourceProof(NamedTuple):
    columns: frozenset[str] | None  # curated columns of the sole source; None if unknown
    domains: dict[str, frozenset[Scalar]]  # exact proven domain per column

    def lacks(self, column: str) -> bool:
        """True when the source is known and does not expose `column`."""
        return self.columns is not None and column not in self.columns


NO_PROOF = SourceProof(None, {})


def source_proof(sql_used: str) -> SourceProof:
    """What the outer WHERE of the sole curated source proves, or NO_PROOF."""
    # Function-local: importing retrieval at module scope would couple the
    # generation and retrieval packages' eager __init__ imports.
    from src.retrieval.text_to_sql import CURATED_VIEW_COLUMNS

    try:
        scope = outer_scope(tokenize(sql_used))
    except SqlSyntaxError:
        return NO_PROOF
    columns = CURATED_VIEW_COLUMNS.get(scope.source) if scope else None
    if scope is None or columns is None:
        return NO_PROOF
    return SourceProof(columns, predicate_domains(scope.where, scope.qualifier, columns))
