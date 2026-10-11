"""Leaf types for the alignment gate: outcome/reason enums and the decision.

Imports neither the gate nor retrieval, so every alignment module can share
these without an import cycle. Diagnostics stay internal: a decision carries
an outcome, a safe reason enum and, for a provider failure, only the
exception *class name* -- never a raw error, evidence, or judge reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AlignmentOutcome = Literal["passed", "rejected", "indeterminate", "judge_error", "timeout"]
AlignmentReason = Literal[
    "role_mismatch",
    "date_mismatch",
    "venue_mismatch",
    "filter_mismatch",
    "caller_question_venue_conflict",
    "semantic_mismatch",
    "unknown",
]
# The only reasons a `rejected` outcome may be replayed from cache for: every
# one is a hard-constraint check that ran before the LLM judge and depends on
# nothing probabilistic. An allowlist, not an exclusion list, so a new reason
# added later is cache-ineligible by default until explicitly reviewed and
# added here.
DETERMINISTIC_REJECT_REASONS: frozenset[str] = frozenset(
    {
        "role_mismatch",
        "date_mismatch",
        "venue_mismatch",
        "filter_mismatch",
        "caller_question_venue_conflict",
    }
)


@dataclass(frozen=True)
class AlignmentDecision:
    outcome: AlignmentOutcome
    reason: AlignmentReason | None = None
    # Safe provider-failure class name only; excluded from equality so a
    # decision still compares by (outcome, reason).
    exception_class: str | None = field(default=None, compare=False)

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    def as_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"outcome": self.outcome}
        if self.reason is not None:
            out["reason"] = self.reason
        if self.exception_class is not None:
            out["exception_class"] = self.exception_class
        return out
