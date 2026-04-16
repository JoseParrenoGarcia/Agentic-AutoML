"""Review-decision record schema for run-history.jsonl (Contract 3 extension).

Defines required fields, valid values, and a builder function for constructing
well-formed records that the reviewer-router agent appends to run-history.jsonl.
"""

from dataclasses import dataclass, asdict
from typing import Optional

# --- Valid value sets ---

VALID_STATUSES = {"completed", "failed"}
VALID_VERDICTS = {"sufficient", "insufficient"}
VALID_ROUTES = {"continue", "rollback", "pivot"}


@dataclass
class PrimaryMetric:
    """Primary metric snapshot for a single iteration."""

    name: str
    value: float
    delta: Optional[float]  # null on iteration 1


@dataclass
class RiskFlag:
    """A single risk flag raised during review."""

    type: str       # leakage | overfitting | underfitting | data_issue
    severity: str   # low | medium | high
    evidence: str


@dataclass
class ReviewDecision:
    """Full review-decision record appended to run-history.jsonl.

    Extends Contract 3 with reviewer and router fields.
    """

    iteration: int
    timestamp: str  # ISO 8601
    status: str     # completed | failed
    plan_summary: str
    primary_metric: PrimaryMetric
    model_family: str
    reviewer_verdict: str       # sufficient | insufficient
    reviewer_reasoning: str
    router_decision: str        # continue | rollback | pivot
    router_reasoning: str
    risk_flags_summary: list    # list of RiskFlag-like dicts
    best_iteration: int

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON output."""
        d = asdict(self)
        return d
