"""Load and summarise run-history.jsonl for the reviewer-router agent.

Provides a structured summary of prior iterations so the agent can
make informed verdict and routing decisions.
"""

import json
from pathlib import Path
from typing import Union


def load_run_history(history_path: Union[str, Path]) -> list[dict]:
    """Load all records from run-history.jsonl.

    Args:
        history_path: Path to run-history.jsonl.

    Returns:
        List of parsed records, ordered by iteration. Empty list if file
        does not exist or is empty.
    """
    path = Path(history_path)
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                )

    records.sort(key=lambda r: r.get("iteration", 0))
    return records


def summarise_history(records: list[dict]) -> dict:
    """Build a structured summary of run history for the agent.

    Args:
        records: List of review-decision records from load_run_history.

    Returns:
        Summary dict with keys:
        - total_iterations: int
        - best_iteration: int (iteration with highest primary metric)
        - best_metric_value: float
        - latest_iteration: int
        - latest_verdict: str
        - latest_route: str
        - model_families_tried: list[str] (unique, ordered by first appearance)
        - metric_trajectory: list[dict] with {iteration, value, delta}
        - has_high_severity_flags: bool
    """
    if not records:
        return {
            "total_iterations": 0,
            "best_iteration": None,
            "best_metric_value": None,
            "latest_iteration": None,
            "latest_verdict": None,
            "latest_route": None,
            "model_families_tried": [],
            "metric_trajectory": [],
            "has_high_severity_flags": False,
        }

    best_iter = None
    best_value = float("-inf")
    families_seen = []
    trajectory = []
    has_high_flags = False

    for rec in records:
        pm = rec.get("primary_metric", {})
        val = pm.get("value")
        iteration = rec.get("iteration")

        if val is not None and val > best_value:
            best_value = val
            best_iter = iteration

        family = rec.get("model_family")
        if family and family not in families_seen:
            families_seen.append(family)

        trajectory.append({
            "iteration": iteration,
            "value": val,
            "delta": pm.get("delta"),
        })

        for flag in rec.get("risk_flags_summary", []):
            if flag.get("severity") == "high":
                has_high_flags = True

    latest = records[-1]

    return {
        "total_iterations": len(records),
        "best_iteration": best_iter,
        "best_metric_value": best_value if best_value != float("-inf") else None,
        "latest_iteration": latest.get("iteration"),
        "latest_verdict": latest.get("reviewer_verdict"),
        "latest_route": latest.get("router_decision"),
        "model_families_tried": families_seen,
        "metric_trajectory": trajectory,
        "has_high_severity_flags": has_high_flags,
    }
