"""Compute deltas, trends, and best-so-far tracking across iterations.

Used by the reviewer-router agent to determine improvement trajectory
and inform verdict/routing decisions.
"""

from typing import Optional


def compute_deltas(
    current_value: float,
    previous_value: Optional[float],
    best_value: Optional[float],
) -> dict:
    """Compute metric deltas for the current iteration.

    Args:
        current_value: Primary metric value for this iteration.
        previous_value: Primary metric value from the previous iteration (None on iter 1).
        best_value: Best primary metric value seen so far (None on iter 1).

    Returns:
        Dict with delta_vs_previous, delta_vs_best, and improved_vs_best flag.
    """
    delta_prev = None
    if previous_value is not None:
        delta_prev = current_value - previous_value

    delta_best = None
    improved_vs_best = None
    if best_value is not None:
        delta_best = current_value - best_value
        improved_vs_best = current_value > best_value

    return {
        "delta_vs_previous": delta_prev,
        "delta_vs_best": delta_best,
        "improved_vs_best": improved_vs_best,
    }


def compute_trend(metric_trajectory: list[dict], window: int = 3) -> str:
    """Determine the improvement trend over the last N iterations.

    Args:
        metric_trajectory: List of {iteration, value, delta} dicts, ordered by iteration.
        window: Number of recent iterations to consider.

    Returns:
        One of: "improving", "plateau", "degrading", "insufficient_data".
    """
    values = [
        t["value"] for t in metric_trajectory
        if t.get("value") is not None
    ]

    if len(values) < 2:
        return "insufficient_data"

    recent = values[-window:] if len(values) >= window else values

    # Compute consecutive deltas
    deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

    if not deltas:
        return "insufficient_data"

    positive = sum(1 for d in deltas if d > 0.005)
    negative = sum(1 for d in deltas if d < -0.005)
    flat = sum(1 for d in deltas if abs(d) <= 0.005)

    if flat == len(deltas):
        return "plateau"
    if positive > negative:
        return "improving"
    if negative > positive:
        return "degrading"
    return "plateau"


def find_best_iteration(records: list[dict]) -> dict:
    """Find the iteration with the best primary metric value.

    Args:
        records: List of review-decision records.

    Returns:
        Dict with best_iteration (int) and best_value (float).
        Returns {best_iteration: None, best_value: None} if no records.
    """
    if not records:
        return {"best_iteration": None, "best_value": None}

    best_iter = None
    best_val = float("-inf")

    for rec in records:
        val = rec.get("primary_metric", {}).get("value")
        if val is not None and val > best_val:
            best_val = val
            best_iter = rec.get("iteration")

    if best_iter is None:
        return {"best_iteration": None, "best_value": None}

    return {"best_iteration": best_iter, "best_value": best_val}
