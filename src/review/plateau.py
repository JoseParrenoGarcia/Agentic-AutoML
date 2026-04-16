"""Plateau detection helper for the reviewer-router agent.

Reads M6's plateau_signal from model-report.json and cross-references
with run-history to confirm plateau status.
"""

from .config import PLATEAU_STALE_THRESHOLD, PLATEAU_CONSECUTIVE_MIN


def check_plateau(
    model_report: dict,
    history_records: list[dict],
) -> dict:
    """Determine plateau status from model report and run history.

    Args:
        model_report: Parsed model-report.json from M6.
        history_records: List of prior review-decision records.

    Returns:
        Dict with:
        - detected: bool — whether plateau is confirmed
        - consecutive_stale: int — number of consecutive stale iterations
        - source: str — "model_report" or "history_computed" or "none"
        - multiple_strategies_tried: bool — whether >1 model family has been tried
    """
    # Count unique model families from history
    families = set()
    for rec in history_records:
        family = rec.get("model_family")
        if family:
            families.add(family)

    # Also count current iteration's family from model report
    current_family = (
        model_report.get("feature_importance", {}).get("model")
    )
    if current_family:
        families.add(current_family)

    multiple_strategies = len(families) > 1

    # Primary source: M6's reviewer_summary.plateau_signal
    reviewer_summary = model_report.get("reviewer_summary", {})
    plateau_signal = reviewer_summary.get("plateau_signal", {})

    if plateau_signal:
        m6_detected = plateau_signal.get("detected", False)
        m6_stale = plateau_signal.get("consecutive_stale_iterations", 0)

        if m6_detected or m6_stale >= PLATEAU_CONSECUTIVE_MIN:
            return {
                "detected": True,
                "consecutive_stale": m6_stale,
                "source": "model_report",
                "multiple_strategies_tried": multiple_strategies,
            }

    # Fallback: compute from history deltas
    if len(history_records) >= PLATEAU_CONSECUTIVE_MIN:
        recent = history_records[-PLATEAU_CONSECUTIVE_MIN:]
        stale_count = 0
        for rec in recent:
            delta = rec.get("primary_metric", {}).get("delta")
            if delta is not None and abs(delta) < PLATEAU_STALE_THRESHOLD:
                stale_count += 1

        if stale_count >= PLATEAU_CONSECUTIVE_MIN:
            return {
                "detected": True,
                "consecutive_stale": stale_count,
                "source": "history_computed",
                "multiple_strategies_tried": multiple_strategies,
            }

    return {
        "detected": False,
        "consecutive_stale": 0,
        "source": "none",
        "multiple_strategies_tried": multiple_strategies,
    }
