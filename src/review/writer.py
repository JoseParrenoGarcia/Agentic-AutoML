"""Validate and append a review-decision record to run-history.jsonl.

Called by the reviewer-router agent after making its verdict and routing decision.
Ensures every appended record passes Contract 3 validation before writing.
Also maintains a human-readable decision-log.md alongside run-history.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from .validator import validate_review_decision, ReviewValidationError


def format_decision_log_entry(record: dict) -> str:
    """Format a review-decision record as a markdown decision-log entry.

    Args:
        record: A validated review-decision dict (Contract 3).

    Returns:
        Markdown string for one iteration entry (with trailing newline).
    """
    iteration = record.get("iteration", "?")
    model_family = record.get("model_family", "unknown")
    pm = record.get("primary_metric", {})
    pm_name = pm.get("name", "metric")
    pm_value = pm.get("value")
    pm_delta = pm.get("delta")
    verdict = record.get("reviewer_verdict", "unknown")
    route = record.get("router_decision", "unknown")
    plan_summary = record.get("plan_summary", "")
    reasoning = record.get("reviewer_reasoning", "")
    risk_flags = record.get("risk_flags_summary", [])

    # Format metric line
    value_str = f"{pm_value:.4f}" if pm_value is not None else "N/A"
    if pm_delta is not None:
        delta_str = f"{pm_delta:+.4f}"
    else:
        delta_str = "N/A (baseline)"

    # Format risk flags
    if risk_flags:
        flag_types = [f"{f.get('severity', '?')}-{f.get('type', '?')}" for f in risk_flags]
        flags_str = f"{len(risk_flags)} ({', '.join(flag_types)})"
    else:
        flags_str = "none"

    # Truncate reasoning to keep entries scannable
    reasoning_short = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning

    return (
        f"## Iteration {iteration} — {model_family}\n"
        f"**Metric:** {pm_name} = {value_str} (delta: {delta_str})  \n"
        f"**Verdict:** {verdict} | **Route:** {route}  \n"
        f"**Summary:** {plan_summary}  \n"
        f"**Reasoning:** {reasoning_short}  \n"
        f"**Risk flags:** {flags_str}\n\n"
    )


def append_decision_log(record: dict, log_path: Union[str, Path]) -> None:
    """Append a formatted markdown entry to decision-log.md.

    Args:
        record: A validated review-decision dict (Contract 3).
        log_path: Path to decision-log.md. Created with header if it does not exist.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create with header if file doesn't exist or is empty
    if not path.exists() or path.stat().st_size == 0:
        header = (
            "# Decision Log\n"
            "<!-- Append-only. One section per completed iteration. -->\n\n"
        )
        with open(path, "w") as f:
            f.write(header)

    entry = format_decision_log_entry(record)
    with open(path, "a") as f:
        f.write(entry)


def append_review_decision(
    record: dict,
    history_path: Union[str, Path],
    iteration_dir: Union[str, Path, None] = None,
    decision_log_path: Union[str, Path, None] = None,
) -> dict:
    """Validate and append a review-decision record to run-history.jsonl.

    Also writes a copy as review-decision.json inside the iteration's reports/
    directory so each iteration is self-contained and auditable.

    If decision_log_path is provided (or can be inferred from history_path),
    also appends a human-readable entry to decision-log.md.

    Args:
        record: Review-decision dict matching Contract 3 extended schema.
        history_path: Path to run-history.jsonl. Created if it does not exist.
        iteration_dir: Path to the iteration directory (e.g., iterations/iteration-1).
            If provided, writes reports/review-decision.json there.
        decision_log_path: Path to decision-log.md. If None, defaults to
            decision-log.md in the same directory as history_path.

    Returns:
        Validation summary from validate_review_decision.

    Raises:
        ReviewValidationError: If the record fails validation.
    """
    # Validate before writing anything
    summary = validate_review_decision(record)

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Append as a single JSON line
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    # Write per-iteration copy for auditability
    if iteration_dir is not None:
        review_path = Path(iteration_dir) / "reports" / "review-decision.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with open(review_path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    # Append to decision-log.md
    if decision_log_path is None:
        decision_log_path = path.parent / "decision-log.md"
    append_decision_log(record, decision_log_path)

    return summary


def build_record(
    iteration: int,
    status: str,
    plan_summary: str,
    primary_metric_name: str,
    primary_metric_value: float,
    primary_metric_delta: float | None,
    model_family: str,
    reviewer_verdict: str,
    reviewer_reasoning: str,
    router_decision: str,
    router_reasoning: str,
    risk_flags_summary: list,
    best_iteration: int,
) -> dict:
    """Build a well-formed review-decision record with auto-generated timestamp.

    Convenience function so the agent doesn't have to construct the full dict manually.
    """
    return {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "plan_summary": plan_summary,
        "primary_metric": {
            "name": primary_metric_name,
            "value": primary_metric_value,
            "delta": primary_metric_delta,
        },
        "model_family": model_family,
        "reviewer_verdict": reviewer_verdict,
        "reviewer_reasoning": reviewer_reasoning,
        "router_decision": router_decision,
        "router_reasoning": router_reasoning,
        "risk_flags_summary": risk_flags_summary,
        "best_iteration": best_iteration,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate and append a review-decision record to run-history.jsonl"
    )
    parser.add_argument("record_json", help="Path to a JSON file with the review-decision record")
    parser.add_argument("history_path", help="Path to run-history.jsonl")
    args = parser.parse_args()

    try:
        with open(args.record_json) as f:
            record = json.load(f)
        summary = append_review_decision(record, args.history_path)
        print(f"SUCCESS: Appended iteration {summary['iteration']} "
              f"(verdict={summary['reviewer_verdict']}, route={summary['router_decision']})")
    except ReviewValidationError as e:
        print(f"VALIDATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
