"""Validates review-decision records against Contract 3 (extended).

Pattern follows src/evaluation/validator.py and src/planning/validator.py:
early-fail on first violation, descriptive error messages.
"""

import json
from pathlib import Path
from typing import Union

from .schemas import VALID_STATUSES, VALID_VERDICTS, VALID_ROUTES


class ReviewValidationError(Exception):
    """Raised when a review-decision record fails schema validation."""


_REQUIRED_KEYS = [
    "iteration",
    "timestamp",
    "status",
    "plan_summary",
    "primary_metric",
    "model_family",
    "reviewer_verdict",
    "reviewer_reasoning",
    "router_decision",
    "router_reasoning",
    "risk_flags_summary",
    "best_iteration",
]

_REQUIRED_METRIC_KEYS = ["name", "value", "delta"]

_VALID_RISK_TYPES = {"leakage", "overfitting", "underfitting", "data_issue"}
_VALID_SEVERITIES = {"low", "medium", "high"}


def validate_review_decision(record: Union[str, Path, dict]) -> dict:
    """Validate a review-decision record against the extended Contract 3 schema.

    Args:
        record: Path to a JSON file, JSON string, or parsed dict.

    Returns:
        Summary dict with key validated fields.

    Raises:
        ReviewValidationError: Descriptive message on first violation.
        FileNotFoundError: If path does not exist.
    """
    if isinstance(record, (str, Path)):
        path = Path(record)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
        else:
            # Might be a JSON string
            try:
                data = json.loads(str(record))
            except json.JSONDecodeError:
                raise FileNotFoundError(f"File not found and not valid JSON: {record}")
    else:
        data = record

    # --- Required keys ---
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ReviewValidationError(f"Missing required key: '{key}'")

    # --- Type checks ---
    if not isinstance(data["iteration"], int) or data["iteration"] < 1:
        raise ReviewValidationError(
            f"'iteration' must be a positive int, got: {data['iteration']}"
        )

    if not isinstance(data["timestamp"], str) or len(data["timestamp"]) < 10:
        raise ReviewValidationError(
            f"'timestamp' must be an ISO 8601 string, got: {data['timestamp']}"
        )

    if data["status"] not in VALID_STATUSES:
        raise ReviewValidationError(
            f"'status' must be one of {VALID_STATUSES}, got: '{data['status']}'"
        )

    if not isinstance(data["plan_summary"], str) or not data["plan_summary"].strip():
        raise ReviewValidationError("'plan_summary' must be a non-empty string")

    # --- primary_metric ---
    pm = data["primary_metric"]
    if not isinstance(pm, dict):
        raise ReviewValidationError(
            f"'primary_metric' must be a dict, got: {type(pm).__name__}"
        )
    for mk in _REQUIRED_METRIC_KEYS:
        if mk not in pm:
            raise ReviewValidationError(
                f"'primary_metric' missing required key: '{mk}'"
            )
    if not isinstance(pm["name"], str) or not pm["name"].strip():
        raise ReviewValidationError("'primary_metric.name' must be a non-empty string")
    if not isinstance(pm["value"], (int, float)):
        raise ReviewValidationError(
            f"'primary_metric.value' must be numeric, got: {type(pm['value']).__name__}"
        )
    if pm["delta"] is not None and not isinstance(pm["delta"], (int, float)):
        raise ReviewValidationError(
            f"'primary_metric.delta' must be numeric or null, got: {type(pm['delta']).__name__}"
        )

    if not isinstance(data["model_family"], str) or not data["model_family"].strip():
        raise ReviewValidationError("'model_family' must be a non-empty string")

    # --- Verdict and route ---
    if data["reviewer_verdict"] not in VALID_VERDICTS:
        raise ReviewValidationError(
            f"'reviewer_verdict' must be one of {VALID_VERDICTS}, got: '{data['reviewer_verdict']}'"
        )

    if not isinstance(data["reviewer_reasoning"], str) or not data["reviewer_reasoning"].strip():
        raise ReviewValidationError("'reviewer_reasoning' must be a non-empty string")

    if data["router_decision"] not in VALID_ROUTES:
        raise ReviewValidationError(
            f"'router_decision' must be one of {VALID_ROUTES}, got: '{data['router_decision']}'"
        )

    if not isinstance(data["router_reasoning"], str) or not data["router_reasoning"].strip():
        raise ReviewValidationError("'router_reasoning' must be a non-empty string")

    # --- risk_flags_summary ---
    flags = data["risk_flags_summary"]
    if not isinstance(flags, list):
        raise ReviewValidationError(
            f"'risk_flags_summary' must be a list, got: {type(flags).__name__}"
        )
    for i, flag in enumerate(flags):
        if not isinstance(flag, dict):
            raise ReviewValidationError(
                f"'risk_flags_summary[{i}]' must be a dict, got: {type(flag).__name__}"
            )
        for fk in ("type", "severity", "evidence"):
            if fk not in flag:
                raise ReviewValidationError(
                    f"'risk_flags_summary[{i}]' missing required key: '{fk}'"
                )
        if flag["type"] not in _VALID_RISK_TYPES:
            raise ReviewValidationError(
                f"'risk_flags_summary[{i}].type' must be one of {_VALID_RISK_TYPES}, got: '{flag['type']}'"
            )
        if flag["severity"] not in _VALID_SEVERITIES:
            raise ReviewValidationError(
                f"'risk_flags_summary[{i}].severity' must be one of {_VALID_SEVERITIES}, got: '{flag['severity']}'"
            )

    # --- best_iteration ---
    if not isinstance(data["best_iteration"], int) or data["best_iteration"] < 1:
        raise ReviewValidationError(
            f"'best_iteration' must be a positive int, got: {data['best_iteration']}"
        )

    # --- Logical consistency ---
    if data["reviewer_verdict"] == "sufficient" and data["router_decision"] != "continue":
        # When sufficient, router_decision should still be set but is informational.
        # We allow any value but the typical case is "continue" as a no-op.
        pass

    return {
        "iteration": data["iteration"],
        "reviewer_verdict": data["reviewer_verdict"],
        "router_decision": data["router_decision"],
        "best_iteration": data["best_iteration"],
        "risk_flag_count": len(flags),
    }
