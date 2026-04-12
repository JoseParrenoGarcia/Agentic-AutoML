from pathlib import Path
from typing import Union

import yaml


class PlanValidationError(Exception):
    """Raised when an experiment plan fails schema validation."""


def validate_plan(plan: Union[str, Path, dict]) -> dict:
    """
    Validate an experiment plan against the required schema.

    Args:
        plan: Path to a YAML file (str or Path) or an already-parsed dict.

    Returns:
        Parsed plan dict if valid.

    Raises:
        PlanValidationError: Descriptive message on first violation found.
        FileNotFoundError: If a path is given and the file does not exist.
    """
    if isinstance(plan, (str, Path)):
        path = Path(plan)
        if not path.exists():
            raise FileNotFoundError(f"Plan file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        data = plan

    required_top_level = [
        "iteration",
        "objective",
        "hypotheses",
        "feature_steps",
        "model_steps",
        "evaluation_focus",
        "expected_win_condition",
        "rollback_or_stop_condition",
    ]
    for key in required_top_level:
        if key not in data:
            raise PlanValidationError(f"Missing required field: {key}")

    if not isinstance(data["iteration"], int) or data["iteration"] < 1:
        raise PlanValidationError(
            f"'iteration' must be an int >= 1, got: {data['iteration']!r}"
        )

    for field in ("objective", "evaluation_focus", "expected_win_condition", "rollback_or_stop_condition"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise PlanValidationError(
                f"'{field}' must be a non-empty string, got: {value!r}"
            )

    hypotheses = data["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) == 0:
        raise PlanValidationError("'hypotheses' must be a non-empty list")
    for i, h in enumerate(hypotheses):
        for sub in ("id", "description", "expected_impact"):
            if sub not in h:
                raise PlanValidationError(
                    f"Hypothesis at index {i} missing required sub-field: '{sub}'"
                )
            if not isinstance(h[sub], str) or not h[sub].strip():
                raise PlanValidationError(
                    f"Hypothesis at index {i}: '{sub}' must be a non-empty string, got: {h[sub]!r}"
                )

    feature_steps = data["feature_steps"]
    if not isinstance(feature_steps, list):
        raise PlanValidationError("'feature_steps' must be a list")
    for i, step in enumerate(feature_steps):
        for sub in ("name", "action", "rationale"):
            if sub not in step:
                raise PlanValidationError(
                    f"feature_steps[{i}] missing required sub-field: '{sub}'"
                )
            if not isinstance(step[sub], str) or not step[sub].strip():
                raise PlanValidationError(
                    f"feature_steps[{i}]: '{sub}' must be a non-empty string, got: {step[sub]!r}"
                )

    model_steps = data["model_steps"]
    if not isinstance(model_steps, list) or len(model_steps) == 0:
        raise PlanValidationError("'model_steps' must be a non-empty list")
    for i, step in enumerate(model_steps):
        if "algorithm" not in step or not isinstance(step["algorithm"], str) or not step["algorithm"].strip():
            raise PlanValidationError(
                f"model_steps[{i}]: 'algorithm' must be a non-empty string"
            )
        if "hyperparameters" not in step or not isinstance(step["hyperparameters"], dict):
            raise PlanValidationError(
                f"model_steps[{i}]: 'hyperparameters' must be a dict (may be empty)"
            )
        if "rationale" not in step or not isinstance(step["rationale"], str) or not step["rationale"].strip():
            raise PlanValidationError(
                f"model_steps[{i}]: 'rationale' must be a non-empty string"
            )

    return data
