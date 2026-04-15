import csv
import json
from pathlib import Path
from typing import Union


class OutputValidationError(Exception):
    """Raised when post-run artifacts fail schema validation."""


_REQUIRED_ARTIFACTS = [
    "outputs/metrics.json",
    "outputs/predictions.csv",
    "outputs/feature_importance.json",
    "outputs/learning_curves.json",
    "outputs/pipeline_metadata.json",
    "outputs/model/model.pkl",
    "outputs/model/metadata.json",
]

_BINARY_PREDICTION_COLUMNS = {"index", "y_true", "y_pred", "y_prob_0", "y_prob_1"}
_REGRESSION_PREDICTION_COLUMNS = {"index", "y_true", "y_pred"}


def validate_outputs(
    iteration_dir: Union[str, Path],
    task_type: str = "binary_classification",
) -> dict:
    """
    Validate post-run artifacts conform to Contract 5 schemas.

    Args:
        iteration_dir: Path to the iteration root.
        task_type: ``"binary_classification"`` or ``"regression"``.

    Returns:
        Summary dict with keys: iteration_dir, artifacts_checked,
        metrics_primary, task_type.

    Raises:
        OutputValidationError: Descriptive message on the first violation.
        FileNotFoundError: If *iteration_dir* does not exist.
    """
    root = Path(iteration_dir)
    if not root.exists():
        raise FileNotFoundError(f"Iteration directory not found: {root}")

    # --- Check 1: Required artifact files exist ---
    for rel in _REQUIRED_ARTIFACTS:
        path = root / rel
        if not path.exists():
            raise OutputValidationError(
                f"Required artifact missing: {rel} (expected at {path})"
            )

    # --- Check 2: metrics.json schema ---
    metrics_path = root / "outputs" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    if not isinstance(metrics, dict):
        raise OutputValidationError("metrics.json did not parse to a dict")

    for key in ("primary", "secondary", "train", "validation"):
        if key not in metrics:
            raise OutputValidationError(
                f"metrics.json missing required key: '{key}'"
            )

    primary = metrics["primary"]
    if not isinstance(primary, dict) or "name" not in primary or "value" not in primary:
        raise OutputValidationError(
            "metrics.json 'primary' must have 'name' and 'value' keys"
        )
    if not isinstance(primary["value"], (int, float)):
        raise OutputValidationError(
            f"metrics.json primary.value must be numeric, got {type(primary['value']).__name__}"
        )

    # --- Check 3: predictions.csv columns ---
    predictions_path = root / "outputs" / "predictions.csv"
    with open(predictions_path, newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])

        expected = (
            _BINARY_PREDICTION_COLUMNS
            if task_type == "binary_classification"
            else _REGRESSION_PREDICTION_COLUMNS
        )
        missing = expected - columns
        if missing:
            raise OutputValidationError(
                f"predictions.csv missing columns for {task_type}: {sorted(missing)}"
            )

        # Check for NaN in required columns (sample first 10 rows)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            for col in expected:
                val = row.get(col, "")
                if val.strip().lower() in ("nan", ""):
                    raise OutputValidationError(
                        f"predictions.csv has NaN/empty in column '{col}' at row {i + 1}"
                    )

    # --- Check 4: feature_importance.json ---
    fi_path = root / "outputs" / "feature_importance.json"
    with open(fi_path) as f:
        fi = json.load(f)

    for key in ("method", "features", "sorted", "model"):
        if key not in fi:
            raise OutputValidationError(
                f"feature_importance.json missing required key: '{key}'"
            )
    if not fi["features"]:
        raise OutputValidationError("feature_importance.json 'features' is empty")
    if fi["sorted"] is not True:
        raise OutputValidationError("feature_importance.json 'sorted' must be true")

    # --- Check 5: learning_curves.json ---
    lc_path = root / "outputs" / "learning_curves.json"
    with open(lc_path) as f:
        lc = json.load(f)

    if "note" not in lc:
        # Must have matching-length arrays
        for key in ("metric_name", "train", "validation", "iterations"):
            if key not in lc:
                raise OutputValidationError(
                    f"learning_curves.json missing key: '{key}' (and no 'note' key)"
                )
        if len(lc["train"]) != len(lc["validation"]) or len(lc["train"]) != len(lc["iterations"]):
            raise OutputValidationError(
                "learning_curves.json: train, validation, and iterations arrays must have equal length"
            )

    # --- Check 6: pipeline_metadata.json ---
    pm_path = root / "outputs" / "pipeline_metadata.json"
    with open(pm_path) as f:
        pm = json.load(f)

    for key in ("stages", "total_duration_s", "python_version", "packages"):
        if key not in pm:
            raise OutputValidationError(
                f"pipeline_metadata.json missing required key: '{key}'"
            )

    # --- Check 7: model/metadata.json ---
    mm_path = root / "outputs" / "model" / "metadata.json"
    with open(mm_path) as f:
        mm = json.load(f)

    for key in ("model_class", "feature_list", "training_timestamp"):
        if key not in mm:
            raise OutputValidationError(
                f"model/metadata.json missing required key: '{key}'"
            )
    if not mm["feature_list"]:
        raise OutputValidationError("model/metadata.json 'feature_list' is empty")

    # --- Check 8: model.pkl non-zero ---
    pkl_path = root / "outputs" / "model" / "model.pkl"
    if pkl_path.stat().st_size == 0:
        raise OutputValidationError("model/model.pkl is empty (0 bytes)")

    return {
        "iteration_dir": str(root),
        "artifacts_checked": _REQUIRED_ARTIFACTS,
        "metrics_primary": metrics["primary"],
        "task_type": task_type,
    }
