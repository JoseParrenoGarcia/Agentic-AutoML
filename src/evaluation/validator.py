"""
M6.4 — Validator for model-report.json (Contract 4).

Follows the project validator pattern: accept path/dict, raise specific
exception on first violation, return summary dict on success.

Runnable as module: python -m src.evaluation.validator <report_path>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Union


class ReportValidationError(Exception):
    """Raised when model-report.json fails schema validation."""


_REQUIRED_KEYS = [
    "schema_version",
    "iteration",
    "task_type",
    "generated_at",
    "headline_metrics",
    "overfitting_check",
    "leakage_indicators",
    "error_analysis",
    "feature_importance",
    "prior_run_comparison",
    "reviewer_summary",
    "plots",
]

_VALID_TASK_TYPES = {"binary_classification", "multiclass", "regression"}
_VALID_VERDICTS = {"improved", "degraded", "neutral", "suspicious"}
_VALID_SEVERITIES = {"low", "medium", "high", "unknown"}
_VALID_RISK_TYPES = {"leakage", "overfitting", "underfitting", "data_issue"}
_VALID_TRENDS = {"improving", "plateau", "diverging", "unavailable"}


def validate_report(report: Union[str, Path, dict]) -> dict:
    """
    Validate a model-report.json against Contract 4 schema.

    Args:
        report: Path to JSON file, or already-parsed dict.

    Returns:
        Summary dict with: iteration, task_type, headline_verdict,
        risk_flag_count, primary_metric.

    Raises:
        ReportValidationError: Descriptive message on first violation.
        FileNotFoundError: If report path does not exist.
    """
    if isinstance(report, (str, Path)):
        path = Path(report)
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")
        with open(path) as f:
            data = json.load(f)
    else:
        data = report

    if not isinstance(data, dict):
        raise ReportValidationError("Report must be a JSON object")

    # Required top-level keys
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ReportValidationError(f"Missing required key: '{key}'")

    # task_type
    if data["task_type"] not in _VALID_TASK_TYPES:
        raise ReportValidationError(
            f"Invalid task_type: '{data['task_type']}'. "
            f"Expected one of: {_VALID_TASK_TYPES}"
        )

    # headline_metrics
    hm = data["headline_metrics"]
    if not isinstance(hm, dict):
        raise ReportValidationError("headline_metrics must be a dict")
    if "primary" not in hm:
        raise ReportValidationError("headline_metrics missing 'primary'")
    primary = hm["primary"]
    if "name" not in primary or "value" not in primary:
        raise ReportValidationError(
            "headline_metrics.primary must have 'name' and 'value'"
        )
    if not isinstance(primary["value"], (int, float)):
        raise ReportValidationError(
            f"headline_metrics.primary.value must be numeric, got {type(primary['value']).__name__}"
        )

    # overfitting_check
    oc = data["overfitting_check"]
    if "severity" not in oc:
        raise ReportValidationError("overfitting_check missing 'severity'")
    if oc["severity"] not in _VALID_SEVERITIES:
        raise ReportValidationError(
            f"Invalid overfitting severity: '{oc['severity']}'"
        )
    if "learning_curve_trend" in oc and oc["learning_curve_trend"] not in _VALID_TRENDS:
        raise ReportValidationError(
            f"Invalid learning_curve_trend: '{oc['learning_curve_trend']}'"
        )

    # reviewer_summary
    rs = data["reviewer_summary"]
    if "headline_verdict" not in rs:
        raise ReportValidationError("reviewer_summary missing 'headline_verdict'")
    if rs["headline_verdict"] not in _VALID_VERDICTS:
        raise ReportValidationError(
            f"Invalid headline_verdict: '{rs['headline_verdict']}'"
        )

    # risk_flags
    for i, flag in enumerate(rs.get("risk_flags", [])):
        if flag.get("type") not in _VALID_RISK_TYPES:
            raise ReportValidationError(
                f"risk_flags[{i}] has invalid type: '{flag.get('type')}'"
            )
        if flag.get("severity") not in _VALID_SEVERITIES:
            raise ReportValidationError(
                f"risk_flags[{i}] has invalid severity: '{flag.get('severity')}'"
            )

    # feature_importance
    fi = data["feature_importance"]
    if "features" not in fi:
        raise ReportValidationError("feature_importance missing 'features'")

    # separation_quality (optional)
    sq = data.get("separation_quality")
    if sq is not None and "quality" in sq:
        if sq["quality"] not in ("strong", "moderate", "weak"):
            raise ReportValidationError(
                f"Invalid separation_quality.quality: '{sq['quality']}'"
            )

    # plots
    plots = data["plots"]
    if not isinstance(plots, dict):
        raise ReportValidationError("plots must be a dict")

    return {
        "iteration": data["iteration"],
        "task_type": data["task_type"],
        "headline_verdict": rs["headline_verdict"],
        "risk_flag_count": len(rs.get("risk_flags", [])),
        "primary_metric": primary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate model-report.json")
    parser.add_argument("report_path", help="Path to model-report.json")
    args = parser.parse_args()

    try:
        summary = validate_report(Path(args.report_path))
        print(
            f"Valid: iteration={summary['iteration']} "
            f"verdict={summary['headline_verdict']} "
            f"risks={summary['risk_flag_count']}"
        )
    except ReportValidationError as e:
        print(f"VALIDATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
