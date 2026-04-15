import json

import pytest

from src.evaluation.validator import ReportValidationError, validate_report


def _valid_report():
    """Return a minimal valid model-report.json dict."""
    return {
        "schema_version": "1.0.0",
        "iteration": 1,
        "task_type": "binary_classification",
        "generated_at": "2026-04-15T10:00:00",
        "headline_metrics": {
            "primary": {"name": "val_auc_roc", "value": 0.835},
            "secondary": {"val_accuracy": 0.79},
            "train": {"train_auc_roc": 0.87},
            "validation": {"val_auc_roc": 0.835},
        },
        "overfitting_check": {
            "severity": "low",
            "learning_curve_trend": "unavailable",
            "train_val_gap": {"metric": "val_auc_roc", "gap_pct": 4.0},
            "verdict": "OK",
        },
        "leakage_indicators": {
            "suspiciously_high_metric": False,
            "feature_importance_anomalies": [],
            "verdict": "No leakage",
        },
        "error_analysis": {
            "task_type": "binary_classification",
            "confusion_matrix": {"tp": 50, "fp": 10, "fn": 15, "tn": 25},
        },
        "feature_importance": {
            "method": "coefficients",
            "model": "LogisticRegression",
            "features": [{"name": "Sex", "importance": 2.5, "rank": 1}],
        },
        "prior_run_comparison": None,
        "reviewer_summary": {
            "headline_verdict": "neutral",
            "metric_summary": {
                "primary_metric": 0.835,
                "delta_vs_previous": None,
                "delta_vs_baseline": None,
            },
            "risk_flags": [],
            "plateau_signal": {"detected": False, "consecutive_stale_iterations": 0},
        },
        "plots": {
            "confusion_matrix": "plots/confusion_matrix.png",
            "actual_vs_predicted": "plots/actual_vs_predicted.png",
            "calibration_curve": None,
            "error_distribution": "plots/error_distribution.png",
            "feature_diagnostics": [],
        },
    }


class TestValidateReport:
    def test_valid_report(self):
        result = validate_report(_valid_report())
        assert result["iteration"] == 1
        assert result["headline_verdict"] == "neutral"
        assert result["risk_flag_count"] == 0

    def test_from_file(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text(json.dumps(_valid_report()))
        result = validate_report(path)
        assert result["iteration"] == 1

    def test_missing_required_key(self):
        report = _valid_report()
        del report["headline_metrics"]
        with pytest.raises(ReportValidationError, match="headline_metrics"):
            validate_report(report)

    def test_invalid_task_type(self):
        report = _valid_report()
        report["task_type"] = "image_classification"
        with pytest.raises(ReportValidationError, match="task_type"):
            validate_report(report)

    def test_invalid_verdict(self):
        report = _valid_report()
        report["reviewer_summary"]["headline_verdict"] = "excellent"
        with pytest.raises(ReportValidationError, match="headline_verdict"):
            validate_report(report)

    def test_invalid_severity(self):
        report = _valid_report()
        report["overfitting_check"]["severity"] = "extreme"
        with pytest.raises(ReportValidationError, match="severity"):
            validate_report(report)

    def test_invalid_risk_flag_type(self):
        report = _valid_report()
        report["reviewer_summary"]["risk_flags"] = [
            {"type": "invalid_type", "severity": "low", "evidence": "test"}
        ]
        with pytest.raises(ReportValidationError, match="risk_flags"):
            validate_report(report)

    def test_missing_primary_in_headline(self):
        report = _valid_report()
        del report["headline_metrics"]["primary"]
        with pytest.raises(ReportValidationError, match="primary"):
            validate_report(report)

    def test_non_numeric_primary_value(self):
        report = _valid_report()
        report["headline_metrics"]["primary"]["value"] = "not_a_number"
        with pytest.raises(ReportValidationError, match="numeric"):
            validate_report(report)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_report(tmp_path / "nonexistent.json")
