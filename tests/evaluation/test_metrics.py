import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    classify_risk_flags,
    compute_bootstrap_ci,
    compute_headline_metrics,
    compute_leakage_indicators,
    compute_overfitting_check,
    compute_plateau_signal,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable metric dicts
# ---------------------------------------------------------------------------

@pytest.fixture
def titanic_metrics():
    """Metrics matching Titanic iteration-1 output."""
    return {
        "primary": {"name": "val_auc_roc", "value": 0.8349},
        "secondary": {"val_accuracy": 0.793, "val_f1": 0.726},
        "train": {"train_auc_roc": 0.8702, "train_accuracy": 0.808},
        "validation": {"val_auc_roc": 0.8349, "val_accuracy": 0.793},
    }


@pytest.fixture
def titanic_feature_importance():
    return {
        "method": "logistic_regression_coefficients",
        "features": [
            {"name": "Sex", "importance": 2.52},
            {"name": "has_cabin", "importance": 0.80},
            {"name": "Pclass", "importance": 0.60},
            {"name": "Fare", "importance": 0.40},
            {"name": "Embarked_Q", "importance": 0.34},
            {"name": "SibSp", "importance": 0.33},
            {"name": "Embarked_S", "importance": 0.27},
            {"name": "Parch", "importance": 0.17},
            {"name": "Age", "importance": 0.04},
        ],
        "sorted": True,
        "model": "LogisticRegression",
    }


@pytest.fixture
def non_iterative_learning_curves():
    return {"note": "model does not support iterative training"}


@pytest.fixture
def iterative_learning_curves():
    return {
        "metric_name": "auc_roc",
        "train": [0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89, 0.90],
        "validation": [0.68, 0.72, 0.76, 0.78, 0.79, 0.80, 0.805, 0.808, 0.81],
        "iterations": list(range(1, 10)),
    }


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

class TestHeadlineMetrics:
    def test_repackage(self, titanic_metrics):
        result = compute_headline_metrics(titanic_metrics)
        assert result["primary"]["name"] == "val_auc_roc"
        assert result["primary"]["value"] == pytest.approx(0.8349)
        assert "val_accuracy" in result["secondary"]
        assert "train_auc_roc" in result["train"]

    def test_missing_optional_keys(self):
        minimal = {"primary": {"name": "acc", "value": 0.9}}
        result = compute_headline_metrics(minimal)
        assert result["secondary"] == {}
        assert result["train"] == {}


# ---------------------------------------------------------------------------
# Overfitting check
# ---------------------------------------------------------------------------

class TestOverfittingCheck:
    def test_titanic_low_severity(self, titanic_metrics, non_iterative_learning_curves):
        result = compute_overfitting_check(
            titanic_metrics, non_iterative_learning_curves, "binary_classification"
        )
        assert result["severity"] == "low"
        assert result["train_val_gap"]["gap"] == pytest.approx(0.0353, abs=0.001)
        assert result["train_val_gap"]["gap_pct"] < 5.0
        assert result["learning_curve_trend"] == "unavailable"

    def test_medium_severity(self, non_iterative_learning_curves):
        metrics = {
            "primary": {"name": "val_auc_roc", "value": 0.75},
            "train": {"train_auc_roc": 0.85},
        }
        result = compute_overfitting_check(
            metrics, non_iterative_learning_curves, "binary_classification"
        )
        assert result["severity"] == "medium"
        assert 5.0 <= result["train_val_gap"]["gap_pct"] < 15.0

    def test_high_severity(self, non_iterative_learning_curves):
        metrics = {
            "primary": {"name": "val_auc_roc", "value": 0.60},
            "train": {"train_auc_roc": 0.95},
        }
        result = compute_overfitting_check(
            metrics, non_iterative_learning_curves, "binary_classification"
        )
        assert result["severity"] == "high"
        assert result["train_val_gap"]["gap_pct"] >= 15.0

    def test_no_matching_train_metric(self, non_iterative_learning_curves):
        metrics = {
            "primary": {"name": "val_auc_roc", "value": 0.80},
            "train": {"something_else": 0.90},
        }
        result = compute_overfitting_check(
            metrics, non_iterative_learning_curves, "binary_classification"
        )
        assert result["severity"] == "unknown"

    def test_with_iterative_learning_curves(self, titanic_metrics, iterative_learning_curves):
        result = compute_overfitting_check(
            titanic_metrics, iterative_learning_curves, "binary_classification"
        )
        assert result["learning_curve_trend"] in ("improving", "plateau", "diverging")


# ---------------------------------------------------------------------------
# Leakage indicators
# ---------------------------------------------------------------------------

class TestLeakageIndicators:
    def test_titanic_no_leakage(self, titanic_metrics, titanic_feature_importance):
        result = compute_leakage_indicators(
            titanic_metrics, titanic_feature_importance, "binary_classification"
        )
        assert result["suspiciously_high_metric"] is False
        assert result["feature_importance_anomalies"] == []
        assert "No leakage" in result["verdict"]

    def test_suspicious_metric(self, titanic_feature_importance):
        metrics = {"primary": {"name": "val_auc_roc", "value": 0.9995}}
        result = compute_leakage_indicators(
            metrics, titanic_feature_importance, "binary_classification"
        )
        assert result["suspiciously_high_metric"] is True
        assert "leakage" in result["verdict"].lower()

    def test_dominant_feature(self, titanic_metrics):
        fi = {
            "features": [
                {"name": "leak_col", "importance": 9.0},
                {"name": "other", "importance": 0.1},
            ]
        }
        result = compute_leakage_indicators(
            titanic_metrics, fi, "binary_classification"
        )
        assert len(result["feature_importance_anomalies"]) == 1
        assert "leak_col" in result["feature_importance_anomalies"][0]


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

class TestRiskFlags:
    def test_no_flags_titanic(
        self, titanic_metrics, titanic_feature_importance, non_iterative_learning_curves
    ):
        overfitting = compute_overfitting_check(
            titanic_metrics, non_iterative_learning_curves, "binary_classification"
        )
        leakage = compute_leakage_indicators(
            titanic_metrics, titanic_feature_importance, "binary_classification"
        )
        flags = classify_risk_flags(
            overfitting, leakage, titanic_metrics, "binary_classification"
        )
        assert flags == []

    def test_overfitting_flag(self, titanic_metrics):
        overfitting = {
            "severity": "high",
            "train_val_gap": {"gap_pct": 20.0, "metric": "val_auc_roc"},
        }
        leakage = {
            "suspiciously_high_metric": False,
            "feature_importance_anomalies": [],
        }
        flags = classify_risk_flags(
            overfitting, leakage, titanic_metrics, "binary_classification"
        )
        assert any(f["type"] == "overfitting" for f in flags)

    def test_underfitting_flag(self):
        metrics = {"primary": {"name": "val_auc_roc", "value": 0.45}}
        overfitting = {"severity": "low", "train_val_gap": {"gap_pct": 1.0, "metric": "auc"}}
        leakage = {"suspiciously_high_metric": False, "feature_importance_anomalies": []}
        flags = classify_risk_flags(
            overfitting, leakage, metrics, "binary_classification"
        )
        assert any(f["type"] == "underfitting" for f in flags)


# ---------------------------------------------------------------------------
# Plateau signal
# ---------------------------------------------------------------------------

class TestPlateauSignal:
    def test_iteration_1_no_plateau(self, tmp_path):
        result = compute_plateau_signal(1, tmp_path)
        assert result["detected"] is False
        assert result["consecutive_stale_iterations"] == 0

    def test_no_prior_reports(self, tmp_path):
        result = compute_plateau_signal(3, tmp_path)
        assert result["detected"] is False

    def test_detects_plateau(self, tmp_path):
        for i in range(1, 5):
            report_dir = tmp_path / f"iteration-{i}" / "reports"
            report_dir.mkdir(parents=True)
            (report_dir / "model-report.json").write_text(json.dumps({
                "headline_metrics": {
                    "primary": {"name": "val_auc_roc", "value": 0.835 + i * 0.001}
                }
            }))

        result = compute_plateau_signal(5, tmp_path)
        assert result["detected"] is True
        assert result["consecutive_stale_iterations"] >= 2

    def test_no_plateau_with_improvements(self, tmp_path):
        for i in range(1, 4):
            report_dir = tmp_path / f"iteration-{i}" / "reports"
            report_dir.mkdir(parents=True)
            (report_dir / "model-report.json").write_text(json.dumps({
                "headline_metrics": {
                    "primary": {"name": "val_auc_roc", "value": 0.70 + i * 0.05}
                }
            }))

        result = compute_plateau_signal(4, tmp_path)
        assert result["detected"] is False


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def _write_predictions(self, path, n=200, seed=42):
        rng = np.random.RandomState(seed)
        y_true = rng.randint(0, 2, size=n)
        y_prob_1 = rng.beta(2, 5, size=n)
        y_prob_1[y_true == 1] = rng.beta(5, 2, size=(y_true == 1).sum())
        y_pred = (y_prob_1 > 0.5).astype(int)
        df = pd.DataFrame({
            "index": range(n),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob_0": 1 - y_prob_1,
            "y_prob_1": y_prob_1,
        })
        df.to_csv(path, index=False)

    def test_binary_classification(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        self._write_predictions(pred_path)
        metrics = {"primary": {"name": "val_auc_roc", "value": 0.835}}
        result = compute_bootstrap_ci(
            pred_path, metrics, "binary_classification", n_bootstrap=200
        )

        assert result["n_samples"] == 200
        assert result["intervals"] is not None
        assert "auc_roc" in result["intervals"]
        ci = result["intervals"]["auc_roc"]
        assert ci["ci_lower"] < ci["ci_upper"]
        assert ci["confidence"] == 0.95
        assert "interpretation" in result

    def test_small_sample(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        self._write_predictions(pred_path, n=50)
        metrics = {"primary": {"name": "val_auc_roc", "value": 0.80}}
        result = compute_bootstrap_ci(
            pred_path, metrics, "binary_classification", n_bootstrap=500
        )
        # Should produce valid intervals even with smaller samples
        assert result["n_samples"] == 50
        if result["primary_ci"]:
            assert result["primary_ci"]["ci_lower"] <= result["primary_ci"]["ci_upper"]
