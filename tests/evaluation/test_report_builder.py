import json

import numpy as np
import pandas as pd
import yaml

from src.evaluation.report_builder import (
    build_model_report,
    compute_prior_run_comparison,
    compute_reviewer_summary,
    load_iteration_inputs,
)


def _create_project_fixtures(tmp_path):
    """
    Create a minimal project + iteration directory structure for testing.
    Mirrors the Titanic iteration-1 layout.
    """
    project_dir = tmp_path / "project"
    iter_dir = project_dir / "iterations" / "iteration-1"

    # Raw data
    rng = np.random.RandomState(42)
    n = 200
    raw_df = pd.DataFrame({
        "Survived": rng.randint(0, 2, n),
        "Sex": rng.choice([0, 1], n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Age": rng.normal(30, 10, n),
        "Fare": rng.exponential(30, n),
    })
    data_dir = project_dir / "data" / "raw"
    data_dir.mkdir(parents=True)
    raw_df.to_csv(data_dir / "train.csv", index=False)

    # Profile
    profile_dir = project_dir / "artifacts" / "data"
    profile_dir.mkdir(parents=True)
    profile = {
        "columns": [
            {"name": "Survived", "inferred_semantic_type": "binary_target",
             "cardinality": {"unique_count": 2}},
            {"name": "Sex", "inferred_semantic_type": "categorical",
             "cardinality": {"unique_count": 2}},
            {"name": "Pclass", "inferred_semantic_type": "ordinal",
             "cardinality": {"unique_count": 3}},
            {"name": "Age", "inferred_semantic_type": "numeric",
             "cardinality": {"unique_count": 88}},
            {"name": "Fare", "inferred_semantic_type": "continuous",
             "cardinality": {"unique_count": 150}},
        ],
    }
    (profile_dir / "profile.json").write_text(json.dumps(profile))

    # Iteration directory
    outputs_dir = iter_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    model_dir = outputs_dir / "model"
    model_dir.mkdir()

    # Config
    config = {
        "iteration": 1,
        "random_seed": 42,
        "data": {"train": "../../data/raw/train.csv"},
        "target_column": "Survived",
        "task_type": "binary_classification",
        "split": {"method": "stratified", "val_ratio": 0.2},
        "hyperparameters": {},
        "output_paths": {},
    }
    with open(iter_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Metrics
    (outputs_dir / "metrics.json").write_text(json.dumps({
        "primary": {"name": "val_auc_roc", "value": 0.835},
        "secondary": {"val_accuracy": 0.793},
        "train": {"train_auc_roc": 0.870},
        "validation": {"val_auc_roc": 0.835},
    }))

    # Feature importance
    (outputs_dir / "feature_importance.json").write_text(json.dumps({
        "method": "coefficients",
        "features": [
            {"name": "Sex", "importance": 2.5},
            {"name": "Pclass", "importance": 0.6},
            {"name": "Age", "importance": 0.4},
            {"name": "Fare", "importance": 0.3},
        ],
        "sorted": True,
        "model": "LogisticRegression",
    }))

    # Learning curves
    (outputs_dir / "learning_curves.json").write_text(json.dumps({
        "note": "model does not support iterative training",
    }))

    # Pipeline metadata
    (outputs_dir / "pipeline_metadata.json").write_text(json.dumps({
        "stages": [],
        "total_duration_s": 1.0,
        "python_version": "3.11.0",
        "packages": {"pandas": "2.0.0"},
    }))

    # Predictions — create from actual split
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df["Survived"]
    )
    val_df = val_df.reset_index(drop=True)
    y_true = val_df["Survived"].values
    y_prob_1 = rng.beta(2, 5, size=len(val_df))
    y_prob_1[y_true == 1] = rng.beta(5, 2, size=(y_true == 1).sum())
    y_pred = (y_prob_1 > 0.5).astype(int)

    pred_df = pd.DataFrame({
        "index": range(len(val_df)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob_0": 1 - y_prob_1,
        "y_prob_1": y_prob_1,
    })
    pred_df.to_csv(outputs_dir / "predictions.csv", index=False)

    return project_dir, iter_dir


class TestLoadIterationInputs:
    def test_loads_all(self, tmp_path):
        project_dir, iter_dir = _create_project_fixtures(tmp_path)
        inputs = load_iteration_inputs(iter_dir, project_dir)

        assert inputs["config"]["iteration"] == 1
        assert inputs["metrics"]["primary"]["name"] == "val_auc_roc"
        assert inputs["predictions_csv"].exists()
        assert "columns" in inputs["profile"]


class TestPriorRunComparison:
    def test_iteration_1_returns_none(self, tmp_path):
        result = compute_prior_run_comparison(1, tmp_path, {})
        assert result is None

    def test_with_prior_report(self, tmp_path):
        prev_dir = tmp_path / "iteration-1" / "reports"
        prev_dir.mkdir(parents=True)
        (prev_dir / "model-report.json").write_text(json.dumps({
            "headline_metrics": {
                "primary": {"name": "val_auc_roc", "value": 0.80},
                "secondary": {"val_accuracy": 0.75},
            },
        }))

        current = {
            "primary": {"name": "val_auc_roc", "value": 0.85},
            "secondary": {"val_accuracy": 0.79},
        }
        result = compute_prior_run_comparison(2, tmp_path, current)
        assert result is not None
        assert result["previous_iteration"] == 1
        assert len(result["deltas"]) == 2
        assert result["deltas"][0]["improved"] is True


class TestReviewerSummary:
    def test_neutral_on_iteration_1(self, tmp_path):
        headline = {"primary": {"name": "auc", "value": 0.85}}
        overfitting = {"severity": "low", "train_val_gap": {"gap_pct": 3.0}}
        leakage = {"suspiciously_high_metric": False, "feature_importance_anomalies": []}
        result = compute_reviewer_summary(
            headline, overfitting, leakage, [], None, 1, tmp_path
        )
        assert result["headline_verdict"] == "neutral"
        assert result["metric_summary"]["delta_vs_previous"] is None

    def test_improved_verdict(self, tmp_path):
        headline = {"primary": {"name": "auc", "value": 0.90}}
        overfitting = {"severity": "low", "train_val_gap": {"gap_pct": 2.0}}
        leakage = {"suspiciously_high_metric": False, "feature_importance_anomalies": []}
        prior = {"deltas": [{"metric": "auc", "delta": 0.05}]}
        result = compute_reviewer_summary(
            headline, overfitting, leakage, [], prior, 2, tmp_path
        )
        assert result["headline_verdict"] == "improved"

    def test_suspicious_verdict_overrides(self, tmp_path):
        headline = {"primary": {"name": "auc", "value": 0.999}}
        overfitting = {"severity": "low", "train_val_gap": {"gap_pct": 0.5}}
        leakage = {"suspiciously_high_metric": True, "feature_importance_anomalies": []}
        prior = {"deltas": [{"metric": "auc", "delta": 0.1}]}
        result = compute_reviewer_summary(
            headline, overfitting, leakage, [], prior, 2, tmp_path
        )
        assert result["headline_verdict"] == "suspicious"


class TestBuildModelReport:
    def test_full_report(self, tmp_path):
        project_dir, iter_dir = _create_project_fixtures(tmp_path)
        report = build_model_report(iter_dir, project_dir)

        # Structure
        assert report["schema_version"] == "1.1.0"
        assert report["iteration"] == 1
        assert report["task_type"] == "binary_classification"

        # Headline metrics
        assert report["headline_metrics"]["primary"]["name"] == "val_auc_roc"

        # Overfitting
        assert report["overfitting_check"]["severity"] == "low"

        # Leakage
        assert report["leakage_indicators"]["suspiciously_high_metric"] is False

        # Calibration (binary classification)
        assert report["calibration"] is not None
        assert "brier_score" in report["calibration"]

        # Error analysis
        assert "confusion_matrix" in report["error_analysis"]

        # Feature importance
        assert report["feature_importance"]["features"][0]["rank"] == 1

        # Prior run (iteration 1 = None)
        assert report["prior_run_comparison"] is None

        # Reviewer summary
        assert report["reviewer_summary"]["headline_verdict"] == "neutral"
        assert report["reviewer_summary"]["risk_flags"] == []

        # Plots
        assert report["plots"]["confusion_matrix"] is not None

        # New enhancements
        assert report["threshold_analysis"] is not None
        assert "optimal_threshold" in report["threshold_analysis"]
        assert "roc_curve" in report["threshold_analysis"]
        assert report["threshold_analysis"]["optimal_threshold"]["threshold"] != 0.5

        assert report["separation_quality"] is not None
        assert report["separation_quality"]["quality"] in ("strong", "moderate", "weak")
        assert report["separation_quality"]["ks_statistic"] > 0

        assert report["bootstrap_ci"] is not None
        assert report["bootstrap_ci"]["n_samples"] > 0
        assert "interpretation" in report["bootstrap_ci"]

        assert isinstance(report["hardest_samples"], list)
        assert len(report["hardest_samples"]) > 0
        assert "loss" in report["hardest_samples"][0]

        # New plots
        assert report["plots"]["roc_curve"] is not None
        assert report["plots"]["precision_recall_curve"] is not None
        assert len(report["plots"]["residual_vs_feature"]) > 0

        # File written
        report_path = iter_dir / "reports" / "model-report.json"
        assert report_path.exists()
        with open(report_path) as f:
            written = json.load(f)
        assert written["iteration"] == 1
