import numpy as np
import pandas as pd
import yaml

from src.evaluation.analysis import (
    compute_calibration,
    compute_error_analysis,
    compute_segment_analysis,
    compute_separation_quality,
    compute_threshold_analysis,
    identify_hardest_samples,
    repackage_feature_importance,
    select_segment_columns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_binary_predictions(path, n=100, seed=42):
    """Write a synthetic binary predictions CSV."""
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
    return df


def _write_regression_predictions(path, n=100, seed=42):
    rng = np.random.RandomState(seed)
    y_true = rng.normal(10, 3, size=n)
    y_pred = y_true + rng.normal(0, 1, size=n)

    df = pd.DataFrame({
        "index": range(n),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    df.to_csv(path, index=False)
    return df


def _make_titanic_profile():
    """Minimal profile.json matching Titanic structure."""
    return {
        "columns": [
            {"name": "Survived", "inferred_semantic_type": "binary_target",
             "cardinality": {"unique_count": 2}},
            {"name": "Pclass", "inferred_semantic_type": "ordinal",
             "cardinality": {"unique_count": 3}},
            {"name": "Sex", "inferred_semantic_type": "categorical",
             "cardinality": {"unique_count": 2}},
            {"name": "Age", "inferred_semantic_type": "numeric",
             "cardinality": {"unique_count": 88}},
            {"name": "Fare", "inferred_semantic_type": "continuous",
             "cardinality": {"unique_count": 248}},
            {"name": "PassengerId", "inferred_semantic_type": "identifier",
             "cardinality": {"unique_count": 891}},
        ],
    }


def _make_feature_importance():
    return {
        "method": "coefficients",
        "features": [
            {"name": "Sex", "importance": 2.52},
            {"name": "Pclass", "importance": 0.60},
            {"name": "Fare", "importance": 0.40},
            {"name": "Age", "importance": 0.04},
        ],
        "sorted": True,
        "model": "LogisticRegression",
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_binary_classification(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_binary_predictions(pred_path, n=200)
        result = compute_calibration(pred_path, "binary_classification")

        assert result is not None
        assert 0.0 <= result["brier_score"] <= 1.0
        assert len(result["reliability_curve"]["mean_predicted"]) > 0
        assert len(result["reliability_curve"]["bin_counts"]) > 0

    def test_regression_returns_none(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_regression_predictions(pred_path)
        result = compute_calibration(pred_path, "regression")
        assert result is None


# ---------------------------------------------------------------------------
# Segment column selection
# ---------------------------------------------------------------------------

class TestSelectSegmentColumns:
    def test_titanic_columns(self):
        profile = _make_titanic_profile()
        fi = _make_feature_importance()
        cols = select_segment_columns(profile, fi)

        names = [c["column"] for c in cols]
        assert "Pclass" in names
        assert "Sex" in names
        assert "PassengerId" not in names
        assert "Survived" not in names

        # Top-2 numeric by importance: Fare and Age
        numeric_cols = [c for c in cols if c["type"] == "numeric_binned"]
        assert len(numeric_cols) <= 2

    def test_empty_profile(self):
        cols = select_segment_columns({"columns": []}, {"features": []})
        assert cols == []


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

class TestSegmentAnalysis:
    def test_basic_segment_analysis(self, tmp_path):
        # Create raw data
        rng = np.random.RandomState(42)
        n = 200
        raw_df = pd.DataFrame({
            "Sex": rng.choice(["male", "female"], n),
            "Pclass": rng.choice([1, 2, 3], n),
            "Age": rng.normal(30, 10, n),
            "Fare": rng.exponential(30, n),
            "Survived": rng.randint(0, 2, n),
        })
        raw_path = tmp_path / "data" / "raw" / "train.csv"
        raw_path.parent.mkdir(parents=True)
        raw_df.to_csv(raw_path, index=False)

        # config.yaml
        config = {
            "data": {"train": "../../data/raw/train.csv"},
            "target_column": "Survived",
            "task_type": "binary_classification",
            "random_seed": 42,
            "split": {"method": "stratified", "val_ratio": 0.2},
        }
        iter_dir = tmp_path / "iterations" / "iteration-1"
        iter_dir.mkdir(parents=True)
        config_path = iter_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Predictions — same split
        from sklearn.model_selection import train_test_split
        _, val_df = train_test_split(
            raw_df, test_size=0.2, random_state=42, stratify=raw_df["Survived"]
        )
        val_df = val_df.reset_index(drop=True)

        pred_df = pd.DataFrame({
            "index": range(len(val_df)),
            "y_true": val_df["Survived"].values,
            "y_pred": rng.randint(0, 2, len(val_df)),
            "y_prob_0": rng.uniform(0.3, 0.7, len(val_df)),
            "y_prob_1": rng.uniform(0.3, 0.7, len(val_df)),
        })
        pred_path = iter_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        profile = {
            "columns": [
                {"name": "Sex", "inferred_semantic_type": "categorical",
                 "cardinality": {"unique_count": 2}},
                {"name": "Pclass", "inferred_semantic_type": "ordinal",
                 "cardinality": {"unique_count": 3}},
                {"name": "Age", "inferred_semantic_type": "numeric",
                 "cardinality": {"unique_count": 50}},
                {"name": "Fare", "inferred_semantic_type": "continuous",
                 "cardinality": {"unique_count": 100}},
                {"name": "Survived", "inferred_semantic_type": "binary_target",
                 "cardinality": {"unique_count": 2}},
            ],
        }
        fi = _make_feature_importance()

        result = compute_segment_analysis(
            pred_path, config_path, profile, fi, "binary_classification"
        )

        assert "segments" in result
        assert len(result["segments"]) > 0
        for seg in result["segments"]:
            assert "column" in seg
            assert "slices" in seg
            assert len(seg["slices"]) > 0


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    def test_binary_classification(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_binary_predictions(pred_path)
        result = compute_error_analysis(pred_path, "binary_classification")

        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == 100
        assert "misclassification_patterns" in result
        assert "error_rate_by_confidence" in result

    def test_regression(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_regression_predictions(pred_path)
        result = compute_error_analysis(pred_path, "regression")

        assert "residual_stats" in result
        assert "mean" in result["residual_stats"]
        assert "confusion_matrix" not in result


# ---------------------------------------------------------------------------
# Feature importance repackaging
# ---------------------------------------------------------------------------

class TestRepackageFeatureImportance:
    def test_adds_rank(self):
        fi = _make_feature_importance()
        result = repackage_feature_importance(fi)

        assert result["method"] == "coefficients"
        assert result["model"] == "LogisticRegression"
        assert len(result["features"]) == 4
        assert result["features"][0]["rank"] == 1
        assert result["features"][0]["name"] == "Sex"
        assert result["features"][-1]["rank"] == 4

    def test_empty_features(self):
        result = repackage_feature_importance({"features": []})
        assert result["features"] == []
        assert result["method"] == "unknown"


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

class TestThresholdAnalysis:
    def test_binary_classification(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_binary_predictions(pred_path, n=200)
        result = compute_threshold_analysis(pred_path, "binary_classification")

        assert result is not None
        assert "roc_curve" in result
        assert result["roc_curve"]["auc"] > 0.5
        assert "precision_recall_curve" in result
        assert "optimal_threshold" in result
        assert 0 < result["optimal_threshold"]["threshold"] < 1
        assert result["optimal_threshold"]["f1"] > 0
        assert "default_threshold" in result
        assert result["default_threshold"]["threshold"] == 0.5
        assert "threshold_delta" in result

    def test_regression_returns_none(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_regression_predictions(pred_path)
        assert compute_threshold_analysis(pred_path, "regression") is None


# ---------------------------------------------------------------------------
# Separation quality
# ---------------------------------------------------------------------------

class TestSeparationQuality:
    def test_binary_classification(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_binary_predictions(pred_path, n=200)
        result = compute_separation_quality(pred_path, "binary_classification")

        assert result is not None
        assert result["ks_statistic"] > 0
        assert result["discrimination_slope"] != 0
        assert 0 <= result["histogram_overlap"] <= 1
        assert result["quality"] in ("strong", "moderate", "weak")

    def test_regression_returns_none(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_regression_predictions(pred_path)
        assert compute_separation_quality(pred_path, "regression") is None


# ---------------------------------------------------------------------------
# Hardest samples
# ---------------------------------------------------------------------------

class TestHardestSamples:
    def test_binary_classification(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_binary_predictions(pred_path, n=100)
        result = identify_hardest_samples(pred_path, "binary_classification", top_n=5)

        assert len(result) == 5
        assert all("loss" in s for s in result)
        assert all("y_prob_1" in s for s in result)
        # Sorted by loss descending
        losses = [s["loss"] for s in result]
        assert losses == sorted(losses, reverse=True)

    def test_regression(self, tmp_path):
        pred_path = tmp_path / "predictions.csv"
        _write_regression_predictions(pred_path, n=100)
        result = identify_hardest_samples(pred_path, "regression", top_n=5)

        assert len(result) == 5
        assert all("loss" in s for s in result)
        assert "y_prob_1" not in result[0]
