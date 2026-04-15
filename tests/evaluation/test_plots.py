import numpy as np
import pandas as pd

from src.evaluation.analysis import compute_calibration
from src.evaluation.plots import (
    generate_all_plots,
    plot_actual_vs_predicted,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_feature_diagnostics,
)


def _write_binary_predictions(path, n=100, seed=42):
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


def _make_val_features(n=100, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "Sex": rng.choice([0, 1], n),
        "Age": rng.normal(30, 10, n),
        "Fare": rng.exponential(30, n),
        "Pclass": rng.choice([1, 2, 3], n),
    })


def _make_fi():
    return {
        "features": [
            {"name": "Sex", "importance": 2.5},
            {"name": "Age", "importance": 0.4},
            {"name": "Fare", "importance": 0.3},
            {"name": "Pclass", "importance": 0.2},
        ],
    }


class TestConfusionMatrix:
    def test_creates_png(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        _write_binary_predictions(pred_path)
        result = plot_confusion_matrix(pred_path, tmp_path, "binary_classification")
        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 0

    def test_regression_returns_none(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        _write_binary_predictions(pred_path)
        assert plot_confusion_matrix(pred_path, tmp_path, "regression") is None


class TestActualVsPredicted:
    def test_binary(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        _write_binary_predictions(pred_path)
        result = plot_actual_vs_predicted(pred_path, tmp_path, "binary_classification")
        assert result.exists()

    def test_regression(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        rng = np.random.RandomState(42)
        pd.DataFrame({
            "index": range(50),
            "y_true": rng.normal(10, 3, 50),
            "y_pred": rng.normal(10, 3, 50),
        }).to_csv(pred_path, index=False)
        result = plot_actual_vs_predicted(pred_path, tmp_path, "regression")
        assert result.exists()


class TestCalibrationCurve:
    def test_creates_png(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        _write_binary_predictions(pred_path, n=200)
        cal_data = compute_calibration(pred_path, "binary_classification")
        result = plot_calibration_curve(cal_data, tmp_path)
        assert result is not None
        assert result.exists()

    def test_none_data(self, tmp_path):
        assert plot_calibration_curve(None, tmp_path) is None


class TestErrorDistribution:
    def test_binary(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        _write_binary_predictions(pred_path)
        result = plot_error_distribution(pred_path, tmp_path, "binary_classification")
        assert result.exists()


class TestFeatureDiagnostics:
    def test_creates_plots(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        pred_df = _write_binary_predictions(pred_path)
        val_df = _make_val_features(n=len(pred_df))
        fi = _make_fi()

        paths = plot_feature_diagnostics(
            val_df, pred_df, fi, tmp_path, "binary_classification", top_n=3
        )
        # Should create plots for numeric features (Sex is 0/1 numeric, Age, Fare)
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0


class TestGenerateAllPlots:
    def test_full_pipeline(self, tmp_path):
        pred_path = tmp_path / "pred.csv"
        _write_binary_predictions(pred_path, n=200)
        cal_data = compute_calibration(pred_path, "binary_classification")
        val_df = _make_val_features(n=200)
        fi = _make_fi()

        report_dir = tmp_path / "reports"
        result = generate_all_plots(
            pred_path, cal_data, fi, val_df, report_dir, "binary_classification"
        )

        assert result["confusion_matrix"] is not None
        assert result["actual_vs_predicted"] is not None
        assert result["calibration_curve"] is not None
        assert result["error_distribution"] is not None
        assert len(result["feature_diagnostics"]) >= 2

        # All paths should be relative strings
        plot_dir = report_dir / "plots"
        assert plot_dir.exists()
        assert (report_dir / result["confusion_matrix"]).exists()
