"""
M6.2 — Evaluation plots: confusion matrix, actual-vs-predicted,
calibration curve, error distribution, per-feature diagnostics.

Follows src/analysis/plots.py patterns: matplotlib Agg backend,
_save helper, STYLE/DPI constants.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE = (8, 5)
DPI = 120


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    predictions_csv: Path,
    out_dir: Path,
    task_type: str,
) -> Path | None:
    """Plot confusion matrix for classification tasks. Returns None for regression."""
    if task_type == "regression":
        return None

    df = pd.read_csv(predictions_csv)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(df["y_true"].unique())

    fig, ax = plt.subplots(figsize=(6, 5))
    with plt.style.context(STYLE):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

    return _save(fig, out_dir / "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Actual vs predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    predictions_csv: Path,
    out_dir: Path,
    task_type: str,
) -> Path:
    """
    Classification: histogram of predicted probabilities by true class.
    Regression: scatter of actual vs predicted.
    """
    df = pd.read_csv(predictions_csv)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        if task_type in ("binary_classification", "multiclass"):
            if "y_prob_1" in df.columns:
                for cls in sorted(df["y_true"].unique()):
                    subset = df[df["y_true"] == cls]["y_prob_1"]
                    ax.hist(
                        subset, bins=20, alpha=0.6,
                        label=f"Class {cls}", edgecolor="white",
                    )
                ax.set_xlabel("Predicted Probability (class 1)")
                ax.set_ylabel("Count")
                ax.set_title("Predicted Probability Distribution by True Class")
                ax.legend()
            else:
                ax.bar(["Correct", "Incorrect"], [
                    (df["y_true"] == df["y_pred"]).sum(),
                    (df["y_true"] != df["y_pred"]).sum(),
                ], color=["#4C72B0", "#DD8452"], edgecolor="white")
                ax.set_title("Prediction Correctness")
        else:
            ax.scatter(df["y_true"], df["y_pred"], alpha=0.4, s=20, color="#4C72B0")
            lims = [
                min(df["y_true"].min(), df["y_pred"].min()),
                max(df["y_true"].max(), df["y_pred"].max()),
            ]
            ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            ax.legend()

    return _save(fig, out_dir / "actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    calibration_data: dict | None,
    out_dir: Path,
) -> Path | None:
    """Plot reliability diagram from pre-computed calibration data."""
    if calibration_data is None:
        return None

    curve = calibration_data["reliability_curve"]
    mean_pred = curve["mean_predicted"]
    frac_pos = curve["fraction_positive"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        ax.plot(mean_pred, frac_pos, "o-", color="#4C72B0", label="Model")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Curve (Brier: {calibration_data['brier_score']:.4f})")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    return _save(fig, out_dir / "calibration_curve.png")


# ---------------------------------------------------------------------------
# Error distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(
    predictions_csv: Path,
    out_dir: Path,
    task_type: str,
) -> Path:
    """
    Classification: error rate by confidence bin.
    Regression: residual histogram.
    """
    df = pd.read_csv(predictions_csv)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        if task_type in ("binary_classification", "multiclass") and "y_prob_1" in df.columns:
            df["confidence"] = df["y_prob_1"].apply(lambda p: max(p, 1 - p))
            df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
            bins = pd.cut(df["confidence"], bins=5)
            error_by_bin = 1 - df.groupby(bins, observed=True)["correct"].mean()
            error_by_bin.plot.bar(ax=ax, color="#DD8452", edgecolor="white")
            ax.set_xlabel("Confidence Bin")
            ax.set_ylabel("Error Rate")
            ax.set_title("Error Rate by Prediction Confidence")
            plt.xticks(rotation=45, ha="right")
        else:
            residuals = df["y_true"] - df["y_pred"]
            ax.hist(residuals, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Residual (Actual - Predicted)")
            ax.set_ylabel("Count")
            ax.set_title("Residual Distribution")

    return _save(fig, out_dir / "error_distribution.png")


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    threshold_analysis: dict | None,
    out_dir: Path,
) -> Path | None:
    """Plot ROC curve from pre-computed threshold analysis data."""
    if threshold_analysis is None:
        return None

    roc = threshold_analysis["roc_curve"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC=0.5)")
        ax.plot(roc["fpr"], roc["tpr"], color="#4C72B0", linewidth=2,
                label=f"Model (AUC={roc['auc']:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    return _save(fig, out_dir / "roc_curve.png")


# ---------------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    threshold_analysis: dict | None,
    out_dir: Path,
) -> Path | None:
    """Plot precision-recall curve from pre-computed threshold analysis data."""
    if threshold_analysis is None:
        return None

    pr = threshold_analysis["precision_recall_curve"]
    opt = threshold_analysis["optimal_threshold"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        ax.plot(pr["recall"], pr["precision"], color="#4C72B0", linewidth=2,
                label="Model")
        ax.axhline(y=opt["precision"], color="#DD8452", linestyle=":",
                   alpha=0.7, label=f"Optimal precision={opt['precision']:.3f}")
        ax.axvline(x=opt["recall"], color="#55A868", linestyle=":",
                   alpha=0.7, label=f"Optimal recall={opt['recall']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve (optimal threshold={opt['threshold']:.3f})")
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    return _save(fig, out_dir / "precision_recall_curve.png")


# ---------------------------------------------------------------------------
# Residual vs feature plots
# ---------------------------------------------------------------------------

def plot_residual_vs_feature(
    val_features_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_importance: dict,
    out_dir: Path,
    task_type: str,
    top_n: int = 5,
) -> list[Path]:
    """
    For top-N features: scatter/strip plot of residual or error indicator
    against feature values. Shows whether errors are systematic across
    feature ranges.

    Classification: strips y_prob_1 colored by correct/incorrect.
    Regression: scatter of residual vs feature value.
    """
    features = feature_importance.get("features", [])[:top_n]
    y_true = predictions_df["y_true"].values
    y_pred = predictions_df["y_pred"].values
    n = min(len(val_features_df), len(predictions_df))

    paths = []
    for feat in features:
        col_name = feat["name"]
        if col_name not in val_features_df.columns:
            continue

        col_data = val_features_df[col_name].values[:n]
        if not pd.api.types.is_numeric_dtype(val_features_df[col_name]):
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        with plt.style.context(STYLE):
            if task_type == "regression":
                residuals = y_true[:n] - y_pred[:n]
                ax.scatter(col_data, residuals, alpha=0.4, s=20, color="#4C72B0")
                ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
                ax.set_ylabel("Residual (Actual - Predicted)")
                ax.set_title(f"Residual vs {col_name}")
            else:
                correct = (y_true[:n] == y_pred[:n])
                if "y_prob_1" in predictions_df.columns:
                    y_prob = predictions_df["y_prob_1"].values[:n]
                    ax.scatter(col_data[correct], y_prob[correct],
                              alpha=0.4, s=20, color="#4C72B0", label="Correct")
                    ax.scatter(col_data[~correct], y_prob[~correct],
                              alpha=0.6, s=30, color="#DD8452", marker="x", label="Incorrect")
                    ax.set_ylabel("Predicted Probability (class 1)")
                else:
                    ax.scatter(col_data[correct], y_pred[:n][correct],
                              alpha=0.4, s=20, color="#4C72B0", label="Correct")
                    ax.scatter(col_data[~correct], y_pred[:n][~correct],
                              alpha=0.6, s=30, color="#DD8452", marker="x", label="Incorrect")
                    ax.set_ylabel("Predicted")
                ax.legend()
                ax.set_title(f"Predictions vs {col_name}")

            ax.set_xlabel(col_name)

        paths.append(_save(fig, out_dir / f"residual_vs_{col_name}.png"))

    return paths


# ---------------------------------------------------------------------------
# Per-feature diagnostics
# ---------------------------------------------------------------------------

def plot_feature_diagnostics(
    val_features_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_importance: dict,
    out_dir: Path,
    task_type: str,
    top_n: int = 5,
) -> list[Path]:
    """
    For top-N important features: box plot of feature values,
    split by correct vs incorrect predictions.
    """
    features = feature_importance.get("features", [])[:top_n]
    correct = (predictions_df["y_true"] == predictions_df["y_pred"]).values

    paths = []
    for feat in features:
        col_name = feat["name"]
        if col_name not in val_features_df.columns:
            continue

        col_data = val_features_df[col_name].copy()
        if not pd.api.types.is_numeric_dtype(col_data):
            continue

        plot_df = pd.DataFrame({
            col_name: col_data.values[:len(correct)],
            "Prediction": np.where(correct[:len(col_data)], "Correct", "Incorrect"),
        })

        fig, ax = plt.subplots(figsize=FIGSIZE)
        with plt.style.context(STYLE):
            sns.boxplot(
                data=plot_df, x="Prediction", y=col_name,
                hue="Prediction",
                palette={"Correct": "#4C72B0", "Incorrect": "#DD8452"},
                legend=False,
                ax=ax,
            )
            ax.set_title(f"Feature: {col_name} — Correct vs Incorrect Predictions")

        paths.append(_save(fig, out_dir / f"feature_diagnostic_{col_name}.png"))

    return paths


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_plots(
    predictions_csv: Path,
    calibration_data: dict | None,
    feature_importance: dict,
    val_features_df: pd.DataFrame | None,
    out_dir: Path,
    task_type: str,
    threshold_analysis: dict | None = None,
) -> dict:
    """
    Generate all evaluation plots. Returns the plots block for model-report.json.

    All paths are relative to the iteration root (reports/plots/).
    """
    plot_dir = out_dir / "plots"

    cm_path = plot_confusion_matrix(predictions_csv, plot_dir, task_type)
    avp_path = plot_actual_vs_predicted(predictions_csv, plot_dir, task_type)
    cal_path = plot_calibration_curve(calibration_data, plot_dir)
    err_path = plot_error_distribution(predictions_csv, plot_dir, task_type)
    roc_path = plot_roc_curve(threshold_analysis, plot_dir)
    pr_path = plot_precision_recall_curve(threshold_analysis, plot_dir)

    feat_paths = []
    residual_paths = []
    if val_features_df is not None:
        predictions_df = pd.read_csv(predictions_csv)
        feat_paths = plot_feature_diagnostics(
            val_features_df, predictions_df, feature_importance,
            plot_dir, task_type,
        )
        residual_paths = plot_residual_vs_feature(
            val_features_df, predictions_df, feature_importance,
            plot_dir, task_type,
        )

    def _rel(p: Path | None) -> str | None:
        return str(p.relative_to(out_dir)) if p else None

    return {
        "confusion_matrix": _rel(cm_path),
        "actual_vs_predicted": _rel(avp_path),
        "calibration_curve": _rel(cal_path),
        "error_distribution": _rel(err_path),
        "roc_curve": _rel(roc_path),
        "precision_recall_curve": _rel(pr_path),
        "feature_diagnostics": [_rel(p) for p in feat_paths],
        "residual_vs_feature": [_rel(p) for p in residual_paths],
    }
