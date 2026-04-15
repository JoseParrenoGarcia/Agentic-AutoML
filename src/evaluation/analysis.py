"""
M6.1 — Analysis functions: calibration, segment analysis, error analysis,
and feature importance repackaging.

All functions are pure deterministic Python — no LLM calls.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Calibration (classification only)
# ---------------------------------------------------------------------------

def compute_calibration(predictions_csv: Path, task_type: str) -> dict | None:
    """
    Compute calibration metrics for classification tasks.

    Returns None for regression. For classification, returns dict with
    brier_score and reliability_curve data.
    """
    if task_type == "regression":
        return None

    df = pd.read_csv(predictions_csv)
    y_true = df["y_true"].values
    y_prob = df["y_prob_1"].values

    brier = float(brier_score_loss(y_true, y_prob))

    n_bins = min(10, max(3, len(df) // 20))
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_counts.append(int(mask.sum()))

    return {
        "brier_score": round(brier, 6),
        "reliability_curve": {
            "bin_edges": [round(float(x), 4) for x in bin_edges],
            "mean_predicted": [round(float(x), 4) for x in mean_predicted],
            "fraction_positive": [round(float(x), 4) for x in fraction_pos],
            "bin_counts": bin_counts,
        },
    }


# ---------------------------------------------------------------------------
# Segment analysis
# ---------------------------------------------------------------------------

def select_segment_columns(
    profile_json: dict,
    feature_importance_json: dict,
) -> list[dict]:
    """
    Choose columns suitable for segment analysis.

    Selection criteria:
    - Categorical/ordinal columns with unique_count <= 10
    - Top-2 numeric columns by feature importance (binned into quartiles)
    """
    columns = profile_json.get("columns", [])
    importance_rank = {
        f["name"]: i for i, f in enumerate(feature_importance_json.get("features", []))
    }

    categorical_segments = []
    numeric_candidates = []

    for col in columns:
        name = col["name"]
        semantic = col.get("inferred_semantic_type", "")
        cardinality = col.get("cardinality", {}).get("unique_count", 999)

        if semantic in ("identifier", "binary_target", "regression_target"):
            continue

        if semantic in ("categorical", "ordinal", "binary_flag") and cardinality <= 10:
            categorical_segments.append({
                "column": name,
                "type": "categorical",
                "unique_count": cardinality,
            })
        elif semantic in ("numeric", "continuous") and name in importance_rank:
            numeric_candidates.append({
                "column": name,
                "type": "numeric_binned",
                "importance_rank": importance_rank[name],
            })

    numeric_candidates.sort(key=lambda x: x["importance_rank"])
    numeric_segments = [
        {"column": c["column"], "type": "numeric_binned"}
        for c in numeric_candidates[:2]
    ]

    return categorical_segments + numeric_segments


def compute_segment_analysis(
    predictions_csv: Path,
    config_yaml: Path,
    profile_json: dict,
    feature_importance_json: dict,
    task_type: str,
) -> dict:
    """
    Compute per-segment performance metrics.

    Re-loads raw data and re-splits using config.yaml (same seed = same split)
    to recover validation-set feature columns. Joins with predictions on index.
    """
    import yaml

    with open(config_yaml) as f:
        config = yaml.safe_load(f)

    segment_cols = select_segment_columns(profile_json, feature_importance_json)
    if not segment_cols:
        return {"segments": []}

    # Re-load and re-split to get val features
    iteration_dir = config_yaml.parent
    train_path = iteration_dir / config["data"]["train"]
    raw_df = pd.read_csv(train_path)
    target = config["target_column"]
    method = config["split"]["method"]
    val_ratio = config["split"]["val_ratio"]
    seed = config["random_seed"]

    if method == "stratified":
        _, val_df = train_test_split(
            raw_df, test_size=val_ratio, random_state=seed, stratify=raw_df[target]
        )
    elif method == "random":
        _, val_df = train_test_split(
            raw_df, test_size=val_ratio, random_state=seed
        )
    elif method == "temporal":
        time_col = config["split"].get("time_column")
        raw_df = raw_df.sort_values(time_col).reset_index(drop=True)
        cutoff = config["split"].get("cutoff")
        if cutoff:
            val_df = raw_df[~(raw_df[time_col] < cutoff)]
        else:
            n = int(len(raw_df) * (1 - val_ratio))
            val_df = raw_df.iloc[n:]
    else:
        return {"segments": []}

    val_df = val_df.reset_index(drop=True)

    # Load predictions and join
    pred_df = pd.read_csv(predictions_csv)
    if len(pred_df) != len(val_df):
        return {"segments": [], "_warning": "Prediction/val size mismatch"}

    # Compute per-segment metrics
    if task_type == "binary_classification":
        from sklearn.metrics import roc_auc_score
        metric_fn = lambda yt, yp_prob: float(roc_auc_score(yt, yp_prob))
    else:
        from sklearn.metrics import mean_squared_error
        metric_fn = lambda yt, yp: float(mean_squared_error(yt, yp))

    segments = []
    for seg in segment_cols:
        col_name = seg["column"]
        if col_name not in val_df.columns:
            continue

        col_values = val_df[col_name].copy()
        if seg["type"] == "numeric_binned":
            col_values = pd.qcut(col_values, q=4, duplicates="drop")

        slices = []
        for value, group_idx in col_values.groupby(col_values).groups.items():
            if len(group_idx) < 5:
                continue
            subset_pred = pred_df.iloc[group_idx]
            y_true = subset_pred["y_true"].values
            y_pred = subset_pred["y_pred"].values

            acc = float(np.mean(y_true == y_pred)) if task_type != "regression" else None

            try:
                if task_type == "binary_classification" and "y_prob_1" in subset_pred.columns:
                    primary_metric = metric_fn(y_true, subset_pred["y_prob_1"].values)
                else:
                    primary_metric = metric_fn(y_true, y_pred)
            except ValueError:
                primary_metric = None

            slice_entry = {
                "value": str(value),
                "n": int(len(group_idx)),
                "primary_metric": round(primary_metric, 4) if primary_metric is not None else None,
            }
            if acc is not None:
                slice_entry["accuracy"] = round(acc, 4)

            slices.append(slice_entry)

        segments.append({
            "column": col_name,
            "type": seg["type"],
            "slices": slices,
        })

    return {"segments": segments}


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def compute_error_analysis(predictions_csv: Path, task_type: str) -> dict:
    """
    Compute error analysis: confusion matrix (classification),
    residual stats (regression), error rate by confidence bin.
    """
    df = pd.read_csv(predictions_csv)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    result: dict = {"task_type": task_type}

    if task_type in ("binary_classification", "multiclass"):
        cm = confusion_matrix(y_true, y_pred)
        if task_type == "binary_classification" and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            result["confusion_matrix"] = {
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
            }
        else:
            result["confusion_matrix"] = cm.tolist()

        # Misclassification patterns
        result["misclassification_patterns"] = _classify_errors(df, task_type)

        # Error rate by confidence bin
        if "y_prob_1" in df.columns:
            result["error_rate_by_confidence"] = _error_by_confidence(df)

    else:  # regression
        residuals = y_true - y_pred
        result["residual_stats"] = {
            "mean": round(float(np.mean(residuals)), 6),
            "std": round(float(np.std(residuals)), 6),
            "median": round(float(np.median(residuals)), 6),
            "min": round(float(np.min(residuals)), 6),
            "max": round(float(np.max(residuals)), 6),
        }

    return result


def _classify_errors(df: pd.DataFrame, task_type: str) -> list[dict]:
    """Identify misclassification patterns for classification."""
    patterns = []
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    incorrect = df[y_true != y_pred]
    total = len(df)

    if total == 0:
        return patterns

    error_rate = len(incorrect) / total
    patterns.append({
        "pattern": "overall_error_rate",
        "detail": f"{error_rate:.2%} ({len(incorrect)}/{total})",
    })

    if task_type == "binary_classification" and "y_prob_1" in df.columns:
        # High-confidence errors
        high_conf = df[df["y_prob_1"].apply(lambda p: max(p, 1 - p) > 0.8)]
        if len(high_conf) > 0:
            hc_errors = high_conf[high_conf["y_true"] != high_conf["y_pred"]]
            patterns.append({
                "pattern": "high_confidence_errors",
                "detail": (
                    f"{len(hc_errors)} errors among {len(high_conf)} "
                    f"high-confidence predictions (>80%)"
                ),
            })

    return patterns


def _error_by_confidence(df: pd.DataFrame) -> list[dict]:
    """Compute error rate by predicted probability bin."""
    df = df.copy()
    df["confidence"] = df["y_prob_1"].apply(lambda p: max(p, 1 - p))
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)

    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    result = []
    for lo, hi in bins:
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi)
        if hi == 1.0:
            mask = (df["confidence"] >= lo) & (df["confidence"] <= hi)
        subset = df[mask]
        if len(subset) == 0:
            continue
        error_rate = float(1 - subset["correct"].mean())
        result.append({
            "confidence_bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(len(subset)),
            "error_rate": round(error_rate, 4),
        })

    return result


# ---------------------------------------------------------------------------
# Decision threshold analysis (classification only)
# ---------------------------------------------------------------------------

def compute_threshold_analysis(predictions_csv: Path, task_type: str) -> dict | None:
    """
    Compute ROC curve, precision-recall curve, and optimal threshold.

    Returns None for regression.
    """
    if task_type == "regression":
        return None

    df = pd.read_csv(predictions_csv)
    y_true = df["y_true"].values

    if "y_prob_1" not in df.columns:
        return None

    y_prob = df["y_prob_1"].values

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_true, y_prob))

    # Precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    # Optimal threshold — maximize F1
    best_f1 = 0.0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_at_t = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, preds_at_t, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    # Metrics at optimal threshold
    preds_optimal = (y_prob >= best_threshold).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    metrics_at_optimal = {
        "threshold": round(best_threshold, 3),
        "f1": round(best_f1, 4),
        "accuracy": round(float(accuracy_score(y_true, preds_optimal)), 4),
        "precision": round(float(precision_score(y_true, preds_optimal, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds_optimal, zero_division=0)), 4),
    }

    # Metrics at default 0.5
    preds_default = (y_prob >= 0.5).astype(int)
    metrics_at_default = {
        "threshold": 0.5,
        "f1": round(float(f1_score(y_true, preds_default, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, preds_default)), 4),
        "precision": round(float(precision_score(y_true, preds_default, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds_default, zero_division=0)), 4),
    }

    # Downsample curves for JSON (keep at most 50 points)
    def _downsample(arr, n=50):
        if len(arr) <= n:
            return [round(float(x), 4) for x in arr]
        idx = np.linspace(0, len(arr) - 1, n, dtype=int)
        return [round(float(arr[i]), 4) for i in idx]

    return {
        "roc_curve": {
            "fpr": _downsample(fpr),
            "tpr": _downsample(tpr),
            "auc": round(auc, 4),
        },
        "precision_recall_curve": {
            "precision": _downsample(precision),
            "recall": _downsample(recall),
        },
        "optimal_threshold": metrics_at_optimal,
        "default_threshold": metrics_at_default,
        "threshold_delta": {
            "f1_gain": round(best_f1 - metrics_at_default["f1"], 4),
            "description": (
                f"Optimal threshold {best_threshold:.3f} vs default 0.5: "
                f"F1 {'improves' if best_f1 > metrics_at_default['f1'] else 'unchanged'} "
                f"by {abs(best_f1 - metrics_at_default['f1']):.4f}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Prediction separation quality
# ---------------------------------------------------------------------------

def compute_separation_quality(predictions_csv: Path, task_type: str) -> dict | None:
    """
    Quantify how well predicted probabilities separate classes.

    Metrics:
    - KS statistic: max separation between class CDFs
    - Discrimination slope: mean(P(y=1|y=1)) - mean(P(y=1|y=0))
    - Histogram overlap coefficient
    """
    if task_type == "regression":
        return None

    df = pd.read_csv(predictions_csv)
    if "y_prob_1" not in df.columns:
        return None

    y_true = df["y_true"].values
    y_prob = df["y_prob_1"].values

    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return None

    # KS statistic
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(pos_probs, neg_probs)

    # Discrimination slope
    disc_slope = float(np.mean(pos_probs) - np.mean(neg_probs))

    # Histogram overlap — approximate via binned overlap coefficient
    bins = np.linspace(0, 1, 21)
    hist_pos, _ = np.histogram(pos_probs, bins=bins, density=True)
    hist_neg, _ = np.histogram(neg_probs, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = float(np.sum(np.minimum(hist_pos, hist_neg)) * bin_width)

    # Qualitative assessment
    if ks_stat > 0.5 and disc_slope > 0.3:
        quality = "strong"
    elif ks_stat > 0.3 and disc_slope > 0.15:
        quality = "moderate"
    else:
        quality = "weak"

    return {
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 6),
        "discrimination_slope": round(disc_slope, 4),
        "mean_prob_positive_class": round(float(np.mean(pos_probs)), 4),
        "mean_prob_negative_class": round(float(np.mean(neg_probs)), 4),
        "histogram_overlap": round(overlap, 4),
        "quality": quality,
        "verdict": (
            f"Separation quality: {quality}. "
            f"KS={ks_stat:.3f}, discrimination slope={disc_slope:.3f}, "
            f"histogram overlap={overlap:.2%}."
        ),
    }


# ---------------------------------------------------------------------------
# Hardest samples
# ---------------------------------------------------------------------------

def identify_hardest_samples(
    predictions_csv: Path,
    task_type: str,
    top_n: int = 10,
) -> list[dict]:
    """
    Identify the top-N highest-loss samples in the validation set.

    For classification: highest cross-entropy loss (most confident wrong predictions).
    For regression: largest absolute residuals.
    """
    df = pd.read_csv(predictions_csv)

    if task_type in ("binary_classification", "multiclass") and "y_prob_1" in df.columns:
        y_true = df["y_true"].values
        y_prob = df["y_prob_1"].values

        # Cross-entropy loss per sample
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        loss = -(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
        df["loss"] = loss
    elif task_type == "regression":
        df["loss"] = np.abs(df["y_true"] - df["y_pred"])
    else:
        return []

    hardest = df.nlargest(top_n, "loss")

    samples = []
    for _, row in hardest.iterrows():
        entry = {
            "index": int(row["index"]),
            "y_true": float(row["y_true"]),
            "y_pred": float(row["y_pred"]),
            "loss": round(float(row["loss"]), 4),
        }
        if "y_prob_1" in df.columns:
            entry["y_prob_1"] = round(float(row["y_prob_1"]), 4)
        samples.append(entry)

    return samples


# ---------------------------------------------------------------------------
# Feature importance repackaging
# ---------------------------------------------------------------------------

def repackage_feature_importance(feature_importance_json: dict) -> dict:
    """Add rank field to features and repackage for model-report.json."""
    features = []
    for i, f in enumerate(feature_importance_json.get("features", [])):
        features.append({
            "name": f["name"],
            "importance": f["importance"],
            "rank": i + 1,
        })

    return {
        "method": feature_importance_json.get("method", "unknown"),
        "model": feature_importance_json.get("model", "unknown"),
        "features": features,
    }
