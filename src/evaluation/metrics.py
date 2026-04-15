"""
M6.1 — Core metric analysis: headline repackaging, overfitting checks,
leakage detection, risk flag classification, and plateau signal.

All functions are pure deterministic Python — no LLM calls.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Overfitting severity thresholds (gap_pct of primary metric)
# ---------------------------------------------------------------------------
_OVERFIT_LOW = 5.0       # < 5%
_OVERFIT_MEDIUM = 15.0   # 5–15%
# >= 15% → high

# Leakage thresholds
_SUSPICIOUS_METRIC_THRESHOLD = 0.99
_DOMINANT_FEATURE_RATIO = 0.80  # single feature > 80% of total importance

# Plateau detection
_STALE_DELTA_THRESHOLD = 0.005  # delta < 0.5% = stale iteration


def compute_headline_metrics(metrics_json: dict) -> dict:
    """Repackage raw metrics.json into the headline_metrics block."""
    return {
        "primary": metrics_json["primary"],
        "secondary": metrics_json.get("secondary", {}),
        "train": metrics_json.get("train", {}),
        "validation": metrics_json.get("validation", {}),
    }


def compute_overfitting_check(
    metrics_json: dict,
    learning_curves_json: dict,
    task_type: str,
) -> dict:
    """
    Analyse train/val gap for the primary metric.

    Returns dict with: train_val_gap, severity, learning_curve_trend, verdict.
    """
    primary_name = metrics_json["primary"]["name"]
    val_value = metrics_json["primary"]["value"]

    # Find matching train metric: strip "val_" prefix and try "train_" prefix
    base_name = primary_name.replace("val_", "")
    train_key = f"train_{base_name}"
    train_metrics = metrics_json.get("train", {})
    train_value = train_metrics.get(train_key)

    if train_value is None:
        # Try exact match in train dict
        for k, v in train_metrics.items():
            if base_name in k:
                train_value = v
                break

    if train_value is None:
        return {
            "train_val_gap": {
                "metric": primary_name,
                "train": None,
                "val": val_value,
                "gap": None,
                "gap_pct": None,
            },
            "severity": "unknown",
            "learning_curve_trend": _learning_curve_trend(learning_curves_json),
            "verdict": "Cannot compute overfitting — no matching train metric found.",
        }

    gap = abs(train_value - val_value)
    gap_pct = (gap / abs(train_value) * 100) if train_value != 0 else 0.0

    if gap_pct < _OVERFIT_LOW:
        severity = "low"
    elif gap_pct < _OVERFIT_MEDIUM:
        severity = "medium"
    else:
        severity = "high"

    trend = _learning_curve_trend(learning_curves_json)

    verdict_parts = [f"Train/val gap: {gap:.4f} ({gap_pct:.1f}%) — severity: {severity}."]
    if trend != "unavailable":
        verdict_parts.append(f"Learning curve trend: {trend}.")

    return {
        "train_val_gap": {
            "metric": primary_name,
            "train": train_value,
            "val": val_value,
            "gap": round(gap, 6),
            "gap_pct": round(gap_pct, 2),
        },
        "severity": severity,
        "learning_curve_trend": trend,
        "verdict": " ".join(verdict_parts),
    }


def _learning_curve_trend(learning_curves_json: dict) -> str:
    """Classify learning curve trend from learning_curves.json."""
    if "note" in learning_curves_json:
        return "unavailable"

    train = learning_curves_json.get("train")
    val = learning_curves_json.get("validation")
    if not train or not val or len(train) < 3:
        return "unavailable"

    # Check last third of training
    n = len(val)
    tail = max(1, n // 3)
    val_tail = val[-tail:]
    train_tail = train[-tail:]

    val_improving = val_tail[-1] > val_tail[0]
    gap_widening = (train_tail[-1] - val_tail[-1]) > (train_tail[0] - val_tail[0])

    if val_improving and not gap_widening:
        return "improving"
    elif not val_improving and gap_widening:
        return "diverging"
    else:
        return "plateau"


def compute_leakage_indicators(
    metrics_json: dict,
    feature_importance_json: dict,
    task_type: str,
) -> dict:
    """
    Detect potential data leakage from suspiciously high metrics
    or dominant feature importance.
    """
    primary_value = metrics_json["primary"]["value"]
    suspicious = primary_value > _SUSPICIOUS_METRIC_THRESHOLD

    features = feature_importance_json.get("features", [])
    total_importance = sum(abs(f["importance"]) for f in features)
    anomalies = []
    if total_importance > 0:
        for f in features:
            ratio = abs(f["importance"]) / total_importance
            if ratio > _DOMINANT_FEATURE_RATIO:
                anomalies.append(
                    f"Feature '{f['name']}' has {ratio:.0%} of total importance"
                )

    verdict_parts = []
    if suspicious:
        verdict_parts.append(
            f"Primary metric ({primary_value:.4f}) exceeds {_SUSPICIOUS_METRIC_THRESHOLD} — possible leakage."
        )
    if anomalies:
        verdict_parts.append(f"Feature anomalies: {'; '.join(anomalies)}.")
    if not verdict_parts:
        verdict_parts.append("No leakage indicators detected.")

    return {
        "suspiciously_high_metric": suspicious,
        "feature_importance_anomalies": anomalies,
        "verdict": " ".join(verdict_parts),
    }


def classify_risk_flags(
    overfitting: dict,
    leakage: dict,
    metrics_json: dict,
    task_type: str,
) -> list[dict]:
    """
    Aggregate risk flags from overfitting and leakage analysis.

    Each flag: {type, severity, evidence}.
    """
    flags = []

    # Overfitting flag
    if overfitting["severity"] in ("medium", "high"):
        gap_info = overfitting["train_val_gap"]
        flags.append({
            "type": "overfitting",
            "severity": overfitting["severity"],
            "evidence": (
                f"Train/val gap of {gap_info['gap_pct']}% on {gap_info['metric']}"
            ),
        })

    # Leakage flag
    if leakage["suspiciously_high_metric"]:
        flags.append({
            "type": "leakage",
            "severity": "high",
            "evidence": (
                f"Primary metric {metrics_json['primary']['value']:.4f} "
                f"exceeds {_SUSPICIOUS_METRIC_THRESHOLD}"
            ),
        })
    if leakage["feature_importance_anomalies"]:
        flags.append({
            "type": "leakage",
            "severity": "medium",
            "evidence": leakage["feature_importance_anomalies"][0],
        })

    # Underfitting check (heuristic: primary val metric < 0.5 for classification)
    if task_type in ("binary_classification", "multiclass"):
        val = metrics_json["primary"]["value"]
        if val < 0.5:
            flags.append({
                "type": "underfitting",
                "severity": "high",
                "evidence": f"Primary metric {val:.4f} below 0.5 — model may not be learning.",
            })

    return flags


def compute_bootstrap_ci(
    predictions_csv: Path,
    metrics_json: dict,
    task_type: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for primary and secondary metrics.

    Resamples the validation predictions n_bootstrap times and recomputes
    each metric to estimate the sampling distribution.
    """
    df = pd.read_csv(predictions_csv)
    rng = np.random.RandomState(seed)
    n = len(df)
    alpha = (1 - confidence) / 2

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    y_prob = df["y_prob_1"].values if "y_prob_1" in df.columns else None

    # Define metric functions
    metric_fns: dict = {}
    if task_type == "binary_classification" and y_prob is not None:
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        metric_fns["auc_roc"] = lambda yt, yp, ypr: float(roc_auc_score(yt, ypr))
        metric_fns["accuracy"] = lambda yt, yp, ypr: float(accuracy_score(yt, yp))
        metric_fns["f1"] = lambda yt, yp, ypr: float(f1_score(yt, yp, zero_division=0))
    elif task_type == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        metric_fns["mse"] = lambda yt, yp, ypr: float(mean_squared_error(yt, yp))
        metric_fns["mae"] = lambda yt, yp, ypr: float(mean_absolute_error(yt, yp))
    else:
        from sklearn.metrics import accuracy_score
        metric_fns["accuracy"] = lambda yt, yp, ypr: float(accuracy_score(yt, yp))

    # Bootstrap
    bootstrap_results: dict[str, list[float]] = {k: [] for k in metric_fns}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        ypr_b = y_prob[idx] if y_prob is not None else None

        # Skip degenerate samples (single class)
        if len(np.unique(yt_b)) < 2 and task_type != "regression":
            continue

        for name, fn in metric_fns.items():
            try:
                bootstrap_results[name].append(fn(yt_b, yp_b, ypr_b))
            except ValueError:
                pass

    # Compute CIs
    intervals = {}
    for name, samples in bootstrap_results.items():
        if len(samples) < 100:
            continue
        arr = np.array(samples)
        lo = float(np.percentile(arr, alpha * 100))
        hi = float(np.percentile(arr, (1 - alpha) * 100))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        intervals[name] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4),
            "confidence": confidence,
            "n_bootstrap": len(samples),
        }

    # Add context for the primary metric
    primary_name = metrics_json["primary"]["name"]
    primary_value = metrics_json["primary"]["value"]
    # Map primary metric name to bootstrap key
    primary_key = None
    for k in intervals:
        if k in primary_name or primary_name.replace("val_", "") in k:
            primary_key = k
            break

    return {
        "intervals": intervals,
        "primary_ci": intervals.get(primary_key) if primary_key else None,
        "primary_metric_name": primary_name,
        "primary_point_estimate": primary_value,
        "n_samples": n,
        "interpretation": _ci_interpretation(
            intervals.get(primary_key), primary_value, n
        ) if primary_key and primary_key in intervals else (
            f"Bootstrap CI not available for {primary_name}."
        ),
    }


def _ci_interpretation(ci: dict, point_est: float, n: int) -> str:
    """Generate human-readable CI interpretation."""
    width = ci["ci_upper"] - ci["ci_lower"]
    return (
        f"Primary metric {point_est:.4f} has 95% CI "
        f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
        f"(width={width:.4f}) based on {n} validation samples. "
        f"Improvements smaller than {width:.4f} may not be statistically meaningful."
    )


def compute_plateau_signal(
    iteration: int,
    prior_reports_dir: Path,
) -> dict:
    """
    Detect metric plateau by reading previous model-report.json files.

    Args:
        iteration: Current iteration number (1-indexed).
        prior_reports_dir: Path to the iterations/ directory.

    Returns:
        Dict with detected (bool), consecutive_stale_iterations (int).
    """
    if iteration <= 1:
        return {"detected": False, "consecutive_stale_iterations": 0}

    # Collect primary metric values from previous reports (newest first)
    values = []
    for i in range(iteration - 1, 0, -1):
        report_path = prior_reports_dir / f"iteration-{i}" / "reports" / "model-report.json"
        if not report_path.exists():
            break
        try:
            with open(report_path) as f:
                report = json.load(f)
            values.append(report["headline_metrics"]["primary"]["value"])
        except (json.JSONDecodeError, KeyError):
            break

    if not values:
        return {"detected": False, "consecutive_stale_iterations": 0}

    # Count consecutive stale iterations (from most recent backward)
    consecutive_stale = 0
    for i in range(len(values) - 1):
        delta = abs(values[i] - values[i + 1])
        if delta < _STALE_DELTA_THRESHOLD:
            consecutive_stale += 1
        else:
            break

    return {
        "detected": consecutive_stale >= 2,
        "consecutive_stale_iterations": consecutive_stale,
    }
