# %% [evaluate] Compute metrics and write all output artifacts
#
# Writes: metrics.json, predictions.csv, feature_importance.json,
#         learning_curves.json, pipeline_metadata.json, model artifact.
#
# DIFF vs iteration 1:
#   - Feature importance uses model.feature_importances_ (Gini importance)
#     instead of logistic regression coefficients.
#   - Learning curves extracted from staged_predict for GradientBoosting
#     (train & val AUC-ROC at each boosting stage).

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    metadata: Any,  # PipelineMetadata from utils.py
) -> dict:
    """
    Compute evaluation metrics, save all output artifacts, and return a metrics dict.

    Args:
        model:    Fitted sklearn GradientBoostingClassifier.
        X_train:  Training feature matrix.
        y_train:  Training labels.
        X_val:    Validation feature matrix.
        y_val:    Validation labels.
        config:   Parsed config.yaml dict. Uses config["output_paths"] for all write paths.
        metadata: PipelineMetadata collector (from utils.py).

    Returns:
        Metrics dict matching the metrics.json schema:
        {
            "primary":    {"name": str,   "value": float},
            "secondary":  {metric_name: float, ...},
            "train":      {metric_name: float, ...},
            "validation": {metric_name: float, ...},
        }
    """
    out = config["output_paths"]

    # Ensure output directories exist
    for path_str in out.values():
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    # --- Predictions ---
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)

    # --- Validation metrics ---
    val_auc = float(roc_auc_score(y_val, val_probs[:, 1]))
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1 = float(f1_score(y_val, val_preds))
    val_precision = float(precision_score(y_val, val_preds))
    val_recall = float(recall_score(y_val, val_preds))

    # --- Train metrics ---
    train_auc = float(roc_auc_score(y_train, train_probs[:, 1]))
    train_acc = float(accuracy_score(y_train, train_preds))

    metrics = {
        "primary": {"name": "val_auc_roc", "value": val_auc},
        "secondary": {
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
        },
        "train": {
            "train_auc_roc": train_auc,
            "train_accuracy": train_acc,
        },
        "validation": {
            "val_auc_roc": val_auc,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
        },
    }

    # --- Write metrics.json ---
    with open(out["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Write predictions.csv ---
    preds_df = pd.DataFrame({
        "index": X_val.index,
        "y_true": y_val.values,
        "y_pred": val_preds,
        "y_prob_0": val_probs[:, 0],
        "y_prob_1": val_probs[:, 1],
    })
    preds_df.to_csv(out["predictions"], index=False)

    # --- Write feature_importance.json (Contract 5 schema) ---
    # DIFF: Use Gini importance from tree ensemble instead of LR coefficients
    feature_names = X_val.columns.tolist()
    importances = model.feature_importances_.tolist()
    feat_imp_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    feature_importance = {
        "method": "gini_importance",
        "features": [{"name": n, "importance": round(v, 6)} for n, v in feat_imp_pairs],
        "sorted": True,
        "model": type(model).__name__,
    }
    with open(out["feature_importance"], "w") as f:
        json.dump(feature_importance, f, indent=2)

    # --- Write learning_curves.json (Contract 5 schema) ---
    # DIFF: GradientBoosting supports staged_predict -> extract per-stage AUC
    learning_curves = _extract_learning_curves(model, X_train, y_train, X_val, y_val)
    with open(out["learning_curves"], "w") as f:
        json.dump(learning_curves, f, indent=2)

    # --- Write pipeline_metadata.json ---
    with open(out["pipeline_metadata"], "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # --- Save model artifact ---
    model_dir = Path(out["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

    from datetime import datetime, timezone
    model_meta = {
        "model_class": type(model).__name__,
        "feature_list": feature_names,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": len(X_train),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    return metrics


def _extract_learning_curves(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """
    Extract per-stage AUC-ROC from GradientBoosting's staged_predict_proba.

    Returns a dict matching the learning_curves.json contract:
        {"metric_name": str, "train": [float], "validation": [float], "iterations": [int]}
    """
    n_estimators = model.n_estimators_

    train_scores = []
    val_scores = []
    iterations = []

    # Sample at regular intervals to keep the file manageable
    step = max(1, n_estimators // 50)
    stage_indices = list(range(step - 1, n_estimators, step))
    if (n_estimators - 1) not in stage_indices:
        stage_indices.append(n_estimators - 1)

    train_staged = list(model.staged_predict_proba(X_train))
    val_staged = list(model.staged_predict_proba(X_val))

    for i in stage_indices:
        train_auc = float(roc_auc_score(y_train, train_staged[i][:, 1]))
        val_auc = float(roc_auc_score(y_val, val_staged[i][:, 1]))
        train_scores.append(round(train_auc, 6))
        val_scores.append(round(val_auc, 6))
        iterations.append(i + 1)  # 1-indexed

    return {
        "metric_name": "auc_roc",
        "train": train_scores,
        "validation": val_scores,
        "iterations": iterations,
    }
