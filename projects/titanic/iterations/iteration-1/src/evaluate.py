# %% [evaluate] Compute metrics and write all output artifacts
#
# Writes: metrics.json, predictions.csv, feature_importance.json,
#         learning_curves.json, pipeline_metadata.json, model artifact.

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
        model:    Fitted sklearn LogisticRegression.
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

    # --- Write feature_importance.json (coefficients) ---
    feature_names = X_val.columns.tolist()
    coefs = model.coef_[0].tolist()
    feature_importance = {
        "type": "logistic_regression_coefficients",
        "features": feature_names,
        "coefficients": coefs,
        "intercept": float(model.intercept_[0]),
    }
    with open(out["feature_importance"], "w") as f:
        json.dump(feature_importance, f, indent=2)

    # --- Write learning_curves.json ---
    # LogisticRegression does not produce iterative learning curves natively.
    learning_curves = {
        "note": "LogisticRegression does not produce iterative learning curves. Null entry.",
        "train": None,
        "validation": None,
    }
    with open(out["learning_curves"], "w") as f:
        json.dump(learning_curves, f, indent=2)

    # --- Write pipeline_metadata.json ---
    with open(out["pipeline_metadata"], "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # --- Save model artifact ---
    model_dir = Path(out["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

    model_meta = {
        "algorithm": type(model).__name__,
        "iteration": config["iteration"],
        "task_type": config["task_type"],
        "hyperparameters": config["hyperparameters"],
        "random_seed": config["random_seed"],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "classes": model.classes_.tolist(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    return metrics
