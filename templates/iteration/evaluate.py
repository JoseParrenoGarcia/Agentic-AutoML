# %% [evaluate] Compute metrics and write all output artifacts
# Writes: metrics.json, predictions.csv, feature_importance.json,
#         learning_curves.json, pipeline_metadata.json, model artifact.

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


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
        model:    Fitted sklearn-compatible model.
        X_train:  Training feature matrix.
        y_train:  Training labels.
        X_val:    Validation feature matrix.
        y_val:    Validation labels.
        config:   Parsed config.yaml dict. Uses config["output_paths"] for all write paths.
        metadata: PipelineMetadata collector (from utils.py). Serialised to pipeline_metadata.json.

    Returns:
        Metrics dict matching the metrics.json schema:
        {
            "primary":    {"name": str,   "value": float},
            "secondary":  {metric_name: float, ...},
            "train":      {metric_name: float, ...},
            "validation": {metric_name: float, ...},
        }

    Output artifacts written (paths from config["output_paths"]):
        metrics.json            — primary + secondary + train + validation metrics
        predictions.csv         — index, y_true, y_pred, y_prob_0, y_prob_1 (binary)
        feature_importance.json — model-native coefficients / importances
        learning_curves.json    — iterative train/val metric curves or null note
        pipeline_metadata.json  — stage shapes, durations, warnings, package versions
        outputs/model/model.pkl — joblib-serialised fitted model
        outputs/model/metadata.json — lightweight model metadata
    """
    out = config["output_paths"]

    # Ensure output directories exist
    for path_str in out.values():
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    # --- PLAN-SPECIFIC LOGIC ---
    # The Coder agent replaces this section with metric computation appropriate
    # for the task_type (binary_classification | multiclass | regression).
    #
    # Required pattern for binary_classification:
    #
    #   from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    #
    #   val_preds  = model.predict(X_val)
    #   val_probs  = model.predict_proba(X_val)
    #   train_preds = model.predict(X_train)
    #
    #   val_auc    = roc_auc_score(y_val, val_probs[:, 1])
    #   val_acc    = accuracy_score(y_val, val_preds)
    #   val_f1     = f1_score(y_val, val_preds)
    #   train_auc  = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    #
    #   metrics = {
    #       "primary":    {"name": "val_auc_roc", "value": val_auc},
    #       "secondary":  {"val_accuracy": val_acc, "val_f1": val_f1},
    #       "train":      {"train_auc_roc": train_auc},
    #       "validation": {"val_auc_roc": val_auc, "val_accuracy": val_acc},
    #   }
    # --- END PLAN-SPECIFIC LOGIC ---

    metrics: dict = {}  # Coder agent assigns above

    # Write metrics.json
    with open(out["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Write predictions.csv
    # --- PLAN-SPECIFIC LOGIC (predictions) ---
    # val_probs = model.predict_proba(X_val)  # shape (n, 2) for binary
    # predictions_df = pd.DataFrame({
    #     "index":    X_val.index,
    #     "y_true":   y_val.values,
    #     "y_pred":   val_preds,
    #     "y_prob_0": val_probs[:, 0],
    #     "y_prob_1": val_probs[:, 1],
    # })
    # predictions_df.to_csv(out["predictions"], index=False)
    # --- END PLAN-SPECIFIC LOGIC ---

    # Write feature_importance.json
    # --- PLAN-SPECIFIC LOGIC (feature importance) ---
    # For LogisticRegression:
    #   importance = [
    #       {"name": col, "importance": float(coef)}
    #       for col, coef in zip(X_train.columns, model.coef_[0])
    #   ]
    #   importance.sort(key=lambda x: abs(x["importance"]), reverse=True)
    #   fi_dict = {
    #       "method": "logistic_regression_coefficients",
    #       "features": importance,
    #       "sorted": True,
    #       "model": type(model).__name__,
    #   }
    # with open(out["feature_importance"], "w") as f:
    #     json.dump(fi_dict, f, indent=2)
    # --- END PLAN-SPECIFIC LOGIC ---

    # Write learning_curves.json
    # Non-iterative models (LogReg, SVM, etc.) write a null note.
    learning_curves = {"note": "model does not support iterative training"}
    # --- PLAN-SPECIFIC LOGIC (learning curves) ---
    # For iterative models (XGBoost, LightGBM, etc.) replace the note with:
    #   learning_curves = {
    #       "metric_name": "auc_roc",
    #       "train":      [...],
    #       "validation": [...],
    #       "iterations": [...],
    #   }
    # --- END PLAN-SPECIFIC LOGIC ---
    with open(out["learning_curves"], "w") as f:
        json.dump(learning_curves, f, indent=2)

    # Write pipeline_metadata.json
    with open(out["pipeline_metadata"], "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Save model artifact
    model_dir = Path(out["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

    import datetime
    model_meta = {
        "model_class": type(model).__name__,
        "feature_list": list(X_train.columns),
        "training_timestamp": datetime.datetime.utcnow().isoformat(),
        "n_train_samples": len(X_train),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    return metrics
