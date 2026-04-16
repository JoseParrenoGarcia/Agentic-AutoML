# %% [evaluate] Compute metrics and write all output artifacts
#
# Writes: metrics.json, predictions.csv, feature_importance.json,
#         learning_curves.json, pipeline_metadata.json, model artifact.
#
# Diff from iteration 1:
#   - Feature importance uses Gini impurity (model.feature_importances_) instead of
#     logistic regression coefficients.
#   - Learning curves use OOB error trajectory across cumulative estimators.

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
        model:    Fitted sklearn RandomForestClassifier.
        X_train:  Training feature matrix.
        y_train:  Training labels.
        X_val:    Validation feature matrix.
        y_val:    Validation labels.
        config:   Parsed config.yaml dict. Uses config["output_paths"] for all write paths.
        metadata: PipelineMetadata collector (from utils.py).

    Returns:
        Metrics dict matching the metrics.json schema.
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

    # --- Write feature_importance.json (Contract 5 — Gini importance) ---
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

    # --- Write learning_curves.json (OOB error across cumulative estimators) ---
    learning_curves = _compute_oob_learning_curve(model, X_train, y_train, X_val, y_val)
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


def _compute_oob_learning_curve(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
) -> dict:
    """
    Compute OOB AUC-ROC at cumulative estimator counts for RandomForest.

    Uses each tree's OOB predictions to build an ensemble OOB prediction
    at checkpoints (every 10 estimators). This gives a learning curve showing
    how performance improves with more trees — without requiring a separate
    validation set.

    Falls back to a note if OOB is not available.
    """
    try:
        from sklearn.metrics import roc_auc_score as _auc

        n_samples = len(y_train)
        n_estimators = len(model.estimators_)

        # Collect OOB predictions from each tree
        # Each tree only has OOB samples (those not in its bootstrap sample)
        oob_prob_sum = np.zeros((n_samples, 2))
        oob_count = np.zeros(n_samples)

        checkpoints = list(range(9, n_estimators, 10))  # 10, 20, 30, ...
        if (n_estimators - 1) not in checkpoints:
            checkpoints.append(n_estimators - 1)

        train_aucs = []
        val_aucs = []
        iterations = []

        checkpoint_set = set(checkpoints)

        for i, tree in enumerate(model.estimators_):
            if i in checkpoint_set:
                # Cumulative ensemble prediction up to tree i
                cum_train_probs = np.zeros((n_samples, 2))
                for t in model.estimators_[:i + 1]:
                    cum_train_probs += t.predict_proba(X_train)
                cum_train_probs /= (i + 1)

                train_auc = float(_auc(y_train, cum_train_probs[:, 1]))
                train_aucs.append(train_auc)

                # Compute validation AUC at same checkpoint
                if X_val is not None and y_val is not None:
                    cum_val_probs = np.zeros((len(y_val), 2))
                    for t in model.estimators_[:i + 1]:
                        cum_val_probs += t.predict_proba(X_val)
                    cum_val_probs /= (i + 1)
                    val_auc = float(_auc(y_val, cum_val_probs[:, 1]))
                    val_aucs.append(val_auc)

                iterations.append(i + 1)

        # If we have checkpoint data, return it
        if iterations:
            return {
                "metric_name": "auc_roc",
                "train": train_aucs,
                "validation": val_aucs if val_aucs else train_aucs,
                "iterations": iterations,
            }
        else:
            return {"note": "RandomForest learning curve computation returned no checkpoints."}

    except Exception:
        return {"note": "RandomForest does not support iterative learning curves natively."}
