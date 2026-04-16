# %% [evaluate] Compute metrics and write all output artifacts
#
# Writes: metrics.json, predictions.csv, feature_importance.json,
#         learning_curves.json, pipeline_metadata.json, model artifact.
#
# Iteration 6: StackingClassifier — feature importance extracted from the
# RF base estimator (named_estimators_["rf"]).
# Learning curves: StackingClassifier is not iterative — write null note.

import json
from datetime import datetime, timezone
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
        model:    Fitted sklearn StackingClassifier.
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
    Path(out["model"]).mkdir(parents=True, exist_ok=True)

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

    # --- Write feature_importance.json ---
    # Extract Gini importance from the RF base estimator inside the stacking model.
    # named_estimators_ contains fitted base estimators keyed by name.
    rf_estimator = model.named_estimators_.get("rf", None)
    if rf_estimator is not None and hasattr(rf_estimator, "feature_importances_"):
        importances = rf_estimator.feature_importances_
        fi_list = [
            {"name": col, "importance": float(imp)}
            for col, imp in zip(X_train.columns, importances)
        ]
        fi_list.sort(key=lambda x: x["importance"], reverse=True)
        fi_dict = {
            "method": "gini_importance_from_rf_base_estimator",
            "features": fi_list,
            "sorted": True,
            "model": "StackingClassifier[rf=RandomForestClassifier]",
        }
    else:
        # Fallback: use meta-learner LR coefficients if RF not available
        final_est = model.final_estimator_
        # final_estimator is a Pipeline(scaler, lr); get the LR step
        if hasattr(final_est, "named_steps") and "lr" in final_est.named_steps:
            lr_step = final_est.named_steps["lr"]
            coef = lr_step.coef_[0]
            n_base = len(model.estimators_)
            feature_names = [f"base_{i}" for i in range(n_base)]
            fi_list = [
                {"name": n, "importance": float(c)}
                for n, c in zip(feature_names, coef)
            ]
            fi_list.sort(key=lambda x: abs(x["importance"]), reverse=True)
            fi_dict = {
                "method": "meta_learner_lr_coefficients",
                "features": fi_list,
                "sorted": True,
                "model": "StackingClassifier[final_estimator=LogisticRegression]",
            }
        else:
            fi_dict = {
                "method": "unavailable",
                "features": [],
                "sorted": True,
                "model": type(model).__name__,
            }

    with open(out["feature_importance"], "w") as f:
        json.dump(fi_dict, f, indent=2)

    # --- Write learning_curves.json ---
    # StackingClassifier is not iterative — write null note per Contract 5.
    learning_curves = {"note": "model does not support iterative training"}
    with open(out["learning_curves"], "w") as f:
        json.dump(learning_curves, f, indent=2)

    # --- Write pipeline_metadata.json ---
    with open(out["pipeline_metadata"], "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # --- Save model artifact ---
    model_dir = Path(out["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")

    model_metadata = {
        "model_class": type(model).__name__,
        "feature_list": list(X_train.columns),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": int(len(X_train)),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)

    return metrics
