# %% [model] Instantiate, train, and return the fitted model
# Hyperparameters are sourced entirely from config.yaml — nothing hardcoded.

from typing import Any

import pandas as pd


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Any:
    """
    Instantiate the model, run the full training procedure, and return the
    fitted model object. Predictions are the responsibility of evaluate.py.

    Args:
        X_train: Feature matrix for the training split (target column removed).
        y_train: Target series for the training split.
        config:  Parsed config.yaml dict. Uses:
                     config["hyperparameters"] — model constructor / training parameters
                     config["random_seed"]     — for reproducibility
                     config["task_type"]       — binary_classification | multiclass | regression

    Returns:
        A fitted model object. evaluate.py will call the following on it:
            model.predict(X)           — class labels or floats
            model.predict_proba(X)     — class probabilities (classification only)
        For non-sklearn models (e.g. a PyTorch nn.Module), the Coder must either
        wrap the model in an sklearn-compatible interface or adapt evaluate.py
        accordingly.

    Notes:
        - For sklearn estimators: instantiate with hyperparameters, call .fit().
        - For gradient-boosted trees with eval_set callbacks: pass (X_val, y_val)
          here if needed for early stopping, but store the learning curve in a
          way evaluate.py can retrieve it (e.g. model.evals_result()).
        - For deep learning: this function contains the full training loop
          (epochs, optimizer, loss, scheduler). Return the trained nn.Module.
    """
    # --- PLAN-SPECIFIC LOGIC ---
    # --- END PLAN-SPECIFIC LOGIC ---

    raise NotImplementedError("Coder agent must implement train_model for this iteration")
