# %% [model] Instantiate, train, and return the fitted model
# Hyperparameters are sourced entirely from config.yaml — nothing hardcoded.

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Any:
    """
    Instantiate LogisticRegression with hyperparameters from config, fit, and return.

    Args:
        X_train: Feature matrix for the training split (target column removed).
        y_train: Target series for the training split.
        config:  Parsed config.yaml dict. Uses:
                     config["hyperparameters"] — model constructor parameters
                     config["random_seed"]     — for reproducibility

    Returns:
        Fitted LogisticRegression model.
    """
    hp = config["hyperparameters"]

    # class_weight stored as null in YAML (plan specifies "None" meaning no weighting)
    class_weight = hp.get("class_weight", None)
    if class_weight in ("None", "none", "null", ""):
        class_weight = None

    model = LogisticRegression(
        solver=hp.get("solver", "lbfgs"),
        max_iter=int(hp.get("max_iter", 1000)),
        C=float(hp.get("C", 1.0)),
        class_weight=class_weight,
        random_state=int(hp.get("random_state", config["random_seed"])),
    )

    model.fit(X_train, y_train)
    return model
