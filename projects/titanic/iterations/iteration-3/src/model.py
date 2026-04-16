# %% [model] Instantiate, train, and return the fitted model
# Hyperparameters are sourced entirely from config.yaml — nothing hardcoded.
#
# Diff from iteration 1: RandomForestClassifier replaces LogisticRegression.

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Any:
    """
    Instantiate RandomForestClassifier with hyperparameters from config, fit, and return.

    Args:
        X_train: Feature matrix for the training split (target column removed).
        y_train: Target series for the training split.
        config:  Parsed config.yaml dict. Uses:
                     config["hyperparameters"] — model constructor parameters
                     config["random_seed"]     — for reproducibility

    Returns:
        Fitted RandomForestClassifier model.
    """
    hp = config["hyperparameters"]

    # class_weight stored as string in YAML; normalise null-like values
    class_weight = hp.get("class_weight", None)
    if class_weight in ("None", "none", "null", ""):
        class_weight = None

    model = RandomForestClassifier(
        n_estimators=int(hp.get("n_estimators", 200)),
        max_depth=int(hp.get("max_depth", 5)),
        min_samples_leaf=int(hp.get("min_samples_leaf", 10)),
        min_samples_split=int(hp.get("min_samples_split", 20)),
        max_features=hp.get("max_features", "sqrt"),
        class_weight=class_weight,
        random_state=int(hp.get("random_state", config["random_seed"])),
        n_jobs=int(hp.get("n_jobs", -1)),
    )

    model.fit(X_train, y_train)
    return model
