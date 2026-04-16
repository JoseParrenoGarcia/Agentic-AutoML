# %% [model] Instantiate, train, and return the fitted model
# Hyperparameters are sourced entirely from config.yaml — nothing hardcoded.
#
# DIFF vs iteration 1: GradientBoostingClassifier replaces LogisticRegression.
# Hyperparameters from plan: n_estimators=200, max_depth=4, learning_rate=0.1,
# subsample=0.8, min_samples_leaf=10, random_state=42.

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Any:
    """
    Instantiate GradientBoostingClassifier with hyperparameters from config, fit, and return.

    Args:
        X_train: Feature matrix for the training split (target column removed).
        y_train: Target series for the training split.
        config:  Parsed config.yaml dict. Uses:
                     config["hyperparameters"] — model constructor parameters
                     config["random_seed"]     — for reproducibility

    Returns:
        Fitted GradientBoostingClassifier model.
    """
    hp = config["hyperparameters"]

    model = GradientBoostingClassifier(
        n_estimators=int(hp.get("n_estimators", 200)),
        max_depth=int(hp.get("max_depth", 4)),
        learning_rate=float(hp.get("learning_rate", 0.1)),
        subsample=float(hp.get("subsample", 0.8)),
        min_samples_leaf=int(hp.get("min_samples_leaf", 10)),
        random_state=int(hp.get("random_state", config["random_seed"])),
    )

    model.fit(X_train, y_train)
    return model
