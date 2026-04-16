# %% [model] Instantiate, train, and return the fitted model
#
# Hyperparameters are sourced entirely from config.yaml — nothing hardcoded.
# Iteration 5: GradientBoostingClassifier with conservative regularisation and
# built-in early stopping via n_iter_no_change.
#
# Key changes from iteration 4 (RandomForestClassifier):
#   - Algorithm: GradientBoostingClassifier instead of RandomForestClassifier
#   - learning_rate: 0.05 (slow, careful fitting to avoid overfitting)
#   - max_depth: 3 (shallower trees than iter 2's depth=4)
#   - min_samples_leaf: 20 (stronger leaf constraints than iter 2's 4)
#   - subsample: 0.8 (stochastic boosting for implicit regularisation)
#   - n_iter_no_change: 30 (automatic early stopping)
#   - validation_fraction: 0.15 (held-out fraction used by early stopping)

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
        n_estimators=int(hp.get("n_estimators", 500)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        max_depth=int(hp.get("max_depth", 3)),
        min_samples_leaf=int(hp.get("min_samples_leaf", 20)),
        min_samples_split=int(hp.get("min_samples_split", 40)),
        subsample=float(hp.get("subsample", 0.8)),
        max_features=hp.get("max_features", "sqrt"),
        n_iter_no_change=int(hp.get("n_iter_no_change", 30)),
        validation_fraction=float(hp.get("validation_fraction", 0.15)),
        tol=float(hp.get("tol", 0.0001)),
        random_state=int(hp.get("random_state", config["random_seed"])),
    )

    model.fit(X_train, y_train)
    return model
