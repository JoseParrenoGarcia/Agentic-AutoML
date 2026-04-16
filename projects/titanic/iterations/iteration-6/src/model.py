# %% [model] Instantiate, train, and return the fitted StackingClassifier
#
# Iteration 6: StackingClassifier with two base estimators:
#   - RandomForestClassifier (rf)
#   - LogisticRegression (lr)
# Meta-learner: LogisticRegression (final_estimator)
# Hyperparameters are sourced entirely from config.yaml — nothing hardcoded.

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Any:
    """
    Instantiate StackingClassifier with hyperparameters from config, fit, and return.

    The LR base estimator and meta-learner are wrapped in a StandardScaler pipeline
    since logistic regression is sensitive to feature scale.

    Args:
        X_train: Feature matrix for the training split (target column removed).
        y_train: Target series for the training split.
        config:  Parsed config.yaml dict. Uses:
                     config["hyperparameters"] — stacking constructor parameters
                     config["random_seed"]     — for reproducibility

    Returns:
        Fitted StackingClassifier model.
    """
    hp = config["hyperparameters"]

    # --- Build base estimators from config ---
    base_estimators = []
    for est_cfg in hp["estimators"]:
        algo = est_cfg["algorithm"]
        name = est_cfg["name"]

        if algo == "RandomForestClassifier":
            class_weight = est_cfg.get("class_weight", None)
            if class_weight in ("None", "none", "null", ""):
                class_weight = None
            estimator = RandomForestClassifier(
                n_estimators=int(est_cfg.get("n_estimators", 200)),
                max_depth=int(est_cfg.get("max_depth", 5)),
                min_samples_leaf=int(est_cfg.get("min_samples_leaf", 10)),
                min_samples_split=int(est_cfg.get("min_samples_split", 20)),
                max_features=est_cfg.get("max_features", "sqrt"),
                class_weight=class_weight,
                random_state=int(est_cfg.get("random_state", config["random_seed"])),
            )
            base_estimators.append((name, estimator))

        elif algo == "LogisticRegression":
            # Wrap LR in a StandardScaler pipeline for proper scaling
            lr = LogisticRegression(
                C=float(est_cfg.get("C", 1.0)),
                max_iter=int(est_cfg.get("max_iter", 1000)),
                solver=est_cfg.get("solver", "lbfgs"),
                random_state=int(est_cfg.get("random_state", config["random_seed"])),
            )
            estimator = Pipeline([("scaler", StandardScaler()), ("lr", lr)])
            base_estimators.append((name, estimator))

        else:
            raise ValueError(f"Unknown base estimator algorithm: '{algo}'")

    # --- Build meta-learner (final estimator) ---
    fe_cfg = hp["final_estimator"]
    fe_algo = fe_cfg["algorithm"]
    if fe_algo == "LogisticRegression":
        final_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=float(fe_cfg.get("C", 0.5)),
                max_iter=int(fe_cfg.get("max_iter", 1000)),
                solver=fe_cfg.get("solver", "lbfgs"),
                random_state=int(fe_cfg.get("random_state", config["random_seed"])),
            )),
        ])
    else:
        raise ValueError(f"Unknown final_estimator algorithm: '{fe_algo}'")

    # --- Build StackingClassifier ---
    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=int(hp.get("cv", 5)),
        stack_method=hp.get("stack_method", "predict_proba"),
        passthrough=bool(hp.get("passthrough", False)),
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model
