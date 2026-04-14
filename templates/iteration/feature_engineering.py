# %% [feature_engineering] Feature transforms for one iteration
#
# EXTERNAL CONTRACT — two functions, always:
#
#   fit_transform(df_train, config)          →  (transformed_df, fitted_params)
#   transform(df, fitted_params, config)     →  transformed_df
#
# The core rule: any value derived from training data (an imputation fill value,
# the bounds of a cap/collar, a category grouping map, a scaler's mean and std,
# a fitted sklearn Pipeline object) must be computed in fit_transform, returned
# as fitted_params, and then reused — without recomputation — in transform.
# transform must be a pure application of fitted_params to new data.
#
# main.py calls fit_transform once on train_df, then transform on val_df and
# (if present) test_df. This is the only way to guarantee no leakage.
#
# IMPLEMENTATION APPROACHES — choose one, or combine:
#
#   A. Atomic functions (recommended for manual/pandas transforms)
#      Write one small function per distinct transform type, e.g.:
#        cap_and_collar(series, lower, upper)
#        group_rare_categories(series, mapping, fill)
#        impute_with_value(series, value)
#      fit_transform computes the parameters (bounds, mapping, fill value) from
#      df_train, stores them in fitted_params, and calls the atomic functions.
#      transform calls the same atomic functions with values from fitted_params.
#      Keep each atomic function to a single, named responsibility.
#
#   B. sklearn Pipeline
#      Build a Pipeline of sklearn transformers in fit_transform, call
#      pipeline.fit_transform(X). fitted_params = the fitted pipeline object.
#      transform calls pipeline.transform(X). The pipeline handles all state.
#
#   C. Any other approach
#      PyTorch preprocessing in a Dataset, a custom transformer class, etc.
#      The contract is the same: fit once, apply identically to all splits.

import pandas as pd
from typing import Any, Tuple


def fit_transform(df_train: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Any]:
    """
    Fit all feature transforms on the training split and return the transformed
    DataFrame alongside everything needed to reproduce those transforms.

    Args:
        df_train: Full training DataFrame including the target column.
        config:   Parsed config.yaml dict.

    Returns:
        (transformed_df, fitted_params)

        transformed_df — training data after all feature steps applied.
        fitted_params  — all state required to call transform() on new data.
                         Shape is implementation-defined: a dict of scalar values,
                         a fitted sklearn Pipeline, a dataclass, etc.

    Raises:
        ValueError: If a required column is missing or a transform cannot be applied.
    """
    # --- PLAN-SPECIFIC LOGIC ---
    # --- END PLAN-SPECIFIC LOGIC ---

    raise NotImplementedError("Coder agent must implement fit_transform for this iteration")


def transform(df: pd.DataFrame, fitted_params: Any, config: dict) -> pd.DataFrame:
    """
    Apply pre-fitted transforms to a val or test DataFrame.

    Args:
        df:            Input DataFrame (val or test split).
        fitted_params: The object returned by fit_transform.
        config:        Parsed config.yaml dict.

    Returns:
        Transformed DataFrame with the same columns produced by fit_transform.

    Contract: do NOT compute any statistic from df. Only apply fitted_params.
    """
    # --- PLAN-SPECIFIC LOGIC ---
    # --- END PLAN-SPECIFIC LOGIC ---

    raise NotImplementedError("Coder agent must implement transform for this iteration")
