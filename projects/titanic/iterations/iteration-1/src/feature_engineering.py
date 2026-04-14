# %% [feature_engineering] Feature transforms for iteration 1
#
# EXTERNAL CONTRACT — two functions, always:
#   fit_transform(df_train, config)      →  (transformed_df, fitted_params)
#   transform(df, fitted_params, config) →  transformed_df
#
# The core rule: any value derived from training data must be computed in
# fit_transform, returned as fitted_params, and reused without recomputation
# in transform. transform is a pure application of fitted_params to new data.
#
# Iteration 1 feature steps (from iteration-1.yaml):
#   1. drop_passengerid      — drop PassengerId (identifier, no signal)
#   2. drop_name             — drop Name (high cardinality text)
#   3. drop_ticket           — drop Ticket (high cardinality alphanumeric)
#   4. cabin_to_has_cabin_flag — replace Cabin with binary has_cabin flag
#   5. impute_age_median     — fill Age nulls with training median
#   6. impute_embarked_mode  — fill Embarked nulls with training mode ('S')
#   7. encode_sex_binary     — binary encode Sex: female=1, male=0
#   8. encode_embarked_onehot — one-hot encode Embarked, drop first
#   9. fare_log1p_transform  — apply log1p to Fare
#  10. pclass_passthrough    — keep Pclass as-is
#  11. sibsp_parch_passthrough — keep SibSp and Parch as-is

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Atomic transform helpers
# ---------------------------------------------------------------------------

def _impute_with_value(series: pd.Series, fill_value: Any) -> pd.Series:
    """Fill nulls with a fixed value computed from training data."""
    return series.fillna(fill_value)


def _binary_encode(series: pd.Series, positive_value: str) -> pd.Series:
    """Encode a binary categorical: positive_value → 1, everything else → 0."""
    return (series == positive_value).astype(int)


def _log1p_transform(series: pd.Series) -> pd.Series:
    """Apply log1p transform. Safe for zero values."""
    return np.log1p(series)


def _has_value_flag(series: pd.Series) -> pd.Series:
    """Return 1 if value is non-null/non-empty, 0 otherwise."""
    return series.notna().astype(int)


def _onehot_encode(
    series: pd.Series,
    categories: list,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode a series using a fixed category list (computed from train).

    Args:
        series:     The column to encode.
        categories: Ordered list of categories seen in training.
        drop_first: Whether to drop the first dummy to avoid collinearity.

    Returns:
        DataFrame of dummy columns, named <col>_<category>.
    """
    col_name = series.name
    dummies = pd.get_dummies(
        series.astype(pd.CategoricalDtype(categories=categories)),
        prefix=col_name,
        drop_first=drop_first,
    )
    return dummies.astype(int)


# ---------------------------------------------------------------------------
# fit_transform — compute params from train, return transformed train df
# ---------------------------------------------------------------------------

def fit_transform(
    df_train: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit all transforms on df_train, apply them, and return (transformed_df, fitted_params).

    Args:
        df_train: Training DataFrame including the target column.
        config:   Parsed config.yaml dict (not used for params in iteration 1,
                  but kept for contract compliance).

    Returns:
        transformed_df: Fully transformed training DataFrame.
        fitted_params:  Dict of all values derived from training data.
    """
    df = df_train.copy()
    fitted_params: Dict[str, Any] = {}

    # --- Step 1: Drop identifier/high-cardinality columns ---
    cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Step 4: cabin_to_has_cabin_flag ---
    if "Cabin" in df.columns:
        df["has_cabin"] = _has_value_flag(df["Cabin"])
        df = df.drop(columns=["Cabin"])

    # --- Step 5: impute_age_median ---
    age_median = float(df["Age"].median())
    fitted_params["age_median"] = age_median
    df["Age"] = _impute_with_value(df["Age"], age_median)

    # --- Step 6: impute_embarked_mode ---
    # Mode is 'S' per plan; compute from train to be safe
    embarked_mode = df["Embarked"].mode()[0] if df["Embarked"].notna().any() else "S"
    fitted_params["embarked_mode"] = embarked_mode
    df["Embarked"] = _impute_with_value(df["Embarked"], embarked_mode)

    # --- Step 7: encode_sex_binary (female=1, male=0) ---
    df["Sex"] = _binary_encode(df["Sex"], positive_value="female")

    # --- Step 8: encode_embarked_onehot (drop first to avoid dummy trap) ---
    embarked_categories = sorted(df["Embarked"].unique().tolist())
    fitted_params["embarked_categories"] = embarked_categories
    embarked_dummies = _onehot_encode(df["Embarked"], embarked_categories, drop_first=True)
    df = df.drop(columns=["Embarked"])
    df = pd.concat([df, embarked_dummies], axis=1)

    # --- Step 9: fare_log1p_transform ---
    df["Fare"] = _log1p_transform(df["Fare"])

    # --- Steps 10 & 11: Pclass, SibSp, Parch pass through unchanged ---
    # No action needed.

    return df, fitted_params


# ---------------------------------------------------------------------------
# transform — apply fitted params to new data (no recomputation)
# ---------------------------------------------------------------------------

def transform(
    df: pd.DataFrame,
    fitted_params: Dict[str, Any],
    config: dict,
) -> pd.DataFrame:
    """
    Apply fitted transforms to val or test data using only values from fitted_params.

    Args:
        df:            DataFrame to transform (val or test split).
        fitted_params: Dict produced by fit_transform; no values may be recomputed.
        config:        Parsed config.yaml dict.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    # --- Step 1: Drop identifier/high-cardinality columns ---
    cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Step 4: cabin_to_has_cabin_flag ---
    if "Cabin" in df.columns:
        df["has_cabin"] = _has_value_flag(df["Cabin"])
        df = df.drop(columns=["Cabin"])

    # --- Step 5: impute_age_median (use training median) ---
    df["Age"] = _impute_with_value(df["Age"], fitted_params["age_median"])

    # --- Step 6: impute_embarked_mode (use training mode) ---
    df["Embarked"] = _impute_with_value(df["Embarked"], fitted_params["embarked_mode"])

    # --- Step 7: encode_sex_binary ---
    df["Sex"] = _binary_encode(df["Sex"], positive_value="female")

    # --- Step 8: encode_embarked_onehot (use training categories) ---
    embarked_dummies = _onehot_encode(
        df["Embarked"],
        fitted_params["embarked_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Embarked"])
    df = pd.concat([df, embarked_dummies], axis=1)

    # --- Step 9: fare_log1p_transform ---
    df["Fare"] = _log1p_transform(df["Fare"])

    # --- Steps 10 & 11: Pclass, SibSp, Parch pass through unchanged ---

    return df
