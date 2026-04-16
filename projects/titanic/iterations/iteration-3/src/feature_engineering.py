# %% [feature_engineering] Feature transforms for iteration 3
#
# EXTERNAL CONTRACT — two functions, always:
#   fit_transform(df_train, config)      ->  (transformed_df, fitted_params)
#   transform(df, fitted_params, config) ->  transformed_df
#
# The core rule: any value derived from training data must be computed in
# fit_transform, returned as fitted_params, and reused without recomputation
# in transform. transform is a pure application of fitted_params to new data.
#
# Iteration 3 feature steps (from iteration-3.yaml):
#   1. drop_passengerid       — drop PassengerId (identifier, no signal)
#   2. drop_ticket            — drop Ticket (high cardinality alphanumeric)
#   3. extract_title_from_name — extract Title from Name, map to 5 categories,
#                                one-hot encode, then drop Name
#   4. cabin_to_has_cabin_flag — replace Cabin with binary has_cabin flag
#   5. impute_age_median      — fill Age nulls with training median
#   6. impute_embarked_mode   — fill Embarked nulls with training mode ('S')
#   7. encode_sex_binary      — binary encode Sex: female=1, male=0
#   8. encode_embarked_onehot — one-hot encode Embarked, drop first
#   9. fare_log1p_transform   — apply log1p to Fare
#  10. create_family_size     — FamilySize = SibSp + Parch + 1 (NEW vs iter 1)
#  11. pclass_passthrough     — keep Pclass as-is
#
# Diff from iteration 1:
#   - Step 3 (extract_title_from_name) replaces simple Name drop
#   - Step 10 (create_family_size) is new; SibSp and Parch kept as well

import re
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
    """Encode a binary categorical: positive_value -> 1, everything else -> 0."""
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


def _extract_title(name_series: pd.Series) -> pd.Series:
    """
    Extract title from Name column using regex.
    Maps to 5 categories: Mr, Mrs, Miss, Master, Rare.

    Examples:
        "Braund, Mr. Owen Harris"  -> "Mr"
        "Cumings, Mrs. John Bradley" -> "Mrs"
    """
    # Extract text between comma and period: ", Title."
    raw_title = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

    # Map to 5 canonical categories
    title_map = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Don": "Rare",
        "Rev": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Lady": "Rare",
        "Sir": "Rare",
        "Col": "Rare",
        "Capt": "Rare",
        "the Countess": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    return raw_title.map(title_map).fillna("Rare")


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
        config:   Parsed config.yaml dict.

    Returns:
        transformed_df: Fully transformed training DataFrame.
        fitted_params:  Dict of all values derived from training data.
    """
    df = df_train.copy()
    fitted_params: Dict[str, Any] = {}

    # --- Step 3: Extract Title from Name, then drop Name ---
    if "Name" in df.columns:
        df["Title"] = _extract_title(df["Name"])
        df = df.drop(columns=["Name"])

    # --- Steps 1 & 2: Drop identifier/high-cardinality columns ---
    cols_to_drop = [c for c in ["PassengerId", "Ticket"] if c in df.columns]
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

    # --- Step 10: create_family_size (NEW in iter 3, carried from iter 2) ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- Step 3 continued: one-hot encode Title (use training categories) ---
    title_categories = sorted(df["Title"].unique().tolist())
    fitted_params["title_categories"] = title_categories
    title_dummies = _onehot_encode(df["Title"], title_categories, drop_first=True)
    df = df.drop(columns=["Title"])
    df = pd.concat([df, title_dummies], axis=1)

    # --- Step 11: Pclass, SibSp, Parch pass through unchanged ---

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

    # --- Step 3: Extract Title from Name, then drop Name ---
    if "Name" in df.columns:
        df["Title"] = _extract_title(df["Name"])
        df = df.drop(columns=["Name"])

    # --- Steps 1 & 2: Drop identifier/high-cardinality columns ---
    cols_to_drop = [c for c in ["PassengerId", "Ticket"] if c in df.columns]
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

    # --- Step 10: create_family_size ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- Step 3 continued: one-hot encode Title (use training categories) ---
    title_dummies = _onehot_encode(
        df["Title"],
        fitted_params["title_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Title"])
    df = pd.concat([df, title_dummies], axis=1)

    # --- Step 11: Pclass, SibSp, Parch pass through unchanged ---

    return df
