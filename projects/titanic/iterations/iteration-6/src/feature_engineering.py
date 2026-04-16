# %% [feature_engineering] Feature transforms for iteration 6
#
# EXTERNAL CONTRACT — two functions, always:
#   fit_transform(df_train, config)      ->  (transformed_df, fitted_params)
#   transform(df, fitted_params, config) ->  transformed_df
#
# The core rule: any value derived from training data must be computed in
# fit_transform, returned as fitted_params, and reused without recomputation
# in transform. transform is a pure application of fitted_params to new data.
#
# Iteration 6 feature steps (from iteration-6.yaml):
#   1. drop_passengerid         — drop PassengerId (identifier, no signal)
#   2. drop_ticket              — drop Ticket (high cardinality alphanumeric)
#   3. drop_cabin_raw           — extract HasCabin flag then drop raw Cabin
#   4. create_has_cabin         — binary HasCabin: 1 if Cabin not null, 0 otherwise
#   5. extract_title_from_name  — extract Title, map to 5 categories, one-hot, drop Name
#   6. impute_age_median        — fill Age nulls with training median
#   7. bin_age_agebands         — bin Age into child/youth/adult/senior, one-hot, drop Age
#   8. log_transform_fare       — Fare_log = log1p(Fare), drop raw Fare
#   9. create_family_size       — FamilySize = SibSp + Parch + 1 (keep SibSp, Parch)
#  10. encode_sex               — binary encode Sex: female=1, male=0
#  11. impute_embarked_mode     — impute 2 nulls with mode 'S', one-hot, drop Embarked
#  12. pclass_passthrough       — keep Pclass as-is (ordinal integer)
#
# Diff from iteration 4/5:
#   - Step 3/4 reverts to HasCabin binary flag (iter 4 used deck letter extraction)
#   - Step 7 NEW: Age binning into AgeBand (child/youth/adult/senior) replaces raw Age
#   - Step 8 already present in iter 4/5 (log1p fare)
# ---------------------------------------------------------------------------

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Atomic transform helpers
# ---------------------------------------------------------------------------

def _has_value_flag(series: pd.Series) -> pd.Series:
    """Return 1 where series is not null/NaN, 0 otherwise."""
    return series.notna().astype(int)


def _extract_title(name_series: pd.Series) -> pd.Series:
    """
    Extract title from Name column using regex.
    Maps to 5 canonical categories: Mr, Mrs, Miss, Master, Rare.
    """
    title_map = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Ms": "Miss",
        "Mlle": "Miss",
        "Mme": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Capt": "Rare",
        "Countess": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Lady": "Rare",
        "Sir": "Rare",
    }
    titles = name_series.str.extract(r",\s*([^\.]+)\.", expand=False)
    titles = titles.str.strip()
    return titles.map(title_map).fillna("Rare")


def _impute_with_value(series: pd.Series, value: Any) -> pd.Series:
    """Fill nulls in series with a precomputed value."""
    return series.fillna(value)


def _bin_age(age_series: pd.Series) -> pd.Series:
    """
    Bin Age into 4 survival-relevant categories.
    Bins: child (<=12), youth (13-24), adult (25-59), senior (>=60).
    """
    bins = [0, 12, 24, 59, 120]
    labels = ["child", "youth", "adult", "senior"]
    return pd.cut(age_series, bins=bins, labels=labels, right=True)


def _log1p_transform(series: pd.Series) -> pd.Series:
    """Apply log1p transform to series."""
    return np.log1p(series)


def _binary_encode(series: pd.Series, positive_value: str) -> pd.Series:
    """Binary encode: positive_value -> 1, everything else -> 0."""
    return (series == positive_value).astype(int)


def _onehot_encode(
    series: pd.Series,
    categories: List[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encode series using a fixed list of categories (from training).
    Categories not seen in training are ignored; unseen values in transform
    are mapped to all-zero rows (handled by reindex).

    Args:
        series:     Categorical series to encode.
        categories: Sorted list of category values from training.
        drop_first: Whether to drop the first category column.

    Returns:
        DataFrame of dummy columns.
    """
    dummies = pd.get_dummies(series, prefix=series.name)
    # Align to training categories
    expected_cols = [f"{series.name}_{c}" for c in categories]
    if drop_first:
        expected_cols = expected_cols[1:]
    dummies = dummies.reindex(columns=expected_cols, fill_value=0)
    return dummies


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

    # --- Steps 3 & 4: Create HasCabin flag then drop raw Cabin ---
    if "Cabin" in df.columns:
        df["HasCabin"] = _has_value_flag(df["Cabin"])
        df = df.drop(columns=["Cabin"])

    # --- Step 5: Extract Title from Name, map to 5 categories ---
    if "Name" in df.columns:
        df["Title"] = _extract_title(df["Name"])
        title_categories = sorted(df["Title"].unique().tolist())
        fitted_params["title_categories"] = title_categories
        df = df.drop(columns=["Name"])

    # --- Steps 1 & 2: Drop identifier/high-cardinality columns ---
    cols_to_drop = [c for c in ["PassengerId", "Ticket"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Step 6: impute_age_median ---
    age_median = float(df["Age"].median())
    fitted_params["age_median"] = age_median
    df["Age"] = _impute_with_value(df["Age"], age_median)

    # --- Step 7: bin_age_agebands — bin into child/youth/adult/senior, one-hot, drop Age ---
    df["AgeBand"] = _bin_age(df["Age"])
    ageband_categories = ["child", "youth", "adult", "senior"]
    fitted_params["ageband_categories"] = ageband_categories
    df["AgeBand"] = df["AgeBand"].astype(str)
    ageband_dummies = _onehot_encode(df["AgeBand"], ageband_categories, drop_first=True)
    df = df.drop(columns=["Age", "AgeBand"])
    df = pd.concat([df, ageband_dummies], axis=1)

    # --- Step 8: log_transform_fare — Fare_log = log1p(Fare), rename column ---
    if "Fare" in df.columns:
        df["Fare_log"] = _log1p_transform(df["Fare"])
        df = df.drop(columns=["Fare"])

    # --- Step 9: create_family_size — FamilySize = SibSp + Parch + 1 (keep originals) ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- Step 10: encode_sex — female=1, male=0 ---
    df["Sex"] = _binary_encode(df["Sex"], positive_value="female")

    # --- Step 11: impute_embarked_mode then one-hot encode ---
    embarked_mode = df["Embarked"].mode()[0] if df["Embarked"].notna().any() else "S"
    fitted_params["embarked_mode"] = embarked_mode
    df["Embarked"] = _impute_with_value(df["Embarked"], embarked_mode)
    embarked_categories = sorted(df["Embarked"].unique().tolist())
    fitted_params["embarked_categories"] = embarked_categories
    embarked_dummies = _onehot_encode(df["Embarked"], embarked_categories, drop_first=True)
    df = df.drop(columns=["Embarked"])
    df = pd.concat([df, embarked_dummies], axis=1)

    # --- Step 5 continued: one-hot encode Title (use training categories) ---
    title_dummies = _onehot_encode(df["Title"], fitted_params["title_categories"], drop_first=True)
    df = df.drop(columns=["Title"])
    df = pd.concat([df, title_dummies], axis=1)

    # --- Step 12: pclass_passthrough — Pclass passes through as integer ---
    # No transform needed

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

    # --- Steps 3 & 4: Create HasCabin flag then drop raw Cabin ---
    if "Cabin" in df.columns:
        df["HasCabin"] = _has_value_flag(df["Cabin"])
        df = df.drop(columns=["Cabin"])

    # --- Step 5: Extract Title from Name ---
    if "Name" in df.columns:
        df["Title"] = _extract_title(df["Name"])
        df = df.drop(columns=["Name"])

    # --- Steps 1 & 2: Drop identifier/high-cardinality columns ---
    cols_to_drop = [c for c in ["PassengerId", "Ticket"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Step 6: impute_age_median (use training median) ---
    df["Age"] = _impute_with_value(df["Age"], fitted_params["age_median"])

    # --- Step 7: bin_age_agebands (use training categories) ---
    df["AgeBand"] = _bin_age(df["Age"])
    df["AgeBand"] = df["AgeBand"].astype(str)
    ageband_dummies = _onehot_encode(
        df["AgeBand"],
        fitted_params["ageband_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Age", "AgeBand"])
    df = pd.concat([df, ageband_dummies], axis=1)

    # --- Step 8: log_transform_fare ---
    if "Fare" in df.columns:
        df["Fare_log"] = _log1p_transform(df["Fare"])
        df = df.drop(columns=["Fare"])

    # --- Step 9: create_family_size (keep SibSp, Parch) ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- Step 10: encode_sex ---
    df["Sex"] = _binary_encode(df["Sex"], positive_value="female")

    # --- Step 11: impute_embarked_mode then one-hot (use training params) ---
    df["Embarked"] = _impute_with_value(df["Embarked"], fitted_params["embarked_mode"])
    embarked_dummies = _onehot_encode(
        df["Embarked"],
        fitted_params["embarked_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Embarked"])
    df = pd.concat([df, embarked_dummies], axis=1)

    # --- Step 5 continued: one-hot encode Title (use training categories) ---
    title_dummies = _onehot_encode(
        df["Title"],
        fitted_params["title_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Title"])
    df = pd.concat([df, title_dummies], axis=1)

    # --- Step 12: pclass_passthrough ---
    # No transform needed

    return df
