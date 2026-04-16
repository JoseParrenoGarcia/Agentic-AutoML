# %% [feature_engineering] Feature transforms for iteration 4
#
# EXTERNAL CONTRACT — two functions, always:
#   fit_transform(df_train, config)      ->  (transformed_df, fitted_params)
#   transform(df, fitted_params, config) ->  transformed_df
#
# The core rule: any value derived from training data must be computed in
# fit_transform, returned as fitted_params, and reused without recomputation
# in transform. transform is a pure application of fitted_params to new data.
#
# Iteration 4 feature steps (from iteration-4.yaml):
#   1. drop_passengerid       — drop PassengerId (identifier, no signal)
#   2. drop_ticket            — drop Ticket (high cardinality alphanumeric)
#   3. extract_title_from_name — extract Title from Name, map to 5 categories,
#                                one-hot encode, then drop Name
#   4. extract_cabin_deck     — extract Deck letter from Cabin, map rare to 'Rare',
#                                create HasCabin flag, one-hot encode Deck, drop Cabin
#                                (NEW vs iter 3: replaces cabin_to_has_cabin_flag)
#   5. impute_age_median      — fill Age nulls with training median
#   6. encode_sex_binary      — binary encode Sex: female=1, male=0
#   7. encode_embarked_onehot — one-hot encode Embarked, drop first
#   8. fare_log1p_transform   — apply log1p to Fare
#   9. create_family_size     — FamilySize = SibSp + Parch + 1
#  10. pclass_passthrough     — keep Pclass as-is
#
# Diff from iteration 3:
#   - Step 4 (extract_cabin_deck) replaces cabin_to_has_cabin_flag:
#     now extracts deck letter (A-G), maps T/G to 'Rare', sets null to 'Unknown',
#     creates HasCabin binary, one-hot encodes Deck with drop_first

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


def _extract_deck(cabin_series: pd.Series) -> pd.Series:
    """
    Extract deck letter from Cabin column.

    - Non-null Cabin: take the first character (A-G, T).
    - Map rare decks (T, G) to 'Rare'.
    - Null Cabin: set to 'Unknown'.

    Returns:
        Series of deck categories.
    """
    # Extract first character; NaN stays NaN
    deck = cabin_series.str[0]

    # Map rare decks to 'Rare'
    rare_decks = {"T", "G"}
    deck = deck.apply(lambda x: "Rare" if x in rare_decks else x)

    # Fill nulls with 'Unknown'
    deck = deck.fillna("Unknown")

    return deck


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

    # --- Step 4: extract_cabin_deck (NEW in iter 4, replaces cabin_to_has_cabin_flag) ---
    if "Cabin" in df.columns:
        # Create HasCabin binary flag before extracting deck
        df["HasCabin"] = _has_value_flag(df["Cabin"])

        # Extract deck letter (A-G, Rare, Unknown)
        df["Deck"] = _extract_deck(df["Cabin"])

        # Drop original Cabin column
        df = df.drop(columns=["Cabin"])

        # Record training deck categories for consistent encoding on val/test
        deck_categories = sorted(df["Deck"].unique().tolist())
        fitted_params["deck_categories"] = deck_categories

        # One-hot encode Deck with drop_first
        deck_dummies = _onehot_encode(df["Deck"], deck_categories, drop_first=True)
        df = df.drop(columns=["Deck"])
        df = pd.concat([df, deck_dummies], axis=1)

    # --- Step 5: impute_age_median ---
    age_median = float(df["Age"].median())
    fitted_params["age_median"] = age_median
    df["Age"] = _impute_with_value(df["Age"], age_median)

    # --- Step 6: encode_sex_binary (female=1, male=0) ---
    df["Sex"] = _binary_encode(df["Sex"], positive_value="female")

    # --- Step 7: impute Embarked then one-hot encode (drop first) ---
    embarked_mode = df["Embarked"].mode()[0] if df["Embarked"].notna().any() else "S"
    fitted_params["embarked_mode"] = embarked_mode
    df["Embarked"] = _impute_with_value(df["Embarked"], embarked_mode)

    embarked_categories = sorted(df["Embarked"].unique().tolist())
    fitted_params["embarked_categories"] = embarked_categories
    embarked_dummies = _onehot_encode(df["Embarked"], embarked_categories, drop_first=True)
    df = df.drop(columns=["Embarked"])
    df = pd.concat([df, embarked_dummies], axis=1)

    # --- Step 8: fare_log1p_transform ---
    df["Fare"] = _log1p_transform(df["Fare"])

    # --- Step 9: create_family_size ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- Step 3 continued: one-hot encode Title (use training categories) ---
    title_categories = sorted(df["Title"].unique().tolist())
    fitted_params["title_categories"] = title_categories
    title_dummies = _onehot_encode(df["Title"], title_categories, drop_first=True)
    df = df.drop(columns=["Title"])
    df = pd.concat([df, title_dummies], axis=1)

    # --- Step 10: Pclass, SibSp, Parch pass through unchanged ---

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

    # --- Step 4: extract_cabin_deck (use training deck categories) ---
    if "Cabin" in df.columns:
        # Create HasCabin binary flag
        df["HasCabin"] = _has_value_flag(df["Cabin"])

        # Extract deck letter
        df["Deck"] = _extract_deck(df["Cabin"])

        # Drop original Cabin column
        df = df.drop(columns=["Cabin"])

        # One-hot encode Deck using training categories
        deck_dummies = _onehot_encode(
            df["Deck"],
            fitted_params["deck_categories"],
            drop_first=True,
        )
        df = df.drop(columns=["Deck"])
        df = pd.concat([df, deck_dummies], axis=1)

    # --- Step 5: impute_age_median (use training median) ---
    df["Age"] = _impute_with_value(df["Age"], fitted_params["age_median"])

    # --- Step 6: encode_sex_binary ---
    df["Sex"] = _binary_encode(df["Sex"], positive_value="female")

    # --- Step 7: impute Embarked then one-hot encode (use training params) ---
    df["Embarked"] = _impute_with_value(df["Embarked"], fitted_params["embarked_mode"])
    embarked_dummies = _onehot_encode(
        df["Embarked"],
        fitted_params["embarked_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Embarked"])
    df = pd.concat([df, embarked_dummies], axis=1)

    # --- Step 8: fare_log1p_transform ---
    df["Fare"] = _log1p_transform(df["Fare"])

    # --- Step 9: create_family_size ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- Step 3 continued: one-hot encode Title (use training categories) ---
    title_dummies = _onehot_encode(
        df["Title"],
        fitted_params["title_categories"],
        drop_first=True,
    )
    df = df.drop(columns=["Title"])
    df = pd.concat([df, title_dummies], axis=1)

    # --- Step 10: Pclass, SibSp, Parch pass through unchanged ---

    return df
