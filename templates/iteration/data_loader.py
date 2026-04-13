# %% [data_loader] Load raw CSVs and produce train/val splits
#
# The split strategy is determined upstream — by the Planner reading profile.json
# signals from the Dataset Analyser (temporal indicators, class imbalance, etc.).
# This module only executes the strategy specified in config["split"]["method"].
# It makes no decisions about which strategy is appropriate.
#
# Supported methods (set by the Planner in iteration YAML → config.yaml):
#   stratified — random split preserving class proportions (classification)
#   random     — simple random split
#   temporal   — sort by time_column, split at a cutoff or val_ratio boundary
#                (no shuffling; preserves temporal ordering in both splits)

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training CSV and split into train/val according to config["split"].

    Args:
        config: Parsed config.yaml dict. Uses:
            config["data"]["train"]          — path to training CSV
            config["target_column"]          — name of the label column
            config["random_seed"]            — seed (stratified | random only)
            config["split"]["method"]        — stratified | random | temporal
            config["split"]["val_ratio"]     — val fraction (stratified | random | temporal fallback)
            config["split"]["time_column"]   — column to sort by (temporal only)
            config["split"]["cutoff"]        — ISO date cutoff string or null (temporal only)

    Returns:
        Tuple of (train_df, val_df). Both DataFrames retain all original columns
        including the target column.

    Raises:
        FileNotFoundError: If the training CSV does not exist.
        ValueError: If a required column or config key is missing for the chosen method.
    """
    train_path = Path(config["data"]["train"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_csv(train_path)

    target = config["target_column"]
    if target not in df.columns:
        raise ValueError(f"target_column '{target}' not found in {train_path}")

    method = config["split"]["method"]

    if method == "stratified":
        train_df, val_df = train_test_split(
            df,
            test_size=config["split"]["val_ratio"],
            random_state=config["random_seed"],
            stratify=df[target],
        )

    elif method == "random":
        train_df, val_df = train_test_split(
            df,
            test_size=config["split"]["val_ratio"],
            random_state=config["random_seed"],
        )

    elif method == "temporal":
        time_col = config["split"].get("time_column")
        if not time_col or time_col not in df.columns:
            raise ValueError(
                f"split.time_column '{time_col}' not found in dataset — required for temporal split"
            )
        df = df.sort_values(time_col).reset_index(drop=True)
        cutoff = config["split"].get("cutoff")
        if cutoff:
            mask = df[time_col] < cutoff
            train_df, val_df = df[mask], df[~mask]
        else:
            n = int(len(df) * (1 - config["split"]["val_ratio"]))
            train_df, val_df = df.iloc[:n], df.iloc[n:]

    else:
        raise ValueError(f"Unknown split method: '{method}'. Expected: stratified | random | temporal")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
