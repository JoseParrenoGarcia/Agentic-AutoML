# %% [main] Orchestration entry point — seed → load → features → train → evaluate
# Run from the iteration root: python src/main.py

import sys
import yaml
from pathlib import Path

# Add iteration src/ to path so sibling modules resolve without installation
sys.path.insert(0, str(Path(__file__).parent))

from utils import PipelineMetadata, capture_warnings, set_seed, setup_logging, timer
from data_loader import load_and_split
from feature_engineering import fit_transform, transform
from model import train_model
from evaluate import evaluate_model


def main() -> None:
    # --- Load config ---
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- Setup ---
    logger = setup_logging(config["output_paths"]["log"])
    logger.info("Iteration %d started", config["iteration"])

    set_seed(config["random_seed"])
    metadata = PipelineMetadata()

    # --- Load data ---
    logger.info("Loading data")
    with timer("data_loading", metadata), capture_warnings("data_loading", metadata):
        train_df, val_df = load_and_split(config)
    metadata.record_stage("data_loading", train_df.shape, val_df.shape)
    logger.info("Train shape: %s  Val shape: %s", train_df.shape, val_df.shape)

    # --- Feature engineering ---
    logger.info("Engineering features")
    target = config["target_column"]

    with timer("feature_engineering", metadata), capture_warnings("feature_engineering", metadata):
        train_df, fitted_params = fit_transform(train_df, config)
        val_df = transform(val_df, fitted_params, config)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    metadata.record_stage("feature_engineering", train_df.shape, X_train.shape)
    logger.info("Feature matrix shape: %s", X_train.shape)

    # --- Train ---
    logger.info("Training model")
    with timer("model_training", metadata), capture_warnings("model_training", metadata):
        model = train_model(X_train, y_train, config)
    metadata.record_stage("model_training", X_train.shape, X_train.shape)
    logger.info("Training complete: %s", type(model).__name__)

    # --- Evaluate ---
    logger.info("Evaluating model and writing artifacts")
    with timer("evaluation", metadata), capture_warnings("evaluation", metadata):
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val, config, metadata)

    primary = metrics.get("primary", {})
    logger.info(
        "Primary metric: %s = %.4f",
        primary.get("name", "unknown"),
        primary.get("value", float("nan")),
    )
    logger.info("Artifacts written to %s", config["output_paths"].get("metrics"))
    logger.info("Iteration %d complete", config["iteration"])


if __name__ == "__main__":
    main()
