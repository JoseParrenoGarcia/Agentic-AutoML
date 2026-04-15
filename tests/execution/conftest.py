import textwrap

import pytest


@pytest.fixture
def valid_iteration_dir(tmp_path):
    """Minimal iteration directory that runs successfully and produces valid outputs."""
    src = tmp_path / "src"
    src.mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    model_dir = outputs / "model"
    model_dir.mkdir()
    execution = tmp_path / "execution"
    execution.mkdir()

    (tmp_path / "config.yaml").write_text(textwrap.dedent("""\
        iteration: 1
        random_seed: 42
        target_column: target
        task_type: binary_classification
        data:
          train: data/train.csv
          test: data/test.csv
        split:
          method: stratified
          val_ratio: 0.2
        hyperparameters: {}
        output_paths:
          metrics: outputs/metrics.json
          predictions: outputs/predictions.csv
          feature_importance: outputs/feature_importance.json
          learning_curves: outputs/learning_curves.json
          pipeline_metadata: outputs/pipeline_metadata.json
          model: outputs/model/model.pkl
          model_metadata: outputs/model/metadata.json
          log: execution/log.txt
    """))

    (tmp_path / "requirements.txt").write_text("")

    # main.py that writes all required artifacts
    (src / "main.py").write_text(textwrap.dedent("""\
        import json, csv, os, sys
        from pathlib import Path

        root = Path(__file__).parent.parent
        os.makedirs(root / "outputs" / "model", exist_ok=True)
        os.makedirs(root / "execution", exist_ok=True)

        (root / "outputs" / "metrics.json").write_text(json.dumps({
            "primary": {"name": "auc_roc", "value": 0.85},
            "secondary": {"accuracy": 0.82},
            "train": {"auc_roc": 0.90},
            "validation": {"auc_roc": 0.85},
        }))

        with open(root / "outputs" / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "y_true", "y_pred", "y_prob_0", "y_prob_1"])
            w.writerow([0, 1, 1, 0.15, 0.85])

        (root / "outputs" / "feature_importance.json").write_text(json.dumps({
            "method": "coefficients", "features": [{"name": "x", "importance": 1.0}],
            "sorted": True, "model": "LR",
        }))

        (root / "outputs" / "learning_curves.json").write_text(json.dumps({
            "note": "model does not support iterative training",
        }))

        (root / "outputs" / "pipeline_metadata.json").write_text(json.dumps({
            "stages": [], "total_duration_s": 0.1, "python_version": "3.11",
            "packages": {},
        }))

        (root / "outputs" / "model" / "metadata.json").write_text(json.dumps({
            "model_class": "LR", "feature_list": ["x"],
            "training_timestamp": "2026-01-01T00:00:00Z", "n_train_samples": 10,
        }))

        (root / "outputs" / "model" / "model.pkl").write_bytes(b"\\x80\\x05fake")

        print("done")

        if __name__ == "__main__":
            pass
    """))

    return tmp_path


@pytest.fixture
def failing_syntax_iteration(tmp_path):
    """Iteration with a SyntaxError in main.py."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def broken(\n")
    return tmp_path


@pytest.fixture
def failing_import_iteration(tmp_path):
    """Iteration with a missing import."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("import nonexistent_package_xyz\n")
    return tmp_path
