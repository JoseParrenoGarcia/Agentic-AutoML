import json

import pytest

from src.execution.output_validator import OutputValidationError, validate_outputs


def _write_valid_artifacts(root, task_type="binary_classification"):
    """Create a minimal set of valid Contract 5 artifacts."""
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    model_dir = outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    (outputs / "metrics.json").write_text(json.dumps({
        "primary": {"name": "auc_roc", "value": 0.85},
        "secondary": {"accuracy": 0.82},
        "train": {"auc_roc": 0.90},
        "validation": {"auc_roc": 0.85},
    }))

    if task_type == "binary_classification":
        lines = [
            "index,y_true,y_pred,y_prob_0,y_prob_1",
            "0,1,1,0.15,0.85",
            "1,0,0,0.90,0.10",
        ]
    else:
        lines = [
            "index,y_true,y_pred",
            "0,3.5,3.4",
            "1,2.1,2.0",
        ]
    (outputs / "predictions.csv").write_text("\n".join(lines) + "\n")

    (outputs / "feature_importance.json").write_text(json.dumps({
        "method": "coefficients",
        "features": [{"name": "age", "importance": 0.5}],
        "sorted": True,
        "model": "LogisticRegression",
    }))

    (outputs / "learning_curves.json").write_text(json.dumps({
        "note": "model does not support iterative training",
    }))

    (outputs / "pipeline_metadata.json").write_text(json.dumps({
        "stages": [{"name": "fe", "input_shape": [100, 5], "output_shape": [100, 8], "duration_s": 0.1, "warnings": []}],
        "total_duration_s": 0.5,
        "python_version": "3.11.0",
        "packages": {"pandas": "2.0.0"},
    }))

    (model_dir / "metadata.json").write_text(json.dumps({
        "model_class": "LogisticRegression",
        "feature_list": ["age", "fare"],
        "training_timestamp": "2026-04-14T07:00:00Z",
        "n_train_samples": 100,
    }))

    (model_dir / "model.pkl").write_bytes(b"\x80\x05fake_pickle_data")


class TestValidateOutputs:
    def test_happy_path(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        result = validate_outputs(tmp_path)
        assert result["task_type"] == "binary_classification"
        assert result["metrics_primary"]["name"] == "auc_roc"

    def test_missing_metrics(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        (tmp_path / "outputs" / "metrics.json").unlink()
        with pytest.raises(OutputValidationError, match="metrics.json"):
            validate_outputs(tmp_path)

    def test_invalid_metrics_schema(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        # primary exists but missing name/value sub-keys
        (tmp_path / "outputs" / "metrics.json").write_text(json.dumps({
            "primary": {},
            "secondary": {},
            "train": {},
            "validation": {},
        }))
        with pytest.raises(OutputValidationError, match="primary.*name.*value"):
            validate_outputs(tmp_path)

    def test_predictions_nan(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        (tmp_path / "outputs" / "predictions.csv").write_text(
            "index,y_true,y_pred,y_prob_0,y_prob_1\n0,1,nan,0.5,0.5\n"
        )
        with pytest.raises(OutputValidationError, match="NaN"):
            validate_outputs(tmp_path)

    def test_predictions_wrong_columns(self, tmp_path):
        _write_valid_artifacts(tmp_path, task_type="regression")
        # File has binary columns but we validate as regression
        (tmp_path / "outputs" / "predictions.csv").write_text(
            "index,y_true,y_pred,y_prob_0,y_prob_1\n0,1,1,0.1,0.9\n"
        )
        # Regression expects index, y_true, y_pred — this should pass
        # since those columns ARE present. Let's test missing columns instead.
        (tmp_path / "outputs" / "predictions.csv").write_text(
            "index,y_pred\n0,3.5\n"
        )
        with pytest.raises(OutputValidationError, match="missing columns"):
            validate_outputs(tmp_path, task_type="regression")

    def test_feature_importance_not_sorted(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        (tmp_path / "outputs" / "feature_importance.json").write_text(json.dumps({
            "method": "coefficients",
            "features": [{"name": "age", "importance": 0.5}],
            "sorted": False,
            "model": "LogisticRegression",
        }))
        with pytest.raises(OutputValidationError, match="sorted.*true"):
            validate_outputs(tmp_path)

    def test_empty_model_pkl(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        (tmp_path / "outputs" / "model" / "model.pkl").write_bytes(b"")
        with pytest.raises(OutputValidationError, match="model.pkl.*empty"):
            validate_outputs(tmp_path)

    def test_missing_pipeline_metadata_key(self, tmp_path):
        _write_valid_artifacts(tmp_path)
        (tmp_path / "outputs" / "pipeline_metadata.json").write_text(json.dumps({
            "stages": [],
        }))
        with pytest.raises(OutputValidationError, match="pipeline_metadata.*total_duration_s"):
            validate_outputs(tmp_path)
