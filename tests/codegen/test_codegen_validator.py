from pathlib import Path

import pytest
import yaml

from src.codegen.validator import CodegenValidationError, validate_codegen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CONFIG = {
    "iteration": 1,
    "random_seed": 42,
    "target_column": "Survived",
    "task_type": "binary_classification",
    "data": {"train": "../../data/raw/train.csv", "test": None},
    "split": {"method": "stratified", "val_ratio": 0.2},
    "hyperparameters": {"C": 1.0, "solver": "lbfgs"},
    "output_paths": {
        "metrics": "outputs/metrics.json",
        "predictions": "outputs/predictions.csv",
        "log": "execution/log.txt",
    },
}

_MINIMAL_MAIN = """\
import sys

def main():
    pass

if __name__ == "__main__":
    main()
"""

_MINIMAL_PY = """\
def placeholder():
    pass
"""


def _make_valid_iteration(tmp_path: Path) -> Path:
    """Create a structurally complete valid iteration directory."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text(_MINIMAL_MAIN)
    for name in ("data_loader", "feature_engineering", "model", "evaluate", "utils"):
        (src / f"{name}.py").write_text(_MINIMAL_PY)

    (tmp_path / "config.yaml").write_text(yaml.dump(_VALID_CONFIG))
    (tmp_path / "requirements.txt").write_text("scikit-learn\npandas\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_complete_directory(tmp_path):
    """A structurally complete directory passes; returns a summary dict."""
    iteration_dir = _make_valid_iteration(tmp_path)
    result = validate_codegen(iteration_dir)
    assert isinstance(result, dict)
    assert "files_checked" in result
    assert "config_keys_present" in result
    assert "python_files_parsed" in result


def test_missing_main_py(tmp_path):
    """Missing src/main.py raises CodegenValidationError mentioning 'main.py'."""
    iteration_dir = _make_valid_iteration(tmp_path)
    (iteration_dir / "src" / "main.py").unlink()
    with pytest.raises(CodegenValidationError, match="main.py"):
        validate_codegen(iteration_dir)


def test_missing_config_yaml(tmp_path):
    """Missing config.yaml raises CodegenValidationError mentioning 'config.yaml'."""
    iteration_dir = _make_valid_iteration(tmp_path)
    (iteration_dir / "config.yaml").unlink()
    with pytest.raises(CodegenValidationError, match="config.yaml"):
        validate_codegen(iteration_dir)


def test_config_missing_required_key(tmp_path):
    """config.yaml without 'random_seed' raises CodegenValidationError mentioning the key."""
    iteration_dir = _make_valid_iteration(tmp_path)
    config = dict(_VALID_CONFIG)
    del config["random_seed"]
    (iteration_dir / "config.yaml").write_text(yaml.dump(config))
    with pytest.raises(CodegenValidationError, match="random_seed"):
        validate_codegen(iteration_dir)


def test_python_syntax_error(tmp_path):
    """A Python file with a syntax error raises CodegenValidationError mentioning 'SyntaxError'."""
    iteration_dir = _make_valid_iteration(tmp_path)
    (iteration_dir / "src" / "model.py").write_text("def broken(\n    pass\n")
    with pytest.raises(CodegenValidationError, match="SyntaxError"):
        validate_codegen(iteration_dir)


def test_hardcoded_absolute_path(tmp_path):
    """A Python file with a hardcoded /Users/ path raises CodegenValidationError mentioning 'hardcoded'."""
    iteration_dir = _make_valid_iteration(tmp_path)
    (iteration_dir / "src" / "data_loader.py").write_text(
        'path = "/Users/jose/data/train.csv"\n'
    )
    with pytest.raises(CodegenValidationError, match="[Hh]ardcoded"):
        validate_codegen(iteration_dir)


def test_empty_feature_engineering_valid(tmp_path):
    """feature_engineering.py with no transforms is valid (empty feature_steps is allowed)."""
    iteration_dir = _make_valid_iteration(tmp_path)
    (iteration_dir / "src" / "feature_engineering.py").write_text(
        "def engineer_features(df, config):\n    return df\n"
    )
    result = validate_codegen(iteration_dir)
    assert isinstance(result, dict)


def test_nonexistent_directory_raises_file_not_found(tmp_path):
    """Passing a path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        validate_codegen(tmp_path / "does_not_exist")
