import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.execution.classifier import ErrorCategory, Stage, classify_error
from src.execution.output_validator import validate_outputs
from src.execution.runner import run_iteration


class TestFixtureIntegration:
    """Integration tests using conftest fixtures."""

    def test_valid_iteration_runs_and_validates(self, valid_iteration_dir):
        result = run_iteration(valid_iteration_dir)
        assert result.exit_code == 0

        validation = validate_outputs(valid_iteration_dir)
        assert validation["metrics_primary"]["value"] == 0.85

        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "success"

    def test_syntax_failure_classified(self, failing_syntax_iteration):
        result = run_iteration(failing_syntax_iteration)
        assert result.exit_code != 0

        classification = classify_error(result.stderr, result.exit_code)
        assert classification.category == ErrorCategory.SYNTAX_ERROR
        assert classification.stage == Stage.STAGE_1

    def test_import_failure_classified(self, failing_import_iteration):
        result = run_iteration(failing_import_iteration)
        assert result.exit_code != 0

        classification = classify_error(result.stderr, result.exit_code)
        assert classification.category == ErrorCategory.IMPORT_ERROR
        assert classification.stage == Stage.STAGE_1


TITANIC_ITERATION_1 = Path("projects/titanic/iterations/iteration-1")


@pytest.fixture
def titanic_copy(tmp_path):
    """Copy iteration-1 to a temp dir with absolutized data paths.

    Skips if the iteration directory or its raw data files are missing
    (data is gitignored, so CI won't have it).
    """
    if not TITANIC_ITERATION_1.exists():
        pytest.skip("Titanic iteration-1 not present")

    # Check that raw data exists (gitignored — not available in CI)
    train_csv = (TITANIC_ITERATION_1 / "../../data/raw/train.csv").resolve()
    if not train_csv.exists():
        pytest.skip("Titanic raw data not present (gitignored)")

    target = tmp_path / "iteration-1"
    shutil.copytree(TITANIC_ITERATION_1, target)

    # Absolutize data paths so the copy can find the CSV files
    config_path = target / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "data" in config:
        for key in ("train", "test"):
            if key in config["data"]:
                abs_path = (TITANIC_ITERATION_1 / config["data"][key]).resolve()
                config["data"][key] = str(abs_path)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return target


class TestTitanicIntegration:
    """End-to-end test against a disposable copy of Titanic iteration-1."""

    def test_titanic_runs_successfully(self, titanic_copy):
        result = run_iteration(titanic_copy)
        assert result.exit_code == 0, f"Failed with stderr:\n{result.stderr[:1000]}"

    def test_titanic_outputs_valid(self, titanic_copy):
        run_iteration(titanic_copy)
        validation = validate_outputs(titanic_copy, task_type="binary_classification")
        assert validation["metrics_primary"]["value"] >= 0.80

    def test_titanic_manifest_written(self, titanic_copy):
        result = run_iteration(titanic_copy)
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "success"
        assert manifest["iteration"] == 1
