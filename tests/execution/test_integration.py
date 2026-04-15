import json
from pathlib import Path

import pytest

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


@pytest.mark.skipif(
    not TITANIC_ITERATION_1.exists(),
    reason="Titanic iteration-1 not present",
)
class TestTitanicIntegration:
    """End-to-end test against the real Titanic iteration-1."""

    def test_titanic_runs_successfully(self):
        result = run_iteration(TITANIC_ITERATION_1)
        assert result.exit_code == 0, f"Failed with stderr:\n{result.stderr[:1000]}"

    def test_titanic_outputs_valid(self):
        validation = validate_outputs(TITANIC_ITERATION_1, task_type="binary_classification")
        assert validation["metrics_primary"]["value"] >= 0.80

    def test_titanic_manifest_written(self):
        result = run_iteration(TITANIC_ITERATION_1)
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "success"
        assert manifest["iteration"] == 1
