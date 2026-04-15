import json

import pytest

from src.execution.runner import ExecutionError, ExecutionResult, run_iteration


def _write_main(iteration_dir, code):
    """Helper: write a src/main.py with the given code."""
    src = iteration_dir / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "main.py").write_text(code)


class TestRunIteration:
    def test_happy_path(self, tmp_path):
        _write_main(tmp_path, 'print("hello world")\n')
        result = run_iteration(tmp_path)
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.manifest_path.exists()
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "success"
        assert manifest["exit_code"] == 0
        assert manifest["artifacts_validated"] is False

    def test_nonzero_exit(self, tmp_path):
        _write_main(tmp_path, "import sys; sys.exit(1)\n")
        result = run_iteration(tmp_path)
        assert result.exit_code == 1
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "failed"

    def test_stderr_capture(self, tmp_path):
        _write_main(tmp_path, 'import sys; sys.stderr.write("oops\\n")\n')
        result = run_iteration(tmp_path)
        assert "oops" in result.stderr

    def test_timeout(self, tmp_path):
        _write_main(tmp_path, "import time; time.sleep(999)\n")
        result = run_iteration(tmp_path, timeout_s=1)
        assert result.exit_code == -9
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["status"] == "failed"

    def test_missing_main_py(self, tmp_path):
        with pytest.raises(ExecutionError, match="src/main.py not found"):
            run_iteration(tmp_path)

    def test_manifest_schema(self, tmp_path):
        _write_main(tmp_path, 'print("ok")\n')
        result = run_iteration(tmp_path)
        manifest = json.loads(result.manifest_path.read_text())
        required_keys = {
            "iteration", "timestamp", "status", "exit_code",
            "duration_s", "python_version", "packages",
            "error_class", "error_summary", "retry_count",
            "artifacts_validated",
        }
        assert required_keys.issubset(manifest.keys())
        assert isinstance(manifest["duration_s"], float)
        assert isinstance(manifest["packages"], dict)
        assert isinstance(manifest["retry_count"], int)
