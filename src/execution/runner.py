import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


class ExecutionError(Exception):
    """Raised when runner infrastructure fails (not when user code fails)."""


@dataclass
class ExecutionResult:
    exit_code: int
    duration_s: float
    stdout: str
    stderr: str
    manifest_path: Path


def _collect_packages() -> dict:
    """Collect versions of common ML packages. Tolerant of missing ones."""
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("pandas", "scikit-learn", "numpy", "xgboost", "lightgbm", "joblib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return packages


def _parse_iteration_number(iteration_dir: Path) -> int:
    """Extract iteration number from directory name like 'iteration-3'."""
    name = iteration_dir.name
    if name.startswith("iteration-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            pass
    return -1


def run_iteration(
    iteration_dir: Union[str, Path],
    timeout_s: int = 600,
) -> ExecutionResult:
    """
    Execute ``python src/main.py`` from the iteration root and capture output.

    Args:
        iteration_dir: Path to iteration root
            (e.g. ``projects/titanic/iterations/iteration-1/``).
        timeout_s: Max wall-clock seconds before killing the process.

    Returns:
        ExecutionResult with exit code, duration, captured output,
        and path to the written manifest.

    Raises:
        ExecutionError: If infrastructure fails (directory missing main.py,
            cannot write manifest).
        FileNotFoundError: If *iteration_dir* does not exist.
    """
    root = Path(iteration_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Iteration directory not found: {root}")

    main_py = root / "src" / "main.py"
    if not main_py.exists():
        raise ExecutionError(
            f"src/main.py not found in {root}. "
            "Was the Coder agent run first?"
        )

    exec_dir = root / "execution"
    exec_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    exit_code: int
    stdout = ""
    stderr = ""

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [python, "src/main.py"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = -9
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    duration_s = round(time.monotonic() - t0, 3)

    status = "success" if exit_code == 0 else "failed"
    error_summary = stderr[:500].strip() if stderr else None

    manifest = {
        "iteration": _parse_iteration_number(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "exit_code": exit_code,
        "duration_s": duration_s,
        "python_version": platform.python_version(),
        "packages": _collect_packages(),
        "error_class": None,
        "error_summary": error_summary if status == "failed" else None,
        "retry_count": 0,
        "artifacts_validated": False,
    }

    manifest_path = exec_dir / "manifest.json"
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except OSError as exc:
        raise ExecutionError(f"Cannot write manifest: {exc}") from exc

    return ExecutionResult(
        exit_code=exit_code,
        duration_s=duration_s,
        stdout=stdout,
        stderr=stderr,
        manifest_path=manifest_path,
    )
