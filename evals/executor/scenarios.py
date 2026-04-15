"""
Executor agent eval scenarios.

Each scenario function takes a base iteration directory, copies it to a
target directory, and injects a specific failure. The executor agent is
then expected to classify the error and repair the code.

Usage:
    from evals.executor.scenarios import SCENARIOS, setup_scenario

    for name, scenario_fn in SCENARIOS.items():
        target = setup_scenario(name, scenario_fn, base_dir, work_dir)
        # invoke executor agent on target...
"""

import shutil
from pathlib import Path
from typing import Callable


def setup_scenario(
    name: str,
    inject_fn: Callable[[Path], dict],
    base_dir: Path,
    work_dir: Path,
) -> tuple[Path, dict]:
    """
    Copy base_dir to work_dir/<name>, fix data paths, and inject the failure.

    Returns:
        (scenario_dir, metadata) where metadata describes what was injected.
    """
    target = work_dir / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(base_dir, target)
    _absolutize_data_paths(target, base_dir)
    metadata = inject_fn(target)
    metadata["scenario"] = name
    return target, metadata


def _absolutize_data_paths(scenario_dir: Path, original_dir: Path) -> None:
    """Resolve relative data paths in config.yaml to absolute paths."""
    import yaml

    config_path = scenario_dir / "config.yaml"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    changed = False
    if "data" in config:
        for key in ("train", "test"):
            if key in config["data"]:
                rel = config["data"][key]
                abs_path = (original_dir / rel).resolve()
                if abs_path.exists():
                    config["data"][key] = str(abs_path)
                    changed = True

    if changed:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Scenario 1: Happy path — no injection
# ---------------------------------------------------------------------------

def inject_happy_path(iteration_dir: Path) -> dict:
    """No changes. Should succeed on first run."""
    return {
        "description": "Clean iteration, no failures injected",
        "expected_exit": 0,
        "expected_retries": 0,
    }


# ---------------------------------------------------------------------------
# Scenario 2: Syntax error — remove closing parenthesis
# ---------------------------------------------------------------------------

def inject_syntax_error(iteration_dir: Path) -> dict:
    """Remove a closing parenthesis in main.py to cause SyntaxError."""
    main_py = iteration_dir / "src" / "main.py"
    content = main_py.read_text()
    # Find the first setup_logging call and remove its closing paren
    original = "setup_logging(log_path)"
    broken = "setup_logging(log_path"
    if original not in content:
        # Fallback: break the first function call found
        content = content.replace(")", "", 1)
    else:
        content = content.replace(original, broken, 1)
    main_py.write_text(content)
    return {
        "description": "Removed closing paren from setup_logging() call in main.py",
        "expected_error_class": "syntax_error",
        "expected_stage": 1,
        "file_modified": "src/main.py",
    }


# ---------------------------------------------------------------------------
# Scenario 3: Import error — wrong class name
# ---------------------------------------------------------------------------

def inject_import_error(iteration_dir: Path) -> dict:
    """Rename LogisticRegression import to a non-existent class."""
    model_py = iteration_dir / "src" / "model.py"
    content = model_py.read_text()
    content = content.replace(
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.linear_model import LogisticRegressionXYZ",
    )
    # Also replace the constructor call
    content = content.replace(
        "model = LogisticRegression(",
        "model = LogisticRegressionXYZ(",
    )
    model_py.write_text(content)
    return {
        "description": "Changed LogisticRegression to LogisticRegressionXYZ in model.py",
        "expected_error_class": "import_error",
        "expected_stage": 1,
        "file_modified": "src/model.py",
    }


# ---------------------------------------------------------------------------
# Scenario 4: Bad column name — reference non-existent column
# ---------------------------------------------------------------------------

def inject_bad_column(iteration_dir: Path) -> dict:
    """Reference a non-existent column in feature_engineering.py."""
    fe_py = iteration_dir / "src" / "feature_engineering.py"
    content = fe_py.read_text()
    # Replace a real column reference with a fake one
    # "Age" is used in imputation — change to "AgeXYZ"
    content = content.replace('"Age"', '"AgeXYZ"')
    fe_py.write_text(content)
    return {
        "description": "Changed column 'Age' to 'AgeXYZ' in feature_engineering.py",
        "expected_error_class": "runtime_error",
        "expected_stage": 2,
        "file_modified": "src/feature_engineering.py",
    }


# ---------------------------------------------------------------------------
# Scenario 5: Invalid hyperparameter — add fake kwarg
# ---------------------------------------------------------------------------

def inject_invalid_param(iteration_dir: Path) -> dict:
    """Add a non-existent parameter to LogisticRegression constructor."""
    model_py = iteration_dir / "src" / "model.py"
    content = model_py.read_text()
    # Add fake_param=True to the constructor
    content = content.replace(
        "model = LogisticRegression(\n",
        "model = LogisticRegression(\n        fake_nonexistent_param=True,\n",
    )
    model_py.write_text(content)
    return {
        "description": "Added fake_nonexistent_param=True to LogisticRegression() in model.py",
        "expected_error_class": "type_error",
        "expected_stage": 1,
        "file_modified": "src/model.py",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, Callable[[Path], dict]] = {
    "happy_path": inject_happy_path,
    "syntax_error": inject_syntax_error,
    "import_error": inject_import_error,
    "bad_column": inject_bad_column,
    "invalid_param": inject_invalid_param,
}
