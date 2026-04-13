import ast
import re
from pathlib import Path
from typing import Union

import yaml


class CodegenValidationError(Exception):
    """Raised when a generated iteration directory fails structural validation."""


_REQUIRED_FILES = [
    "src/main.py",
    "src/data_loader.py",
    "src/feature_engineering.py",
    "src/model.py",
    "src/evaluate.py",
    "src/utils.py",
    "config.yaml",
    "requirements.txt",
]

_REQUIRED_CONFIG_KEYS = [
    "iteration",
    "random_seed",
    "target_column",
    "task_type",
    "data",
    "split",
    "hyperparameters",
    "output_paths",
]

_HARDCODED_PATH_PATTERN = re.compile(r"(/Users/|/home/|C:\\\\)")


def validate_codegen(
    iteration_dir: Union[str, Path],
    plan_path: Union[str, Path, None] = None,
) -> dict:
    """
    Validate a generated iteration directory against structural requirements.

    Args:
        iteration_dir: Path to the iteration root (e.g. projects/titanic/iterations/iteration-1/).
        plan_path: Optional path to the iteration YAML. When provided, enables the
                   feature-step count sanity check (check 6).

    Returns:
        Summary dict with keys: iteration_dir, files_checked, config_keys_present,
        python_files_parsed, feature_step_check.

    Raises:
        CodegenValidationError: Descriptive message on the first violation found.
        FileNotFoundError: If iteration_dir does not exist.
    """
    root = Path(iteration_dir)
    if not root.exists():
        raise FileNotFoundError(f"Iteration directory not found: {root}")

    # --- Check 1: Required files exist ---
    for rel in _REQUIRED_FILES:
        if not (root / rel).exists():
            raise CodegenValidationError(
                f"Required file missing: {rel} (expected at {root / rel})"
            )

    # --- Check 2: config.yaml has all required keys ---
    config_path = root / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise CodegenValidationError("config.yaml did not parse to a dict")
    for key in _REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise CodegenValidationError(
                f"config.yaml missing required key: '{key}'"
            )

    # --- Check 3: All Python files parse without SyntaxError ---
    python_files = list((root / "src").glob("*.py"))
    for py_file in python_files:
        source = py_file.read_text()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise CodegenValidationError(
                f"SyntaxError in {py_file.relative_to(root)}: {exc}"
            ) from exc

    # --- Check 4: No hardcoded absolute paths ---
    for py_file in python_files:
        source = py_file.read_text()
        if _HARDCODED_PATH_PATTERN.search(source):
            raise CodegenValidationError(
                f"Hardcoded absolute path found in {py_file.relative_to(root)}. "
                "All paths must be sourced from config.yaml."
            )

    # --- Check 5: main.py contains if __name__ guard ---
    main_source = (root / "src" / "main.py").read_text()
    if "__name__" not in main_source:
        raise CodegenValidationError(
            "src/main.py is missing the 'if __name__ == \"__main__\"' guard"
        )

    # --- Check 6: Feature step count sanity (optional) ---
    feature_step_check = "skipped (no plan path provided)"
    if plan_path is not None:
        plan_file = Path(plan_path)
        if not plan_file.exists():
            raise FileNotFoundError(f"Plan file not found: {plan_file}")
        with open(plan_file) as f:
            plan = yaml.safe_load(f)
        plan_step_count = len(plan.get("feature_steps", []))

        fe_source = (root / "src" / "feature_engineering.py").read_text()
        # Count df assignments that look like transform operations
        transform_count = len(re.findall(r"df\[.+?\]\s*=|df\s*=\s*df\.", fe_source))

        if plan_step_count > 0:
            ratio = transform_count / plan_step_count
            if ratio < 0.5:
                feature_step_check = (
                    f"WARNING: plan has {plan_step_count} feature steps but only "
                    f"{transform_count} transform operations detected in feature_engineering.py"
                )
            else:
                feature_step_check = (
                    f"ok (plan={plan_step_count} steps, detected={transform_count} transforms)"
                )
        else:
            feature_step_check = "ok (plan has 0 feature steps)"

    return {
        "iteration_dir": str(root),
        "files_checked": _REQUIRED_FILES,
        "config_keys_present": _REQUIRED_CONFIG_KEYS,
        "python_files_parsed": [str(f.relative_to(root)) for f in python_files],
        "feature_step_check": feature_step_check,
    }
