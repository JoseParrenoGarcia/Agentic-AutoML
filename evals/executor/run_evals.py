"""
Executor agent eval runner.

Sets up scenarios, runs the executor utilities (runner + validator),
and reports pass/fail for each. This tests the Python infrastructure
that the executor agent relies on.

For full agent-level evals (testing the LLM's ability to diagnose and
repair), run each scenario via a subagent in a worktree.

Usage:
    # From repo root:
    .venv/bin/python evals/executor/run_evals.py

    # Run a single scenario:
    .venv/bin/python evals/executor/run_evals.py --scenario happy_path
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.executor.scenarios import SCENARIOS, setup_scenario
from src.execution.classifier import classify_error
from src.execution.output_validator import OutputValidationError, validate_outputs
from src.execution.runner import run_iteration


BASE_ITERATION = REPO_ROOT / "projects" / "titanic" / "iterations" / "iteration-1"
WORK_DIR = REPO_ROOT / "evals" / "executor" / "results"


def run_single_eval(name: str, inject_fn, base_dir: Path, work_dir: Path) -> dict:
    """Run a single eval scenario. Returns a result dict."""
    result = {
        "scenario": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "setup_ok": False,
        "run_ok": False,
        "classification_ok": False,
        "output_validation_ok": False,
        "passed": False,
        "details": {},
    }

    # Step 1: Setup
    try:
        scenario_dir, metadata = setup_scenario(name, inject_fn, base_dir, work_dir)
        result["setup_ok"] = True
        result["details"]["metadata"] = metadata
    except Exception as e:
        result["details"]["setup_error"] = str(e)
        return result

    # Step 2: Run
    try:
        exec_result = run_iteration(scenario_dir, timeout_s=120)
        result["run_ok"] = True
        result["details"]["exit_code"] = exec_result.exit_code
        result["details"]["duration_s"] = exec_result.duration_s
    except Exception as e:
        result["details"]["run_error"] = str(e)
        return result

    # Step 3: Classification (for failure scenarios)
    expected_exit = metadata.get("expected_exit", None)
    expected_error_class = metadata.get("expected_error_class", None)

    if exec_result.exit_code != 0 and expected_error_class:
        classification = classify_error(exec_result.stderr, exec_result.exit_code)
        result["details"]["classified_as"] = classification.category.value
        result["details"]["expected_class"] = expected_error_class
        result["details"]["stage"] = classification.stage.value
        result["details"]["file_hint"] = classification.file_hint
        result["details"]["summary"] = classification.summary[:200]
        result["classification_ok"] = classification.category.value == expected_error_class
    elif exec_result.exit_code == 0 and expected_exit == 0:
        result["classification_ok"] = True  # No error to classify

    # Step 4: Output validation (only for successful runs)
    if exec_result.exit_code == 0:
        try:
            validate_outputs(scenario_dir, task_type="binary_classification")
            result["output_validation_ok"] = True
        except OutputValidationError as e:
            result["details"]["validation_error"] = str(e)

    # Step 5: Overall pass/fail
    if expected_exit == 0:
        # Happy path: must run successfully and pass validation
        result["passed"] = (
            result["run_ok"]
            and exec_result.exit_code == 0
            and result["output_validation_ok"]
        )
    else:
        # Failure scenario: must fail AND classify correctly
        result["passed"] = (
            result["run_ok"]
            and exec_result.exit_code != 0
            and result["classification_ok"]
        )

    return result


def print_report(results: list[dict]):
    """Print a formatted eval report."""
    print("\n" + "=" * 70)
    print("EXECUTOR EVAL REPORT")
    print("=" * 70)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        icon = "v" if r["passed"] else "x"
        scenario = r["scenario"]
        details = r["details"]

        print(f"\n  [{icon}] {status}  {scenario}")
        if "exit_code" in details:
            print(f"      exit_code: {details['exit_code']}")
        if "classified_as" in details:
            expected = details.get("expected_class", "?")
            actual = details["classified_as"]
            match = "==" if actual == expected else "!="
            print(f"      classified: {actual} {match} expected {expected}")
        if "validation_error" in details:
            print(f"      validation: {details['validation_error'][:100]}")
        if "summary" in details:
            print(f"      error: {details['summary'][:100]}")

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {passed}/{total} scenarios passed")
    print(f"{'=' * 70}\n")

    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Run executor eval scenarios")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="Run a single scenario (default: all)",
    )
    args = parser.parse_args()

    if not BASE_ITERATION.exists():
        print(f"ERROR: Base iteration not found at {BASE_ITERATION}")
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios_to_run = SCENARIOS

    results = []
    for name, inject_fn in scenarios_to_run.items():
        print(f"Running scenario: {name}...", end=" ", flush=True)
        result = run_single_eval(name, inject_fn, BASE_ITERATION, WORK_DIR)
        results.append(result)
        print("PASS" if result["passed"] else "FAIL")

    # Save results to JSON
    results_file = WORK_DIR / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    all_passed = print_report(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
