"""
M6.4 — Report builder: orchestrates all evaluation functions to produce
model-report.json. Runnable as a module: python -m src.evaluation.report_builder <iter_dir> <project_dir>
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.evaluation.analysis import (
    compute_calibration,
    compute_error_analysis,
    compute_segment_analysis,
    compute_separation_quality,
    compute_threshold_analysis,
    identify_hardest_samples,
    repackage_feature_importance,
)
from src.evaluation.metrics import (
    classify_risk_flags,
    compute_bootstrap_ci,
    compute_headline_metrics,
    compute_leakage_indicators,
    compute_overfitting_check,
    compute_plateau_signal,
)
from src.evaluation.plots import generate_all_plots


def load_iteration_inputs(iteration_dir: Path, project_dir: Path) -> dict:
    """
    Load all Contract 5 artifacts + config.yaml + profile.json.

    Returns a dict with all parsed artifacts.
    """
    def _load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    config_path = iteration_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        "config": config,
        "config_path": config_path,
        "metrics": _load_json(iteration_dir / "outputs" / "metrics.json"),
        "feature_importance": _load_json(iteration_dir / "outputs" / "feature_importance.json"),
        "learning_curves": _load_json(iteration_dir / "outputs" / "learning_curves.json"),
        "pipeline_metadata": _load_json(iteration_dir / "outputs" / "pipeline_metadata.json"),
        "predictions_csv": iteration_dir / "outputs" / "predictions.csv",
        "profile": _load_json(project_dir / "artifacts" / "data" / "profile.json"),
    }


def _reconstruct_val_features(config: dict, config_path: Path) -> pd.DataFrame | None:
    """
    Re-load raw data and re-split to recover validation-set features.

    Same seed + same split method = identical val set.
    """
    iteration_dir = config_path.parent
    train_path = iteration_dir / config["data"]["train"]
    if not train_path.exists():
        return None

    raw_df = pd.read_csv(train_path)
    target = config["target_column"]
    method = config["split"]["method"]
    val_ratio = config["split"]["val_ratio"]
    seed = config["random_seed"]

    if method == "stratified":
        _, val_df = train_test_split(
            raw_df, test_size=val_ratio, random_state=seed, stratify=raw_df[target]
        )
    elif method == "random":
        _, val_df = train_test_split(
            raw_df, test_size=val_ratio, random_state=seed
        )
    elif method == "temporal":
        time_col = config["split"].get("time_column")
        raw_df = raw_df.sort_values(time_col).reset_index(drop=True)
        cutoff = config["split"].get("cutoff")
        if cutoff:
            val_df = raw_df[~(raw_df[time_col] < cutoff)]
        else:
            n = int(len(raw_df) * (1 - val_ratio))
            val_df = raw_df.iloc[n:]
    else:
        return None

    return val_df.reset_index(drop=True)


def compute_prior_run_comparison(
    iteration: int,
    iterations_dir: Path,
    current_metrics: dict,
) -> dict | None:
    """
    Compare current metrics against the previous iteration's report.

    Returns None on iteration 1 or if no prior report exists.
    """
    if iteration <= 1:
        return None

    prev_report_path = (
        iterations_dir / f"iteration-{iteration - 1}" / "reports" / "model-report.json"
    )
    if not prev_report_path.exists():
        return None

    with open(prev_report_path) as f:
        prev_report = json.load(f)

    prev_metrics = prev_report.get("headline_metrics", {})
    deltas = []

    # Primary metric
    curr_primary = current_metrics["primary"]
    prev_primary = prev_metrics.get("primary", {})
    if prev_primary.get("value") is not None:
        delta = curr_primary["value"] - prev_primary["value"]
        deltas.append({
            "metric": curr_primary["name"],
            "previous": prev_primary["value"],
            "current": curr_primary["value"],
            "delta": round(delta, 6),
            "improved": delta > 0,
        })

    # Secondary metrics
    curr_secondary = current_metrics.get("secondary", {})
    prev_secondary = prev_metrics.get("secondary", {})
    for name, curr_val in curr_secondary.items():
        prev_val = prev_secondary.get(name)
        if prev_val is not None:
            delta = curr_val - prev_val
            deltas.append({
                "metric": name,
                "previous": prev_val,
                "current": curr_val,
                "delta": round(delta, 6),
                "improved": delta > 0,
            })

    return {
        "previous_iteration": iteration - 1,
        "deltas": deltas,
    }


def compute_reviewer_summary(
    headline_metrics: dict,
    overfitting: dict,
    leakage: dict,
    risk_flags: list[dict],
    prior_comparison: dict | None,
    iteration: int,
    iterations_dir: Path,
) -> dict:
    """
    Compute the deterministic reviewer summary block.

    Verdict logic:
    - suspicious: if leakage detected
    - improved: if delta_vs_previous > 0
    - degraded: if delta_vs_previous < 0
    - neutral: iteration 1 or delta == 0
    """
    primary_value = headline_metrics["primary"]["value"]

    delta_vs_previous = None
    if prior_comparison and prior_comparison["deltas"]:
        primary_delta = prior_comparison["deltas"][0]
        delta_vs_previous = primary_delta["delta"]

    # Verdict
    if leakage["suspiciously_high_metric"]:
        verdict = "suspicious"
    elif delta_vs_previous is not None:
        if delta_vs_previous > 0:
            verdict = "improved"
        elif delta_vs_previous < 0:
            verdict = "degraded"
        else:
            verdict = "neutral"
    else:
        verdict = "neutral"

    plateau = compute_plateau_signal(iteration, iterations_dir)

    return {
        "headline_verdict": verdict,
        "metric_summary": {
            "primary_metric": primary_value,
            "delta_vs_previous": delta_vs_previous,
            "delta_vs_baseline": None,
        },
        "risk_flags": risk_flags,
        "plateau_signal": plateau,
    }


def build_model_report(iteration_dir: Path, project_dir: Path) -> dict:
    """
    Main orchestrator: build the complete model-report.json.

    Calls all M6.1-M6.3 functions, writes report to reports/model-report.json,
    and returns the full report dict.
    """
    iteration_dir = Path(iteration_dir)
    project_dir = Path(project_dir)

    inputs = load_iteration_inputs(iteration_dir, project_dir)
    config = inputs["config"]
    task_type = config["task_type"]
    iteration = config["iteration"]
    iterations_dir = iteration_dir.parent

    # M6.1: Core metrics
    headline = compute_headline_metrics(inputs["metrics"])
    overfitting = compute_overfitting_check(
        inputs["metrics"], inputs["learning_curves"], task_type
    )
    leakage = compute_leakage_indicators(
        inputs["metrics"], inputs["feature_importance"], task_type
    )
    risk_flags = classify_risk_flags(
        overfitting, leakage, inputs["metrics"], task_type
    )

    # M6.2: Analysis
    calibration = compute_calibration(inputs["predictions_csv"], task_type)
    segment_analysis = compute_segment_analysis(
        inputs["predictions_csv"],
        inputs["config_path"],
        inputs["profile"],
        inputs["feature_importance"],
        task_type,
    )
    error_analysis = compute_error_analysis(inputs["predictions_csv"], task_type)
    feature_importance = repackage_feature_importance(inputs["feature_importance"])
    threshold_analysis = compute_threshold_analysis(inputs["predictions_csv"], task_type)
    separation_quality = compute_separation_quality(inputs["predictions_csv"], task_type)
    hardest_samples = identify_hardest_samples(inputs["predictions_csv"], task_type)
    bootstrap_ci = compute_bootstrap_ci(
        inputs["predictions_csv"], inputs["metrics"], task_type
    )

    # Prior-run comparison
    prior_comparison = compute_prior_run_comparison(
        iteration, iterations_dir, inputs["metrics"]
    )

    # Reviewer summary
    reviewer_summary = compute_reviewer_summary(
        headline, overfitting, leakage, risk_flags,
        prior_comparison, iteration, iterations_dir,
    )

    # M6.3: Plots
    report_dir = iteration_dir / "reports"
    val_features = _reconstruct_val_features(config, inputs["config_path"])
    plots = generate_all_plots(
        inputs["predictions_csv"],
        calibration,
        inputs["feature_importance"],
        val_features,
        report_dir,
        task_type,
        threshold_analysis=threshold_analysis,
    )

    # Assemble report
    report = {
        "schema_version": "1.1.0",
        "iteration": iteration,
        "task_type": task_type,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "headline_metrics": headline,
        "overfitting_check": overfitting,
        "leakage_indicators": leakage,
        "calibration": calibration,
        "threshold_analysis": threshold_analysis,
        "separation_quality": separation_quality,
        "bootstrap_ci": bootstrap_ci,
        "segment_analysis": segment_analysis,
        "error_analysis": error_analysis,
        "hardest_samples": hardest_samples,
        "feature_importance": feature_importance,
        "prior_run_comparison": prior_comparison,
        "reviewer_summary": reviewer_summary,
        "plots": plots,
    }

    # Write
    report_path = report_dir / "model-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M6 model report builder")
    parser.add_argument("iteration_dir", help="Path to iteration directory")
    parser.add_argument("project_dir", help="Path to project directory")
    args = parser.parse_args()

    try:
        report = build_model_report(Path(args.iteration_dir), Path(args.project_dir))
        primary = report["headline_metrics"]["primary"]
        verdict = report["reviewer_summary"]["headline_verdict"]
        risk_count = len(report["reviewer_summary"]["risk_flags"])
        print(
            f"Report built: iteration={report['iteration']} "
            f"primary={primary['name']}={primary['value']:.4f} "
            f"verdict={verdict} risks={risk_count}"
        )
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
