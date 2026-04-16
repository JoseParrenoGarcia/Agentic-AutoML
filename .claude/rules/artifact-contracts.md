# Artifact Contracts

Unconditional. Applies to all agents that read or write structured artifacts. Agents must fail loudly (not silently skip) if any required key is missing.

## Contract 1: `profile.json`

Written by Dataset Analyser. Read by Planner.

Required top-level keys: `profiler_version`, `generated_at`, `source`, `columns`, `correlation`, `target_validation`, `leakage_flags`, `feature_risk_flags`, `mutual_information`.

Each item in `columns[]` must have: `name`, `pandas_dtype`, `inferred_semantic_type`, `description`, `sample_values`, `basic_stats`, `null_analysis`, `cardinality`, `risk_flags`.

## Contract 2: `iteration-<n>.yaml`

Written by Planner. Read by Coder.

Required fields: per `templates/plans/iteration.yaml`. Validated by `src/planning/validator.py` before the Coder agent reads it. Must be written to `artifacts/plans/iteration-<n>.yaml` (1-indexed).

## Contract 3: `run-history.jsonl`

Written by Reviewer-Router agent via `src/review/writer.py`. Read by Planner on iteration > 1. Validated by `src/review/validator.py`.

Append-only. Each line is a self-contained JSON object. Agents must not rewrite or delete existing lines.

Required fields per record: `iteration` (int), `timestamp` (ISO 8601), `status` (completed|failed), `plan_summary` (str), `primary_metric` (object with `name`, `value`, `delta`), `model_family` (str), `reviewer_verdict` (sufficient|insufficient), `reviewer_reasoning` (str), `router_decision` (continue|rollback|pivot), `router_reasoning` (str), `risk_flags_summary` (list of {type, severity, evidence}), `best_iteration` (int).

```json
{
  "iteration": "<int>",
  "timestamp": "<ISO 8601>",
  "status": "completed | failed",
  "plan_summary": "<str>",
  "primary_metric": {"name": "<str>", "value": "<float>", "delta": "<float | null>"},
  "model_family": "<str>",
  "reviewer_verdict": "sufficient | insufficient",
  "reviewer_reasoning": "<str>",
  "router_decision": "continue | rollback | pivot",
  "router_reasoning": "<str>",
  "risk_flags_summary": [{"type": "leakage | overfitting | underfitting | data_issue", "severity": "low | medium | high", "evidence": "<str>"}],
  "best_iteration": "<int>"
}
```

## Contract 4: `model-report.json`

Written by Model Report Builder (M6). Read by Reviewer (M7) and Action Router (M7). Path: `iterations/iteration-<n>/reports/model-report.json`. Validated by `src/evaluation/validator.py`.

Required top-level keys: `schema_version`, `iteration`, `task_type`, `generated_at`, `headline_metrics`, `overfitting_check`, `leakage_indicators`, `error_analysis`, `feature_importance`, `prior_run_comparison` (null on iter 1), `reviewer_summary`, `plots`.

Optional (classification only): `calibration`, `segment_analysis`.

```json
{
  "schema_version": "1.0.0",
  "iteration": "<int>",
  "task_type": "binary_classification | multiclass | regression",
  "generated_at": "<ISO 8601>",
  "headline_metrics": {
    "primary": {"name": "<str>", "value": "<float>"},
    "secondary": {"<metric_name>": "<float>"},
    "train": {"<metric_name>": "<float>"},
    "validation": {"<metric_name>": "<float>"}
  },
  "overfitting_check": {
    "train_val_gap": {"metric": "<str>", "train": "<float>", "val": "<float>", "gap": "<float>", "gap_pct": "<float>"},
    "severity": "low | medium | high",
    "learning_curve_trend": "improving | plateau | diverging | unavailable",
    "verdict": "<str>"
  },
  "leakage_indicators": {
    "suspiciously_high_metric": "<bool>",
    "feature_importance_anomalies": ["<str>"],
    "verdict": "<str>"
  },
  "calibration": {
    "brier_score": "<float>",
    "reliability_curve": {"bin_edges": ["<float>"], "mean_predicted": ["<float>"], "fraction_positive": ["<float>"], "bin_counts": ["<int>"]}
  },
  "segment_analysis": {
    "segments": [{"column": "<str>", "type": "categorical | numeric_binned", "slices": [{"value": "<str>", "n": "<int>", "primary_metric": "<float>", "accuracy": "<float>"}]}]
  },
  "error_analysis": {
    "confusion_matrix": {"tp": "<int>", "fp": "<int>", "fn": "<int>", "tn": "<int>"},
    "misclassification_patterns": [{"pattern": "<str>", "detail": "<str>"}],
    "error_rate_by_confidence": [{"confidence_bin": "<str>", "n": "<int>", "error_rate": "<float>"}]
  },
  "feature_importance": {
    "method": "<str>", "model": "<str>",
    "features": [{"name": "<str>", "importance": "<float>", "rank": "<int>"}]
  },
  "prior_run_comparison": null | {
    "previous_iteration": "<int>",
    "deltas": [{"metric": "<str>", "previous": "<float>", "current": "<float>", "delta": "<float>", "improved": "<bool>"}]
  },
  "reviewer_summary": {
    "headline_verdict": "improved | degraded | neutral | suspicious",
    "metric_summary": {"primary_metric": "<float>", "delta_vs_previous": "<float | null>", "delta_vs_baseline": "<float | null>"},
    "risk_flags": [{"type": "leakage | overfitting | underfitting | data_issue", "severity": "low | medium | high", "evidence": "<str>"}],
    "plateau_signal": {"detected": "<bool>", "consecutive_stale_iterations": "<int>"}
  },
  "plots": {
    "confusion_matrix": "<relative path | null>",
    "actual_vs_predicted": "<relative path>",
    "calibration_curve": "<relative path | null>",
    "error_distribution": "<relative path>",
    "feature_diagnostics": ["<relative path>"]
  }
}
```

## Contract 5: Iteration code outputs

Written by Coder agent (M4). Read by Executor (M5) and Model Report Builder (M6). All paths are relative to the iteration root. Agents must fail loudly if any required key or column is missing.

**`outputs/metrics.json`**
```json
{
  "primary":    {"name": "<str>", "value": "<float>"},
  "secondary":  {"<metric_name>": "<float>"},
  "train":      {"<metric_name>": "<float>"},
  "validation": {"<metric_name>": "<float>"}
}
```

**`outputs/predictions.csv`**
Required columns (binary classification): `index`, `y_true`, `y_pred`, `y_prob_0`, `y_prob_1`
Required columns (regression): `index`, `y_true`, `y_pred`

**`outputs/feature_importance.json`**
```json
{
  "method":   "<str>",
  "features": [{"name": "<str>", "importance": "<float>"}],
  "sorted":   true,
  "model":    "<str>"
}
```

**`outputs/learning_curves.json`**
For iterative models:
```json
{"metric_name": "<str>", "train": ["<float>"], "validation": ["<float>"], "iterations": ["<int>"]}
```
For non-iterative models (e.g. LogisticRegression):
```json
{"note": "model does not support iterative training"}
```

**`outputs/pipeline_metadata.json`**
```json
{
  "stages": [
    {"name": "<str>", "input_shape": ["<int>", "<int>"], "output_shape": ["<int>", "<int>"], "duration_s": "<float>", "warnings": ["<str>"]}
  ],
  "total_duration_s": "<float>",
  "python_version":   "<str>",
  "packages":         {"<name>": "<version>"}
}
```

**`outputs/model/model.pkl`** — joblib-serialised fitted model. Gitignored.

**`outputs/model/metadata.json`**
```json
{
  "model_class":          "<str>",
  "feature_list":         ["<str>"],
  "training_timestamp":   "<ISO 8601>",
  "n_train_samples":      "<int>"
}
```

**`config.yaml`** — required keys: `iteration`, `random_seed`, `target_column`, `task_type`, `data`, `split`, `hyperparameters`, `output_paths`. Validated by `src/codegen/validator.py`.

## Contract 6: Execution artifacts

Written by Executor agent (M5). Read by Reviewer (M6) and Planner (iteration > 1).

**`execution/manifest.json`**
```json
{
  "iteration":            "<int>",
  "timestamp":            "<ISO 8601>",
  "status":               "success | failed",
  "exit_code":            "<int>",
  "duration_s":           "<float>",
  "python_version":       "<str>",
  "packages":             {"<name>": "<version>"},
  "error_class":          "<str | null>",
  "error_summary":        "<str | null>",
  "retry_count":          "<int>",
  "artifacts_validated":  "<bool>"
}
```

**`execution/retry-log.jsonl`** — append-only, one JSON object per line. Written only when debug retries occur.
```json
{
  "attempt":           "<int>",
  "timestamp":         "<ISO 8601>",
  "error_class":       "<str>",
  "stage":             "<int — 1 or 2>",
  "error_summary":     "<str>",
  "file_modified":     "<str>",
  "patch_description": "<str>",
  "outcome":           "fixed | still_failing | regression"
}
```
