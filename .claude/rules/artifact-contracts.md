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

Written by post-run hooks. Read by Planner on iteration > 1.

Append-only. Each line is a self-contained JSON object. Required fields per record: `iteration` (int), `timestamp` (ISO 8601), `status` (completed|failed), `plan_summary` (str), `primary_metric` (object with `name`, `value`, `delta`), `model_family` (str), `reviewer_verdict` (str), `router_decision` (str). Agents must not rewrite or delete existing lines.

## Contract 4: `model-report.json` (stub)

Written by Model Report Builder at M6. Required fields (stub): `iteration` (int), `primary_metric` (object), `secondary_metrics` (object), `feature_importance` (object), `overfitting_check` (object).

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
