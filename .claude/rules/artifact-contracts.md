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
