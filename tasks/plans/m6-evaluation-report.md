# M6 Implementation Plan: Evaluation & Model Reporting

## Context

M5 (Execution & Debugging) is complete. The system runs iterations and produces raw outputs (metrics.json, predictions.csv, feature_importance.json, learning_curves.json, pipeline_metadata.json, execution/manifest.json). M6 transforms these into a comprehensive evaluation package — `model-report.json` + `model-report.md` + plots — so M7 (Reviewer) can make decisions without any computation.

**Design decisions confirmed with Jose:**
- M6 pre-computes a deterministic verdict (improved/degraded/neutral/suspicious) + risk flags. M7 can override with contextual reasoning.
- The model-report-builder agent writes the narrative model-report.md (follows dataset-analyser pattern).
- Start with consolidated files (~5 src files), split if any exceeds 300 lines.

---

## Key Design Decisions

### D1: Segment analysis data access
**Problem**: predictions.csv only has y_true/y_pred/y_prob — no feature columns for slicing.
**Solution**: Re-load raw data via config.yaml, re-split with same seed = identical val set. Join val features with predictions on index. Contract 5 stays unchanged.

### D2: Which columns to slice by
Use profile.json `inferred_semantic_type` and `cardinality`:
- Categorical/ordinal with unique_count <= 10
- Top-2 numeric columns by importance, binned into quartiles

### D3: Per-feature diagnostic plots — top 5 by importance

### D4: model-report.json is self-contained for M7
M7 reads ONLY model-report.json + model-report.md. Never reads metrics.json or feature_importance.json directly.

### D5: Prior-run comparison
Iteration 1: all delta fields are null. Iteration > 1: reads previous `reports/model-report.json`.

---

## Contract 4: `model-report.json` Full Schema

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
    "train_val_gap": {
      "metric": "<str>", "train": "<float>", "val": "<float>",
      "gap": "<float>", "gap_pct": "<float>"
    },
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
    "reliability_curve": {
      "bin_edges": ["<float>"], "mean_predicted": ["<float>"],
      "fraction_positive": ["<float>"], "bin_counts": ["<int>"]
    }
  },

  "segment_analysis": {
    "segments": [{
      "column": "<str>", "type": "categorical | numeric_binned",
      "slices": [{"value": "<str>", "n": "<int>", "primary_metric": "<float>", "accuracy": "<float>"}]
    }]
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
    "metric_summary": {
      "primary_metric": "<float>",
      "delta_vs_previous": "<float | null>",
      "delta_vs_baseline": "<float | null>"
    },
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

Required top-level keys: `schema_version`, `iteration`, `task_type`, `generated_at`, `headline_metrics`, `overfitting_check`, `leakage_indicators`, `error_analysis`, `feature_importance`, `prior_run_comparison` (null on iter 1), `reviewer_summary`, `plots`.
Optional (classification only): `calibration`, `segment_analysis`.

---

## Milestone Breakdown

### M6.1: Core analysis functions
**File: `src/evaluation/metrics.py`** (~200 lines)

Functions:
- `compute_headline_metrics(metrics_json: dict) -> dict` — repackage metrics.json
- `compute_overfitting_check(metrics_json: dict, learning_curves_json: dict, task_type: str) -> dict` — train/val gap, gap_pct, severity (<5% low, 5-15% medium, >15% high), learning_curve_trend
- `compute_leakage_indicators(metrics_json: dict, feature_importance_json: dict, task_type: str) -> dict` — primary > 0.99 = suspicious, single feature > 80% total importance = anomaly
- `classify_risk_flags(overfitting: dict, leakage: dict, metrics: dict, task_type: str) -> list[dict]` — aggregates {type, severity, evidence}
- `compute_plateau_signal(iteration: int, prior_reports_dir: Path) -> dict` — reads previous model-report.json files, delta < 0.005 = stale

**File: `src/evaluation/analysis.py`** (~250 lines)

Functions:
- `compute_calibration(predictions_csv: Path, task_type: str) -> dict | None` — classification only, sklearn calibration_curve + brier_score_loss
- `select_segment_columns(profile_json: dict, feature_importance_json: dict) -> list[dict]` — picks categorical (unique <= 10) + top-2 numeric
- `compute_segment_analysis(predictions_csv: Path, config_yaml: Path, profile_json: dict, feature_importance_json: dict, task_type: str) -> dict` — re-loads data, re-splits, joins, computes per-slice metrics
- `compute_error_analysis(predictions_csv: Path, task_type: str) -> dict` — confusion matrix (binary), residual stats (regression), error rate by confidence bin
- `repackage_feature_importance(feature_importance_json: dict) -> dict` — adds rank field

**File: `src/evaluation/__init__.py`** — empty

**Tests: `tests/evaluation/test_metrics.py`, `tests/evaluation/test_analysis.py`**
- Titanic assertions: gap=3.5%, severity=low, no leakage, no anomalies
- Calibration: Brier score from 178 predictions
- Segments: Sex (2 vals), Pclass (3 vals) as categorical slices
- Confusion matrix sums to ~178

---

### M6.2: Plots
**File: `src/evaluation/plots.py`** (~250 lines)

Follows `src/analysis/plots.py` patterns: matplotlib Agg backend, `_save` helper, STYLE/DPI constants.

Functions:
- `plot_confusion_matrix(predictions_csv, out_dir, task_type) -> Path | None` — classification only
- `plot_actual_vs_predicted(predictions_csv, out_dir, task_type) -> Path` — probability histogram (binary), scatter (regression)
- `plot_calibration_curve(calibration_data, out_dir) -> Path | None` — classification only
- `plot_error_distribution(predictions_csv, out_dir, task_type) -> Path` — confidence vs error rate (binary), residual histogram (regression)
- `plot_feature_diagnostics(val_features_df, predictions_df, feature_importance, out_dir, task_type, top_n=5) -> list[Path]` — box plot of feature values for correct vs misclassified
- `generate_all_plots(predictions_csv, calibration_data, feature_importance, val_features_df, out_dir, task_type) -> dict` — orchestrator, returns plots block for report

**Tests: `tests/evaluation/test_plots.py`**
- Assert PNGs created, non-zero file sizes
- Correct count (4 standard + up to 5 feature diagnostics)

---

### M6.3: Report builder, validator, agent, contract update
**File: `src/evaluation/report_builder.py`** (~180 lines)

Functions:
- `load_iteration_inputs(iteration_dir: Path, project_dir: Path) -> dict` — loads all Contract 5 artifacts + config.yaml + resolves profile.json path
- `build_model_report(iteration_dir: Path, project_dir: Path) -> dict` — main orchestrator: calls metrics.py + analysis.py + plots.py, assembles model-report.json, writes to `reports/model-report.json`
- `compute_prior_run_comparison(iteration: int, iterations_dir: Path, current_metrics: dict) -> dict | None` — null on iter 1, else loads previous report + computes deltas
- `compute_reviewer_summary(headline_metrics, overfitting, leakage, prior_comparison, iteration, prior_reports_dir) -> dict` — deterministic verdict: improved (delta > 0), degraded (delta < 0), neutral (iter 1 or delta == 0), suspicious (leakage detected)

**File: `src/evaluation/validator.py`** (~120 lines)

- `class ReportValidationError(Exception)`
- `validate_report(report: Path | dict) -> dict` — checks all required keys, valid enum values, numeric types. Returns summary dict. Raises ReportValidationError.

**File: `.claude/agents/model-report-builder.md`** (~100 lines)

Agent spec:
```yaml
name: model-report-builder
description: >
  Produces the evaluation package for an iteration. Runs deterministic Python
  analysis scripts (src/evaluation/) then writes a human-readable narrative
  (model-report.md). Invoke after executor completes successfully.
  Invoke with: "run the model-report-builder agent on projects/<project-name>"
tools: Read, Write, Bash, Glob
model: claude-sonnet-4-6
maxTurns: 15
```

Workflow (6 steps):
1. **Locate iteration** — Read project.yaml, find latest iteration with execution/manifest.json status=success
2. **Run report builder** — `python -m src.evaluation.report_builder <iteration_dir> <project_dir>`
3. **Validate report** — `python -m src.evaluation.validator <report_path>`
4. **Read model-report.json** — load the structured report
5. **Write model-report.md** — agent writes interpretive narrative with plot references (agentic part, like dataset-analyser writes profile.md)
6. **Report completion** — print iteration, primary metric, verdict, risk flag count, plot count

**Update: `.claude/rules/artifact-contracts.md`** — replace Contract 4 stub with full schema

**Tests: `tests/evaluation/test_report_builder.py`, `tests/evaluation/test_validator.py`**

---

## Dependency Graph & Build Order

```
M6.1  metrics.py + analysis.py (core analysis functions)
  │
  ├── M6.2  plots.py (needs calibration_data, val_features_df from analysis.py)
  │
  └── M6.3  report_builder.py + validator.py + agent + contract update
              (needs all of M6.1 + M6.2)
```

**Recommended sequence:** M6.1 → M6.2 → M6.3

---

## File Inventory

### New files (7 src + 5 test + 1 agent = 13 files):

| File | Type | Est. Lines |
|------|------|-----------|
| `src/evaluation/__init__.py` | empty | 0 |
| `src/evaluation/metrics.py` | Python | ~200 |
| `src/evaluation/analysis.py` | Python | ~250 |
| `src/evaluation/plots.py` | Python | ~250 |
| `src/evaluation/report_builder.py` | Python | ~180 |
| `src/evaluation/validator.py` | Python | ~120 |
| `.claude/agents/model-report-builder.md` | Agent | ~100 |
| `tests/evaluation/__init__.py` | empty | 0 |
| `tests/evaluation/test_metrics.py` | Test | ~150 |
| `tests/evaluation/test_analysis.py` | Test | ~150 |
| `tests/evaluation/test_plots.py` | Test | ~80 |
| `tests/evaluation/test_report_builder.py` | Test | ~120 |
| `tests/evaluation/test_validator.py` | Test | ~80 |

### Modified files:

| File | Change |
|------|--------|
| `.claude/rules/artifact-contracts.md` | Replace Contract 4 stub with full schema |

---

## Verification Strategy

### Against Titanic iteration-1:

1. **headline_metrics**: primary = {name: "val_auc_roc", value: 0.835}
2. **overfitting**: gap = 0.035, gap_pct ~ 4.02%, severity = "low"
3. **leakage**: suspicious = false (0.835 < 0.99), no feature anomalies
4. **calibration**: Brier score from 178 predictions with y_prob_1
5. **segments**: Sex (2 vals), Pclass (3 vals) as categorical slices
6. **errors**: confusion matrix sums to ~178
7. **importance**: 9 features, rank 1-9, Sex at rank 1
8. **prior_run_comparison**: null (iteration 1)
9. **reviewer_summary**: verdict = "neutral", delta_vs_previous = null, 0 risk flags
10. **plots**: ~9 PNGs (confusion matrix, actual_vs_predicted, calibration, error_dist, 5 feature diagnostics)

### Integration test:
```python
def test_full_report_titanic():
    report = build_model_report(
        Path("projects/titanic/iterations/iteration-1"),
        Path("projects/titanic")
    )
    summary = validate_report(report)
    assert summary["headline_verdict"] == "neutral"
    assert summary["risk_flag_count"] == 0
    assert report["headline_metrics"]["primary"]["value"] == pytest.approx(0.835, abs=0.001)
    assert report["prior_run_comparison"] is None
    assert Path("projects/titanic/iterations/iteration-1/reports/model-report.json").exists()
    assert Path("projects/titanic/iterations/iteration-1/reports/plots/confusion_matrix.png").exists()
```

### Agent smoke test:
Run `model-report-builder` agent on `projects/titanic`. Verify:
- `reports/model-report.json` validates
- `reports/model-report.md` has all expected sections
- `reports/plots/` has ~9 PNGs
- No errors in agent output

---

## Existing code to reuse

| What | Where | How |
|------|-------|-----|
| Plot style/save pattern | `src/analysis/plots.py` | Copy Agg backend, `_save` helper, DPI/STYLE constants |
| Validator pattern | `src/execution/output_validator.py` | Same structure: accept path/dict, raise specific error, return summary |
| Agent frontmatter | `.claude/agents/executor.md` | Same YAML format, similar workflow step structure |
| Narrative writing pattern | `.claude/agents/dataset-analyser.md` | Agent reads structured JSON then writes interpretive .md |
| Data re-split for segments | `templates/iteration/data_loader.py` | Same split logic (stratified, same seed) |
