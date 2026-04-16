# Gate Checks Reference

Detailed validation commands for each agent step. All paths use `<project>` (project root relative to repo) and `<n>` (iteration number) as placeholders.

All validators use `.venv/bin/python3`. All raise specific `*ValidationError` on failure and return a summary `dict` on success.

---

## Dataset Analyser Gate

**Required artifacts:**
- `<project>/artifacts/data/profile.json`
- `<project>/artifacts/data/profile.md`
- `<project>/artifacts/data/plots/` (directory with at least 1 PNG)

**Validation command (no dedicated validator — structural check):**
```bash
.venv/bin/python3 -c "
import json, pathlib
p = json.load(open('<project>/artifacts/data/profile.json'))
required = ['profiler_version', 'generated_at', 'source', 'columns', 'correlation', 'target_validation', 'leakage_flags', 'feature_risk_flags', 'mutual_information']
missing = [k for k in required if k not in p]
assert not missing, f'Missing keys: {missing}'
plots = list(pathlib.Path('<project>/artifacts/data/plots').glob('*.png'))
print(f'OK: {len(p[\"columns\"])} columns profiled, {len(plots)} plots')
"
```

**On failure:** Retry agent once. If still failing, escalate.

---

## Planner Gate

**Required artifacts:**
- `<project>/artifacts/plans/iteration-<n>.yaml`
- `<project>/artifacts/plans/iteration-<n>.md`

**Discover N after planner returns:**
```bash
.venv/bin/python3 -c "
import pathlib
plans = sorted(pathlib.Path('<project>/artifacts/plans').glob('iteration-*.yaml'))
n = int(plans[-1].stem.split('-')[1])
print(f'ITERATION_N={n}')
"
```

**Validation command:**
```bash
.venv/bin/python3 -c "
from src.planning.validator import validate_plan
result = validate_plan('<project>/artifacts/plans/iteration-<n>.yaml')
print(f'OK: iteration={result[\"iteration\"]}')
"
```

**On failure:** Retry agent once. If still failing, escalate.

---

## Coder Gate

**Required artifacts:**
- `<project>/iterations/iteration-<n>/src/` (directory with Python files)
- `<project>/iterations/iteration-<n>/config.yaml`

**Validation command:**
```bash
.venv/bin/python3 -c "
from src.codegen.validator import validate_codegen
result = validate_codegen(
    '<project>/iterations/iteration-<n>',
    plan_path='<project>/artifacts/plans/iteration-<n>.yaml'
)
print(f'OK: {result[\"files_checked\"]} files checked, config valid')
"
```

**On failure:** Retry agent once. If still failing, escalate.

---

## Executor Gate

**Required artifacts:**
- `<project>/iterations/iteration-<n>/execution/manifest.json`
- `<project>/iterations/iteration-<n>/outputs/metrics.json`
- `<project>/iterations/iteration-<n>/outputs/predictions.csv`
- `<project>/iterations/iteration-<n>/outputs/feature_importance.json`
- `<project>/iterations/iteration-<n>/outputs/model/model.pkl`

**Step 1 — Check manifest status:**
```bash
.venv/bin/python3 -c "
import json
m = json.load(open('<project>/iterations/iteration-<n>/execution/manifest.json'))
if m['status'] != 'success':
    print(f'FAILED: status={m[\"status\"]}, error_class={m.get(\"error_class\")}, error_summary={m.get(\"error_summary\")}')
    exit(1)
print(f'OK: status=success, duration={m[\"duration_s\"]}s, retries={m[\"retry_count\"]}')
"
```

**Step 2 — Validate outputs (only if manifest status=success):**
```bash
.venv/bin/python3 -c "
from src.execution.output_validator import validate_outputs
result = validate_outputs('<project>/iterations/iteration-<n>', task_type='<task_type>')
print(f'OK: primary={result[\"metrics_primary\"]}, artifacts_checked={result[\"artifacts_checked\"]}')
"
```

**On failure:** Do NOT retry. Executor has internal 5-attempt retry budget. Escalate immediately with manifest error details.

---

## Model Report Builder Gate

**Required artifacts:**
- `<project>/iterations/iteration-<n>/reports/model-report.json`
- `<project>/iterations/iteration-<n>/reports/model-report.md`
- `<project>/iterations/iteration-<n>/reports/plots/` (directory)

**Validation command:**
```bash
.venv/bin/python3 -c "
from src.evaluation.validator import validate_report
result = validate_report('<project>/iterations/iteration-<n>/reports/model-report.json')
print(f'OK: verdict={result[\"headline_verdict\"]}, primary_metric={result[\"primary_metric\"]}, risk_flags={result[\"risk_flag_count\"]}')
"
```

**On failure:** Retry agent once. If still failing, escalate.

---

## Reviewer-Router Gate

**Required artifacts:**
- `<project>/memory/run-history.jsonl` (must have a new line appended)
- `<project>/iterations/iteration-<n>/reports/review-decision.json`
- `<project>/memory/decision-log.md` (must have new content)

**Count lines before invoking reviewer (for verification):**
```bash
wc -l < <project>/memory/run-history.jsonl 2>/dev/null || echo 0
```

**Validation command (after reviewer returns):**
```bash
.venv/bin/python3 -c "
import json
from src.review.validator import validate_review_decision
with open('<project>/memory/run-history.jsonl') as f:
    lines = f.readlines()
last = json.loads(lines[-1])
result = validate_review_decision(last)
print(f'OK: verdict={last[\"reviewer_verdict\"]}, route={last[\"router_decision\"]}, best={last[\"best_iteration\"]}, metric={last[\"primary_metric\"][\"value\"]}')
"
```

**Extract fields for progress report:**
- `reviewer_verdict` → stop condition check
- `router_decision` → reported in progress
- `primary_metric.name`, `primary_metric.value`, `primary_metric.delta`
- `best_iteration`
- `model_family`

**On failure:** Retry agent once. If still failing, escalate.
