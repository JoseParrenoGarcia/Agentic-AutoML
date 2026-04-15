# M5 — Execution & Debugging Loop

## Context

M4 (Coder) generates Python code in `iterations/iteration-<n>/`. Currently, running that code requires manual `python src/main.py`. M5 automates execution, captures results, classifies failures, and performs bounded LLM-driven self-repair. The Titanic iteration-1 smoke test (val AUC-ROC = 0.835) is the baseline to reproduce automatically.

## Design Decisions

1. **Single combined agent** — one `executor.md` handles both execution and debugging (no separate debugger agent)
2. **Python utilities + agent** — `src/execution/` modules the agent invokes, matching the `src/codegen/validator.py` pattern
3. **LLM-driven fixes with classification hints** — Python classifier categorizes errors, agent (LLM) reads error + source and patches
4. **Runner does NOT write log.txt** — generated code already writes `execution/log.txt` via `setup_logging()`. Runner captures stdout/stderr for error analysis and writes `manifest.json` only.

---

## M5.1 — Runner & Log Capture

**Files:** `src/execution/__init__.py`, `src/execution/runner.py`, `tests/execution/__init__.py`, `tests/execution/test_runner.py`

### `runner.py`

```python
@dataclass
class ExecutionResult:
    exit_code: int
    duration_s: float
    stdout: str
    stderr: str
    manifest_path: Path

class ExecutionError(Exception):
    """Raised when runner infrastructure fails (not when user code fails)."""

def run_iteration(iteration_dir, timeout_s=600) -> ExecutionResult:
```

Core logic:
1. Validate `iteration_dir` exists and contains `src/main.py`
2. Create `execution/` subdirectory
3. Run `subprocess.run([sys.executable, "src/main.py"], cwd=iteration_dir, capture_output=True, timeout=timeout_s, text=True)`
4. Compute duration via `time.monotonic()`
5. Collect package versions via `importlib.metadata` (pandas, scikit-learn, numpy — tolerant of missing)
6. Write `execution/manifest.json` (see schema below)
7. Return `ExecutionResult`

Non-zero exit is NOT an exception — it's a normal return with `exit_code != 0`. Only infrastructure errors (missing dir, can't write) raise.

### Tests (6)
1. Happy path: mini main.py exits 0 → manifest.json valid, exit_code 0
2. Non-zero exit: main.py does `sys.exit(1)` → exit_code 1, status "failed"
3. Stderr capture: main.py writes to stderr → captured in result
4. Timeout: `time.sleep(999)` with `timeout_s=1` → handled gracefully
5. Missing `src/main.py` → `ExecutionError`
6. Manifest schema: all required keys present and typed correctly

---

## M5.2 — Error Classifier

**Files:** `src/execution/classifier.py`, `tests/execution/test_classifier.py`

### `classifier.py`

```python
class ErrorCategory(str, Enum):
    SYNTAX_ERROR, IMPORT_ERROR, TYPE_ERROR,          # Stage 1
    DATA_SHAPE_ERROR, NAN_PROPAGATION,               # Stage 2
    CONVERGENCE_ERROR, RUNTIME_ERROR, TIMEOUT, UNKNOWN

class Stage(int, Enum):
    STAGE_1 = 1   # max 3 retries
    STAGE_2 = 2   # max 2 retries

TOTAL_ATTEMPT_CAP = 5

@dataclass
class ErrorClassification:
    category: ErrorCategory
    stage: Stage
    summary: str          # first ~500 chars of relevant error
    file_hint: str | None # e.g. "model.py"
    line_hint: int | None # e.g. 42

def classify_error(stderr: str, exit_code: int) -> ErrorClassification:
```

Classification rules (first match wins):
1. `SyntaxError` → SYNTAX_ERROR (Stage 1)
2. `ModuleNotFoundError` / `ImportError` → IMPORT_ERROR (Stage 1)
3. `TypeError` → TYPE_ERROR (Stage 1)
4. `shape` / `dimension` / `mismatch` / `broadcast` → DATA_SHAPE_ERROR (Stage 2)
5. NaN/nan/inf in ValueError → NAN_PROPAGATION (Stage 2)
6. `ConvergenceWarning` / `did not converge` → CONVERGENCE_ERROR (Stage 2)
7. exit_code == -9 or timeout → TIMEOUT (Stage 2)
8. Any other traceback → RUNTIME_ERROR (Stage 2)
9. No traceback → UNKNOWN (Stage 2)

Also extracts `file_hint` / `line_hint` from last `File "..."` line in traceback.

Pure functions, no I/O, stdlib only.

### Tests (10 parametrized)
Synthetic stderr strings for each category + traceback file/line extraction.

---

## M5.3 — Output Validator (Post-run)

**Files:** `src/execution/output_validator.py`, `tests/execution/test_output_validator.py`

### `output_validator.py`

```python
class OutputValidationError(Exception):
    """Raised when post-run artifacts fail schema validation."""

def validate_outputs(iteration_dir, task_type="binary_classification") -> dict:
```

Follows codegen validator pattern: raise-on-first-violation, return summary dict on success.

Checks (from Contract 5):
1. All 7 required artifact files exist
2. `metrics.json`: has `primary` (name + value), `secondary`, `train`, `validation`; values numeric
3. `predictions.csv`: correct columns per task_type; no NaN in required columns
4. `feature_importance.json`: has `method`, non-empty `features`, `sorted == true`, `model`
5. `learning_curves.json`: either has `note` key or matching-length arrays
6. `pipeline_metadata.json`: has `stages`, `total_duration_s`, `python_version`, `packages`
7. `model/metadata.json`: has `model_class`, non-empty `feature_list`, `training_timestamp`
8. `model/model.pkl`: non-zero file size

### Tests (8)
Happy path + one test per violation type with synthetic artifacts.

---

## M5.4 — Executor Agent + Integration Tests

**Files:** `.claude/agents/executor.md`, `tests/execution/conftest.py`, `tests/execution/test_integration.py`

### `executor.md`

Frontmatter:
```yaml
name: executor
description: >
  Runs generated iteration code, captures output, classifies failures,
  and attempts bounded self-repair. Invoke after the coder has written code.
  Invoke with: "run the executor agent on projects/<project-name>"
tools: [Read, Edit, Bash, Glob]
model: claude-sonnet-4-6
maxTurns: 30
```

**Scope Guardrails:**
- CAN: Read iteration files, Edit `src/` files (repairs only), Edit `requirements.txt`, Run Python via Bash, Create `execution/` artifacts
- CANNOT: Modify `artifacts/`, `project.yaml`; write outside iteration dir; exceed retry limits; change model architecture/hyperparameters/FE logic during repair

**Workflow:**
1. Pre-flight: run codegen validator → fail = Coder problem, stop
2. Install deps: `pip install -r requirements.txt`
3. Execute: invoke `runner.run_iteration()`
4. Success path: if exit_code == 0 → `validate_outputs()` → update manifest → Done
5. Debug loop (bounded):
   - Classify error → check budget (stage limits + total cap)
   - Read relevant source file (using file_hint)
   - Repair with minimal fix (no architecture changes)
   - Re-run → check for regression (rollback if new error introduced)
   - Append to `execution/retry-log.jsonl`
   - If exit_code == 0 → validate outputs
6. Terminal: update manifest with final status, retry_count, artifacts_validated

**Done message:**
- Success: `Done  Execution complete | iteration: <n> | status: success | retries: <count> | primary_metric: <value>`
- Failure: `Done  Execution FAILED | iteration: <n> | status: failed | retries: <count>/<cap> | error: <class>`

### Integration test
Run `run_iteration()` on `projects/titanic/iterations/iteration-1/` → assert exit_code 0, `validate_outputs()` passes, primary metric >= 0.80.

---

## Schemas

### `execution/manifest.json` (new — add to artifact-contracts.md as Contract 6)
```json
{
  "iteration": "<int>",
  "timestamp": "<ISO 8601>",
  "status": "success | failed",
  "exit_code": "<int>",
  "duration_s": "<float>",
  "python_version": "<str>",
  "packages": {"<name>": "<version>"},
  "error_class": "<str | null>",
  "error_summary": "<str | null>",
  "retry_count": "<int>",
  "artifacts_validated": "<bool>"
}
```

### `execution/retry-log.jsonl` (each line, new — add to Contract 6)
```json
{
  "attempt": "<int>",
  "timestamp": "<ISO 8601>",
  "error_class": "<str>",
  "stage": "<int>",
  "error_summary": "<str>",
  "file_modified": "<str>",
  "patch_description": "<str>",
  "outcome": "fixed | still_failing | regression"
}
```

---

## Implementation Order

```
M5.1  runner.py + tests          ← start here (no deps)
M5.2  classifier.py + tests      ← parallel with M5.1
M5.3  output_validator.py + tests ← parallel with M5.1/M5.2
      artifact-contracts.md update (Contract 6) ← with M5.3
M5.4  executor.md + conftest + integration tests ← depends on M5.1-M5.3
      CLAUDE.md update            ← after M5.4
```

M5.1, M5.2, M5.3 are independent and can be built in parallel.
M5.4 depends on all three.

## Files Summary (12 total)

| # | File | Sub-milestone |
|---|------|---------------|
| 1 | `src/execution/__init__.py` | M5.1 |
| 2 | `src/execution/runner.py` | M5.1 |
| 3 | `src/execution/classifier.py` | M5.2 |
| 4 | `src/execution/output_validator.py` | M5.3 |
| 5 | `.claude/agents/executor.md` | M5.4 |
| 6 | `.claude/rules/artifact-contracts.md` (update) | M5.3 |
| 7 | `tests/execution/__init__.py` | M5.1 |
| 8 | `tests/execution/test_runner.py` | M5.1 |
| 9 | `tests/execution/test_classifier.py` | M5.2 |
| 10 | `tests/execution/test_output_validator.py` | M5.3 |
| 11 | `tests/execution/conftest.py` | M5.4 |
| 12 | `tests/execution/test_integration.py` | M5.4 |

## Verification

1. All unit tests pass: `pytest tests/execution/ -v`
2. Integration: `run_iteration("projects/titanic/iterations/iteration-1/")` → success, AUC-ROC >= 0.80
3. Agent smoke test: "run the executor agent on projects/titanic" → produces manifest.json + validated outputs
4. CI green on the branch
