# M4 — Plan-to-Code Layer

## Context

M4 bridges the gap between experiment plans (iteration YAML from M3) and executable Python code. The Planner has produced `iteration-1.yaml` for the Titanic project. M4 must translate that plan into working code, run it, debug any failures, and produce rich artifacts that a future Validator (M5) can evaluate.

**Three components, two agents + one harness:**
- **Coder Agent** — LLM reads YAML plan, writes Python code
- **Executor Harness** — deterministic Python, runs subprocess, captures logs
- **Debugger Agent** — LLM-powered 2-stage DS-STAR debugging

---

## Sub-milestones

### M4.1 — Coder Agent + Infrastructure

**Goal:** Create the Coder agent file and define all schemas/conventions for generated code.

**Deliverables:**

1. **`.claude/agents/coder.md`** — Agent file with:
   - Frontmatter: `name: coder`, `tools: Read, Write, Glob`, `model: sonnet`, `maxTurns: 20`
   - Scope guardrails (CAN: write code; CANNOT: execute code, modify plans)
   - Step-by-step workflow:
     1. Read inputs (iteration YAML, project.yaml, profile.json, coding-rules.md)
     2. Resolve paths (project root, data paths, output paths)
     3. Generate config.yaml
     4. Generate 6 Python files in order: utils.py, data_loader.py, feature_engineering.py, model.py, evaluate.py, main.py
     5. Generate requirements.txt
     6. Self-validate (check all files exist, check imports resolve, check config paths)
   - Output contract: `✓ Code generated` with file count, iteration number, path

2. **config.yaml schema** — the Coder generates this per run:
   ```yaml
   iteration: 1
   seed: 42
   project_root: "../../"          # relative from runs/iteration-<n>/
   data:
     train: "../../data/raw/train.csv"
   target_column: "Survived"
   task_type: "binary_classification"
   split:
     method: "stratified"
     val_ratio: 0.2
   output_paths:
     metrics: "outputs/metrics.json"
     predictions: "outputs/predictions.csv"
     learning_curves: "outputs/learning_curves.json"
     feature_importance: "outputs/feature_importance.json"
     pipeline_metadata: "outputs/pipeline_metadata.json"
     log: "execution/log.txt"
   ```

3. **Output artifact schemas** (documented in the agent file as guidance):

   **metrics.json** (per coding-rules.md):
   ```json
   {
     "primary": {"name": "roc_auc", "value": 0.85},
     "secondary": {"accuracy": 0.82, "f1": 0.79, "precision": 0.81, "recall": 0.77},
     "train": {"roc_auc": 0.87, "accuracy": 0.84},
     "validation": {"roc_auc": 0.85, "accuracy": 0.82}
   }
   ```

   **predictions.csv**: columns = `index, y_true, y_pred, y_prob` (for classification)

   **learning_curves.json**:
   ```json
   {
     "model": "LogisticRegression",
     "curves": [{"epoch": 1, "train_loss": null, "val_metric": 0.85}],
     "note": "Single-step model; one data point"
   }
   ```

   **feature_importance.json**:
   ```json
   {
     "type": "coefficient",
     "features": ["Sex", "Fare_log1p", "..."],
     "values": [2.31, 0.45, "..."],
     "model": "LogisticRegression"
   }
   ```

   **pipeline_metadata.json**:
   ```json
   {
     "stages": [
       {"name": "data_loading", "input_shape": [891, 12], "output_shape": [891, 12], "duration_s": 0.05, "warnings": []},
       {"name": "feature_engineering", "input_shape": [891, 12], "output_shape": [712, 9], "duration_s": 0.1, "warnings": []},
       {"name": "training", "input_shape": [712, 9], "output_shape": null, "duration_s": 0.3, "warnings": ["ConvergenceWarning: ..."]}
     ],
     "split_indices": {"train": [0, 2, 5], "val": [1, 3, 7]},
     "total_duration_s": 1.2,
     "python_version": "3.11.5",
     "packages": {"pandas": "2.1.0", "scikit-learn": "1.3.0"}
   }
   ```

4. **Directory structure** — created by the Coder:
   ```
   projects/titanic/runs/iteration-1/
   ├── src/
   │   ├── main.py
   │   ├── data_loader.py
   │   ├── feature_engineering.py
   │   ├── model.py
   │   ├── evaluate.py
   │   └── utils.py
   ├── config.yaml
   └── requirements.txt
   ```
   (`outputs/` and `execution/` are created at runtime by the code itself)

**Verification:** Invoke the coder agent on `projects/titanic` with iteration-1. Check that all 8 files are created in the correct directory structure.

---

### M4.2 — Working Code Generation

**Goal:** The Coder produces Python code that runs to completion and emits all artifacts.

**Deliverables:**

1. **data_loader.py** — reads train.csv via config path, performs stratified train/val split, returns DataFrames
2. **feature_engineering.py** — implements all 11 feature_steps from iteration-1.yaml (drops, imputations, encodings, transforms, passthroughs). Applied identically to train and val sets via a single `transform(df)` function.
3. **model.py** — instantiates LogisticRegression with exact hyperparameters from YAML, fits on train, returns trained model
4. **evaluate.py** — computes AUC-ROC (primary), accuracy/f1/precision/recall (secondary), on both train and val. Writes metrics.json, predictions.csv, feature_importance.json, learning_curves.json
5. **utils.py** — logging setup (stdout + log file), timing context manager, warning capture, seed setting, pipeline metadata collector
6. **main.py** — orchestrates the pipeline: seed, load, split, feature eng, train, evaluate, write pipeline_metadata.json. All wrapped in try/except that logs and exits non-zero on failure.

**Key implementation details:**
- Feature engineering must use a function that takes a DataFrame and returns a transformed DataFrame — same function applied to train and val (no leakage)
- All paths resolved from config.yaml (no hardcoding)
- Logging via Python `logging` module with dual handler (stdout + file)
- Warnings captured via `warnings.catch_warnings(record=True)`
- Timing via `time.perf_counter()` wrapped in a context manager

**Verification:**
```bash
cd projects/titanic/runs/iteration-1
../../.venv/bin/python src/main.py
# Check: exit code 0
# Check: outputs/metrics.json exists with correct schema
# Check: outputs/predictions.csv exists with correct columns
# Check: outputs/learning_curves.json exists
# Check: outputs/feature_importance.json exists
# Check: outputs/pipeline_metadata.json exists
# Check: execution/log.txt has content
```

---

### M4.3 — Executor Harness

**Goal:** Deterministic Python module that runs generated code and captures structured metadata.

**Deliverables:**

1. **`src/execution/__init__.py`**

2. **`src/execution/runner.py`** — public API:
   ```python
   def run_iteration(run_dir: Path, timeout: int = 300) -> RunResult:
       """
       Run python src/main.py in subprocess, capture all output.
       Returns RunResult with exit_code, log_path, manifest_path, error_class.
       """
   ```

   **RunResult** dataclass:
   ```python
   @dataclass
   class RunResult:
       exit_code: int
       status: str          # "success" | "syntax_error" | "import_error" | "runtime_error"
       log_path: Path
       manifest_path: Path
       error_summary: str | None   # first 500 chars of traceback if failed
       duration_s: float
   ```

   **Behaviour:**
   - Creates `execution/` dir if not exists
   - Runs `python src/main.py` via `subprocess.run()` with `capture_output=True, text=True, timeout=timeout, cwd=run_dir`
   - Writes stdout+stderr to `execution/log.txt`
   - Classifies error:
     - `SyntaxError`, `IndentationError` → `"syntax_error"`
     - `ImportError`, `ModuleNotFoundError` → `"import_error"`
     - Everything else non-zero → `"runtime_error"`
     - Exit 0 → `"success"`
   - Writes `execution/manifest.json`:
     ```json
     {
       "iteration": 1,
       "status": "success",
       "exit_code": 0,
       "start_time": "2026-04-13T10:00:00Z",
       "end_time": "2026-04-13T10:00:02Z",
       "duration_s": 1.8,
       "python_version": "3.11.5",
       "error_class": null,
       "error_summary": null
     }
     ```

3. **Error classification function** (internal):
   ```python
   def classify_error(stderr: str) -> str:
       """Parse stderr/traceback to determine error class."""
   ```
   Uses regex on the last traceback line to identify the error type.

**Verification:** Run the executor on the iteration-1 code from M4.2. Check manifest.json is written correctly. Also test with a deliberately broken file to verify error classification.

---

### M4.4 — Debugger Agent + Retry Loop

**Goal:** LLM agent that reads failure logs, summarizes them, and patches code. Plus retry orchestration.

**Deliverables:**

1. **`.claude/agents/debugger.md`** — Agent file with:
   - Frontmatter: `name: debugger`, `tools: Read, Edit, Grep, Bash`, `model: sonnet`, `maxTurns: 8`
   - Two-stage workflow:

   **Stage 1 — Syntax & Import Repair** (max 3 retries):
   - Read `execution/log.txt` (tail ~50 lines)
   - Identify the exact error: missing import, syntax error, indentation
   - Apply targeted fix via Edit tool (minimal, surgical patches — not full rewrites)

   **Stage 2 — Logic & Runtime Repair** (max 2 retries):
   - Read `execution/log.txt` (tail ~100 lines for context)
   - Read the failing source file identified in traceback
   - Summarize the error in 2-3 sentences (DS-STAR: compress traceback before reasoning)
   - Diagnose root cause (data shape mismatch, NaN, type error, convergence)
   - Apply fix via Edit tool

   **Output contract:**
   ```
   ✓ Fix applied | stage: 1|2 | error_class: ... | file_changed: <path> | summary: <one-line>
   ```
   OR
   ```
   ✗ Cannot fix | stage: 1|2 | error_class: ... | attempts: <n> | reason: <why>
   ```

2. **`src/execution/retry.py`** — Retry orchestration (deterministic Python):
   ```python
   def retry_loop(run_dir: Path, max_stage1: int = 3, max_stage2: int = 2) -> RetryResult:
       """
       Run → classify → dispatch to debugger → re-run → repeat.
       """
   ```

   **RetryResult** dataclass:
   ```python
   @dataclass
   class RetryResult:
       final_status: str       # "success" | "failed"
       total_attempts: int
       stage1_attempts: int
       stage2_attempts: int
       retry_log_path: Path
   ```

   **Behaviour:**
   - Calls `runner.run_iteration()` first
   - If success → return immediately
   - If failure → classify → dispatch to debugger agent (Stage 1 or 2)
   - After fix → re-run → check
   - Separate counters per stage
   - Append each attempt to `execution/retry-log.jsonl`:
     ```json
     {"attempt": 1, "stage": 1, "error_class": "import_error", "error_summary": "...", "fix_summary": "...", "result": "success"}
     ```
   - On exhaustion → structured failure artifact

**Design note:** `retry.py` invokes the debugger agent. The natural mechanism: retry logic lives within an orchestrating Claude agent context that calls the Agent tool. Alternative: standalone Python calling `claude` CLI. To be resolved during implementation.

**Verification:**
- Introduce a syntax error → verify Stage 1 fixes it in ≤3 attempts
- Introduce a runtime error (wrong column name) → verify Stage 2 fixes it
- Exhaust retries → verify structured failure artifact written

---

### M4.5 — End-to-End Integration + Smoke Test

**Goal:** Full pipeline runs cleanly from YAML plan to artifacts.

**Deliverables:**

1. **End-to-end verification checklist:**
   - [ ] Invoke Coder on `projects/titanic` for iteration-1
   - [ ] `runs/iteration-1/src/` has 6 .py files
   - [ ] `runs/iteration-1/config.yaml` and `requirements.txt` present
   - [ ] Run Executor → `execution/manifest.json` status=success
   - [ ] If failure → Debugger fixes it within retry budget
   - [ ] `outputs/metrics.json` matches schema
   - [ ] `outputs/predictions.csv` has columns: index, y_true, y_pred, y_prob
   - [ ] `outputs/learning_curves.json` present
   - [ ] `outputs/feature_importance.json` present
   - [ ] `outputs/pipeline_metadata.json` present
   - [ ] `execution/log.txt` has content
   - [ ] No hardcoded paths (grep test)
   - [ ] Seed recorded in config.yaml
   - [ ] No internet imports
   - [ ] Feature engineering applied to both train and val

2. **run-history.jsonl append** — on success, append to `projects/titanic/memory/run-history.jsonl`:
   ```json
   {
     "iteration": 1,
     "timestamp": "2026-04-13T12:00:00Z",
     "status": "completed",
     "plan_summary": "Logistic regression baseline with binary encoding, median imputation, log1p fare",
     "primary_metric": {"name": "roc_auc", "value": 0.85, "delta": null},
     "model_family": "LogisticRegression",
     "reviewer_verdict": "pending",
     "router_decision": "pending"
   }
   ```

3. **CLAUDE.md update** — reflect M4 completion status.

---

## File creation/modification summary

| File | Action | Sub-milestone |
|------|--------|---------------|
| `.claude/agents/coder.md` | Create | M4.1 |
| `.claude/agents/debugger.md` | Create | M4.4 |
| `src/execution/__init__.py` | Create | M4.3 |
| `src/execution/runner.py` | Create | M4.3 |
| `src/execution/retry.py` | Create | M4.4 |
| `projects/titanic/runs/iteration-1/src/*.py` (6 files) | Generated by Coder | M4.2 |
| `projects/titanic/runs/iteration-1/config.yaml` | Generated by Coder | M4.2 |
| `projects/titanic/runs/iteration-1/requirements.txt` | Generated by Coder | M4.2 |
| `projects/titanic/memory/run-history.jsonl` | Append | M4.5 |
| `.claude/CLAUDE.md` | Update | M4.5 |

## Dependencies

```
M4.1 (Coder agent + infra)
  └──→ M4.2 (Working code gen)
         └──→ M4.3 (Executor harness)
                └──→ M4.4 (Debugger + retry)
                       └──→ M4.5 (E2E integration)
```

## Open decisions (resolve during implementation)

1. **How does retry.py invoke the debugger agent?** Orchestrating Claude agent context (Agent tool) vs standalone Python calling `claude` CLI.
2. **Virtual environment for runs:** Use project-root `.venv/` (already has pandas, sklearn). Generated `requirements.txt` documents deps but doesn't create a separate venv.
