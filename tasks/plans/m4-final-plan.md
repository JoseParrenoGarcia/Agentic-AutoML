# M4 — Plan-to-Code Layer: Final Implementation Plan

## Context

M4 bridges experiment plans (iteration YAML from M3) and executable Python code. The Planner has produced `iteration-1.yaml` for the Titanic project. M4 must translate that plan into structurally valid, syntactically correct Python code that a future Executor (M5) can run.

**Scope boundary:** M4 produces code. M5 runs and debugs it. This matches PRD §17 exactly.

**Key design inputs:**
- PRD §6.3 (Coder agent spec), §6.4 (Executor spec — forward reference)
- DS-STAR: structural templates + incremental code building + Verifier pattern
- Karpathy autoresearch: single editable scope, grep-friendly metrics, redirect stdout to log
- AutoKaggle: multi-phase pipeline separation, atomic file writes
- Existing patterns: `src/planning/validator.py` (custom exception + single validate function + early-fail), pytest conventions with `tmp_path` and `pytest.raises`

**Directory naming:** `iterations/` (user preference). Requires updating coding-rules.md path scope.

---

## Sub-Milestones

### M4.1 — Base Code Templates (`templates/iteration/`)

Create structural reference files that show the Coder agent the expected code organization, imports, function signatures, and output contracts. These are **not** Jinja templates — they are reference documents the LLM reads for structure.

**Files to create:**

| File | Purpose | Key contract |
|------|---------|-------------|
| `templates/iteration/main.py` | Orchestration skeleton | Entry point: seed -> load -> features -> train -> evaluate. Dual logging (stdout + file). Warning capture. Pipeline metadata collection. |
| `templates/iteration/data_loader.py` | `load_and_split(config)` function | Reads paths from config.yaml. Stratified split for classification. Returns train/val DataFrames. |
| `templates/iteration/feature_engineering.py` | `engineer_features(df, config)` function | Same function applied to train AND val. Returns transformed DataFrame. One function per feature step group (drops -> imputation -> encoding -> transforms -> passthrough). |
| `templates/iteration/model.py` | `train_model(X_train, y_train, config)` function | Returns trained model + train predictions. Captures learning curves when model supports callbacks (null otherwise). |
| `templates/iteration/evaluate.py` | `evaluate_model(model, X_train, y_train, X_val, y_val, config)` function | Computes all metrics. Writes: metrics.json, predictions.csv, feature_importance.json, learning_curves.json. Saves model artifact. |
| `templates/iteration/utils.py` | Shared utilities | Logging dual-handler setup. Timing context manager. Warning capture context manager. Seed setting. Pipeline metadata collector class. |
| `templates/iteration/config.yaml` | Config schema template | Required keys documented with types and examples. |

**Template conventions:**
- Each file includes imports, function signatures, docstrings with input/output types, and `# --- PLAN-SPECIFIC LOGIC ---` markers where the Coder fills in iteration-specific code
- Output-writing boilerplate is in the template (JSON serialization, CSV writing) — the Coder fills in the computation
- `# %%` cell markers for future jupytext conversion (costs nothing)

**config.yaml required keys:**
```yaml
iteration: 1
random_seed: 42
project_root: "../../"
data:
  train: "../../data/raw/train.csv"
  test: "../../data/raw/test.csv"     # null if no test set
target_column: "Survived"
task_type: "binary_classification"
split:
  method: "stratified"
  val_ratio: 0.2
hyperparameters: {}                    # from plan model_steps
output_paths:
  metrics: "outputs/metrics.json"
  predictions: "outputs/predictions.csv"
  learning_curves: "outputs/learning_curves.json"
  feature_importance: "outputs/feature_importance.json"
  pipeline_metadata: "outputs/pipeline_metadata.json"
  model: "outputs/model/"
  log: "execution/log.txt"
```

---

### M4.2 — Output Artifact Schemas + Contract 5

Define schemas for all M4 code outputs. Add Contract 5 to `.claude/rules/artifact-contracts.md`.

**Artifacts:**

| Artifact | Schema | Notes |
|----------|--------|-------|
| `metrics.json` | `{primary: {name: str, value: float}, secondary: {name: float, ...}, train: {name: float, ...}, validation: {name: float, ...}}` | Per coding-rules.md rule 3 |
| `predictions.csv` | Columns: `index, y_true, y_pred, y_prob_0, y_prob_1` (binary classification) or `index, y_true, y_pred` (regression) | Per-class probabilities for calibration analysis |
| `feature_importance.json` | `{method: str, features: [{name: str, importance: float}], sorted: true, model: str}` | Model-native importance when available |
| `learning_curves.json` | `{metric_name: str, train: [float], validation: [float], iterations: [int]}` or `{note: "model does not support iterative training"}` | Graceful null for non-iterative models (LogReg) |
| `pipeline_metadata.json` | `{stages: [{name: str, input_shape: [int, int], output_shape: [int, int], duration_s: float, warnings: [str]}], total_duration_s: float, python_version: str, packages: {name: version}}` | Used by M6 Model Report Builder |
| `config.yaml` | See M4.1 schema | Generated per iteration |
| `outputs/model/model.pkl` | joblib-serialized trained model | gitignored but referenced in metadata |
| `outputs/model/metadata.json` | `{model_class: str, feature_list: [str], training_timestamp: str, n_train_samples: int}` | Lightweight metadata alongside the artifact |

---

### M4.3 — Codegen Validator (`src/codegen/validator.py`)

Follow the exact pattern from `src/planning/validator.py`:
- Custom `CodegenValidationError(Exception)`
- Single entry function: `validate_codegen(iteration_dir: Union[str, Path]) -> dict`
- Early-fail semantics (raise on first violation with contextual message)
- Returns summary dict on success

**Validation checks (in order):**
1. Required files exist: `src/main.py`, `src/data_loader.py`, `src/feature_engineering.py`, `src/model.py`, `src/evaluate.py`, `src/utils.py`, `config.yaml`, `requirements.txt`
2. `config.yaml` has all required keys (iteration, random_seed, target_column, task_type, data, split, hyperparameters, output_paths)
3. All Python files parse without `SyntaxError` via `ast.parse()`
4. No hardcoded absolute paths in Python files (regex: `/Users/`, `/home/`, `C:\\`)
5. `main.py` contains `if __name__` guard
6. Feature step count sanity check: count function calls or transform operations in `feature_engineering.py`, warn if count differs significantly from plan's `feature_steps` count (pass plan path as optional arg)

**Also create:** `src/codegen/__init__.py` (empty)

---

### M4.4 — Coder Agent (`.claude/agents/coder.md`)

Follow `.claude/skills/create-agent/SKILL.md` conventions.

**Frontmatter:**
```yaml
name: coder
description: >
  Translates an iteration plan YAML into executable Python scripts.
  Use when: a validated iteration plan exists and code needs to be generated for it.
tools: Read, Write, Glob
model: sonnet
maxTurns: 20
```

**Inputs (read from disk):**
- `artifacts/plans/iteration-<n>.yaml` — validated plan (Contract 2)
- `project.yaml` — target column, task type, data paths
- `artifacts/data/profile.json` — column types, stats, null analysis
- `.claude/rules/coding-rules.md` — 10 mandatory rules
- `templates/iteration/` (iteration 1) OR prior iteration code (iteration > 1)

**Outputs (write to disk):**
```
projects/<project>/iterations/iteration-<n>/
  src/
    main.py
    data_loader.py
    feature_engineering.py
    model.py
    evaluate.py
    utils.py
  config.yaml
  requirements.txt
```

**10-step workflow:**
1. Read inputs: iteration YAML, project.yaml, profile.json
2. Create directory structure: `iterations/iteration-<n>/src/`
3. Generate config.yaml: merge plan hyperparams, project metadata, paths, seed
4. Generate utils.py: logging, timing, warnings, seed, metadata collector
5. Generate data_loader.py: load CSVs from config paths, stratified train/val split
6. Generate feature_engineering.py: translate EACH feature_step from plan into code
7. Generate model.py: instantiate model with hyperparams from config, fit, predict
8. Generate evaluate.py: compute metrics, write all output artifacts, save model
9. Generate main.py: orchestrate steps 4-8, logging setup, seed setting
10. Self-validate: run codegen validator on the generated directory

**Feature step -> code mapping table (embedded in agent):**

| Plan Action | Code Pattern |
|-------------|-------------|
| `Drop <column>` | `df = df.drop(columns=['col'])` |
| `Impute <col> with median/mode(<value>)` | `df['col'] = df['col'].fillna(value)` |
| `Binary encode <col>: A=1, B=0` | `df['col'] = df['col'].map({'A': 1, 'B': 0})` |
| `One-hot encode <col>` | `df = pd.get_dummies(df, columns=['col'], drop_first=True)` |
| `log1p transform <col>` | `df['col'] = np.log1p(df['col'])` |
| `Replace <col> with binary flag` | `df['has_col'] = df['col'].notna().astype(int); df = df.drop(columns=['col'])` |
| `Keep as-is / passthrough` | No-op, document in comments |

**Scope guardrails:**
- CAN: Write Python files in `iterations/iteration-<n>/`, create directories, declare dependencies
- CANNOT: Execute code, modify plans, modify profile, modify anything outside the iteration directory, add features not in the plan, skip feature steps from the plan

**For iteration > 1:** Read `iterations/iteration-<n-1>/src/` as base code. Apply targeted diffs based on the new plan rather than rewriting from scratch.

**Output contract:**
```
Done Code generated | iteration: <n> | files: <count> | path: iterations/iteration-<n>/
```

---

### M4.5 — Validator Unit Tests (`tests/codegen/test_codegen_validator.py`)

Follow existing test patterns from `tests/planning/test_plan_schema.py`.

**Test fixtures** (`tests/codegen/conftest.py` or inline):
- `valid_iteration_dir(tmp_path)` — creates a complete valid iteration directory with all files
- Helper to create minimal valid Python files

**Test cases:**

| Test | Expectation |
|------|-------------|
| `test_valid_complete_directory` | Passes, returns summary dict |
| `test_missing_main_py` | `CodegenValidationError` matching "main.py" |
| `test_missing_config_yaml` | `CodegenValidationError` matching "config.yaml" |
| `test_config_missing_required_keys` | `CodegenValidationError` matching key name |
| `test_python_syntax_error` | `CodegenValidationError` matching "SyntaxError" |
| `test_hardcoded_absolute_path` | `CodegenValidationError` matching "hardcoded" |
| `test_empty_feature_engineering_valid` | Passes (empty features list is valid per plan schema) |

**Also create:** `tests/codegen/__init__.py` (empty)

---

### M4.6 — Smoke Test on Titanic Iteration-1

Run the Coder agent on `projects/titanic` for iteration-1.

**Structural checks (required pass criteria):**
- [ ] All 8 files exist in `iterations/iteration-1/` (6 .py + config.yaml + requirements.txt)
- [ ] config.yaml has: random_seed=42, target_column=Survived, task_type=binary_classification
- [ ] config.yaml hyperparameters match plan: `{C: 1.0, solver: lbfgs, max_iter: 1000, random_state: 42}`
- [ ] feature_engineering.py implements all 11 feature steps from the plan
- [ ] model.py uses `LogisticRegression` with correct hyperparams
- [ ] evaluate.py computes AUC-ROC as primary metric
- [ ] All Python files pass `ast.parse` (no syntax errors)
- [ ] codegen validator passes on the generated directory
- [ ] No coding-rules.md violations (no hardcoded paths, seed set, no internet imports)

**Stretch execution (optional, informs M5):**
- [ ] `python src/main.py` exits with code 0
- [ ] `outputs/metrics.json` exists with correct schema
- [ ] `outputs/predictions.csv` has correct columns
- [ ] If stretch fails -> document the error for M5's first test case

---

## Ancillary Changes

| Change | File | When |
|--------|------|------|
| Update path scope `runs/**` -> `iterations/**` | `.claude/rules/coding-rules.md` | With M4.3 |
| Add Contract 5: iteration code output schemas | `.claude/rules/artifact-contracts.md` | M4.2 |
| Mark M4 as in-progress | `.claude/CLAUDE.md` | Start of M4 |

---

## Files Created/Modified Summary

| File | Action | Sub-milestone |
|------|--------|---------------|
| `templates/iteration/main.py` | Create | M4.1 |
| `templates/iteration/data_loader.py` | Create | M4.1 |
| `templates/iteration/feature_engineering.py` | Create | M4.1 |
| `templates/iteration/model.py` | Create | M4.1 |
| `templates/iteration/evaluate.py` | Create | M4.1 |
| `templates/iteration/utils.py` | Create | M4.1 |
| `templates/iteration/config.yaml` | Create | M4.1 |
| `.claude/rules/artifact-contracts.md` | Modify (add Contract 5) | M4.2 |
| `src/codegen/__init__.py` | Create | M4.3 |
| `src/codegen/validator.py` | Create | M4.3 |
| `.claude/agents/coder.md` | Create | M4.4 |
| `tests/codegen/__init__.py` | Create | M4.5 |
| `tests/codegen/test_codegen_validator.py` | Create | M4.5 |
| `.claude/rules/coding-rules.md` | Modify (path scope) | Ancillary |
| `.claude/CLAUDE.md` | Modify (status) | Ancillary |
| `projects/titanic/iterations/iteration-1/**` | Generated by Coder agent | M4.6 |

---

## Dependency Graph

```
M4.1 (templates) ---\
M4.2 (schemas)  -----+--> M4.4 (coder agent) --> M4.6 (smoke test)
M4.3 (validator) ---/
                     \--> M4.5 (validator tests)  [parallel with M4.4]
```

- M4.1, M4.2, M4.3 proceed in parallel
- M4.5 proceeds in parallel with M4.4
- M4.6 requires M4.4 complete

---

## Verification Criteria

1. **Structural**: `src/codegen/validator.py` passes on Titanic iteration-1 generated code
2. **Syntax**: All generated .py files parse without SyntaxError via `ast.parse`
3. **Config**: config.yaml has correct hyperparams from plan + all required keys
4. **Coding-rules compliance**: No hardcoded paths, seed set, feature transforms applied to both splits
5. **Template coherence**: main.py calls load -> features -> train -> evaluate in order
6. **Test suite green**: `pytest tests/codegen/` all pass
7. **Smoke test**: Coder generates structurally valid code from Titanic iteration-1.yaml
8. **Stretch**: Generated code runs to completion and produces metrics.json (optional)

---

## Forward-Compatibility Notes (NOT M4 deliverables)

These inform M4 design decisions but are delivered in later milestones:

- **Executor manifest schema** (M5): `execution/manifest.json` with status, exit_code, duration, error_class
- **Timeout default** (M5): 300s, documented in config.yaml schema as reference
- **DS-STAR Debugger compression** (M5): Debugger agent should summarise tracebacks before reasoning
- **Verifier semantic check** (M5): LLM verification pass checking plan->code fidelity
- **Git-based rollback** (M7): Iteration directories are self-contained, enabling rollback by directory deletion
- **`.gitignore`** (M5): Add rules for `model.pkl`, large CSVs, `.venv/`
- **Retrieval-augmented coding** (future): For iteration > 1, Coder could search `knowledge-base/tactics/` for proven patterns

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| M4 scope | Code generation only (execution = M5) | Matches PRD §17. Keeps milestones focused and reviewable. |
| Directory naming | `iterations/` not `runs/` | User preference. Already exists in project. Update coding-rules path scope. |
| Template approach | Structural references, not Jinja | LLM Coder reads templates for structure/contracts, doesn't fill placeholders. DS-STAR pattern. |
| Coder vs Debugger | Separate agents | Coder (M4) generates code, Debugger (M5) fixes runtime errors. DS-STAR separation. |
| Codegen validation | Deterministic Python (`ast.parse` + schema + regex) | More reliable than LLM self-checking. Reusable. Follows planning validator pattern. |
| predictions.csv format | Per-class probabilities (`y_prob_0, y_prob_1`) | Supports multi-class generalisation and calibration analysis. |
| pipeline_metadata.json | Include in M4 output contract | M6 Model Report Builder needs stage-level shapes, durations, and warnings. |
| Model artifact | Persist `model.pkl` + `metadata.json` | Enables downstream comparison, ensembling, and analysis. |
| utils.py | Explicit 6th generated file | Timing, warning capture, dual logging are too important to leave implicit. |
