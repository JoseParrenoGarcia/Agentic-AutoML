# M4 — Plan-to-Code Layer

**Status:** Draft
**Created:** 2026-04-13
**Milestone:** M4
**Depends on:** M2 (profiler), M3 (planner + validator)

---

## Goal

Build a Coder agent that translates an iteration plan YAML into executable Python scripts, backed by base templates, config schema, and output format contracts. M4 produces structurally correct, syntactically valid code. M5 (next milestone) runs and debugs it.

**No context is lost between M4 and M5.** The plan YAML, profile.json, and coding-rules all live as files on disk. When M5's Debugger fixes code, it reads: (1) the plan file to understand *intent*, (2) the profile for data context, (3) the broken code, and (4) the traceback. This is exactly DS-STAR's debugger pattern: `s_fixed = Debugger(code, traceback, data_descriptions)`.

---

## Approach

**Structural reference templates + LLM Coder agent.** Templates define code organization, import patterns, and output contracts. The Coder fills in actual logic from the plan. For iteration > 1, prior iteration code replaces templates as the starting point.

---

## Research Inputs

Key patterns extracted from the three reference papers/repos:

### DS-STAR (arxiv 2509.21825)
- **Debugger agent** receives code + traceback + data descriptions (not just traceback alone). Our M5 Debugger should follow this pattern — read plan + profile.json alongside the error.
- **Coder agent** is given the cumulative plan + data descriptions + base code (for iteration > 1). Incremental code building — each round extends the prior code rather than rewriting.
- **Verifier agent** checks sufficiency of plan+code against the query. For our M4, the codegen validator is the deterministic analogue.
- **Router agent** decides whether to add a new step or correct an existing one. Maps to our M7 Action Router.
- **Two-stage debugging**: syntax/import repair (fast, deterministic) vs logic/runtime repair (LLM-assisted). Directly adopted in PRD §6.4 for M5.

### Karpathy autoresearch
- **One editable file, one metric, one loop.** For M4: maximal clarity on what the Coder can change and what is fixed infrastructure.
- **Context management**: redirect stdout to log, extract only key metrics via grep. Our evaluate.py should produce grep-friendly summary lines alongside structured JSON.
- **Git-as-rollback**: atomic experiments with revert on failure. Each iteration directory is self-contained and immutable — same principle without git mechanics.
- **Simplicity criterion**: "A 0.001 improvement that adds 20 lines of hacky code? Not worth it." Relevant for iteration > 1 when Coder patches code.

### AutoKaggle
- Competition-oriented code generation with submission file handling. Defer to M10, but predictions.csv covers the same purpose for now.
- Multi-phase pipeline (data cleaning → feature engineering → model training → evaluation) maps directly to our file separation.

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Directory naming | `iterations/` not `runs/` | Matches existing repo structure. coding-rules.md path scope updated accordingly. |
| Coder vs Debugger | Separate Claude Code agents | Coder (M4) generates code, Debugger (M5) fixes runtime errors. Both read plan + profile from disk. DS-STAR pattern. |
| Template approach | Structural references, not Jinja | LLM Coder reads templates for structure/contracts, doesn't fill in placeholders. Leverages LLM flexibility while maintaining consistency. |
| evaluate.py scope | M4 scope; model-report is M6 | evaluate.py produces raw data artifacts (metrics, predictions, importance). Model Report Builder (M6) creates the analysis narrative. |
| Code execution | NOT in M4 | M4 produces syntactically valid code. M5 handles execution + debugging. |

---

## Phase A: Infrastructure (Templates + Schemas)

### M4.1 — Base code templates (`templates/iteration/`)

Create skeleton Python files that show expected structure, imports, function signatures, and output contracts. These are reference documents for the Coder agent, not Jinja templates.

**Files to create:**

| File | Purpose | Key Contract |
|---|---|---|
| `templates/iteration/main.py` | Fixed orchestration: set seed → load → features → train → evaluate | Entry point with `if __name__ == "__main__"` |
| `templates/iteration/data_loader.py` | `load_and_split()` function | Reads paths from config.yaml. Returns train/val/test DataFrames. Stratified split for classification. |
| `templates/iteration/feature_engineering.py` | `engineer_features(train, val, test)` function | Identical transforms applied to all three splits. Returns transformed DataFrames. |
| `templates/iteration/model.py` | `train_model(train_df, target_col, config)` function | Returns trained model + train predictions. Captures learning curves when model supports callbacks. |
| `templates/iteration/evaluate.py` | `evaluate_and_report(model, train, val, test, config)` function | Computes all metrics. Writes: metrics.json, predictions.csv, feature_importance.json, learning_curves.json, model artifact. |
| `templates/iteration/config.yaml` | Config schema template | Required keys: iteration, random_seed, data_paths, target_column, task_type, split_ratios, hyperparameters, output_paths. |

**Relevant existing code:** PRD §10 (base template structure), coding-rules.md (10 rules).

### M4.2 — Output artifact schemas

Define schemas for all M4 code outputs. Add as Contract 5 to `artifact-contracts.md`.

| Artifact | Schema | Notes |
|---|---|---|
| `metrics.json` | `{primary: {name: str, value: float}, secondary: {name: float}, train: {name: float}, validation: {name: float}}` | Already in coding-rules.md rule 3 |
| `predictions.csv` | Columns: index, actual, predicted, probability_class_0, probability_class_1 (classification) or index, actual, predicted (regression) | Index-aligned to input data |
| `feature_importance.json` | `{method: str, features: [{name: str, importance: float}], sorted: true}` | Model-native importance when available |
| `learning_curves.json` | `{metric_name: str, train: [float], validation: [float], iterations: [int]}` | null/empty when model doesn't support iterative training (e.g., LogReg) |
| `config.yaml` | Required keys: iteration, random_seed, data_paths, target_column, task_type, split_ratios, hyperparameters, feature_columns, output_paths | Generated per iteration |
| Model artifact | `outputs/model/model.pkl` + `outputs/model/metadata.json` (model class name, feature list, training timestamp) | joblib or pickle serialised |

### M4.3 — Update coding-rules.md path scope

Change `paths: ["runs/**"]` → `paths: ["iterations/**"]` and update body text references from `runs/` to `iterations/`.

### M4.4 — Codegen validator (`src/codegen/validator.py`)

Lightweight Python validator analogous to `src/planning/validator.py`. Checks:

1. Required files exist: main.py, data_loader.py, feature_engineering.py, model.py, evaluate.py, config.yaml, requirements.txt
2. config.yaml has all required keys
3. Python files parse without SyntaxError (`ast.parse`)
4. No hardcoded absolute paths in Python files (basic regex check)

Raises `CodegenValidationError` on first failure.

---

## Phase B: Coder Agent

### M4.5 — Create Coder agent (`.claude/agents/coder.md`)

Follow `.claude/skills/create-agent/SKILL.md` for authoring conventions.

**Inputs:**
- `artifacts/plans/iteration-<n>.yaml` — validated plan (Contract 2)
- `project.yaml` — target column, task type, data paths
- `artifacts/data/profile.json` — column types, stats, null analysis (used for data loading context)
- `.claude/rules/coding-rules.md` — 10 mandatory rules
- `templates/iteration/` — structural reference (iteration 1); prior iteration code (iteration > 1)

**Outputs:**
```
iterations/iteration-<n>/
├── src/
│   ├── main.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluate.py
├── config.yaml
└── requirements.txt
```

**10-step workflow:**

1. **Read inputs**: iteration-<n>.yaml, project.yaml, profile.json
2. **Create directory structure**: `iterations/iteration-<n>/{src/, outputs/, execution/, outputs/model/}`
3. **Generate config.yaml**: merge plan hyperparams, project metadata, paths, seed from plan
4. **Generate data_loader.py**: load CSVs from config paths, stratified train/val split (70/15/15 or per project.yaml), return DataFrames
5. **Generate feature_engineering.py**: translate EACH feature_step from plan into code. Key invariant: same function applied to train, val, test. Group by: drops → imputation → encoding → transforms → passthrough.
6. **Generate model.py**: translate model_step into code. Import model class, instantiate with hyperparams from config, fit on train, predict on train+val. Capture learning curves if model supports it (e.g., LightGBM eval_set).
7. **Generate evaluate.py**: compute primary + secondary metrics (based on task_type from config) on train+val sets. Write: metrics.json, predictions.csv, feature_importance.json, learning_curves.json. Save model artifact.
8. **Generate main.py**: orchestrate steps 4-7 in order, with logging setup and seed setting.
9. **Generate requirements.txt**: list all package dependencies.
10. **Self-validate**: run `src/codegen/validator.py` on the generated directory.

**Scope guardrails:**
- **CAN**: Write Python files in `iterations/iteration-<n>/`, create directories, declare dependencies
- **CANNOT**: Execute code, modify plans, modify profile, modify anything outside `iterations/iteration-<n>/`, add features not in the plan, change the evaluation harness, skip feature steps from the plan

**For iteration > 1**: Read `iterations/iteration-<n-1>/src/` as the base code. Apply targeted diffs based on the new plan rather than rewriting from scratch. Preserve: import structure, logging, file layout, output format.

### M4.6 — Feature step → code translation patterns

Embedded in the Coder agent instructions. Mapping from plan action verbs to Python code patterns:

| Plan Action | Code Pattern | Example Plan Step |
|---|---|---|
| `Drop <column>` | `df = df.drop(columns=['col'])` | drop_passengerid |
| `Impute <col> with median/mode(<value>)` | `df['col'] = df['col'].fillna(value)` | impute_age_median |
| `Binary encode <col>: A=1, B=0` | `df['col'] = df['col'].map({'A': 1, 'B': 0})` | encode_sex_binary |
| `One-hot encode <col>` | `df = pd.get_dummies(df, columns=['col'], drop_first=True)` | encode_embarked |
| `log1p transform <col>` | `df['col'] = np.log1p(df['col'])` | fare_log1p |
| `Replace <col> with binary flag` | `df['has_col'] = df['col'].notna().astype(int); df = df.drop(columns=['col'])` | cabin_to_has_cabin |
| `Keep as-is / passthrough` | No-op, document in comments | pclass_passthrough |

**Key principle**: ALL transforms live in a single `engineer_features()` function, applied identically to train/val/test DataFrames. No fit-on-train-apply-on-val pattern for simple transforms (that complexity comes with sklearn Pipelines in later iterations).

---

## Phase C: Tests + Smoke Test

### M4.7 — Unit tests for codegen validator

`tests/codegen/test_codegen_validator.py`:

| Test | Expectation |
|---|---|
| Valid complete directory | Passes |
| Missing main.py | `CodegenValidationError` |
| config.yaml missing required keys | `CodegenValidationError` |
| Python file with syntax error | `CodegenValidationError` |
| Hardcoded absolute path in .py file | `CodegenValidationError` |
| Empty feature_engineering.py (valid) | Passes (empty features list is valid per plan schema) |

### M4.8 — Smoke test on Titanic iteration-1

Run the Coder agent on `projects/titanic/artifacts/plans/iteration-1.yaml`.

**Verification checklist:**

- [ ] All files exist in `projects/titanic/iterations/iteration-1/src/`
- [ ] config.yaml has: seed=42, target=Survived, task_type=binary_classification
- [ ] feature_engineering.py implements all 11 feature steps from the plan
- [ ] model.py uses `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)`
- [ ] evaluate.py computes AUC-ROC as primary metric
- [ ] All Python files pass `ast.parse` (no syntax errors)
- [ ] No violations of coding-rules.md
- [ ] codegen validator passes on the generated directory

**Stretch (M5 preview)**: Actually execute `python main.py` from the iteration directory and verify it runs to completion and produces metrics.json.

---

## Files Affected

### Create
| File | Purpose |
|---|---|
| `templates/iteration/main.py` | Orchestration skeleton |
| `templates/iteration/data_loader.py` | Load + split skeleton |
| `templates/iteration/feature_engineering.py` | Feature pipeline skeleton |
| `templates/iteration/model.py` | Train + predict skeleton |
| `templates/iteration/evaluate.py` | Metrics + artifacts skeleton |
| `templates/iteration/config.yaml` | Config schema template |
| `.claude/agents/coder.md` | Coder agent instructions |
| `src/codegen/__init__.py` | Package init |
| `src/codegen/validator.py` | Codegen validation utility |
| `tests/codegen/__init__.py` | Package init |
| `tests/codegen/test_codegen_validator.py` | Validator tests |

### Modify
| File | Change |
|---|---|
| `.claude/rules/coding-rules.md` | Path scope `runs/**` → `iterations/**` |
| `.claude/rules/artifact-contracts.md` | Add Contract 5: iteration code output schemas |
| `docs/PRD.md` | Update M4 status, directory references |

---

## Verification Criteria

1. **Structural**: `src/codegen/validator.py` passes on the Titanic iteration-1 generated code
2. **Syntax**: All generated .py files parse without SyntaxError via `ast.parse`
3. **Config correctness**: config.yaml has correct hyperparams from plan + all required keys
4. **Coding-rules compliance**: No hardcoded paths, seed is set, feature transforms applied to all splits
5. **Template coherence**: main.py calls `load_and_split → engineer_features → train_model → evaluate_and_report` in order
6. **Test suite green**: `tests/codegen/test_codegen_validator.py` all pass
7. **Smoke test**: Coder generates structurally valid code from Titanic iteration-1.yaml

---

## Open Questions / Further Considerations

1. **How prescriptive should templates be?** Start moderately detailed (include imports, logging setup, output-writing boilerplate, clear section comments for plan-specific logic). Test with Titanic smoke test and adjust if the Coder over-relies or ignores them.

2. **Learning curves for non-iterative models**: For LogReg (iteration 1), learning_curves.json should be `null` or `{"note": "model does not support iterative training"}`. The code should handle this gracefully so it works for both LogReg and gradient boosting.

3. **Competition submission file**: PRD §7 mentions submission.csv for Kaggle. Defer to M10 (benchmarking) — for now, predictions.csv on test set covers it.

4. **Notebook conversion**: PRD §18.8 mentions jupytext script→notebook. Not M4 scope, but keep code style notebook-friendly (clear section comments, cell-like blocks).

5. **sklearn Pipeline vs manual transforms**: For iteration 1 (simple baseline), manual transforms in a function are cleaner and more readable. sklearn Pipelines become valuable for iteration > 1 when transforms need fit/transform semantics (e.g., target encoding). The Coder agent should know when to upgrade to Pipelines — likely a knowledge-base tactic entry.

---

## Dependency Graph

```
M4.1 (templates) ──┐
M4.2 (schemas)  ───┤
M4.3 (rules fix) ──┼──→ M4.5 (coder agent) ──→ M4.8 (smoke test)
M4.4 (validator) ──┤              │
                    │              └──→ M4.6 (feature mapping)
                    │
                    └──→ M4.7 (validator tests) ─── [parallel with M4.5]
```

**Parallelizable**: M4.1–M4.4 can all proceed in parallel. M4.7 can proceed in parallel with M4.5–M4.6.
