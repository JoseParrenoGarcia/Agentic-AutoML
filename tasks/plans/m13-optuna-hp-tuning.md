# M13 — Optuna Hyperparameter Tuning: Implementation Plan

**Status:** Draft
**Created:** 2026-04-17

---

## Context

The agentic loop (M0–M9) is complete. Currently, the Planner agent manually specifies hyperparameters in plan YAML, and the Coder hardcodes them into `model.py`. This works for early exploration but leaves performance on the table once a promising model family is identified. M13 adds Optuna-based HP search as an integrated option — triggered by the Planner when strategically appropriate, not by default.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Flow integration | **Option A: Integrated into model.py** | No extra agent, no extra Coder+Executor cycle. The Coder generates a single model.py that includes Optuna search + final training. Flow stays: Planner → Coder → Executor → Report Builder → Reviewer. |
| New agent? | **No** | HP tuning is a variant of how model.py gets written. The Coder agent is extended, not replaced. |
| Search space storage | **Python code** (`src/tuning/search_spaces.py`) | Type-safe, importable by generated scripts, testable. Planner can override ranges in plan YAML. |
| Learnings extraction | **Hybrid** | Deterministic Python (`src/tuning/analysis.py`) computes structured fields (param importance, range exhaustion, anti-patterns). LLM writes a short narrative summary interpreting results for the Planner. |
| CV during HP search | **Single inner split from training** | Avoids validation leakage. Optuna objective uses an inner tuning split carved from `X_train`/`y_train` (e.g., stratified when classification). The pipeline validation split remains untouched for final evaluation. |
| Warm-starting | **Planner-mediated** | Planner reads prior `optuna_results.json` and adjusts search ranges in plan YAML. No Optuna-level study persistence. Balance of explore vs exploit — don't narrow to the exact previous range, but don't re-explore clearly dead regions. |
| Ensemble HP tuning | **Out of scope for M13** | Single-model iterations only. When plan uses ensembling, `hyperparameter_strategy` must be `manual`. Tracked as future enhancement in PRD. |
| Pruning | **Built into templates** | XGBoost/LightGBM use native pruning callbacks. Others use MedianPruner(n_startup_trials=5). Not left to LLM reasoning. |
| Sampler | **TPESampler (default)** | Good for most cases. Planner can override if needed. For < 20 trials, effectively similar to random. |
| Reproducibility | **Fixed seed** | `TPESampler(seed=config['random_seed'])` — aligns with existing pipeline reproducibility contract. |
| Dependency strategy | **Global environment (repo requirements)** | The current Executor runs iteration code in the repo Python environment (no per-iteration install step). Add `optuna` to the repo `requirements.txt` so tuning runs reliably; iteration `requirements.txt` remains an audit trail. |

---

## Architecture Overview

### What changes (no new agents or skills)

```
1. Plan YAML schema          — new fields: hyperparameter_strategy, optuna_budget, optuna_search_space_overrides
2. src/tuning/search_spaces.py — predefined search spaces per model family
3. src/tuning/analysis.py      — post-study analysis (importance, anti-patterns, range exhaustion)
4. templates/iteration/model_optuna.py — Optuna-aware model template
5. Coder agent instructions    — teach it to generate Optuna code when plan says so
6. Executor output validator   — check for optuna_results.json when HP tuning was used
7. artifact-contracts.md       — new Contract 7: optuna_results.json schema
8. Model Report Builder        — add HP tuning section to model-report.json when optuna_results.json exists
9. Planner agent instructions  — (a) how to decide when HP tuning is warranted, (b) how to read prior optuna_results.json, (c) how to set search space overrides
```

### Flow (unchanged, Option A)

```
Planner → Coder → Executor → Report Builder → Reviewer → Router
              │
              └─ if hyperparameter_strategy: optuna
                 → generates model.py with Optuna search built in
                 → Executor runs it (same two-stage repair)
                 → produces standard outputs + optuna_results.json
```

---

## Plan YAML Schema Extension

New fields under `model_steps[0]`:

```yaml
model_steps:
  - algorithm: xgboost
    hyperparameter_strategy: optuna     # "optuna" | "manual" (default: manual)
    hyperparameters:                     # used when strategy is "manual"
      max_depth: 6
      learning_rate: 0.1
    optuna_budget:                       # used when strategy is "optuna"
      n_trials: 50                       # default: 50
      time_limit_s: 600                  # default: 600 (10 min), hard cap
      proxy_training:                    # optional: make tuning feasible for slow models
        enabled: true
        row_subsample_fraction: 0.3      # train on a sample during tuning only
        max_estimators: 300              # cap iterations/estimators during tuning only
    optuna_search_space_overrides:       # optional: Planner narrows/expands ranges
      max_depth: [4, 7]                  # override default [3, 10]
      learning_rate: [0.01, 0.1]         # override default [0.001, 0.3]
    optuna_notes: "Previous iteration found max_depth > 8 consistently underperformed. Narrowing range."
    rationale: "XGBoost showed promise in iter-3. HP tuning to find optimal regularisation."
```

The Planner agent decides `hyperparameter_strategy: optuna` when:
- A model family is decided and we want to maximise its potential
- Reviewer said "refine" (continue same direction)
- Overfitting detected → tune regularisation params specifically
- Previous Optuna run was compute-limited and we have more budget

The Planner sets `hyperparameter_strategy: manual` (or omits it) when:
- Early exploration iterations (trying different model families)
- First iteration (establish a baseline first)
- Pivoting to a different model family
- Score is already near the AutoGluon ceiling
- Feature engineering is the bottleneck, not model config
- Plan uses ensembling (StackingClassifier etc.)

Additional Planner constraint:
- Prefer HP tuning only after feature engineering stabilises (avoid spending budget tuning a moving pipeline).

---

## config.yaml Integration (Runtime Contract)

The generated iteration code reads only `config.yaml` at runtime. Therefore, when `hyperparameter_strategy: optuna` is selected in the plan, the Coder must also write a `tuning` block into `config.yaml`.

Recommended shape (exact field names can be finalised during implementation; include enough info to avoid any dependence on the plan YAML at runtime):

```yaml
hyperparameters: {}  # baseline/default params (used for fallback + comparison)

tuning:
  strategy: optuna | manual
  metric_name: "val_auc_roc"           # must match evaluate.py primary metric name
  direction: maximize | minimize
  budget:
    n_trials: 50
    time_limit_s: 600
  tuning_split:
    method: stratified | random
    val_ratio: 0.2
  proxy_training:
    enabled: true
    row_subsample_fraction: 0.3
    max_estimators: 300
  search_space_overrides: {}
  warm_start:
    enqueue_baseline: true
    enqueue_previous_best: true
```

Key requirement:
- The Optuna objective **must** use the inner tuning split from training, not the pipeline validation split.

---

## Predefined Search Spaces

`src/tuning/search_spaces.py`:

```python
"""Predefined HP search spaces per model family for Optuna.

Each space is a dict mapping param_name to a tuple:
  (suggest_type, *args)

suggest_type is one of:
  'float'     → trial.suggest_float(name, low, high)
  'float_log' → trial.suggest_float(name, low, high, log=True)
  'int'       → trial.suggest_int(name, low, high)
  'int_log'   → trial.suggest_int(name, low, high, log=True)
  'categorical' → trial.suggest_categorical(name, choices)
"""

SEARCH_SPACES = {
    'xgboost': {
        'learning_rate':    ('float_log', 1e-3, 0.3),
        'max_depth':        ('int', 3, 10),
        'n_estimators':     ('int', 100, 2000),
        'subsample':        ('float', 0.5, 1.0),
        'colsample_bytree': ('float', 0.5, 1.0),
        'reg_alpha':        ('float_log', 1e-8, 10.0),
        'reg_lambda':       ('float_log', 1e-8, 10.0),
        'min_child_weight': ('int', 1, 10),
        'gamma':            ('float_log', 1e-8, 5.0),
    },
    'lightgbm': {
        'learning_rate':    ('float_log', 1e-3, 0.3),
        'max_depth':        ('int', 3, 12),
        'n_estimators':     ('int', 100, 2000),
        'num_leaves':       ('int', 20, 150),
        'subsample':        ('float', 0.5, 1.0),
        'colsample_bytree': ('float', 0.5, 1.0),
        'reg_alpha':        ('float_log', 1e-8, 10.0),
        'reg_lambda':       ('float_log', 1e-8, 10.0),
        'min_child_samples':('int', 5, 100),
    },
    'random_forest': {
        'n_estimators':     ('int', 100, 1000),
        'max_depth':        ('int', 5, 30),
        'min_samples_split':('int', 2, 20),
        'min_samples_leaf': ('int', 1, 10),
        'max_features':     ('categorical', ['sqrt', 'log2', 0.5, 0.7, 0.9]),
    },
    'logistic_regression': {
        'C':                ('float_log', 1e-4, 100.0),
        'penalty':          ('categorical', ['l1', 'l2', 'elasticnet']),
        'l1_ratio':         ('float', 0.0, 1.0),  # only when penalty=elasticnet
    },
    'catboost': {
        'learning_rate':    ('float_log', 1e-3, 0.3),
        'depth':            ('int', 3, 10),
        'iterations':       ('int', 100, 2000),
        'l2_leaf_reg':      ('float_log', 1e-3, 10.0),
        'bagging_temperature': ('float', 0.0, 1.0),
        'random_strength':  ('float', 0.0, 10.0),
    },
}

# Pruning callback registry per model family
PRUNING_CALLBACKS = {
    'xgboost': 'optuna.integration.XGBoostPruningCallback',
    'lightgbm': 'optuna.integration.LightGBMPruningCallback',
    'catboost': 'optuna.integration.CatBoostPruningCallback',
    # Others use generic MedianPruner via study.pruner
}
```

The Planner can override individual ranges via `optuna_search_space_overrides` in the plan YAML. The Coder merges overrides with defaults.

Search space guardrail (important for reliability):
- Some models require **conditional** hyperparameters (e.g., LogisticRegression: `penalty`/`solver`/`l1_ratio` combos). The search space implementation must enforce valid combos to avoid wasting trials on FAIL states.

---

## Feasibility Timing Probe

Built into the generated Optuna script, not a separate step. The Coder template includes:

```python
# Phase 0: Timing probe
import time

SUBSAMPLE_FRACTION = 0.1
MIN_SUBSAMPLE_ROWS = 1000
subsample_n = max(MIN_SUBSAMPLE_ROWS, int(len(X_train) * SUBSAMPLE_FRACTION))
X_probe = X_train.sample(n=min(subsample_n, len(X_train)), random_state=SEED)
y_probe = y_train.loc[X_probe.index]

probe_start = time.time()
probe_model = train_with_params(DEFAULT_PARAMS, X_probe, y_probe)
probe_duration = time.time() - probe_start

# Estimate per-trial time on full data (conservative 2x multiplier for safety)
scale_factor = len(X_train) / len(X_probe)
estimated_per_trial_s = probe_duration * scale_factor * 2.0

# Adapt budget
max_feasible_trials = max(3, int(time_limit_s / estimated_per_trial_s))
effective_n_trials = min(n_trials, max_feasible_trials)

feasibility = {
    'probe_duration_s': probe_duration,
    'subsample_fraction': len(X_probe) / len(X_train),
    'estimated_per_trial_s': estimated_per_trial_s,
    'requested_n_trials': n_trials,
    'effective_n_trials': effective_n_trials,
    'budget_limited': effective_n_trials < n_trials,
}

if estimated_per_trial_s > time_limit_s:
    logger.warning(
        f"Single trial estimated at {estimated_per_trial_s:.0f}s exceeds "
        f"time limit {time_limit_s}s. Falling back to manual HP strategy."
    )
    # Fall back: train with default params, skip Optuna
    ...
```

This feasibility data is recorded in `optuna_results.json` so the Reviewer and Planner can see compute constraints.

Feasibility policy:
- If the probe suggests the requested budget is infeasible, reduce `effective_n_trials`.
- If a single trial is estimated to exceed the time limit, fall back to `manual` (train once with baseline/default hyperparameters).
- For slow models, prefer proxy tuning during Optuna (row subsample and/or capped estimators) followed by full retrain.

---

## Study Convergence Early Stopping

Beyond per-trial pruning, the entire study stops early if trials plateau:

```python
class ConvergenceCallback:
    """Stop study if no meaningful improvement in last N trials."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta

    def __call__(self, study, trial):
        if len(study.trials) < self.patience + 5:
            return  # too early to judge
        recent = [t.value for t in study.trials[-self.patience:]
                  if t.value is not None]
        if not recent:
            return
        best_recent = max(recent) if study.direction == optuna.study.StudyDirection.MAXIMIZE else min(recent)
        if abs(study.best_value - best_recent) < self.min_delta:
            study.stop()
```

---

## Post-Study Analysis (Hybrid: Deterministic + LLM)

### Deterministic layer: `src/tuning/analysis.py`

Computes from the Optuna study object:

1. **Param importance** — `optuna.importance.get_param_importances(study)` (fANOVA-based)
2. **Range exhaustion** — for each param, find the effective range where 90% of top-20% trials fell. If this range is < 30% of the search space, flag as "exhausted" with a recommended narrowed range.
3. **Anti-patterns** — identify param combinations where all trials were pruned or scored in bottom 20%. Output as structured rules: `{"condition": "max_depth > 8 AND learning_rate > 0.1", "outcome": "always_underperformed", "n_trials": 5}`.
4. **Convergence info** — was the study stopped early? How many trials before best was found?

Safety guardrails for analysis:
- Only compute param importance / anti-pattern mining if there are enough completed trials (e.g., >= 10 COMPLETE trials). Otherwise emit empty structures plus a short reason.

### LLM layer: Coder agent or Report Builder

After the deterministic analysis writes structured fields to `optuna_results.json`, the Model Report Builder (or the Coder's script epilogue) writes a short narrative summary:

> "Optuna explored 42 trials for XGBoost. Learning rate was the most important HP (42% importance). The optimal region was learning_rate=[0.02, 0.08], max_depth=[4,7]. Trials with max_depth > 8 consistently underperformed — avoid in future runs. The study converged after trial 27; remaining trials confirmed the optimum. Budget was not limiting."

This narrative is included in `optuna_results.json` under `summary_narrative` and also appears in `model-report.md`.

---

## optuna_results.json Schema (Contract 7)

```json
{
  "schema_version": "1.0.0",
  "study_name": "<str>",
  "model_family": "<str>",
  "direction": "maximize | minimize",
  "metric_name": "<str>",

  "tuning_protocol": {
    "tuning_split": {"method": "stratified | random", "val_ratio": "<float>"},
    "uses_inner_split": "<bool>",
    "proxy_training": {
      "enabled": "<bool>",
      "row_subsample_fraction": "<float | null>",
      "max_estimators": "<int | null>"
    }
  },

  "budget": {
    "requested_n_trials": "<int>",
    "effective_n_trials": "<int>",
    "time_limit_s": "<int>",
    "budget_limited": "<bool>"
  },

  "timing_probe": {
    "probe_duration_s": "<float>",
    "subsample_fraction": "<float>",
    "estimated_per_trial_s": "<float>",
    "fallback_to_manual": "<bool>"
  },

  "results": {
    "n_trials_completed": "<int>",
    "n_trials_pruned": "<int>",
    "time_elapsed_s": "<float>",
    "converged_early": "<bool>",
    "best_trial_number": "<int>"
  },

  "best_trial": {
    "number": "<int>",
    "value": "<float>",
    "params": {"<param_name>": "<value>"}
  },

  "comparisons": {
    "baseline_inner": {"metric": "<str>", "value": "<float>"},
    "best_inner": {"metric": "<str>", "value": "<float>"},
    "final_outer": {"metric": "<str>", "value": "<float>"}
  },

  "param_importance": {"<param_name>": "<float>"},

  "param_ranges_exhausted": [
    {
      "param": "<str>",
      "search_range": ["<low>", "<high>"],
      "effective_range": ["<low>", "<high>"],
      "recommendation": "<str>"
    }
  ],

  "anti_patterns": [
    {
      "condition": "<str>",
      "outcome": "<str>",
      "n_trials": "<int>"
    }
  ],

  "summary_narrative": "<str — LLM-generated interpretive summary>",

  "all_trials": [
    {
      "number": "<int>",
      "params": {"<param_name>": "<value>"},
      "value": "<float | null>",
      "duration_s": "<float>",
      "state": "COMPLETE | PRUNED | FAIL"
    }
  ]
}
```

---

## Integration Touchpoints

### Planner agent updates

Add to Planner instructions:
- **New decision: when to use HP tuning.** Clear guidance table (tune vs don't tune, from the brainstorm).
- **Reading prior optuna_results.json.** When continuing the same model family across iterations, read `param_importance`, `param_ranges_exhausted`, and `anti_patterns`. Use these to set `optuna_search_space_overrides` — narrowing dead regions but not collapsing the full space (explore-exploit balance).
- **Setting optuna_budget.** Consider dataset size and model complexity. Large datasets with slow models → lower `n_trials` or tighter `time_limit_s`.
- **Constraint: no HP tuning with ensembles.** When plan uses StackingClassifier or similar, `hyperparameter_strategy` must be `manual`.

### Coder agent updates

Add to Coder instructions:
- **Template selection.** If `hyperparameter_strategy: optuna`, use `model_optuna.py` template instead of `model.py` template.
- **Search space merging.** Read defaults from `src/tuning/search_spaces.py`, apply overrides from plan YAML's `optuna_search_space_overrides`.
- **Output contract.** Must produce `optuna_results.json` in addition to standard outputs.
- **Feasibility probe.** Always include the timing probe in generated code.
- **Pruning callbacks.** Use model-family-specific callbacks from `PRUNING_CALLBACKS` registry.
- **Inner-split objective.** The Optuna objective must create an inner tuning split from training data (do not evaluate trials on the pipeline validation split).
- **Warm-start enqueues.** Enqueue baseline params and previous best params (when available) before free search.

### Executor agent — no changes

Standard two-stage repair handles Optuna scripts (they're just Python). Output validator extended to check for `optuna_results.json` when config says `hyperparameter_strategy: optuna`.

Dependency note:
- The current Executor does not install per-iteration dependencies. `optuna` must be available in the environment (repo `requirements.txt`).

### Model Report Builder updates

When `optuna_results.json` exists:
- Add "Hyperparameter Tuning" section to `model-report.json` and `model-report.md`
- Include: trials completed/pruned, best params, param importance, budget status, default vs tuned metric comparison
- Surface anti-patterns and range exhaustion for the Reviewer

### Reviewer-Router — no direct changes

Reviews the model report as usual. The HP tuning section gives additional context for judging whether to continue tuning, pivot, or finalise.

---

## Minor Milestones (revised)

| # | Deliverable |
|---|---|
| M13.1 | `src/tuning/search_spaces.py` — predefined search spaces + pruning callback registry per model family. Unit tests for all spaces. |
| M13.2 | `src/tuning/analysis.py` — post-study analysis (param importance, range exhaustion, anti-patterns, convergence). Unit tests. |
| M13.3 | `optuna_results.json` Contract 7 schema added to `artifact-contracts.md`. Schema validator in `src/tuning/validator.py`. |
| M13.4 | `templates/iteration/model_optuna.py` — Optuna-aware model template with timing probe, convergence callback, search+retrain flow. |
| M13.5 | Plan YAML schema extended — `hyperparameter_strategy`, `optuna_budget`, `optuna_search_space_overrides`. Plan validator (`src/planning/validator.py`) updated. |
| M13.6 | Coder agent instructions updated — template selection, search space merging, optuna output contract. |
| M13.7 | Planner agent instructions updated — when to tune, reading prior optuna_results.json, setting overrides, ensemble constraint. |
| M13.8 | Executor output validator extended — check for `optuna_results.json` when HP tuning active. |
| M13.9 | Model Report Builder updated — HP tuning section in `model-report.json` and `model-report.md`. |
| M13.10 | End-to-end smoke test on Titanic — Planner requests Optuna for XGBoost, full loop completes, optuna_results.json is valid, report includes HP tuning section. |

---

## Dependencies

- **Requires:** M0–M9 (all complete)
- **No dependency on:** M10 (cost tracking), M11 (human-in-loop), M12 (AutoGluon baseline)
- **Enhances:** M17 (benchmark validation) — tuned models should outperform default HP models

Implementation dependency:
- Add `optuna` to the repo `requirements.txt` (global env). Start simple; additional model-family packages can be added later as needed.

## Risks

| Risk | Mitigation |
|------|------------|
| Optuna import errors in generated code | Ensure `optuna` is in project `requirements.txt`. Executor handles import errors via Stage 1 repair. |
| Timing probe gives bad estimates | Conservative 2x multiplier. Fallback to manual if probe > time_limit. |
| Generated Optuna code is subtly wrong (e.g., data leakage in objective) | Template is reviewed and tested. Coder fills in model-specific parts only. |
| Tuning overfits by double-dipping validation | Objective uses an inner tuning split carved from training; outer pipeline validation is used only once for final evaluation. This protocol is recorded in `optuna_results.json`. |
| LLM ignores `hyperparameter_strategy: optuna` in plan | Coder agent instructions make this explicit. Output validator enforces optuna_results.json presence. |
| Too many trials produce unreadable results | Cap `all_trials` output. Summary narrative distills key insights. |

---

## Open Questions / Implementation Notes

- PRD consistency: ensure the PRD’s Optuna tuner section describes the integrated-in-`model.py` approach (Option A) to avoid implying an extra pipeline step.
