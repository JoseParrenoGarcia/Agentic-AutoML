# M3 Planning Layer + M0.4 Rule Stubs — Refined Plan

**Status:** Ready for implementation
**Date:** 2026-04-10
**Milestone:** M3 (Planning Layer) + M0.4 (Rule stubs)
**Supersedes:** `m3-planning-layer-plan.md`, `m3-plan-independent.md`

**TL;DR:** Build the Planner agent. Four phases: (1) foundation — plan schema, validator, rule stubs, memory scaffold, all in parallel; (2) planner agent; (3) tests (can overlap with Phase 2); (4) smoke-test on Titanic producing `iteration-1.yaml`. M3 is not done until the smoke test passes.

---

## Context

### What already exists (M2 complete ✅)

| Artifact | Path |
|----------|------|
| Dataset profiler | `src/analysis/profiler.py` |
| Plot generator | `src/analysis/plots.py` |
| Dataset analyser agent | `.claude/agents/dataset-analyser.md` |
| Titanic profile JSON | `projects/titanic/artifacts/data/profile.json` |
| Titanic profile narrative | `projects/titanic/artifacts/data/profile.md` |
| EDA plots (22 files) | `projects/titanic/artifacts/data/plots/` |
| Project config | `projects/titanic/project.yaml` |
| Authoring skills (4) | `.claude/skills/create-{agent,hook,rule,skill}/SKILL.md` |
| Authoring guardrail | `.claude/rules/authoring.md` |
| Test suite (profiler + plots) | `tests/analysis/` |
| Test fixtures (CSV) | `tests/fixtures/` |
| Requirements | `requirements.txt` — PyYAML, scikit-learn, pandas already present |

### profile.json key sections (planner inputs)

The profile at `projects/titanic/artifacts/data/profile.json` has the following top-level keys the planner must consume:

| Key | What it contains | Why the planner needs it |
|-----|-----------------|------------------------|
| `columns[]` | Per-column objects: `name`, `inferred_semantic_type`, `description`, `basic_stats`, `null_analysis`, `cardinality`, `outliers`, `risk_flags` | Core feature-level facts for every planning decision |
| `correlation.pearson.top_pairs` | Ranked numeric-numeric correlation pairs | Interaction feature candidates |
| `correlation.cramers_v.top_pairs` | Ranked categorical-categorical association pairs | Categorical interaction candidates |
| `feature_risk_flags.flagged_columns` | Columns with high skew or other flags | Transformation candidates (log, Winsorize) |
| `feature_risk_flags.near_duplicate_pairs` | Feature pairs with r ≥ 0.98 | Redundancy — drop one or both |
| `target_validation` | Class counts, imbalance ratio | Task confirmation, class weighting decisions |
| `leakage_flags.flagged_columns` | Columns with r > 0.95 to target | **Hard exclusion** — must NOT appear in feature steps |
| `mutual_information.scores` | MI score per feature, descending | Primary signal of feature-target relevance |

### Titanic profile highlights

These concrete values ground the smoke-test checklist and serve as the context the planner will reason over for iteration 1:

- **Severe null:** Cabin 77.10% missing — needs special handling or exclusion
- **Moderate null:** Age 19.87% missing, skewness 0.39 — needs imputation
- **Trivial null:** Embarked 0.22% missing
- **High cardinality:** Name (891 unique), Ticket (681 unique), Cabin (147 unique) — need engineering or exclusion
- **High skew + outliers:** Fare (skewness 4.79, outlier_pct 13.02%), SibSp, Parch flagged
- **Top MI signals:** Sex (0.1509) > Fare (0.139) > Pclass (0.0409) > Age (0.0351)
- **No leakage detected** — `leakage_flags.flagged_columns` is empty
- **Task:** binary classification, target = `Survived`, 549 / 342 class split, `is_imbalanced: false`

---

## Design Decisions

### Why a single planner agent (not two)

DS-STAR (Google, 2025) uses two distinct prompts for planning:
1. **Init prompt** — no execution history, just data summaries → produce the first step
2. **Continuation prompt** — accumulated history + execution results → produce the next step

This separation reflects a real cognitive difference: "what's the best first experiment?" is a different task from "given what we just learned, what should we try next?"

Both tasks live inside one agent for M3 because:
- **Only the init path is exercisable at M3.** No iteration history exists until M5 (Executor) produces run results. The continuation path is dead code until then.
- **We cannot tune the continuation prompt without real data.** Splitting prematurely means maintaining two agents with no feedback signal on the second one.
- **The branch is trivial.** A single `if iteration > 1` check is simpler than orchestrating two agents.
- **DS-STAR operates step-by-step; we operate iteration-by-iteration.** DS-STAR's planner emits one step, the executor runs it, then the planner emits the next step. Our planner emits a complete iteration plan (all feature steps + model steps + eval criteria), then the Coder implements it all. The cognitive load difference between the two prompts is smaller in our design because both produce the same structured output.

**Future signal:** When M7–M8 produce real iteration history, evaluate whether plan quality for iteration ≥ 2 degrades compared to iteration 1. If it does, split into `planner-init.md` and `planner-refine.md` with deterministic dispatch (a shell oneliner or 10-line Python function checks `artifacts/plans/iteration-*.yaml | wc -l`). This dispatch is deterministic — no LLM orchestrator needed. A full LLM orchestrator ("Iteration Controller" from PRD §5) is deferred to M9–M10.

### DS-STAR differences that matter

| Dimension | DS-STAR | Agentic-AutoML (this plan) |
|-----------|---------|---------------------------|
| Planning granularity | One step at a time | Full iteration plan |
| Prompt structure | Two templates (init / continue) | One agent, conditional branch |
| Input | Question + file summaries | `project.yaml` + `profile.json` + memory |
| Output | Free-text next action | Structured YAML with required fields |
| Loop control | Planner called after every step execution | Planner called once per iteration |
| Reasoning depth | "Just propose a simple next step" | Grounded hypotheses, profile citations, rationale |

Key takeaway we adopt: **the planner prompt is the most important deliverable, not just the schema.** The agent `.md` body is effectively the prompt. Phase 2 must produce a high-quality, specific instruction set — not a generic "plan well" instruction.

### Lessons from AutoKaggle and Karpathy's autoresearch

Full research reports at [tasks/research/autokaggle-planning-analysis.md](tasks/research/autokaggle-planning-analysis.md) and [references/autoresearch-analysis.md](references/autoresearch-analysis.md).

Two systems were analysed: **ShaneZhong/autokaggle** (a Claude Code Kaggle agent with 3 persistent agents — Builder, Reviewer, Researcher) and **Karpathy's autoresearch** (a minimal infinite-loop research agent driven by a single `program.md` file). Neither has a separate "Planner" agent identical to ours, but both encode planning logic that reveals gaps in our design.

#### Concepts we adopt into the M3 planner

| Concept | Source | How we integrate it |
|---------|--------|---------------------|
| **Baseline-first protocol** | autoresearch | Iteration 1 must always establish a baseline before any creative experimentation. The planner must explicitly frame iteration 1 as "establish baseline performance" — not "try clever feature engineering." |
| **Simplicity criterion** | autoresearch | When the planner proposes feature steps, each must justify its complexity cost. A marginal improvement from a complex transform is worse than a simpler approach with comparable gain. The agent instructions must encode: "all else being equal, simpler is better." |
| **Scope guardrails (CAN / CANNOT)** | autoresearch, academic AutoKaggle | The agent body must include explicit CAN/CANNOT lists. The planner CAN propose feature transforms, model choices, hyperparameters. The planner CANNOT propose modifying the evaluation harness, changing the train/val/test split strategy, or adding new Python dependencies. |
| **Configurable thresholds with sensible defaults** | academic AutoKaggle | Instead of hardcoding "drop if > 50% null," the feature step decision table references project-level or profile-derived thresholds. Defaults: null drop threshold = 0.5, skewness threshold for log transform = 1.0, cardinality threshold for encoding switch = 10. These live in the decision table, not in `project.yaml` (no schema change needed). |
| **Feature tracking across iterations** | academic AutoKaggle | For iteration > 1, the continuation path must note which features existed before and after the prior iteration. The planner reads `run-history.jsonl` which contains `feature_changes` per record. |
| **Cost-benefit framing in plans** | ShaneZhong/autokaggle | Each hypothesis should note its complexity cost, not just expected impact. "H2: Log-transform Fare (skewness 4.79) — low complexity, expected moderate impact on LR performance." |

#### Concepts we note but defer (not M3)

| Concept | Source | Why deferred |
|---------|--------|-------------|
| CV/LB divergence tracking | ShaneZhong/autokaggle | Requires execution results (M5+) and submission infrastructure |
| Ensemble-aware planning (OOF correlation) | ShaneZhong/autokaggle | No ensemble until iteration 3+ at earliest; Reviewer (M7) is the right place |
| Fold-1 kill gates as pre-commit diagnostic | ShaneZhong/autokaggle | Executor concern (M5), not planner |
| Tool-grounded planning via RAG | academic AutoKaggle | No tool library yet; Coder (M4) owns tool selection |
| Retrospective every N rounds | ShaneZhong/autokaggle | Reviewer (M7) and Local Maxima Challenger (M7.5) own this |
| Agent re-spawn for context hygiene | ShaneZhong/autokaggle | Not needed until we run multi-iteration loops (M5+) |
| "Think harder" escape hatch | autoresearch | Relevant to the full loop, not the planner in isolation. The Local Maxima Challenger (M7.5) serves this role. |

#### Concepts we deliberately reject

| Concept | Source | Why rejected |
|---------|--------|-------------|
| Infinite autonomous loop without human gates | autoresearch, ShaneZhong | PRD §2.5 principle 5: "Human review remains first class." Our system has human gates at plan approval and finalization. |
| No structured plans (LLM reasoning IS the plan) | autoresearch | Works when search space is narrow (one file, one metric). Our tabular ML has many orthogonal dimensions — structured YAML plans prevent the agent from losing track. |
| Score-based retry of the planner | academic AutoKaggle | Better to produce a good plan via better prompts and grounding than to retry poor ones. |
| Phase-only planning (plan "how to do data cleaning" globally) | academic AutoKaggle | Our experiment-iteration model is more aligned with hypothesis testing than phase-gated workflows. |

### Other decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Validator dependencies | PyYAML + custom exceptions only | No new deps. `pydantic` not in requirements.txt and adds weight. |
| Memory tooling | No Python module | Planner reads raw JSONL/markdown directly. Intelligence stays in the agent. Python retrieval deferred to M8. |
| Planner tools | `Read` + `Write` only | No Bash needed. Pure reasoning + file I/O. Restricting tools prevents scope creep. |
| Rule stubs timing | Phase 1 (before agent) | Planner agent references both rules. They must exist before the agent is authored. |
| `feature_steps` cardinality | Empty list is valid | A plan that changes only the model (no new feature steps) must be expressible. |
| Iteration 1 model choice | LR + RF (not boosting) | Interpretable baselines first. Diagnose data issues before throwing boosting at it. |
| Agent model | `claude-sonnet-4-5` | Sufficient for planning. Upgrade to Opus only if smoke test plan quality is poor. |
| `maxTurns` | 12 | 10 is tight if the agent needs to create `artifacts/plans/` directory. 12 gives margin. |
| Rationale `.md` file | Required (not optional) | Auditability is a core principle (PRD §2.5). Every decision must be traceable. |
| Smoke test | Blocking — M3 is not done until it passes | Tests alone do not prove the agent works end-to-end. |

---

## Phase 1 — Foundation (all 4 steps in parallel)

All Phase 1 steps are independent. Build them together.

---

### 1a. Plan schema template

**File:** `templates/plans/iteration.yaml`

A filled-in reference template showing all required fields with placeholder comments. Used as both human documentation and the template the planner copies when writing plans.

```yaml
# Agentic-AutoML — Experiment Plan
# Reference: PRD §6.2
# Generated by: planner agent
# Validated by: src/planning/validator.py

iteration: 1
objective: "<one sentence: what this experiment is testing>"

hypotheses:
  - id: H1
    description: "<specific, testable statement grounded in a profile finding>"
    expected_impact: "<expected effect on primary metric>"
  - id: H2
    description: "<second hypothesis if applicable>"
    expected_impact: "<expected effect>"

feature_steps:
  - name: "<step name>"
    action: "<what to do — e.g. median impute Age, drop Cabin, ordinal encode Pclass>"
    rationale: "<why — cite the specific profile observation>"

model_steps:
  - algorithm: "<algorithm name>"
    hyperparameters:
      "<param>": "<value>"
    rationale: "<why this algorithm for this dataset>"

evaluation_focus: "<what to examine most closely this iteration>"
expected_win_condition: "<metric threshold that confirms success — e.g. val AUC-ROC > 0.80>"
rollback_or_stop_condition: "<what signals this approach is not working>"
```

**Field semantics:**
- `iteration` — positive int, 1-indexed
- `objective` — non-empty string
- `hypotheses` — non-empty list; each entry has `id`, `description`, `expected_impact` (all non-empty strings)
- `feature_steps` — list, **may be empty** (valid edge case: model-only change)
- `model_steps` — non-empty list; each entry has `algorithm` (non-empty string), `hyperparameters` (dict, may be `{}`), `rationale` (non-empty string)
- `evaluation_focus` — non-empty string
- `expected_win_condition` — non-empty string
- `rollback_or_stop_condition` — non-empty string

---

### 1b. Plan validator

**Files:**
- `src/planning/__init__.py` — empty module init
- `src/planning/validator.py`

**Interface:**

```python
from pathlib import Path
from typing import Union

class PlanValidationError(Exception):
    """Raised when an experiment plan fails schema validation."""

def validate_plan(plan: Union[str, Path, dict]) -> dict:
    """
    Validate an experiment plan against the required schema.

    Args:
        plan: Path to a YAML file (str or Path) or an already-parsed dict.

    Returns:
        Parsed plan dict if valid.

    Raises:
        PlanValidationError: Descriptive message on first violation found.
        FileNotFoundError: If a path is given and the file does not exist.
    """
```

**Validation rules (checked in order, fail fast on first violation):**

1. If `plan` is str or Path → load with `yaml.safe_load`. Non-existent file → `FileNotFoundError`.
2. **Required top-level keys:** `iteration`, `objective`, `hypotheses`, `feature_steps`, `model_steps`, `evaluation_focus`, `expected_win_condition`, `rollback_or_stop_condition`. Any missing → `PlanValidationError("Missing required field: <key>")`.
3. `iteration` must be `int` and `>= 1`. Wrong type or value → error.
4. String fields (`objective`, `evaluation_focus`, `expected_win_condition`, `rollback_or_stop_condition`) must be non-empty strings. Empty string or wrong type → error.
5. `hypotheses` must be a non-empty list. Each item must have `id`, `description`, `expected_impact` — all non-empty strings. Empty list or missing sub-field → error.
6. `feature_steps` must be a list (may be empty). If non-empty, each item must have `name`, `action`, `rationale` — all non-empty strings.
7. `model_steps` must be a non-empty list. Each item must have `algorithm` (non-empty string), `hyperparameters` (dict, may be `{}`), `rationale` (non-empty string). Empty list → error.
8. Return the dict on success.

**Dependencies:** PyYAML only (already in `requirements.txt`).

---

### 1c. Rule stubs (M0.4)

**Files:** `.claude/rules/coding-rules.md` and `.claude/rules/artifact-contracts.md`

**Before writing either file:** Read `.claude/skills/create-rule/SKILL.md` and follow its conventions exactly.

#### `coding-rules.md`

**Scope:** Path-scoped to `runs/`. Applies to all Python code generated by the Coder agent.

Ten rules from PRD §10, written as enforceable constraints (not guidelines):

1. No notebooks. Python scripts only.
2. Every run must be executable via `python main.py` from the run directory.
3. All metrics written to `outputs/metrics.json`. Schema: `{"primary": {"name": str, "value": float}, "secondary": {name: float}, "train": {name: float}, "validation": {name: float}}`.
4. Predictions written to `outputs/predictions.csv` with index alignment to input data.
5. Feature engineering applied consistently to train, validation, and test splits. No train-only transforms leaking into val/test.
6. No hardcoded paths. All paths from `config.yaml`.
7. Logging via Python `logging` module to both stdout and `execution/log.txt`.
8. Random seed set at top of `main.py` and recorded in `config.yaml`.
9. No internet access during training. No `requests`, `urllib`, or external API calls.
10. All dependencies declared in run-level `requirements.txt`.

#### `artifact-contracts.md`

**Scope:** Unconditional — applies to all agents that read or write structured artifacts.

Four contracts:

**Contract 1: `profile.json`** (written by Dataset Analyser, read by Planner)
- Required top-level keys: `profiler_version`, `generated_at`, `source`, `columns`, `correlation`, `target_validation`, `leakage_flags`, `feature_risk_flags`, `mutual_information`
- Each column object must have: `name`, `pandas_dtype`, `inferred_semantic_type`, `description`, `sample_values`, `basic_stats`, `null_analysis`, `cardinality`, `risk_flags`
- Agents must fail loudly (not silently skip) if any required key is missing

**Contract 2: `iteration-<n>.yaml`** (written by Planner, read by Coder)
- Required fields: per the plan schema template in `templates/plans/iteration.yaml`
- Validated by `src/planning/validator.py` before the Coder agent reads it
- Must be written to `artifacts/plans/iteration-<n>.yaml` (1-indexed)

**Contract 3: `run-history.jsonl`** (written by post-run hooks, read by Planner on iteration > 1)
- Append-only. Each line is a self-contained JSON object.
- Required fields per record: `iteration` (int), `timestamp` (ISO 8601), `status` (completed|failed), `plan_summary` (str), `primary_metric` (object with `name`, `value`, `delta`), `model_family` (str), `reviewer_verdict` (str), `router_decision` (str)
- Agents must not rewrite or delete existing lines

**Contract 4: `model-report.json`** (stub — written by Model Report Builder at M6)
- Required fields (stub, to be expanded): `iteration` (int), `primary_metric` (object), `secondary_metrics` (object), `feature_importance` (object), `overfitting_check` (object)
- Stub included here so Reviewer (M7) knows what to expect

---

### 1d. Memory scaffold

**Files:**
- `projects/titanic/memory/run-history.jsonl` — empty file (populated from M5 onwards)
- `projects/titanic/memory/decision-log.md`:

```markdown
# Decision Log — titanic
<!-- Append-only. One section per completed iteration. -->
<!-- Format: ## Iteration N\n<one-paragraph summary of decisions and outcomes> -->
```

No Python module. The planner agent reads these files directly.

---

## Phase 2 — Planner Agent (after Phase 1)

**Depends on:** Phase 1 complete (schema defined, validator exists, rule stubs exist, memory scaffold in place).

**File:** `.claude/agents/planner.md`

**Before writing:** Read `.claude/skills/create-agent/SKILL.md` and follow its conventions exactly.

### Frontmatter

```yaml
---
name: planner
description: >
  Converts project context into a structured experiment plan (iteration-<n>.yaml).
  Invoke after dataset-analyser has produced profile.json. Reads project.yaml,
  artifacts/data/profile.json, and memory/ (iteration > 1). Writes
  artifacts/plans/iteration-<n>.yaml and artifacts/plans/iteration-<n>.md.
tools:
  - Read
  - Write
model: claude-sonnet-4-5
maxTurns: 12
---
```

**Tool rationale:** Planning is pure reasoning + file I/O. No Bash required. Restricting to Read + Write enforces single responsibility and prevents the agent from running code, installing packages, or modifying data.

### Agent Scope Guardrails

The agent `.md` body must include explicit CAN/CANNOT lists to prevent scope creep:

**The planner CAN:**
- Propose feature transforms (imputation, encoding, scaling, log transforms, exclusions)
- Choose model algorithms and starting hyperparameters
- Define evaluation focus, win conditions, and rollback conditions
- Reference profile statistics, MI scores, and risk flags to justify decisions
- Propose interaction features when correlation or domain evidence supports them

**The planner CANNOT:**
- Write or execute Python code
- Modify the evaluation harness, metric definitions, or train/val/test split strategy
- Add new Python dependencies beyond what's in `requirements.txt`
- Propose changes to `project.yaml` or the profiler output
- Plan actions that belong to future phases (ensembling before baselines exist, hyperparameter grid search on iteration 1)
- Include any column from `leakage_flags.flagged_columns` in feature steps — no exceptions

### Agent Workflow — Init Path (iteration = 1)

The agent body must encode the following workflow. At M3 only the init path is exercisable. The continuation (iteration > 1) path is included for completeness but will only be tested at M5+.

**Iteration 1 is always a baseline.** The planner must frame iteration 1 as establishing a credible baseline — not as a creative feature engineering exercise. The goal is to get a working pipeline with sensible defaults so that subsequent iterations have a reference point. This mirrors the baseline-first protocol from autoresearch: "your first run should always establish the baseline."

**Step 1 — Discover iteration number**

List `<project_root>/artifacts/plans/`. Count files matching `iteration-*.yaml`. Next iteration = count + 1. If directory does not exist, create it; iteration = 1.

**Step 2 — Read inputs (all reads before any reasoning)**

Read upfront — do not interleave reads with planning reasoning:

- `<project_root>/project.yaml` → extract `name`, `task_type`, `target_column`, `description`
- `<project_root>/artifacts/data/profile.json` → read in full. Key sections to focus on:
  - `mutual_information.scores` — primary signal of feature-target relevance, in descending order
  - `leakage_flags.flagged_columns` — these must be excluded from all feature steps (hard constraint)
  - `feature_risk_flags.flagged_columns` — skewed features needing transformation
  - `feature_risk_flags.near_duplicate_pairs` — feature pairs with r ≥ 0.98 (redundancy candidates)
  - `columns[]` where `null_analysis.null_pct > 0` — imputation candidates
  - `columns[]` where `cardinality.is_high_cardinality == true` — encoding or exclusion candidates
  - `correlation.pearson.top_pairs` and `correlation.cramers_v.top_pairs` — interaction candidates
  - `target_validation` — class balance, task confirmation

### Agent Workflow — Continuation Path (iteration > 1)

If iteration > 1, additionally read:
- `<project_root>/memory/run-history.jsonl` — all lines; parse as JSONL; extract:
  - What was tried and what metrics were achieved
  - What failed and what was explicitly abandoned (status = `failed` or `discarded`)
  - `feature_changes` per record — know which features exist now vs. before
  - If 3+ consecutive iterations show declining delta on primary metric, note plateau signal
- `<project_root>/memory/decision-log.md` — narrative context from prior iterations
- Identify the delta: what changed between the last plan and its results? What hypotheses were confirmed or refuted?
- Do not repeat any approach already recorded in `run-history.jsonl` as tried. If the same technique is proposed, it must differ in a material way (different parameters, different target column, combined with a new step).

If `knowledge-base/` contains `.md` files relevant to the task type, read them.

### Agent Workflow — Common Steps (both paths)

**Step 3 — Form hypotheses (reason before writing)**

Before writing anything, reason through the profile findings and produce 2–3 testable hypotheses. Each must be:
- **Grounded:** cites a specific profile finding (column name, statistic, or flag). Example: "Age has 19.87% nulls with skewness 0.39, suggesting median imputation is appropriate"
- **Testable:** names the feature(s) or technique being tested
- **Attributable:** small enough in scope that if metrics improve, the cause is identifiable
- **Non-redundant:** if iteration > 1, must not repeat an approach recorded as tried in `run-history.jsonl`

**Step 4 — Plan feature steps**

One step per decision. Each step must cite its profile evidence and justify its complexity cost (simpler is better — a marginal gain from a complex transform is worse than a simpler approach). Use this decision table with sensible default thresholds:

| Condition | Strategy | Default Threshold |
|-----------|----------|-------------------|
| `null_pct > 0`, numeric, low skew | mean imputation | skewness < 1.0 |
| `null_pct > 0`, numeric, high skew (in `feature_risk_flags`) | median imputation | skewness ≥ 1.0 |
| `null_pct > 0`, categorical | mode or dedicated `"Missing"` category | — |
| `null_pct > threshold` (severe missingness) | consider exclusion; document cost-benefit trade-off | threshold = 0.5 |
| `inferred_semantic_type = low_cardinality_categorical` | ordinal or one-hot encoding; justify choice | unique count < 10 |
| `inferred_semantic_type = high_cardinality_categorical` | frequency encoding, target encoding, or exclusion; justify | unique count ≥ 10 |
| `inferred_semantic_type = identifier` | exclude (PassengerId, Name, Ticket) | — |
| Column in `leakage_flags.flagged_columns` | **must exclude** — no exceptions | — |
| High skew + high `outlier_pct` | consider log transform or Winsorize; cite the skewness value | skewness > 1.0 |
| `near_duplicate_pairs` (r ≥ 0.98) | drop one of the pair; cite the correlation value | r ≥ 0.98 |

**Interaction features:** Do not add speculatively. Only if `correlation.pearson.top_pairs` or strong domain reasoning supports it. On iteration 1, prefer no interaction features — keep the baseline simple.

**Simplicity criterion:** When choosing between two approaches for the same problem (e.g., median impute vs. KNN impute for Age), prefer the simpler one unless there is strong evidence the complex approach will yield meaningfully better results. Cite the expected benefit.

**Step 5 — Plan model steps**

For `task_type = binary_classification`:
- **Iteration 1:** Two algorithms — `LogisticRegression` (interpretable baseline) and `RandomForestClassifier` (non-linear baseline). Sensible default hyperparameters (not a full grid). Rationale for each must reference dataset characteristics (e.g., "mixed numeric/categorical features after encoding suggest a tree-based model as complement to the linear baseline").
- **Iteration 2+:** Introduce `LightGBM` or `XGBoost` if baselines are established. Upgrade hyperparameters only if prior iteration's performance warrants it.

For `task_type = multiclass_classification`: Same approach; macro F1 as primary metric.
For `task_type = regression`: `Ridge` and `RandomForestRegressor` as baselines; RMSE as primary metric.

**Step 6 — Set evaluation metadata**

- `evaluation_focus`: what to examine most closely this iteration (e.g., "feature importance distribution to identify dominant features and validate that Sex/Fare dominate as MI scores suggest")
- `expected_win_condition`: concrete metric threshold (e.g., "validation AUC-ROC > 0.80" for Titanic baselines)
- `rollback_or_stop_condition`: what would indicate failure (e.g., "validation AUC-ROC < 0.70 or train/val gap > 0.15 suggesting overfitting")

**Step 7 — Self-validate before writing**

Before writing any files, verify the plan against these constraints:
- All 8 required top-level fields are present and non-empty
- `iteration` matches the number determined in Step 1
- No column from `leakage_flags.flagged_columns` appears in any `feature_steps` entry
- `model_steps` has at least one entry with `algorithm`, `hyperparameters`, `rationale`
- `hypotheses` has at least one entry with `id`, `description`, `expected_impact`
- If iteration = 1: plan frames itself as a baseline (no exotic models, no complex ensembles)
- If iteration > 1: no hypothesis repeats an approach already tried in `run-history.jsonl`
- Every feature step references a specific column and profile finding (not generic advice)
- Every hypothesis includes a complexity assessment (low / medium / high)

**Step 8 — Write outputs**

1. Write `<project_root>/artifacts/plans/iteration-<n>.yaml` — machine-readable plan (**required**)
2. Write `<project_root>/artifacts/plans/iteration-<n>.md` — human-readable rationale narrative (**required**). Must explain: why each hypothesis was chosen, what profile finding drove each feature step, and what the expected behaviour of each model step is.

**Output contract (print exactly on completion):**
```
✓ Plan written
plan:      <path>/artifacts/plans/iteration-N.yaml
rationale: <path>/artifacts/plans/iteration-N.md
iteration: N
hypotheses: <count>
feature_steps: <count>
model_steps: <count>
```

If any required output file could not be written, print `✗ FAILED: <reason>` and stop.

---

## Phase 3 — Tests (after Phase 1b; can overlap with Phase 2)

**Depends on:** `src/planning/validator.py` exists (Phase 1b).

**Files:**
- `tests/planning/__init__.py` — empty
- `tests/planning/test_plan_schema.py` — 13 test cases
- `tests/fixtures/valid_plan.yaml` — complete well-formed plan for Titanic iteration 1
- `tests/fixtures/invalid_plan_missing_objective.yaml` — same plan with `objective` removed

### Test cases

```python
# tests/planning/test_plan_schema.py

def test_valid_plan_passes():
    """Load valid_plan.yaml, assert no exception, return value is dict."""

def test_valid_plan_as_dict_passes():
    """Pass a pre-parsed dict directly, assert passes."""

def test_missing_objective_raises():
    """PlanValidationError with message mentioning 'objective'."""

def test_missing_hypotheses_raises():
    """PlanValidationError with message mentioning 'hypotheses'."""

def test_missing_model_steps_raises():
    """PlanValidationError with message mentioning 'model_steps'."""

def test_empty_hypotheses_list_raises():
    """hypotheses = [] should raise PlanValidationError."""

def test_empty_model_steps_list_raises():
    """model_steps = [] should raise PlanValidationError."""

def test_feature_steps_may_be_empty():
    """feature_steps = [] is valid — should NOT raise."""

def test_invalid_iteration_type_raises():
    """iteration = 'first' (string) should raise PlanValidationError."""

def test_iteration_zero_raises():
    """iteration = 0 should raise (must be >= 1)."""

def test_missing_sub_field_in_hypothesis_raises():
    """Hypothesis with no 'expected_impact' should raise PlanValidationError."""

def test_missing_rationale_in_model_step_raises():
    """Model step with no 'rationale' should raise PlanValidationError."""

def test_nonexistent_file_raises_file_not_found():
    """Passing a path that doesn't exist should raise FileNotFoundError."""
```

### Fixture: `tests/fixtures/valid_plan.yaml`

A complete, concrete Titanic iteration-1 plan with real field values — not placeholder text. This serves as both a test fixture and documentation of what good output looks like.

### Fixture: `tests/fixtures/invalid_plan_missing_objective.yaml`

Same as above but with the `objective` key removed entirely.

---

## Phase 4 — Smoke Test (after Phase 2 + Phase 3 pass)

**Step 1 — Run unit tests**
```bash
pytest tests/planning/ -v
```
All tests must pass before proceeding.

**Step 2 — Run planner agent on Titanic**
```
> run the planner agent on projects/titanic
```

Expected agent behaviour:
1. Detects iteration = 1 (no prior plans exist)
2. Reads `project.yaml` + `profile.json`
3. Reads `memory/run-history.jsonl` (empty — handles gracefully, does not crash)
4. Produces `artifacts/plans/iteration-1.yaml`
5. Produces `artifacts/plans/iteration-1.md`

**Step 3 — Programmatic validation**
```bash
python -c "
from src.planning.validator import validate_plan
plan = validate_plan('projects/titanic/artifacts/plans/iteration-1.yaml')
print('✓ Valid. iteration:', plan['iteration'])
print('  hypotheses:', len(plan['hypotheses']))
print('  feature_steps:', len(plan['feature_steps']))
print('  model_steps:', len(plan['model_steps']))
"
```

**Step 4 — Manual inspection checklist**

- [ ] `iteration` = 1
- [ ] `objective` frames this as a baseline establishment, referencing Titanic / survival prediction
- [ ] At least 2 hypotheses, each citing a specific profile finding and a complexity assessment
- [ ] Age null imputation addressed (19.87% nulls, skewness 0.39)
- [ ] Cabin handling addressed (77.10% nulls — likely exclusion with documented cost-benefit trade-off)
- [ ] Name and Ticket excluded or engineered (identifiers / high cardinality)
- [ ] No column from `leakage_flags.flagged_columns` in feature_steps (none expected, but confirm)
- [ ] Fare skew addressed (skewness 4.79, outlier_pct 13.02%)
- [ ] `model_steps` has LogisticRegression and/or RandomForestClassifier (not XGBoost on iteration 1)
- [ ] `expected_win_condition` contains a concrete AUC-ROC threshold
- [ ] Plan is Titanic-specific — not generic placeholder text
- [ ] `iteration-1.md` explains the reasoning in plain language, citing profile numbers
- [ ] No interaction features or complex stacking (iteration 1 = simple baseline)
- [ ] Feature steps prefer simple strategies (median impute over KNN impute, one-hot over target encoding)

---

## File Inventory

| File | Action | Phase |
|------|--------|-------|
| `templates/plans/iteration.yaml` | CREATE — schema template | 1a |
| `src/planning/__init__.py` | CREATE — empty module init | 1b |
| `src/planning/validator.py` | CREATE — validator + PlanValidationError | 1b |
| `.claude/rules/coding-rules.md` | CREATE — 10 coding rules (load create-rule skill first) | 1c |
| `.claude/rules/artifact-contracts.md` | CREATE — 4 artifact contracts (load create-rule skill first) | 1c |
| `projects/titanic/memory/run-history.jsonl` | CREATE — empty | 1d |
| `projects/titanic/memory/decision-log.md` | CREATE — header only | 1d |
| `.claude/agents/planner.md` | CREATE — planner agent (load create-agent skill first) | 2 |
| `tests/planning/__init__.py` | CREATE — empty | 3 |
| `tests/planning/test_plan_schema.py` | CREATE — 13 test cases | 3 |
| `tests/fixtures/valid_plan.yaml` | CREATE — Titanic iteration-1 fixture | 3 |
| `tests/fixtures/invalid_plan_missing_objective.yaml` | CREATE — invalid fixture | 3 |

**Total:** 12 new files. No existing files modified.

---

## Parallelism Map

```
Phase 1: [1a schema] [1b validator] [1c rules] [1d memory]  — all independent

                     ↓ all Phase 1 done
Phase 2: [planner agent]   ←── depends on schema (1a), validator (1b), rules (1c), memory (1d)
Phase 3: [tests]            ←── depends on validator (1b) only, can overlap with Phase 2

                     ↓ Phase 2 + Phase 3 done
Phase 4: [smoke test]      ←── sequential: pytest → agent run → validate → inspect
```

---

## Exit Criteria

M3 is complete when ALL of these are true:

1. `pytest tests/planning/ -v` passes (13/13 tests green)
2. `projects/titanic/artifacts/plans/iteration-1.yaml` exists and passes `validate_plan()`
3. `projects/titanic/artifacts/plans/iteration-1.md` exists and contains profile-grounded reasoning
4. Manual inspection checklist has no failures
5. Both rule stubs exist in `.claude/rules/`
6. Memory scaffold exists at `projects/titanic/memory/`
