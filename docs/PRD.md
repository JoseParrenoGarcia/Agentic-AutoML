# Agentic AutoML — Product Requirements Document

**Status:** Draft v0.4
**Last updated:** 2026-04-12

---

## 1. Executive Summary

Agentic AutoML is a Claude Code-centered orchestration system for end-to-end applied machine learning. It combines deterministic Python analysis scripts, structured multi-agent planning and execution loops, persistent project memory, and a reusable knowledge base for ML tactics and research findings. The system helps a data scientist move from raw tabular datasets to a well-documented, reproducible, reviewable model candidate.

The initial scope targets tabular supervised learning (classification and regression) on Kaggle-style datasets. The system operates as a controlled workflow where each stage emits artifacts that become inputs to the next stage, with human oversight at key decision points. Every run leaves a complete trail of decisions, code, metrics, and reviewer outputs.

The long-term vision is an internal AI co-worker for data scientists. The short-term proof point is strong performance and strong documentation on well-scoped benchmark datasets, compared against AutoGluon as the baseline for what brute-force AutoML can achieve.

---

## 2. Problem Statement & Motivation

Current machine learning work is slowed by four recurring bottlenecks:

1. **Limited exploration bandwidth.** A single human data scientist has limited time to explore many plausible modeling directions. Promising ideas go untested because the human cannot parallelize themselves.

2. **Fragmented experiment tracking.** Experiment tracking is frequently scattered across notebooks, ad hoc scripts, local notes, and partially remembered decisions. When revisiting a project after days or weeks, critical context is lost.

3. **Inconsistent LLM-driven coding.** LLM-assisted coding can be productive, but becomes unreliable when not grounded in deterministic tooling, clear contracts, and high-quality reference material. Without structure, agents hallucinate plausible but wrong strategies.

4. **Local-optima traps.** Teams repeatedly refine similar approaches instead of exploring materially different alternatives. Without deliberate divergence mechanisms, incremental polishing dominates over breakthrough exploration.

**Why now:** Claude Code's agent, skill, hook, and rule primitives provide a mature enough harness to build a structured agentic loop without inventing custom infrastructure. The combination of tool-use, persistent memory, and structured prompting makes a reliable DS agent feasible for the first time at reasonable cost.

**Why this approach:** Existing AutoML tools (AutoGluon, H2O, FLAML) solve a different problem — they brute-force search over model configurations. They do not understand *why* a feature matters, cannot creatively engineer domain-aware features, and produce opaque results. This system reasons about data, documents its decisions, and collaborates with humans in natural language.

---

## 2.5. Product Principles

1. **Start narrow, then expand.** Focus first on predefined datasets and supervised tabular ML instead of open-ended dataset discovery or every ML modality.
2. **Small loops beat giant plans.** Planning, execution, and correction happen in short, inspectable cycles. Each iteration should be attributable.
3. **Deterministic analysis first, agentic reasoning second.** Agents lean on scripts and structured artifacts, not free-form guesses. If a question can be answered by code, statistics, or schema inspection, the deterministic layer answers it.
4. **Every run must leave a trail.** Decisions, code versions, metrics, and reviewer outputs are stored in a reusable, auditable format. No ephemeral reasoning.
5. **Human review remains first class.** The system helps a principal data scientist inspect the logic, not hide it. Artifacts are the communication layer.
6. **Reuse research deliberately.** Papers, repos, and internal learnings are indexed into a knowledge base the system consults repeatedly — not dumped as loose links.
7. **Avoid context bloat.** Prefer scripts, markdown, JSON, and YAML artifacts over notebooks. Summaries passed to agents, not raw logs.

---

## 3. Goals & Success Criteria

### Primary Goals

1. Build a reliable iterative loop: data understanding → planning → code generation → execution → evaluation → review → next decision.
2. Produce complete, traceable artifacts at every step so a senior data scientist can audit the full chain of reasoning.
3. Demonstrate competitive performance on tabular benchmarks while generating richer documentation than any existing AutoML tool.
4. Avoid repeating failed experiments by maintaining persistent project memory.
5. Escape local optima through explicit divergence detection and challenge mechanisms.

### Success Criteria

| Criterion | Target |
|---|---|
| Time to first credible baseline | Under 15 minutes from project setup |
| Artifact completeness | 100% of runs produce plan, code, metrics, report, review |
| Benchmark performance vs AutoGluon | Within 5% on standard tabular datasets; exceed on at least 2 of 10 |
| Repeated-mistake rate | Less than 10% of iterations retry a previously-failed approach |
| Reviewer satisfaction | Senior DS can understand and critique every decision from artifacts alone |
| End-to-end benchmark completion | At least 5 Kaggle datasets run fully without manual intervention |
| Distinct experiment branches | At least 3 materially different strategies explored per benchmark |

---

## 4. Scope

### In Scope (v1)

- Tabular supervised learning: binary classification, multiclass classification, regression
- Claude Code-based orchestration using agents, skills, hooks, and rules
- Monorepo structure with all components in a single repository
- Project-scoped workflows operating on user-supplied CSV/Parquet datasets
- Deterministic dataset profiling and EDA reporting
- Structured experiment planning for feature engineering and modeling
- Code generation into Python scripts (no notebooks)
- Execution, debugging, and retry loops with bounded self-repair
- Rich evaluation and model audit reporting (Markdown + JSON)
- Reviewer and action-routing steps for iterative improvement
- Local-maxima detection and divergent exploration triggers
- Persistent project memory and run logs
- Knowledge base (wiki) of reusable ML tactics and research notes
- Benchmarking against Classic Kaggle, Playground Series, and AutoGluon
- File-based experiment tracking with JSONL metrics

### Out of Scope (v1)

- Image, text, time-series, or multimodal ML
- Notebook-first workflows
- Full MLOps deployment and production serving
- Multi-tenant enterprise features
- Graphical UI
- Autonomous internet crawling for datasets
- Multi-provider model routing (designed as extension point only)
- Real-time or streaming data pipelines

---

## 5. System Architecture Overview

### Architectural Position

The system is an orchestrated loop with deterministic analysis and reporting components wrapped by narrow, role-specific agents. It is not a single autonomous blob. It is a controlled workflow where each stage emits structured artifacts that become the inputs to the next stage.

### Design Principles

1. **Provider-agnostic seams.** The architecture assumes Claude Code models for orchestration but agent interfaces should not hard-code Claude-specific behaviour. No abstraction layer is built in v1, but the design leaves explicit extension points for future open-source model routing (e.g., via Ollama or alternative API providers). When that time comes, routine tasks like code generation or simple reviews could be routed to smaller models while reserving frontier models for planning and complex reasoning.

### Logical Layers

1. **Orchestration Layer** — Manages project runs, iteration order, retry limits, approval boundaries, and experiment state.
2. **Deterministic Analysis Layer** — Profiles datasets, calculates metrics, builds plots, validates schemas and artifacts. No LLM reasoning; pure code.
3. **Agent Layer** — Interprets structured inputs, proposes actions, writes plans, generates code, reviews outcomes. LLM-powered reasoning.
4. **Knowledge Layer** — Stores reusable tactics, research notes, project memory. Provides retrievable context for decisions.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                         │
│  ┌───────────┐  ┌──────────────┐  ┌───────────┐  ┌─────────────┐  │
│  │  Project   │  │  Iteration   │  │   State   │  │   Retry &   │  │
│  │ Bootstrap  │  │  Controller  │  │  Loader/  │  │  Approval   │  │
│  │           │  │              │  │   Saver   │  │   Gates     │  │
│  └───────────┘  └──────┬───────┘  └───────────┘  └─────────────┘  │
└─────────────────────────┼───────────────────────────────────────────┘
                          │
    ┌─────────────────────▼──────────────────────────────────┐
    │                    AGENT LOOP                           │
    │                                                        │
    │  ┌──────────┐    ┌─────────┐    ┌──────────────────┐   │
    │  │ Dataset  │───▶│ Planner │───▶│     Coder        │   │
    │  │ Analyser │    │         │    │ (Plan-to-Code)   │   │
    │  └──────────┘    └────┬────┘    └────────┬─────────┘   │
    │                       │                  │             │
    │                       │    ┌─────────────▼──────────┐  │
    │                       │    │  Executor & Debugger   │  │
    │                       │    │  (two-stage DS-STAR)   │  │
    │                       │    └─────────────┬──────────┘  │
    │                       │                  │             │
    │  ┌──────────────┐     │    ┌─────────────▼──────────┐  │
    │  │ Local Maxima │     │    │  Model Report Builder  │  │
    │  │  Challenger  │◀────┤    └─────────────┬──────────┘  │
    │  └──────┬───────┘     │                  │             │
    │         │             │    ┌─────────────▼──────────┐  │
    │         └─────────────┼───▶│      Reviewer          │  │
    │                       │    └─────────────┬──────────┘  │
    │                       │                  │             │
    │                       │    ┌─────────────▼──────────┐  │
    │                       └────│    Action Router       │  │
    │                            └──┬──────────┬──────────┘  │
    │                               │          │             │
    │                          iterate      finalize         │
    │                          (loop)       (exit)           │
    └───────────────────────────┼──────────────┼─────────────┘
                                │              │
    ┌───────────────────────────▼──────────────▼─────────────┐
    │                   KNOWLEDGE LAYER                       │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │   Project    │  │  Wiki /      │  │  Reference   │  │
    │  │   Memory     │  │  Knowledge   │  │  Index       │  │
    │  │  (per-run)   │  │  Base        │  │              │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────┐
    │               DETERMINISTIC ANALYSIS LAYER              │
    │  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
    │  │ Dataset  │  │  Evaluation  │  │    Artifact       │  │
    │  │ Profiler │  │  Scripts     │  │    Validators     │  │
    │  │ Scripts  │  │              │  │                   │  │
    │  └──────────┘  └──────────────┘  └───────────────────┘  │
    └─────────────────────────────────────────────────────────┘

    Supporting Agents (triggered on demand):
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Researcher  │  │  PDF Skill   │  │  Wiki Scribe │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### Repository Layout

```
.
├── docs/
│   ├── PRD.md
│   └── spec-kit/
├── references/
│   ├── index.md
│   └── source-notes/
├── knowledge-base/
│   ├── patterns/
│   ├── tactics/
│   ├── evaluations/
│   └── glossary/
├── agents/
│   ├── dataset-analyser/
│   ├── planner/
│   ├── coder/
│   ├── executor-debugger/
│   ├── model-report-builder/
│   ├── reviewer/
│   ├── action-router/
│   ├── local-maxima-challenger/
│   └── researcher/
├── .claude/
│   ├── skills/          # Claude Code skills (authoring guides + workflow skills)
│   ├── rules/           # Behavioural rules (coding-rules.md, artifact-contracts.md, etc.)
│   └── hooks/           # Automation hooks (post-run-memory-update, wiki-scribe)
├── templates/
│   ├── project/
│   ├── run/
│   ├── reports/
│   └── plans/
├── src/
│   ├── orchestrator/
│   ├── analysis/
│   ├── evaluation/
│   ├── storage/
│   └── utils/
└── projects/
    └── <project-id>/
```

---

## 6. Agent Specifications

Each agent owns one decision class, receives structured file inputs, and emits machine-readable outputs. Agents fail loudly on missing prerequisites rather than inventing context.

### 6.1 Dataset Analyser

**Purpose:** Produce a reliable dataset understanding package before any planning begins.

**Inputs:**
- Project metadata (`project.yaml`)
- Raw datasets (CSV/Parquet) or dataset manifests
- Optional user-provided semantic hints

**Outputs:**
- `artifacts/data/profile.json` — structured profile for downstream agents
- `artifacts/data/profile.md` — human-readable EDA summary
- `artifacts/data/plots/` — optional visualisations

**Responsibilities:**
- Schema and type inference
- Descriptive statistics (numerical summaries, categorical distributions)
- Null analysis with severity ratings
- High-cardinality detection and flagging
- Outlier detection and distribution characterisation
- Correlation analysis (numerical and categorical)
- Target variable identification and validation
- Leakage-risk flagging (identifier columns, future-leak candidates)
- **Semantic layer construction.** Generate a rich description for every column (data type, role, domain meaning, value distribution summary). If no user-provided semantic hints exist, the analyser produces best-guess descriptions and flags each one for human review in the profile report. This semantic layer is a first-class output — downstream agents (Planner, Coder, Reviewer) rely on it to reason about features with domain context rather than raw column names.
- Dataset-level semantic description (what is this dataset, its domain, its purpose)
- Class imbalance assessment (classification tasks)
- Feature-risk flags (skew, zero-variance, near-duplicates)

**Non-responsibilities:**
- Choosing the modelling strategy
- Mutating the dataset beyond read-only profiling

**Error handling:** If dataset cannot be loaded or parsed, emit a structured error artifact and halt. Do not guess schema.

---

### 6.2 Planner

**Purpose:** Convert current project context into the next experiment plan.

**Inputs:**
- Dataset profile (`profile.json`)
- Project objective and target variable
- Prior plan (if iteration > 1)
- Previous run results and metrics
- Reviewer report and router decision
- Relevant knowledge-base entries
- Project memory summary

**Outputs:**
- `artifacts/plans/iteration-<n>.yaml` — structured plan (machine-readable, validated by `src/planning/validator.py`)
- `artifacts/plans/iteration-<n>.md` — human-readable rationale narrative (required; must cite specific profile findings for each decision)

**Plan Schema (minimum fields):**
```yaml
iteration: <int>
objective: <string>
hypotheses:
  - id: H1
    description: <string>
    expected_impact: <string>
feature_steps:
  - name: <string>
    action: <string>
    rationale: <string>
model_steps:
  - algorithm: <string>
    hyperparameters: <dict>
    rationale: <string>
evaluation_focus: <string>
expected_win_condition: <string>
rollback_or_stop_condition: <string>
```

**Responsibilities:**
- Define testable hypotheses
- Plan feature-engineering steps grounded in profile findings
- Plan modelling steps appropriate to problem type
- Define expected evidence of success or failure
- Keep changes small enough to attribute outcomes to specific decisions
- Reference knowledge-base patterns (high-cardinality handling, missing data strategies, etc.)

**Non-responsibilities:**
- Writing executable code
- Judging whether a completed run is acceptable

---

### 6.3 Coder (Plan-to-Code)

**Purpose:** Translate the current approved plan into concrete Python implementation.

**Inputs:**
- Current plan (`iteration-<n>.yaml`)
- Prior code template or previous iteration code
- Coding rules (`rules/coding-rules.md`)
- Artifact contracts (`rules/artifact-contracts.md`)

**Outputs:**
- `runs/iteration-<n>/src/` — generated Python files
- `runs/iteration-<n>/config.yaml` — run-specific configuration
- `runs/iteration-<n>/requirements.txt` — dependency declarations if needed

**Responsibilities:**
- Preserve project templates and stable file layout
- Implement planned feature transformations and modelling logic
- Generate clean, readable, reviewable code
- Apply incremental patches to existing templates rather than rewriting from scratch
- Maintain separation between data loading, feature engineering, model training, and evaluation
- Emit deterministic evaluation outputs (metrics JSON, predictions CSV)

**Non-responsibilities:**
- Executing the code
- Deciding on the next experiment branch

**Code structure convention:**
```
runs/iteration-<n>/src/
├── main.py              # Entry point
├── data_loader.py       # Data loading and splitting
├── feature_engineering.py  # Feature transforms
├── model.py             # Model definition and training
├── evaluate.py          # Metric computation and reporting
└── utils.py             # Shared utilities
```

---

### 6.4 Executor & Debugger

**Purpose:** Run the experiment, capture outcomes, and repair bounded failures.

**Inputs:**
- Generated code (`runs/iteration-<n>/src/`)
- Run configuration (`config.yaml`)
- Project environment

**Outputs:**
- `runs/iteration-<n>/execution/log.txt` — full stdout/stderr
- `runs/iteration-<n>/execution/manifest.json` — runtime metadata, exit status
- `runs/iteration-<n>/execution/retry-log.jsonl` — repair attempt history
- `runs/iteration-<n>/outputs/learning_curves.json` — epoch/iteration-level train and validation metrics captured during training (e.g., loss per boosting round, validation metric per epoch)

**Two-Stage Debugging (DS-STAR inspired):**

1. **Stage 1 — Syntax and Import Repair:** Parse errors, missing imports, type mismatches. Fast, deterministic fixes. Max 3 retries.
2. **Stage 2 — Logic and Runtime Repair:** Data shape errors, NaN propagation, convergence failures. LLM-assisted diagnosis. Max 2 retries.

**Retry Policy:**
- Separate retry counters per error class (syntax, dependency, runtime logic)
- Total retry cap: 5 attempts per iteration
- On exhaustion: emit structured failure artifact, update memory, escalate to human
- Rollback: if repair breaks previously-passing code, revert to last known-good state

**Non-responsibilities:**
- Strategic experiment decisions
- Rewriting historical runs

---

### 6.5 Model Report Builder

**Purpose:** Transform raw run outputs into a comprehensive evaluation package that gives a principal data scientist (and the M7 Reviewer) everything needed to judge an iteration without re-running any computation.

**Inputs:**
- Contract 5 artifacts: `metrics.json`, `predictions.csv`, `feature_importance.json`, `learning_curves.json`, `pipeline_metadata.json`
- Contract 6 artifacts: `execution/manifest.json`
- `config.yaml` (for re-splitting to recover validation features)
- `profile.json` (for segment column selection)
- Previous `model-report.json` files (for prior-run comparison and plateau detection)

**Outputs:**
- `iterations/iteration-<n>/reports/model-report.json` — machine-readable evaluation (Contract 4, schema v1.1.0). Self-contained: M7 reads only this file.
- `iterations/iteration-<n>/reports/model-report.md` — human-readable narrative written by the agent
- `iterations/iteration-<n>/reports/plots/` — PNG evaluation plots

**Implementation:** Deterministic Python in `src/evaluation/` (metrics, analysis, plots, report assembly, validation). Agent (`.claude/agents/model-report-builder.md`) orchestrates the Python scripts then writes the interpretive narrative.

**Report Contents:**
- **Headline metrics:** Primary + secondary metrics, train vs validation split
- **Overfitting check:** Train/val gap with severity classification (low < 5%, medium 5–15%, high > 15%), learning curve trend analysis
- **Leakage indicators:** Suspiciously high metrics (> 0.99), dominant feature detection (> 80% of total importance)
- **Calibration:** Brier score + reliability curve with bin counts (classification only)
- **Decision threshold analysis:** ROC curve, precision-recall curve, optimal threshold via F1 grid search, comparison of metrics at optimal vs default (0.5) threshold. Answers "should we use a different cutoff?"
- **Bootstrap confidence intervals:** 1000-resample CIs (95%) on primary and secondary metrics. Quantifies whether observed improvements are statistically meaningful given the validation set size
- **Prediction separation quality:** KS statistic, discrimination slope, histogram overlap coefficient. Qualitative assessment: strong/moderate/weak
- **Segment analysis:** Auto-selected slicing columns (categorical ≤ 10 unique, top-2 numeric by importance binned into quartiles). Per-slice primary metric and accuracy
- **Error analysis:** Confusion matrix (classification), residual stats (regression), misclassification patterns, error rate by confidence bin
- **Hardest samples:** Top-10 highest-loss predictions (cross-entropy for classification, absolute residual for regression). Surfaces data quality issues and model blind spots
- **Feature importance:** Repackaged with rank. Model-native method (coefficients, Gini, etc.)
- **Per-feature diagnostic plots:** Top-5 features — box plots of feature values for correct vs incorrect predictions
- **Residual vs feature plots:** Top-5 features — scatter of prediction/residual against feature value, coloured by correctness. Reveals systematic error patterns across feature ranges
- **Comparison to prior runs:** Delta table for all metrics vs previous iteration. Null on iteration 1
- **Reviewer summary:** Pre-computed deterministic verdict (improved/degraded/neutral/suspicious), risk flags ({type, severity, evidence}), plateau signal (consecutive stale iterations)

**Plots (10 on Titanic baseline):**
- `confusion_matrix.png`, `actual_vs_predicted.png`, `calibration_curve.png`, `error_distribution.png`
- `roc_curve.png`, `precision_recall_curve.png`
- `feature_diagnostic_<name>.png` (top features)
- `residual_vs_<name>.png` (top features)

**Non-responsibilities:**
- Deciding whether results are good enough (that is the Reviewer's job)
- Choosing next steps (that is the Action Router's job)
- Recommending experiments or strategy changes

---

### 6.6 Reviewer

**Purpose:** Judge experiment quality and explain what the results imply.

**Inputs:**
- Model report (`model-report.json` + `model-report.md`)
- Previous run comparisons
- Project objective and current plan
- Project memory summary

**Outputs:**
- `runs/iteration-<n>/review/reviewer-report.yaml`

**Reviewer Schema:**
```yaml
iteration: <int>
headline_verdict: improved | degraded | neutral | suspicious
metric_summary:
  primary_metric: <float>
  delta_vs_previous: <float>
  delta_vs_baseline: <float>
comparison_to_previous:
  what_changed: <string>
  likely_causes: [<string>]
risk_flags:
  - type: leakage | overfitting | underfitting | data_issue
    severity: low | medium | high
    evidence: <string>
plateau_signal:
  detected: <bool>
  consecutive_stale_iterations: <int>
  recommendation: continue | challenge | escalate
recommended_next_action: <string>
confidence: low | medium | high
```

**Responsibilities:**
- Assess whether the run improved meaningfully (not just numerically)
- Explain why improvement or degradation likely happened
- Detect warning signs (leakage, overfitting, suspicious patterns)
- Identify diminishing returns and trigger plateau signals
- Recommend tactical next actions with evidence

---

### 6.7 Action Router

**Purpose:** Decide the next workflow action after review.

**Inputs:**
- Reviewer report (`reviewer-report.yaml`)
- Current plan
- Current run metadata
- Project memory summary

**Outputs:**
- `runs/iteration-<n>/review/router-decision.yaml`

**Router Schema:**
```yaml
iteration: <int>
decision: finalize | refine | replace_step | add_step | diverge | escalate
reasoning: <string>
required_next_inputs: [<string>]
human_approval_required: <bool>
estimated_remaining_iterations: <int>
```

**Allowed Actions:**
| Action | Trigger | Description |
|---|---|---|
| `finalize` | Target met or budget exhausted | Stop iterating, package final model |
| `refine` | Improvement detected, same direction viable | Tweak hyperparameters, add regularisation |
| `replace_step` | One step identified as bottleneck | Swap a feature or model step |
| `add_step` | Gap identified in pipeline | Add ensembling, new features, post-processing |
| `diverge` | Plateau detected or challenger triggered | Try a fundamentally different approach |
| `escalate` | Ambiguity, risk, or budget concern | Pause for human decision |

**Design principle:** The router explains *why* it chose the action and what evidence threshold drove the decision. It never inspects raw execution output directly — it works from the structured reviewer report.

---

### 6.8 Local Maxima Challenger

**Purpose:** Detect diminishing returns and propose divergent strategy branches.

**Trigger Conditions:**
- Score improvement below threshold for 3+ consecutive iterations
- Same model family unchanged for 4+ iterations
- Error profile stagnates even when headline metric moves slightly
- Feature importance distribution remains static across iterations

**Inputs:**
- Full run history from project memory
- Current model family and feature set
- Reviewer plateau signals

**Outputs:**
- Challenge report with:
  - Evidence of stagnation (metric trajectories, error pattern analysis)
  - 2-3 materially different strategy proposals (different model families, different feature philosophies, different target transformations)
  - Expected risk/reward assessment for each proposal

**Interaction with Router:** When the Challenger fires, its output feeds into the Router as an additional input. The Router may accept a divergent proposal, escalate to the human, or override if strong evidence suggests the current path still has headroom.

---

### 6.9 Researcher

**Purpose:** Investigate external references and bring back tactics relevant to the active problem type.

**Inputs:**
- Problem description and dataset characteristics
- Current knowledge-base gaps identified by Planner or Reviewer
- Specific research questions (e.g., "best approaches for high-cardinality categorical encoding in gradient boosting")

**Outputs:**
- Structured research notes added to `knowledge-base/`
- Tactic summaries with applicability conditions
- Reference entries added to `references/index.md`

**Phase recommendation:** Keep as a Phase 2 component until the core local loop is stable. In Phase 1, the knowledge base is populated manually.

**Implementation note:** When fetching arXiv papers, prefer HTML version URLs (`https://arxiv.org/html/<id>`) over PDF URLs (`https://arxiv.org/pdf/<id>`). The PDF URL returns raw binary that indexes as garbage; the HTML version converts cleanly to searchable markdown.

**Non-responsibilities:**
- Autonomous web crawling without human oversight
- Replacing the Planner's decision-making role

---

### 6.10 PDF Skill

**Purpose:** Ingest research papers and technical documents into the knowledge base without bloating agent context.

**Inputs:**
- PDF file path or URL
- Ingestion instructions (what to extract, which sections matter)

**Outputs:**
- Structured markdown summary in `references/source-notes/`
- Extracted tactics added to `knowledge-base/tactics/`
- Updated reference index entry

**Workflow:**
1. Convert PDF to text (extraction, not OCR for digital PDFs)
2. Chunk by sections/headings
3. Summarise each section with relevance to project goals
4. Extract reusable patterns and tactics
5. Store in knowledge base with YAML frontmatter metadata

**Design note:** Based on the existing `newsletter-engine` PDF import skill pattern. Prioritises text-extractable content. Does not attempt to parse figures or equations.

---

## 7. Pipeline Flow

### Single Iteration Flow

```
     ┌──────────────────────────────────────────────────────────────┐
     │                     HUMAN: Project Setup                     │
     │  - Provide dataset(s)                                        │
     │  - Define objective, target variable, problem type            │
     │  - Approve project.yaml                                      │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 1: Dataset Analyser                                    │
     │  → profile.json, profile.md, plots                           │
     │  → Update project memory                                     │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 2: Planner                                             │
     │  ← profile, memory, wiki, prior results, router decision     │
     │  → iteration-<n>.yaml                                        │
     │  → [HUMAN GATE on iteration 1: approve initial plan]         │
     │  → Update project memory                                     │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 3: Coder (Plan-to-Code)                                │
     │  ← plan, prior code, templates, rules                        │
     │  → Python scripts + config                                   │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 4: Executor & Debugger                                 │
     │  ← code, config, environment                                 │
     │  → logs, manifest, metrics, predictions                      │
     │  → [ON FAILURE: two-stage repair, max 5 retries]             │
     │  → [ON EXHAUST: escalate to human]                           │
     │  → Update project memory                                     │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 5: Model Report Builder                                │
     │  ← predictions, metrics, model artifacts                     │
     │  → model-report.json, model-report.md                        │
     │  → Update project memory                                     │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 6: Reviewer                                            │
     │  ← model report, prior runs, objective, memory               │
     │  → reviewer-report.yaml                                      │
     │  → [IF plateau detected: trigger Local Maxima Challenger]     │
     │  → Update project memory                                     │
     └──────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
     ┌──────────────────────────────────────────────────────────────┐
     │  STEP 7: Action Router                                       │
     │  ← reviewer report, challenger output (if any), memory       │
     │  → router-decision.yaml                                      │
     │  → Update project memory                                     │
     └──────────┬───────────────────────────────────┬───────────────┘
                │                                   │
           ┌────▼────┐                         ┌────▼────┐
           │ ITERATE │                         │FINALIZE │
           │ → Back  │                         │ → Final │
           │ to Step │                         │ model + │
           │    2    │                         │ summary │
           └─────────┘                         └─────────┘
```

### Iteration Control

- **Max iterations per project:** Configurable, default 15
- **Convergence threshold:** If primary metric improves less than 0.1% for 3 consecutive iterations, trigger Local Maxima Challenger
- **Cost budget:** Optional token/compute budget cap; Router escalates when 80% consumed
- **Human interrupt:** User can pause the loop at any point and inject new direction
- **Final review:** Human approves or rejects the final candidate before packaging

### Finalization Artifacts

When the Router decides `finalize`, the system produces a defined output package:

```
projects/<project-id>/final/
├── model/                  # Chosen model artifact (serialised) + metadata
├── summary.md              # Winning approach narrative: strategy, evidence, caveats,
│                           # performance trajectory, and recommendation
└── submission/             # Optional: competition submission file if applicable
```

The final summary must be self-contained — a senior DS should be able to read `summary.md` alone and understand what was tried, what won, and why.

---

## 8. Knowledge Base (Wiki) Design

### Purpose

Separate reusable, generalised ML knowledge from project-specific memory. The knowledge base grows across projects; project memory is scoped to a single project.

### Structure

```
knowledge-base/
├── patterns/
│   ├── high-cardinality-categoricals.md
│   ├── missing-data-strategies.md
│   ├── class-imbalance-handling.md
│   ├── target-transformation.md
│   └── ...
├── tactics/
│   ├── gradient-boosting-tuning.md
│   ├── ensemble-stacking.md
│   ├── feature-selection-methods.md
│   ├── unsupervised-clustering.md
│   └── ...
├── evaluations/
│   ├── classification-metrics-guide.md
│   ├── regression-metrics-guide.md
│   ├── leakage-detection-checklist.md
│   └── ...
└── glossary/
    └── terms.md
```

### Entry Format

Every wiki entry uses markdown with YAML frontmatter:

```markdown
---
title: Handling High-Cardinality Categoricals
category: pattern
tags: [categorical, feature-engineering, encoding]
applicability:
  problem_types: [classification, regression]
  triggers: ["cardinality > 50", "categorical feature with many unique values"]
confidence: high
sources: ["knowledge-base/references/source-notes/catboost-paper.md"]
created: 2026-04-08
updated: 2026-04-08
---

# Handling High-Cardinality Categoricals

## Problem
[When this pattern applies]

## Approaches
[Ranked list of strategies with tradeoffs]

## Recommended Default
[What to try first and why]

## Anti-patterns
[What to avoid and why]

## Evidence
[Links to experiments or papers that support this]
```

### Evolution Path

1. **Phase 1 (v1):** Markdown files in the repo. Manual curation. Agents search by filename and frontmatter tags.
2. **Phase 2:** Obsidian-compatible structure with bidirectional links. Human can browse and edit in Obsidian.
3. **Phase 3:** RAG-based retrieval. Embeddings generated from wiki entries. Agents query semantically rather than by filename.

### Wiki Scribe

Automated knowledge capture after runs:
1. After a run produces a reusable lesson with sufficient evidence, the Wiki Scribe drafts a new entry.
2. Drafts are stored in `knowledge-base/_drafts/`.
3. Human (or a reviewer agent) promotes approved drafts to the main knowledge base.
4. Hybrid model: automatic drafting, promotion requires approval.

---

## 9. Experiment Tracking & State Management

### Design Decision: File-Based Tracking

No MLflow, no Weights & Biases, no database. All experiment tracking is file-based within the project directory. This keeps the system self-contained, version-controllable, and auditable.

### Project-Level State

```
projects/<project-id>/
├── project.yaml                    # Metadata, objective, target, constraints
├── memory/
│   ├── run-history.jsonl           # Append-only, one JSON object per iteration
│   └── decision-log.md            # Human-readable narrative of decisions
└── runs/
    ├── iteration-1/
    ├── iteration-2/
    └── ...
```

### Run History Format (JSONL)

Each line in `run-history.jsonl` is a self-contained record:

```json
{
  "iteration": 3,
  "timestamp": "2026-04-08T14:23:00Z",
  "status": "completed",
  "plan_summary": "Add target encoding for postal_code, switch from RF to LightGBM",
  "primary_metric": {"name": "rmse", "value": 0.1423, "delta": -0.0081},
  "secondary_metrics": {"r2": 0.891, "mae": 0.098},
  "model_family": "lightgbm",
  "feature_changes": ["added: postal_code_target_enc", "removed: postal_code_onehot"],
  "reviewer_verdict": "improved",
  "router_decision": "refine",
  "failure_count": 0,
  "duration_seconds": 127,
  "notes": "Target encoding reduced dimensionality from 847 to 1 feature"
}
```

### Iteration Directory Structure

```
runs/iteration-<n>/
├── src/                            # Generated Python code
│   ├── main.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluate.py
├── config.yaml                     # Run-specific settings
├── execution/
│   ├── log.txt                     # Full stdout/stderr
│   ├── manifest.json               # Runtime metadata, environment, exit status
│   └── retry-log.jsonl             # Repair attempt history
├── outputs/
│   ├── metrics.json                # Raw metric values
│   ├── predictions.csv             # Model predictions
│   ├── feature_importance.json     # Feature importance scores
│   └── plots/                      # Generated visualisations
├── reports/
│   ├── model-report.json           # Machine-readable evaluation
│   └── model-report.md            # Human-readable evaluation
├── review/
│   ├── reviewer-report.yaml        # Reviewer assessment
│   └── router-decision.yaml        # Action router output
└── artifacts/
    └── model/                      # Serialised model (pickle/joblib)
```

### State Immutability

Each iteration is immutable after completion. Corrections happen in the next iteration, not by rewriting history. The only exception: explicitly marked repair metadata appended during the debug stage.

---

## 10. Code Framework Design

### Design Decision: Templates + Incremental Patches

Rather than generating entirely new code each iteration, the Coder agent works from a stable template and applies incremental patches. This reduces hallucination risk, maintains code quality, and makes diffs between iterations reviewable.

### Base Template Structure

The template provides a fixed skeleton that every iteration inherits:

```python
# templates/run/main.py
"""
Iteration {iteration} — {objective}
Generated: {timestamp}
Plan: {plan_reference}
"""

import logging
from data_loader import load_and_split
from feature_engineering import engineer_features
from model import train_model
from evaluate import evaluate_and_report

logger = logging.getLogger(__name__)

def main():
    # 1. Load and split data
    train_df, val_df, test_df = load_and_split()

    # 2. Feature engineering
    train_df, val_df, test_df = engineer_features(train_df, val_df, test_df)

    # 3. Train model
    model, train_predictions = train_model(train_df)

    # 4. Evaluate
    evaluate_and_report(model, train_df, val_df, test_df)

if __name__ == "__main__":
    main()
```

### Patch Strategy

- **Iteration 1:** Full template instantiation from the plan
- **Iteration 2+:** Coder receives previous iteration code + new plan diff. Applies targeted changes rather than rewriting
- **Preserved across iterations:** Import structure, logging, file layout, evaluation output format
- **Changed per iteration:** Feature engineering logic, model definition, hyperparameters, evaluation focus

### Coding Rules (enforced via `rules/coding-rules.md`)

1. No notebooks — Python scripts only
2. Every script must be runnable standalone via `python main.py`
3. All metrics must be written to `metrics.json` in a standard schema
4. Predictions must be written to `predictions.csv` with index alignment to input data
5. Feature engineering must be applied consistently to train, validation, and test splits
6. No hardcoded file paths — use config.yaml for all paths
7. Logging to both console and `log.txt`
8. Random seeds must be set and recorded in config
9. No internet access during training runs
10. Dependencies declared in `requirements.txt`

---

## 11. Data Handling

### Supported Formats (v1)

- CSV (primary)
- Parquet
- Single-table datasets only (multi-table support deferred to v2)

### Data Flow

```
User provides dataset(s)
        │
        ▼
projects/<id>/data/raw/          # Immutable original files
        │
        ▼
Dataset Analyser reads            # Read-only profiling
        │
        ▼
projects/<id>/data/processed/    # Generated by feature engineering
        │                         # (created during execution, not by analyser)
        ▼
Train / Validation / Test split   # Managed by data_loader.py
```

### Splitting Strategy

- Default: 70/15/15 train/validation/test split
- Stratified splitting for classification tasks
- Configurable via `project.yaml`
- If competition provides train/test split, honour it (validation carved from train)
- Random seed recorded and fixed across iterations for comparability

### Data Contracts

- Raw data is never modified. All transforms produce new columns or new files.
- The Dataset Analyser operates read-only on raw data.
- Feature engineering scripts in `runs/iteration-<n>/src/` generate processed data during execution.
- Processed data is stored in the iteration directory, not shared across iterations (each iteration is self-contained).

### Large Dataset Handling

- Profiling uses sampling for datasets over 1M rows (configurable threshold)
- Memory-efficient loading with chunked reads for very large files
- Feature engineering operates on the full dataset during training (sampling only for profiling)

### Missing Data Policy

- The Dataset Analyser reports missing data patterns and severity
- The Planner decides imputation strategy based on profile findings and wiki patterns
- The Coder implements the chosen strategy
- No silent dropping of rows or columns

---

## 12. Human Interaction Model

### Design Philosophy

The system is an autonomous loop with strategic human gates — not a chatbot that asks for permission at every step. The human approves the initial plan, then the system runs autonomously. The human can interrupt at any time and must approve the final output.

### Gate Structure

| Gate | When | Required? | Purpose |
|---|---|---|---|
| Project setup | Before first iteration | Yes | Confirm objective, target, constraints |
| Initial plan approval | After first plan generated | Yes | Ensure the starting direction makes sense |
| Iteration loop | Steps 2-7 each iteration | No (autonomous) | System runs without human input |
| Human interrupt | Any time during loop | Optional | User injects new direction or correction |
| Escalation | Router decides `escalate` | Triggered | Ambiguity, risk, or budget concern |
| Divergent branch approval | Before major strategy shift | Optional | Configurable; default is autonomous |
| Final review | After Router decides `finalize` | Yes | Human approves or rejects final candidate |

### Interaction Modes

1. **Full autonomy:** Human sets up project, approves initial plan, reviews final output. Everything in between is automatic.
2. **Guided autonomy:** Same as above, but human also approves divergent branches and receives progress summaries every N iterations.
3. **Supervised:** Human reviews every plan before execution. Slower but maximum control.

Default mode: **Full autonomy** with interrupt capability.

### Artifact-Based Communication

The system communicates with the human through artifacts, not conversational back-and-forth:
- Plans are readable YAML + markdown
- Reports are structured evaluation narratives
- Memory logs provide full history
- The human reads artifacts and injects corrections via project configuration or direct plan edits

### Progress Reporting

- After each iteration: one-line summary appended to `memory/decision-log.md`
- On escalation: structured question with context and options
- On completion: final summary with performance trajectory and recommendation

---

## 13. Evaluation Strategy

### Metric Selection by Problem Type

| Problem Type | Primary Metric | Secondary Metrics |
|---|---|---|
| Binary classification | AUC-ROC | F1, precision, recall, log loss, accuracy |
| Multiclass classification | Macro F1 | Weighted F1, accuracy, per-class F1, log loss |
| Regression | RMSE | MAE, R², MAPE, median absolute error |

Primary metric is configurable via `project.yaml`. If a Kaggle competition specifies a custom metric, that becomes primary.

### Evaluation Package (per iteration)

Every completed iteration produces:

1. **Metric scores** — Primary + secondary metrics on train, validation, and test sets
2. **Actual vs predicted** — Scatter plot (regression) or confusion matrix (classification)
3. **Segment analysis** — Performance broken down by key categorical features or quantile bins
4. **Feature importance** — Permutation importance or model-native (e.g., LightGBM split importance)
5. **Error analysis** — Distribution of residuals (regression) or misclassification patterns (classification)
6. **Overfitting check** — Train/validation metric gap, learning curves where available
7. **Leakage indicators** — Features with suspiciously high importance, near-perfect single-feature performance
8. **Calibration** — Probability calibration curve (classification only)
9. **Cross-iteration comparison** — Delta table against all prior iterations

### Benchmarking Strategy

Three tiers of benchmarking, in order of priority:

**Tier 1 — Classic Kaggle Datasets:**
- Titanic (binary classification)
- House Prices (regression)
- Spaceship Titanic (binary classification)
- Used as initial development and smoke-testing targets

**Tier 2 — Kaggle Playground Series:**
- Monthly Playground competitions provide fresh tabular problems
- Tests generalisation to unseen problem structures
- 3-5 recent Playground problems selected as benchmarks

**Tier 3 — AutoGluon Comparison:**
- Run AutoGluon with default settings on the same datasets
- Compare final metric, time to result, and artifact quality
- Goal: within 5% of AutoGluon metric while producing dramatically richer documentation
- Stretch goal: exceed AutoGluon on 2+ datasets through creative feature engineering

### What Counts as Success

A benchmark run is successful if:
1. The system completes end-to-end without manual intervention
2. All required artifacts are produced and well-formed
3. The final metric is competitive (within 5% of AutoGluon baseline)
4. A senior DS can understand and critique the full decision chain from artifacts alone
5. At least 3 materially different strategies were explored

---

## 14. Differentiation from Existing AutoML

### Core Difference

Traditional AutoML tools treat ML as a search problem: enumerate configurations, evaluate them, pick the best. This system treats ML as a reasoning problem: understand the data, form hypotheses, test them, learn from results, and adapt.

### Comparison Table

| Dimension | AutoGluon / H2O / FLAML | Agentic AutoML |
|---|---|---|
| **Search strategy** | Brute-force / Bayesian optimisation | LLM-reasoned (understands *why* a strategy should work) |
| **Feature engineering** | Fixed transforms (one-hot, scaling, imputation) | Creative, context-aware (domain-informed encoding, interaction features, target transforms) |
| **Explainability** | Post-hoc SHAP / feature importance | Built into every iteration (leakage checks, segment analysis, error attribution) |
| **Adaptability** | Config-driven; same pipeline regardless of data | Learns from wiki + past runs; adapts strategy to data characteristics |
| **Human collaboration** | Knob-turning (set `time_limit`, `presets`) | Natural language (approve plans, inject direction, review narratives) |
| **Experiment documentation** | Leaderboard table with scores | Full decision chain: plan → code → results → review → next action |
| **Local-optima escape** | Stacking/ensembling as only diversification | Explicit divergence detection + challenger agent proposes new directions |
| **Error handling** | Crash or silent skip | Two-stage debugger with structured repair and rollback |
| **Iteration memory** | None (each run is independent) | Persistent project memory prevents repeating failed approaches |
| **Code output** | Black-box model object | Readable Python scripts, auditable and modifiable by humans |
| **Cost model** | Compute time as primary cost | Token cost + compute time; budget-aware routing |

### What This System Does NOT Replace

- AutoGluon remains better for "give me the best model in 10 minutes with no thinking"
- This system is for when you need to *understand* the model, *document* the process, and *learn* from the experience
- This system is slower per iteration but produces more value per iteration

---

## 15. Technical Requirements

### Runtime Environment

- **Language:** Python 3.11+
- **Orchestration:** Claude Code (agents, skills, hooks, rules)
- **LLM:** Claude (Sonnet for routine tasks, Opus for planning/review)
- **Package management:** pip + single shared virtual environment (`.venv/` at repo root)
- **OS:** macOS (primary development), Linux (CI/production)

### Dependencies

Dependencies are declared in `requirements.txt` at repo root and added as each milestone requires them.

### Performance Requirements

- Dataset profiling: under 60 seconds for datasets up to 1M rows
- Single iteration (plan → code → execute → report → review): under 10 minutes
- Full project (15 iterations): under 2.5 hours
- Token budget per iteration: configurable, default 50K tokens

### Storage Requirements

- Each iteration: approximately 1-10 MB (code, logs, reports, model artifacts)
- Full project (15 iterations): approximately 50-150 MB
- Knowledge base: grows over time, typically under 50 MB

### Reproducibility Requirements

- All random seeds recorded in config and fixed across runs
- Python environment captured in `requirements.txt` at repo root
- Dataset version (hash) recorded in manifest
- Full code stored per iteration (not just diffs)
- Environment metadata (Python version, OS, package versions) in manifest

### Testing Strategy

Each component type has a defined testing approach to ensure reliability without requiring full end-to-end runs:

| Component | Testing Approach |
|---|---|
| Dataset Analyser | Schema fixtures, edge-case datasets (all nulls, single row, high cardinality), known leakage examples |
| Planner | Simulation tests using fixed upstream artifacts; checks that plans stay within allowed action boundaries |
| Coder (Plan-to-Code) | Template conformance tests; code-generation smoke tests against known plans |
| Executor & Debugger | Retry-limit tests; failure classification tests (syntax vs runtime vs logic errors) |
| Model Report Builder | Metric-calculation tests; artifact-presence tests; report schema validation |
| Reviewer & Router | Fixture-based decision tests using known model reports; regression tests on plateau detection and escalation behaviour |
| Project Memory | Historical consistency tests; retrieval quality checks against planted entries |
| Knowledge Base | Entry schema validation; frontmatter conformance tests |

---

## 16. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **Over-automation without grounding.** Agents generate plausible but weak ML strategies when not anchored by deterministic analysis. | High | High | Deterministic analysis layer runs first; agents receive structured profiles, not raw data. Template-based code prevents unconstrained generation. |
| R2 | **Context window overflow.** Too much raw text, log output, or unstructured research degrades agent reasoning quality. | High | High | File-based artifacts instead of inline context. Summaries passed to agents, not raw logs. Strict artifact contracts limit payload sizes. JSONL for machine-readable data, markdown for human-readable. |
| R3 | **Local-optima behaviour.** System keeps refining the same family of ideas, producing diminishing returns. | Medium | High | Local Maxima Challenger agent with explicit trigger thresholds. Router can force divergent exploration. Convergence threshold and max iteration limits. |
| R4 | **Poor artifact contracts.** Inconsistent intermediate outputs cause downstream agents to fail or hallucinate. | Medium | High | Schema validation at every handoff. Agents fail loudly on missing prerequisites. Artifact contracts documented in `rules/artifact-contracts.md` and enforced programmatically. |
| R5 | **Hidden data leakage.** Strong metrics mask invalid modelling choices (future data in features, target leakage, train/test contamination). | Medium | Critical | Dataset Analyser flags leakage-risk columns. Model Report Builder checks for suspicious feature importance. Reviewer has explicit leakage-risk assessment. Feature engineering applies transforms consistently across splits via pipeline pattern. |
| R6 | **Tooling sprawl.** Too many agents or skills too early creates complexity before the core loop is reliable. | Medium | Medium | Milestone plan starts with minimal viable loop (8 core components). Supporting agents (Researcher, PDF Skill) deferred to Phase 2. Each milestone leaves the repo in a runnable state. |
| R7 | **Premature generalisation.** Supporting every ML modality or deployment target early delays useful progress on the core problem. | Low | High | v1 scope is strictly tabular classification + regression. Time series, images, text, and other modalities are explicitly out of scope. Scope expansion only after benchmark validation succeeds. |
| R8 | **Runaway costs.** Unbounded LLM calls during debugging or exploration consume excessive tokens. | Medium | Medium | Per-iteration token budget (configurable, default 50K). Router escalates at 80% budget consumption. Retry caps on Executor (max 5 attempts). Cost tracking in run manifest. |
| R9 | **Irreproducible results.** Non-deterministic model training or data splitting makes comparison across iterations meaningless. | Low | High | Fixed random seeds in config. Full code + environment captured per iteration. Dataset hashing. Immutable iteration directories. |
| R10 | **Knowledge base pollution.** Low-quality or incorrect entries in the wiki mislead future planning. | Medium | Medium | Hybrid Wiki Scribe model: automatic drafting, human-approved promotion. Confidence ratings on entries. Evidence links required. Entries can be deprecated, not just deleted. |

---

## 17. Milestone Plan

### Milestone Scoping Principle

Each minor milestone should be scoped to a single reviewable PR. PRs must be targeted and testable — not massive multi-hundred-line changes. The goal is that each minor milestone can be reviewed together, tested against a simulation scenario or dataset, and wrapped with appropriate tests. No ultra-atomic PRs either, but nothing that requires scrolling through hundreds of lines of unrelated changes.

### M0 — Documentation & Contracts Foundation
**Type:** Major | **Outcome:** Clear PRD, specs, artifact contracts, and repository structure.

| Minor Milestone | Deliverable |
|---|---|
| M0.1 ✅ | PRD and spec-kit documents finalised |
| M0.2 | Lightweight artifact templates — `project.yaml`, `iteration-<n>.yaml` plan, and `run-history.jsonl` entry. `templates/plans/iteration.yaml` ✅ created at M3. `project.yaml` template and run-history entry format pending. |
| M0.3 ✅ | Top-level folder scaffold created: `references/`, `knowledge-base/`, `templates/`, `src/`, `projects/` — each with a README stub. Agent instruction files live at `.claude/agents/` (Claude Code convention) rather than repo root. |
| M0.4 ✅ | `rules/coding-rules.md` (path-scoped to `runs/`, 10 coding rules) and `rules/artifact-contracts.md` (unconditional, 4 artifact schema contracts) created. `rules/authoring.md` already existed. |
| M0.5 ✅ | Claude Code authoring skills: four skills in `.claude/skills/` (`create-agent`, `create-hook`, `create-rule`, `create-skill`) each containing DOs, DON'Ts, anti-patterns, and official reference links for building Claude Code primitives. Enforced at authoring time via `authoring.md` rule. These skills serve as the living best-practices reference consumed whenever a primitive is created or significantly restructured. |

### M1 — Single-Project Runtime Skeleton
**Type:** Major | **Outcome:** A project can be initialised and run through an empty but structured loop.

> **Status: Deferred.** M1 was built incrementally inside M2. The Titanic project (`projects/titanic/`) acts as the runtime skeleton — `project.yaml`, raw data, processed data, artifacts, and iteration directories are all in place. M2 is now complete. Full orchestration (state loader/saver, iteration controller) remains for a future milestone. M1 milestones below remain as reference for what must eventually exist.

| Minor Milestone | Deliverable |
|---|---|
| M1.1 | Project bootstrap command and directory template |
| M1.2 | Run-iteration skeleton and state loader/saver |
| M1.3 | Base config files and environment conventions |
| M1.4 | Smoke tests for project creation and iteration folder creation |

### M2 — Deterministic Dataset Understanding ✅
**Type:** Major | **Outcome:** Credible dataset profile and EDA summary for any supplied tabular dataset.

| Minor Milestone | Deliverable |
|---|---|
| M2.1 ✅ | Schema and basic stats profiler (`src/analysis/profiler.py`) |
| M2.2 ✅ | Null, cardinality, outlier, and correlation analysis (Pearson + Cramér's V) |
| M2.3 ✅ | Generate `profile.json` and `profile.md` (validated against Titanic dataset) |
| M2.4 ✅ | Plot generation (`src/analysis/plots.py`): distribution and target-vs-feature plots |
| M2.5 ✅ | Test suite (`tests/analysis/test_profiler.py`) covering schema, stats, nulls, cardinality, outliers, and correlation |

### M3 — Planning Layer ✅
**Type:** Major | **Outcome:** System turns project context into a structured first experiment plan.

| Minor Milestone | Deliverable |
|---|---|
| M3.1 ✅ | Plan schema template (`templates/plans/iteration.yaml`) and validator (`src/planning/validator.py` + `PlanValidationError`). Enforces all required fields including exactly one model step per iteration. |
| M3.2 ✅ | Planner agent (`.claude/agents/planner.md`): 8-step workflow, scope guardrails (CAN/CANNOT), init and continuation paths, one-model-per-iteration constraint. Rewritten post-smoke-test to remove prescriptive decision tables — agent reasons from profile data directly. |
| M3.3 ✅ | Memory scaffold (`projects/titanic/memory/run-history.jsonl`, `projects/titanic/memory/decision-log.md`). Planner reads both on iteration > 1. |
| M3.4 ✅ | 14 tests in `tests/planning/test_plan_schema.py` (validator schema, edge cases, exactly-one-model-step enforcement). Smoke test on Titanic produced a valid, profile-grounded `artifacts/plans/iteration-1.yaml` and `iteration-1.md`. |

### M4 — Plan-to-Code Layer ✅
**Type:** Major | **Outcome:** System translates a validated iteration plan into executable Python code.

| Minor Milestone | Deliverable |
|---|---|
| M4.1 ✅ | Structural code templates (`templates/iteration/`): `main.py`, `data_loader.py`, `feature_engineering.py`, `model.py`, `evaluate.py`, `utils.py`, `config.yaml`. Establishes `fit_transform`/`transform` contract for leak-free feature engineering and split-strategy dispatch (`stratified`, `random`, `temporal`). |
| M4.2 ✅ | Contract 5 added to `artifact-contracts.md` (iteration code outputs: metrics.json, predictions.csv, feature_importance.json, learning_curves.json, pipeline_metadata.json, model artifact). `coding-rules.md` path scope updated `runs/` → `iterations/`. |
| M4.3 ✅ | Codegen validator (`src/codegen/validator.py` + `CodegenValidationError`): 6 checks — required files, config keys, syntax via `ast.parse`, no hardcoded paths, `if __name__` guard, feature-step count sanity. 8 tests in `tests/codegen/test_codegen_validator.py`, all green. |
| M4.4 ✅ | Coder agent (`.claude/agents/coder.md`): 10-step workflow reading plan YAML + profile.json, generating all 8 iteration files, self-validating via codegen validator. Smoke test on Titanic iteration-1: validator passed, `python src/main.py` ran to completion, val AUC-ROC = 0.835 (win condition > 0.80 ✓). |

### M5 — Execution & Debugging Loop ✅
**Type:** Major | **Outcome:** Generated experiment can run, fail safely, and attempt bounded self-repair.

| Minor Milestone | Deliverable |
|---|---|
| M5.1 ✅ | Run executor (`src/execution/runner.py`) and log capture. `ExecutionResult` dataclass with exit code, stdout, stderr, duration. |
| M5.2 ✅ | Runtime manifest generation (`execution/manifest.json`). Contract 6 added to `artifact-contracts.md`. |
| M5.3 ✅ | Two-stage bounded debugger with retry classification (`src/execution/classifier.py`). `ErrorCategory` enum (syntax, import, type, data_shape, nan, convergence, runtime, timeout, unknown). Stage 1: max 3 retries. Stage 2: max 2 retries. Total cap 5. |
| M5.4 ✅ | Executor agent (`.claude/agents/executor.md`): 7-step workflow with pre-flight validation, output validation, and bounded debug loop. Tests: `test_runner.py`, `test_classifier.py`, `test_output_validator.py`, `test_integration.py` (including Titanic end-to-end). |

### M6 — Evaluation & Model Reporting ✅
**Type:** Major | **Outcome:** Every run ends with a deep evaluation package.

| Minor Milestone | Deliverable |
|---|---|
| M6.1 ✅ | Core metric analysis (`src/evaluation/metrics.py`): headline repackaging, overfitting check (train/val gap with severity thresholds), leakage detection (suspicious metrics + dominant feature), risk flag classification, plateau signal, bootstrap confidence intervals (1000 resamples, 95% CI). |
| M6.2 ✅ | Analysis functions (`src/evaluation/analysis.py`): calibration (Brier score + reliability curve), segment analysis (auto-select categorical ≤10 unique + top-2 numeric by importance), error analysis (confusion matrix + error rate by confidence bin), decision threshold analysis (ROC/PR curves + optimal threshold via F1 grid search), prediction separation quality (KS statistic + discrimination slope + histogram overlap), hardest samples (top-10 highest-loss predictions). |
| M6.3 ✅ | Evaluation plots (`src/evaluation/plots.py`): confusion matrix, actual-vs-predicted, calibration curve, error distribution, ROC curve, precision-recall curve, per-feature diagnostics (box plots), residual-vs-feature scatter plots. 10 plots on Titanic baseline. |
| M6.4 ✅ | Report assembly (`src/evaluation/report_builder.py`), validator (`src/evaluation/validator.py` + `ReportValidationError`), Model Report Builder agent (`.claude/agents/model-report-builder.md`). Contract 4 expanded from stub to full schema (v1.1.0, 18 top-level keys). Agent runs deterministic Python then writes interpretive `model-report.md` narrative. 60 tests across 5 test files, all green. Smoke-tested on Titanic iteration-1. |

### M7 — Reviewer & Action Router
**Type:** Major | **Outcome:** System judges outcomes and chooses the next loop step.

| Minor Milestone | Deliverable |
|---|---|
| M7.1 | Reviewer schema and decision rubric |
| M7.2 | Reviewer with prior-run comparison logic |
| M7.3 | Router schema and allowed actions |
| M7.4 | Router decision logic and stop criteria |
| M7.5 | Local-maxima detection helper integration |

### M8 — Project Memory
**Type:** Major | **Outcome:** System learns from prior iterations and avoids repeating weak ideas.

| Minor Milestone | Deliverable |
|---|---|
| M8.1 | Append-only run history (JSONL) |
| M8.2 | Decision-log summaries (markdown) |
| M8.3 | Retrieval helper for planner and reviewer inputs |
| M8.4 | Tests for historical consistency and retrieval quality |

### M9 — Knowledge Base & Reference Ingestion
**Type:** Major | **Outcome:** Reusable tactics and research findings inform multiple projects.

| Minor Milestone | Deliverable |
|---|---|
| M9.1 | Knowledge-base taxonomy and wiki entry template |
| M9.2 | Reference inventory and source-ingestion conventions |
| M9.3 | Wiki Scribe flow for approved insights |
| M9.4 | Retrieval hooks from planner and reviewer |

### M10 — Benchmark Validation
**Type:** Major | **Outcome:** System proves value on benchmark datasets.

| Minor Milestone | Deliverable |
|---|---|
| M10.1 | Select benchmark datasets and target metrics |
| M10.2 | Run end-to-end baseline experiments |
| M10.3 | AutoGluon comparison runs |
| M10.4 | Benchmark retrospectives and gap analysis |

### M11+ — Expansion Tracks
**Type:** Future | **Outcome:** Broader capability.

Candidate tracks (prioritised after M10):
- Researcher agent for external reference investigation
- PDF Skill for paper ingestion
- Optional open-source model routing
- Time-series support
- Obsidian-compatible knowledge graph export
- RAG-based wiki retrieval
- Packaging for internal team adoption

---

## 18. Future Enhancements

### Near-Term (post-v1)

1. **Time-series support.** Extend data handling, feature engineering, and evaluation to temporal data with proper temporal cross-validation and leakage prevention.
2. **Researcher agent activation.** Automated investigation of papers and repos for tactics relevant to the active problem. Web search with structured extraction.
3. **PDF Skill deployment.** Bulk ingestion of ML papers into the knowledge base with automatic tactic extraction.
4. **Obsidian integration.** Export knowledge base to Obsidian-compatible format with bidirectional links, enabling human browsing and curation.
5. **Bayesian hyperparameter optimisation.** Integrate Optuna for focused hyperparameter search within the agent-chosen model family.
6. **SHAP integration.** Richer explainability in model reports with SHAP value analysis.
7. **Parametric distribution identification.** Extend the Dataset Analyser to fit parametric distribution families (normal, gamma, log-normal, etc.) to numerical features with goodness-of-fit scores. Optionally characterise feature-target relationships with simple parametric fits (linear, polynomial) and report R² quality. Useful for both human understanding and Planner decision-making about transformations.
8. **Script-to-notebook conversion.** Use a tool like jupytext to convert Python scripts into notebook format for human readability. Not a core workflow requirement, but a convenience for reviewing experiment code in a more visual format.

### Medium-Term

9. **Multi-provider model routing.** Route routine tasks (code generation, simple reviews) to smaller/cheaper models while reserving Opus for planning and complex reasoning.
10. **RAG-based knowledge retrieval.** Embed wiki entries and retrieve semantically rather than by filename matching.
11. **Ensemble orchestration.** Planner can propose multi-model ensembles; Coder can generate stacking/blending code.
12. **Multi-table dataset support.** Handle relational datasets with join strategies proposed by the Planner.
13. **Custom metric plugins.** User-defined evaluation metrics with automatic integration into the reporting pipeline.

### Long-Term

14. **Internal team adoption.** Package the system for use by a data science team on internal business problems, with shared knowledge base across projects.
15. **Competition automation.** End-to-end Kaggle competition workflow including submission generation and leaderboard tracking.
16. **Multimodal ML.** Extend beyond tabular to text, image, and mixed-modal problems.
17. **Continuous learning.** System improves its own planning and reviewing quality based on historical outcomes across many projects.
18. **Deployment pipeline.** Generate deployment-ready model packages (Docker, API endpoints) from final candidates.

---

## 19. Reference Materials

### Primary Inspirations

| Source | Relevance | What to Extract |
|---|---|---|
| [AI Analyst Plugin](https://github.com/ai-analyst-lab/ai-analyst-plugin) | Agentic analytics workflow patterns | Orchestration patterns, artifact design, agent role definitions |
| [Karpathy autoresearch](https://github.com/karpathy/autoresearch) | Iterative research loops and memory patterns | Iteration structure, history handling, research workflow design |
| [AutoKaggle](https://github.com/ShaneZhong/autokaggle) | Benchmark-oriented autonomous competition workflow | Competition loop structure, evaluation strategy, run organisation |
| [DS-STAR papers](https://arxiv.org/pdf/2509.21825) | Planning, execution, review, and corrective loops for data science agents | Planner-reviewer-router patterns, two-stage debugging, control loops |
| [Data Science Agent paper](https://arxiv.org/pdf/2410.02958) | Small-step iteration philosophy for agent-driven DS | Agent decomposition, prompt structures, recovery mechanisms |
| [Karpathy personal KB](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) | Reusable wiki/knowledge-base layer design | Organisational principles, note templates, retrieval-friendly structure |
| [Newsletter Engine](https://github.com/JoseParrenoGarcia/newsletter-engine) | Existing reusable patterns, skills, and rules from prior agentic work | Import/indexing skills, rule structure, workflow conventions |
| [PDF Import Skill](https://github.com/JoseParrenoGarcia/newsletter-engine/blob/main/.claude/skills/import-pdf/SKILL.md) | Paper ingestion without context bloat | Document-conversion workflow, ingestion rules, summary patterns |

### Reference Ingestion Principles

1. Do not store references as a loose pile of links.
2. Every reference must have: a short summary, relevance to this project, sections worth focusing on, and reusable tactics extracted from it.
3. Prefer text-extractable versions of sources when building local searchable notes.
4. Separate raw source captures from cleaned notes and reusable patterns.

### Standards and Conventions

- **Artifact formats:** YAML for plans and routing decisions. JSON for metric-heavy machine-readable outputs. Markdown for human-facing summaries. JSONL for append-only logs.
- **Naming:** kebab-case for files and directories. Snake_case for Python code. Iteration numbers are 1-indexed.
- **Documentation:** Markdown throughout. No notebooks. No Word docs. No Google Docs.
- **Version control:** All code, configs, and artifacts are git-tracked. Model binaries are gitignored but referenced by hash in manifests.

---

*End of PRD. This document should be updated as decisions are made and milestones are completed.*
