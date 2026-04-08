# Agentic AutoML — Product Requirements Document

**Status:** Draft v0.2
**Last updated:** 2026-04-08

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
├── skills/
│   ├── pdf-import/
│   ├── dataset-analysis/
│   ├── report-formatting/
│   └── memory-update/
├── rules/
│   ├── coding-rules.md
│   ├── artifact-contracts.md
│   └── wiki-entry-rules.md
├── hooks/
│   ├── post-run-memory-update/
│   └── wiki-scribe/
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
- Dataset semantic description at dataset and column level
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
- `artifacts/plans/iteration-<n>.yaml` — structured plan
- `artifacts/plans/iteration-<n>.md` — optional human-readable rationale

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

**Purpose:** Transform raw run outputs into a comprehensive evaluation package.

**Inputs:**
- Trained model outputs (predictions, probabilities)
- Raw metrics from evaluation script
- Plots and learning curves
- Run metadata and manifest

**Outputs:**
- `runs/iteration-<n>/reports/model-report.json` — machine-readable metrics and diagnostics
- `runs/iteration-<n>/reports/model-report.md` — human-readable evaluation narrative

**Report Contents:**
- **Headline metrics:** Primary optimisation metric + secondary metrics
- **Actual vs predicted:** Scatter plots (regression), confusion matrix (classification)
- **Segment analysis:** Performance breakdown by key features or data slices
- **Feature importance:** Permutation importance or model-native importance scores
- **Error distribution:** Residual analysis (regression), misclassification patterns (classification)
- **Calibration assessment:** Probability calibration curves (classification)
- **Leakage indicators:** Suspiciously high metrics, feature importance anomalies
- **Overfitting signals:** Train/validation gap, learning curve analysis
- **Comparison to prior runs:** Delta metrics table against previous iterations

**Non-responsibilities:**
- Deciding whether results are good enough (that is the Reviewer's job)
- Choosing next steps

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
- **Package management:** pip + virtual environments (one per project)
- **OS:** macOS (primary development), Linux (CI/production)

### Core Python Dependencies

| Package | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Classical ML models, metrics, preprocessing |
| lightgbm | Gradient boosting |
| xgboost | Gradient boosting (alternative) |
| catboost | Gradient boosting (categorical-native) |
| matplotlib / seaborn | Visualisation |
| pyyaml | YAML artifact reading/writing |
| jsonlines | JSONL experiment tracking |

### Optional Dependencies (Phase 2+)

| Package | Purpose |
|---|---|
| shap | Explainability |
| optuna | Bayesian hyperparameter optimisation |
| autogluon | Benchmarking comparison baseline |
| sentence-transformers | Knowledge base RAG embeddings |

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
- Python environment captured in `requirements.txt` per iteration
- Dataset version (hash) recorded in manifest
- Full code stored per iteration (not just diffs)
- Environment metadata (Python version, OS, package versions) in manifest

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

### M0 — Documentation & Contracts Foundation
**Type:** Major | **Outcome:** Clear PRD, specs, artifact contracts, and repository structure.

| Minor Milestone | Deliverable |
|---|---|
| M0.1 | PRD and spec-kit documents finalised |
| M0.2 | Artifact templates for project metadata, plans, reviews, memory logs |
| M0.3 | Repository folders and placeholder structure |
| M0.4 | Coding rules and artifact contract docs |

### M1 — Single-Project Runtime Skeleton
**Type:** Major | **Outcome:** A project can be initialised and run through an empty but structured loop.

| Minor Milestone | Deliverable |
|---|---|
| M1.1 | Project bootstrap command and directory template |
| M1.2 | Run-iteration skeleton and state loader/saver |
| M1.3 | Base config files and environment conventions |
| M1.4 | Smoke tests for project creation and iteration folder creation |

### M2 — Deterministic Dataset Understanding
**Type:** Major | **Outcome:** Credible dataset profile and EDA summary for any supplied tabular dataset.

| Minor Milestone | Deliverable |
|---|---|
| M2.1 | Schema and basic stats profiler |
| M2.2 | Null, cardinality, outlier, and correlation analysis |
| M2.3 | Generate `profile.json` and `profile.md` |
| M2.4 | Plot generation as optional human-facing output |
| M2.5 | Test fixtures for common data-quality edge cases |

### M3 — Planning Layer
**Type:** Major | **Outcome:** System turns project context into a structured first experiment plan.

| Minor Milestone | Deliverable |
|---|---|
| M3.1 | Plan schema definition and validation |
| M3.2 | Initial planner prompt and templates |
| M3.3 | Project-memory summary input integration |
| M3.4 | Fixture-based planner tests using stable dataset profiles |

### M4 — Plan-to-Code Layer
**Type:** Major | **Outcome:** System translates a plan into executable Python code using templates.

| Minor Milestone | Deliverable |
|---|---|
| M4.1 | Baseline experiment code template |
| M4.2 | Feature step rendering into code |
| M4.3 | Model step rendering into code |
| M4.4 | Code-generation smoke tests |

### M5 — Execution & Debugging Loop
**Type:** Major | **Outcome:** Generated experiment can run, fail safely, and attempt bounded self-repair.

| Minor Milestone | Deliverable |
|---|---|
| M5.1 | Run executor and log capture |
| M5.2 | Runtime manifest generation |
| M5.3 | Two-stage bounded debugger with retry classification |
| M5.4 | Tests for syntax, dependency, and runtime failure scenarios |

### M6 — Evaluation & Model Reporting
**Type:** Major | **Outcome:** Every run ends with a deep evaluation package.

| Minor Milestone | Deliverable |
|---|---|
| M6.1 | Base metric calculators by problem type |
| M6.2 | Actual-vs-predicted and segment analysis |
| M6.3 | Feature-importance and explainability hooks |
| M6.4 | Emit `model-report.json` and `model-report.md` |

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

### Medium-Term

7. **Multi-provider model routing.** Route routine tasks (code generation, simple reviews) to smaller/cheaper models while reserving Opus for planning and complex reasoning.
8. **RAG-based knowledge retrieval.** Embed wiki entries and retrieve semantically rather than by filename matching.
9. **Ensemble orchestration.** Planner can propose multi-model ensembles; Coder can generate stacking/blending code.
10. **Multi-table dataset support.** Handle relational datasets with join strategies proposed by the Planner.
11. **Custom metric plugins.** User-defined evaluation metrics with automatic integration into the reporting pipeline.

### Long-Term

12. **Internal team adoption.** Package the system for use by a data science team on internal business problems, with shared knowledge base across projects.
13. **Competition automation.** End-to-end Kaggle competition workflow including submission generation and leaderboard tracking.
14. **Multimodal ML.** Extend beyond tabular to text, image, and mixed-modal problems.
15. **Continuous learning.** System improves its own planning and reviewing quality based on historical outcomes across many projects.
16. **Deployment pipeline.** Generate deployment-ready model packages (Docker, API endpoints) from final candidates.

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
