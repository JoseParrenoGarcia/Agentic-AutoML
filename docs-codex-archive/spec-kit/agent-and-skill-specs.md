# Agent And Skill Specs

## Purpose

Define the first set of components, their responsibilities, and their interfaces. The emphasis is on stable contracts and narrow responsibilities rather than cleverness.

## Component Design Rules

1. Each component owns one decision class.
2. Inputs should come from structured files whenever possible.
3. Outputs should be machine-readable first and human-readable second.
4. Prompts should reference templates and rules, not rely on hidden assumptions.
5. Components should fail loudly on missing prerequisites instead of inventing context.

## Phase 1 Core Components

### 1. Dataset Analyzer

Purpose:

- produce a reliable dataset understanding package before any planning begins

Inputs:

- project metadata
- raw datasets or dataset manifests
- optional user-provided semantic hints

Outputs:

- `profile.json`
- `profile.md`
- plots and statistical summaries
- warnings about likely identifiers, leakage risks, null-heavy columns, skew, cardinality, and target ambiguity

Responsibilities:

- schema and type inference
- descriptive statistics
- target and feature observations
- dataset summary narrative
- feature-risk flags

Non-responsibilities:

- choosing the final modeling strategy
- mutating the dataset beyond read-only profiling unless explicitly part of the analysis contract

### 2. Planner

Purpose:

- convert current project context into the next experiment plan

Inputs:

- dataset profile
- project objective
- prior plan
- previous run results
- reviewer report
- router decision
- relevant knowledge-base entries
- project memory summary

Outputs:

- `iteration-<n>.yaml`
- optional `iteration-<n>.md` rationale

Responsibilities:

- define hypotheses
- define planned feature-engineering steps
- define planned modeling steps
- define expected evidence of success or failure
- keep changes small enough to attribute outcomes

Non-responsibilities:

- writing executable code
- deciding whether a completed run is acceptable

### 3. Plan-to-Code

Purpose:

- translate the current approved plan into concrete Python implementation

Inputs:

- current plan
- prior code template or previous run code
- coding rules
- artifact contracts

Outputs:

- generated Python files
- run config
- dependency declarations if needed

Responsibilities:

- preserve project templates
- implement planned transformations and modeling logic
- maintain stable file layout
- keep code readable and reviewable

Non-responsibilities:

- executing the code
- deciding on the next experiment branch

### 4. Executor And Debugger

Purpose:

- run the experiment, capture outcomes, and repair bounded failures

Inputs:

- generated code
- run config
- project environment

Outputs:

- execution logs
- runtime manifest
- retry records
- final exit status

Responsibilities:

- execute code safely
- capture failures in structured form
- apply bounded repair loops
- avoid infinite retries

Non-responsibilities:

- strategic experiment decisions
- rewriting historical runs

### 5. Model Report Builder

Purpose:

- transform raw run outputs into a comprehensive evaluation package

Inputs:

- trained model outputs
- predictions
- metrics
- plots
- run metadata

Outputs:

- `model-report.json`
- `model-report.md`

Responsibilities:

- summarize metrics
- compare actual vs predicted behavior
- segment evaluation
- feature importance or explainability artifacts where applicable
- surface audit concerns such as leakage or suspicious performance patterns

### 6. Reviewer

Purpose:

- judge experiment quality and explain what the results imply

Inputs:

- model report
- previous run comparisons
- project objective
- current plan
- project memory

Outputs:

- `reviewer-report.yaml`

Responsibilities:

- assess whether the run improved meaningfully
- explain why improvement may have happened
- detect warning signs
- decide whether the current direction is plateauing
- recommend tactical next actions

### 7. Action Router

Purpose:

- decide the next workflow action after review

Inputs:

- reviewer report
- current plan
- current run metadata
- project memory summary

Outputs:

- `router-decision.yaml`

Allowed actions:

- finalize
- replan with refinement
- replan with replacement of one step
- replan with additional step
- trigger divergent exploration
- escalate to user

Design note:

The router should explain why it chose the next action and what evidence threshold drove the decision.

### 8. Project Memory Updater

Purpose:

- keep an accurate durable record of what happened and what was learned

Inputs:

- plan
- execution manifest
- model report
- reviewer report
- router decision

Outputs:

- appended memory records
- summarized lessons learned

Responsibilities:

- preserve historical sequence
- normalize key facts from each run
- make later retrieval easier

## Supporting Components

### Local-Maxima Challenger

Role:

- specialized review helper that detects diminishing returns and proposes more divergent strategy branches

When to trigger:

- score improvement falls below threshold across several consecutive iterations
- model family remains unchanged for too long
- error profile stagnates even when headline metric moves slightly

### Researcher

Role:

- investigate external references and bring back tactics relevant to the active problem type

Phase recommendation:

- keep as a later phase component until the core local loop is stable

### Wiki Scribe

Role:

- convert approved insights into structured knowledge-base entries

Possible trigger:

- after a run produces a reusable lesson with enough evidence

## Skills And Rules To Support The System

### Skills

Recommended initial skills:

- reference-import skill
  Converts papers or repos into indexed notes.
- dataset-analysis skill
  Runs deterministic profiling scripts and report generation.
- report-formatting skill
  Ensures output reports match templates.
- memory-update skill
  Summarizes iteration outcomes consistently.

### Rules

Recommended initial rules:

- no notebooks for baseline implementation
- always emit required artifacts
- use stable templates for run directories
- store decisions in machine-readable formats where possible
- fail if required upstream artifacts are missing

## Suggested Artifact Schemas

### Plan schema

Minimum fields:

- `iteration`
- `objective`
- `hypotheses`
- `feature_steps`
- `model_steps`
- `evaluation_focus`
- `expected_win_condition`
- `rollback_or_stop_condition`

### Reviewer schema

Minimum fields:

- `iteration`
- `headline_verdict`
- `metric_summary`
- `comparison_to_previous`
- `likely_causes`
- `risk_flags`
- `plateau_signal`
- `recommended_next_action`

### Router schema

Minimum fields:

- `iteration`
- `decision`
- `reasoning`
- `required_next_inputs`
- `human_approval_required`

## Testing Strategy Per Component

### Dataset Analyzer

- schema fixtures
- edge-case datasets
- known leakage examples

### Planner

- simulation tests using fixed upstream artifacts
- checks that plans stay within allowed action boundaries

### Plan-to-Code

- template conformance tests
- code-generation smoke tests

### Executor And Debugger

- retry-limit tests
- failure classification tests

### Model Report Builder

- metric-calculation tests
- artifact-presence tests

### Reviewer And Router

- fixture-based decision tests
- regression tests on plateau detection and escalation behavior

## Recommended Build Order

1. Dataset analyzer
2. Plan schema and planner
3. Code template and plan-to-code
4. Executor/debugger
5. Model report builder
6. Reviewer
7. Router
8. Memory updater
9. Knowledge-base ingestion
10. Optional helper agents
