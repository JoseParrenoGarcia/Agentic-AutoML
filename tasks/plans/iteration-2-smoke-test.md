# Iteration-2 Smoke Test Plan — Titanic Project

## Context
Iteration 1 is complete (LogisticRegression, AUC-ROC 0.835). Reviewer said "insufficient", router said "pivot" to tree-based models. All signals are in `run-history.jsonl`. Goal: run the full 5-agent loop autonomously with zero human gates.

## Execution Sequence

Each step is a subagent invocation. Sequential — each depends on the prior step's output.

### Step 1 — Planner
- Invoke: `run the planner agent on projects/titanic`
- Expects: reads `run-history.jsonl` (pivot signal), profile, project.yaml
- Produces: `artifacts/plans/iteration-2.yaml` + `iteration-2.md`
- Quick check: plan file exists, references a tree-based model

### Step 2 — Coder
- Invoke: `run the coder agent on projects/titanic`
- Expects: iteration-2.yaml plan, iteration-1/src/ as base code
- Produces: `iterations/iteration-2/src/*` (7 files + config.yaml + requirements.txt)
- Quick check: `model.py` imports a tree-based algorithm

### Step 3 — Executor
- Invoke: `run the executor agent on projects/titanic`
- Expects: iteration-2/src/* code
- Produces: `execution/manifest.json`, `outputs/*` (metrics, predictions, feature importance, etc.)
- Quick check: manifest status = "success"

### Step 4 — Model Report Builder
- Invoke: `run the model-report-builder agent on projects/titanic`
- Expects: iteration-2/outputs/*
- Produces: `reports/model-report.json`, `model-report.md`, `plots/*.png`
- Quick check: report files exist, `prior_run_comparison` is not null

### Step 5 — Reviewer-Router
- Invoke: `run the reviewer-router agent on projects/titanic`
- Expects: model-report.json, run-history.jsonl
- Produces: appends line 2 to `run-history.jsonl`, writes `review-decision.json`
- Quick check: run-history.jsonl has 2 lines

## Final Verification
- `run-history.jsonl` has exactly 2 valid JSONL entries
- Iteration-2 model family is tree-based (not LogisticRegression)
- AUC-ROC ideally > 0.835 baseline
- All artifact directories populated
- Zero human intervention required

## Success Criteria
- Plan respects pivot signal
- Execution succeeds without human help
- Metrics improve or are competitive
- Full artifact chain produced
- The loop "just works" end-to-end
