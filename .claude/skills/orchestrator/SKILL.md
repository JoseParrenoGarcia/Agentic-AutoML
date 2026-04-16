---
name: orchestrator
description: >
  Runs the full ML experiment loop end-to-end for any project. Chains six agents
  sequentially (dataset-analyser, planner, coder, executor, model-report-builder,
  reviewer-router) with artifact gate checks between each step. Loops automatically
  until the reviewer verdict is "sufficient" or MAX_ITERATIONS (10) is reached.
  Use when: running a complete AutoML experiment, starting a new project from raw
  CSV to final model, resuming an experiment from prior iteration state, running
  iterations until convergence or budget exhaustion. Invoke with: "run the
  orchestrator skill on projects/<project-name>". Reads project.yaml to determine
  task type and data paths. Reports structured progress after each iteration and
  a final experiment summary on completion. NOT for: running a single agent in
  isolation, debugging a specific iteration, modifying project configuration,
  analysing data without running experiments, or editing existing artifacts.
version: 1.0.0
---

# Orchestrator

Run the full ML experiment loop: dataset analysis, planning, coding, execution, evaluation, and review — repeated until the model is sufficient or the iteration budget is exhausted.

## Constants

- `MAX_ITERATIONS = 10` — hard cap on total iterations (including any previously completed)
- `.venv/bin/python3` — all validator commands use the project virtual environment
- Agent prompt pattern: `"run the <agent-name> agent on projects/<project-name>"`

---

## Invocation

```
run the orchestrator skill on projects/<project-name>
```

Extract `<project-name>` from the invocation. All paths below are relative to `projects/<project-name>/`.

---

## Step 0 — Determine state

Before running any agents, assess the current project state.

1. **Read `project.yaml`** — extract `task_type` (needed for executor gate check).

2. **Check profile:** Does `artifacts/data/profile.json` exist? → `profile_exists`

3. **Count completed iterations:** List `iterations/iteration-*/execution/manifest.json` files. For each, read the manifest and count those with `"status": "success"`. → `completed_iterations`

4. **Read last verdict (if iterations exist):** Read the last line of `memory/run-history.jsonl`:
   ```bash
   .venv/bin/python3 -c "
   import json, pathlib
   p = pathlib.Path('projects/<project>/memory/run-history.jsonl')
   if p.exists():
       last = json.loads(p.read_text().strip().split('\n')[-1])
       print(f'verdict={last[\"reviewer_verdict\"]}')
       print(f'route={last[\"router_decision\"]}')
       print(f'best={last[\"best_iteration\"]}')
   else:
       print('No run history found')
   "
   ```

5. **Note start time** for elapsed tracking.

6. **Print state detection report:**
   ```
   +--------------------------------------------------------------+
   | ORCHESTRATOR — STATE DETECTION                                |
   +--------------------------------------------------------------+
   | Project:          <project-name>                              |
   | Task type:        <task_type>                                 |
   | Profile exists:   <yes/no>                                    |
   | Completed iters:  <completed_iterations>                      |
   | Last verdict:     <verdict or N/A>                            |
   | Last route:       <route or N/A>                              |
   | Best iteration:   <best or N/A>                               |
   | Action:           <see below>                                 |
   +--------------------------------------------------------------+
   ```

7. **Early exit check:**
   - If `last_verdict == "sufficient"` → print "Experiment already complete" + final summary (Step 3). Stop.
   - If `completed_iterations >= MAX_ITERATIONS` → print "Max iterations reached" + final summary (Step 3). Stop.
   - Otherwise → proceed to Step 1.

---

## Step 1 — Ensure dataset profile exists

**Condition:** Only run this step if `profile_exists` is false.

**Action:** Launch the dataset-analyser agent:
```
run the dataset-analyser agent on projects/<project-name>
```

**Gate check after agent returns:**
1. Verify `artifacts/data/profile.json` exists
2. Verify `artifacts/data/profile.md` exists
3. Run structural validation:
   ```bash
   .venv/bin/python3 -c "
   import json, pathlib
   p = json.load(open('projects/<project>/artifacts/data/profile.json'))
   required = ['profiler_version', 'generated_at', 'source', 'columns', 'correlation', 'target_validation', 'leakage_flags', 'feature_risk_flags', 'mutual_information']
   missing = [k for k in required if k not in p]
   assert not missing, f'Missing keys: {missing}'
   plots = list(pathlib.Path('projects/<project>/artifacts/data/plots').glob('*.png'))
   print(f'OK: {len(p[\"columns\"])} columns profiled, {len(plots)} plots')
   "
   ```

**On failure:** Retry the dataset-analyser agent once. If still failing, escalate (see Error Handling).

---

## Step 2 — Iteration loop

Repeat this step until a stop condition is met (Step 2g).

### 2a — Planner

Launch the planner agent:
```
run the planner agent on projects/<project-name>
```

**Gate check:**
1. Discover the iteration number N from the newest plan file:
   ```bash
   .venv/bin/python3 -c "
   import pathlib
   plans = sorted(pathlib.Path('projects/<project>/artifacts/plans').glob('iteration-*.yaml'))
   n = int(plans[-1].stem.split('-')[1])
   print(f'ITERATION_N={n}')
   "
   ```
   Record `N` — use it for all subsequent gate checks in this loop iteration.

2. Verify `artifacts/plans/iteration-<n>.md` exists.

3. Validate the plan:
   ```bash
   .venv/bin/python3 -c "
   from src.planning.validator import validate_plan
   result = validate_plan('projects/<project>/artifacts/plans/iteration-<n>.yaml')
   print(f'OK: iteration={result[\"iteration\"]}')
   "
   ```

**On failure:** Retry planner once. If still failing, escalate.

---

### 2b — Coder

Launch the coder agent:
```
run the coder agent on projects/<project-name>
```

**Gate check:**
1. Verify `iterations/iteration-<n>/src/` directory exists with Python files.
2. Verify `iterations/iteration-<n>/config.yaml` exists.
3. Validate:
   ```bash
   .venv/bin/python3 -c "
   from src.codegen.validator import validate_codegen
   result = validate_codegen(
       'projects/<project>/iterations/iteration-<n>',
       plan_path='projects/<project>/artifacts/plans/iteration-<n>.yaml'
   )
   print(f'OK: {result[\"files_checked\"]} files checked, config valid')
   "
   ```

**On failure:** Retry coder once. If still failing, escalate.

---

### 2c — Executor

Launch the executor agent:
```
run the executor agent on projects/<project-name>
```

**Gate check:**
1. Read `iterations/iteration-<n>/execution/manifest.json`.
2. Check manifest status:
   ```bash
   .venv/bin/python3 -c "
   import json
   m = json.load(open('projects/<project>/iterations/iteration-<n>/execution/manifest.json'))
   if m['status'] != 'success':
       print(f'FAILED: status={m[\"status\"]}, error_class={m.get(\"error_class\")}, error_summary={m.get(\"error_summary\")}')
       exit(1)
   print(f'OK: status=success, duration={m[\"duration_s\"]}s, retries={m[\"retry_count\"]}')
   "
   ```
3. If status is success, validate outputs:
   ```bash
   .venv/bin/python3 -c "
   from src.execution.output_validator import validate_outputs
   result = validate_outputs('projects/<project>/iterations/iteration-<n>', task_type='<task_type>')
   print(f'OK: primary={result[\"metrics_primary\"]}, artifacts_checked={result[\"artifacts_checked\"]}')
   "
   ```

**On failure:** Do NOT retry the executor. It has an internal 5-attempt retry budget. If the manifest shows `status: "failed"`, escalate immediately with the `error_class` and `error_summary` from the manifest.

---

### 2d — Model Report Builder

Launch the model-report-builder agent:
```
run the model-report-builder agent on projects/<project-name>
```

**Gate check:**
1. Verify `iterations/iteration-<n>/reports/model-report.json` exists.
2. Verify `iterations/iteration-<n>/reports/model-report.md` exists.
3. Validate:
   ```bash
   .venv/bin/python3 -c "
   from src.evaluation.validator import validate_report
   result = validate_report('projects/<project>/iterations/iteration-<n>/reports/model-report.json')
   print(f'OK: verdict={result[\"headline_verdict\"]}, primary_metric={result[\"primary_metric\"]}, risk_flags={result[\"risk_flag_count\"]}')
   "
   ```

**On failure:** Retry model-report-builder once. If still failing, escalate.

---

### 2e — Reviewer-Router

**Before launching:** Count lines in `memory/run-history.jsonl` (for verification):
```bash
wc -l < projects/<project>/memory/run-history.jsonl 2>/dev/null || echo 0
```

Launch the reviewer-router agent:
```
run the reviewer-router agent on projects/<project-name>
```

**Gate check:**
1. Verify `memory/run-history.jsonl` has one more line than before.
2. Verify `iterations/iteration-<n>/reports/review-decision.json` exists.
3. Validate and extract the verdict:
   ```bash
   .venv/bin/python3 -c "
   import json
   from src.review.validator import validate_review_decision
   with open('projects/<project>/memory/run-history.jsonl') as f:
       lines = f.readlines()
   last = json.loads(lines[-1])
   result = validate_review_decision(last)
   print(f'OK: verdict={last[\"reviewer_verdict\"]}, route={last[\"router_decision\"]}, best={last[\"best_iteration\"]}')
   print(f'metric_name={last[\"primary_metric\"][\"name\"]}')
   print(f'metric_value={last[\"primary_metric\"][\"value\"]}')
   print(f'metric_delta={last[\"primary_metric\"][\"delta\"]}')
   print(f'model_family={last[\"model_family\"]}')
   "
   ```

Extract these fields for the progress report:
- `reviewer_verdict`, `router_decision`, `best_iteration`
- `primary_metric.name`, `primary_metric.value`, `primary_metric.delta`
- `model_family`

**On failure:** Retry reviewer-router once. If still failing, escalate.

---

### 2f — Progress report

Print a structured status after each iteration:

```
+--------------------------------------------------------------+
| ITERATION <n> COMPLETE                                        |
+--------------------------------------------------------------+
| Metric:    <name> = <value> (delta: <delta>)                  |
| Model:     <model_family>                                     |
| Verdict:   <reviewer_verdict>                                 |
| Route:     <router_decision>                                  |
| Best:      iteration-<best_iteration>                         |
| Elapsed:   <Xm Ys> (this iteration)                           |
| Progress:  <n> / 10 iterations                                |
+--------------------------------------------------------------+
```

Use the fields extracted from the reviewer gate check (Step 2e). Calculate elapsed time since the iteration started.

---

### 2g — Check stop conditions

Evaluate in priority order:

1. **`reviewer_verdict == "sufficient"`** → exit loop, proceed to Step 3. Stop reason: `sufficient`.
2. **`N >= MAX_ITERATIONS` (10)** → exit loop, proceed to Step 3. Stop reason: `max_iterations`.
3. **Otherwise** → loop back to Step 2a for the next iteration.

---

## Step 3 — Final summary

Read all records from `memory/run-history.jsonl` and print:

```
+======================================================================+
| EXPERIMENT COMPLETE                                                    |
+======================================================================+
| Project:          <project-name>                                       |
| Total iterations:  <n>                                                 |
| Stop reason:       <sufficient | max_iterations | failure>             |
| Best iteration:    iteration-<best>                                    |
| Best metric:       <name> = <value>                                    |
| Total elapsed:     <total time since Step 0>                           |
+----------------------------------------------------------------------+
| Iteration History:                                                     |
|   iter-1: <metric>=<value> (<model>) [<verdict>/<route>]              |
|   iter-2: <metric>=<value> (<model>) [<verdict>/<route>]              |
|   ...                                                                  |
+======================================================================+
```

Build the iteration history by reading each line of `run-history.jsonl` and formatting it.

---

## Error Handling

### Retry policy

| Agent | Max external retries | Rationale |
|-------|---------------------|-----------|
| dataset-analyser | 1 | Deterministic — transient env issue possible |
| planner | 1 | LLM-based — may fail on edge cases |
| coder | 1 | LLM-based — validation may catch fixable issues |
| executor | **0** | Has internal 5-attempt retry loop — never retry externally |
| model-report-builder | 1 | Mostly deterministic scripts |
| reviewer-router | 1 | LLM-based — may fail on edge cases |

### Retry procedure

When an agent's gate check fails:
1. Log the failure: which agent, what was missing/invalid, the error message.
2. If retries remain: re-launch the same agent with the same prompt. Run the gate check again.
3. If no retries remain: escalate.

### Escalation

When retries are exhausted, print the escalation report and **stop the loop**:

```
+======================================================================+
| ORCHESTRATOR ESCALATION — HUMAN ACTION REQUIRED                       |
+======================================================================+
| Agent:           <agent-name>                                          |
| Step:            <step>                                                |
| Iteration:       <n>                                                   |
| Project:         <project-name>                                        |
+----------------------------------------------------------------------+
| What failed:     <missing files or validation error>                   |
| Error detail:    <validator message or manifest error_summary>         |
| Attempts:        <count> (max: <max>)                                  |
+----------------------------------------------------------------------+
| Suggested action: <specific guidance based on the error>               |
+======================================================================+
```

After printing, stop the orchestrator. Do not attempt further iterations.

---

## Gotchas

1. **Executor has internal retries** — the executor agent has a 5-attempt internal retry budget (3 Stage 1 + 2 Stage 2). Do NOT retry it externally. If its manifest shows `status: "failed"`, escalate immediately.

2. **Agents self-discover iteration N** — never pass the iteration number in the agent prompt. The planner counts plan files; the coder reads the plan YAML; the executor/report-builder/reviewer scan iteration directories. Passing N creates coupling and risks mismatches.

3. **Planner reads run-history.jsonl for routing signals** — the orchestrator does NOT encode the routing decision (continue/rollback/pivot) in the planner prompt. The planner reads `memory/run-history.jsonl` itself and interprets the last record's `router_decision`.

4. **Gate checks run AFTER Agent tool returns** — each agent call is blocking. The gate check runs in the orchestrator's context after the agent completes.

5. **profile.json is write-once** — only run the dataset-analyser if `artifacts/data/profile.json` does not exist. Never re-profile mid-experiment.

6. **run-history.jsonl is append-only** — count lines before the reviewer runs and verify one new line was added after. Never rewrite or delete existing lines.

7. **All validators use `.venv/bin/python3`** — the project virtual environment, not system Python. Validators import from `src/` which requires the venv's PYTHONPATH.

8. **When resuming, do not re-run completed iterations** — the orchestrator picks up from where the last verdict left off. If 3 iterations are complete with verdict=insufficient, the next loop starts at iteration 4.
