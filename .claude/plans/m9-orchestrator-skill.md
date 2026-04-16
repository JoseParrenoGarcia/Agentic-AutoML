# M9 — Orchestrator Skill: Implementation Plan

## Context

The Agentic-AutoML project has 6 working agents that run sequentially to perform iterative ML experiments. Currently they're chained manually (as done in the iteration-2 smoke test). M9 encodes this exact sequence as a reusable, invocable Claude Code skill that loops automatically with artifact gate checks and stop conditions — zero human intervention.

**What prompted this:** M8 (memory/run-history) is complete. All 6 agents, 5 validators, and artifact contracts are in place. The orchestrator is the capstone that makes the system truly end-to-end.

**Intended outcome:** `run the orchestrator skill on projects/<project-name>` runs from raw CSV to final model autonomously.

---

## Files to Create

| # | File | Purpose |
|---|------|---------|
| 1 | `.claude/skills/orchestrator/SKILL.md` | Main skill (~350 lines) |
| 2 | `.claude/skills/orchestrator/references/gate-checks.md` | Detailed validation commands per agent step |
| 3 | `.claude/skills/orchestrator/references/progress-templates.md` | Output format templates with examples |

**No new Python code needed.** All 5 existing validators are sufficient:
- `src/planning/validator.validate_plan(plan_path)` → `dict`
- `src/codegen/validator.validate_codegen(iteration_dir, plan_path=None)` → `dict`
- `src/execution/output_validator.validate_outputs(iteration_dir, task_type)` → `dict`
- `src/evaluation/validator.validate_report(report_path)` → `dict`
- `src/review/validator.validate_review_decision(record_dict)` → `dict`

---

## SKILL.md Structure

### Frontmatter
```yaml
name: orchestrator
description: >
  Runs the full ML experiment loop end-to-end for any project. Chains six agents
  (dataset-analyser, planner, coder, executor, model-report-builder, reviewer-router)
  with artifact gate checks between each step. Loops until reviewer verdict is
  "sufficient" or MAX_ITERATIONS (10) reached. Use when: running a complete AutoML
  experiment, starting a new project end-to-end, resuming from prior state, running
  iterations until convergence. Invoke with: "run the orchestrator skill on
  projects/<project-name>". NOT for: running a single agent, debugging an iteration,
  modifying project config, or data analysis without experiments.
```

### Body Sections (~350 lines)

```
# Orchestrator
## Constants                    — MAX_ITERATIONS=10, .venv path
## Invocation                   — "run the orchestrator skill on projects/<project-name>"
## Step 0 — Determine state     — Read project.yaml, check profile, count iterations, read last verdict
## Step 1 — Dataset profile     — Conditional: only if no profile.json
## Step 2 — Iteration loop      — The core loop
  ### 2a — Planner              — Agent call + gate check (validate_plan)
  ### 2b — Coder                — Agent call + gate check (validate_codegen)
  ### 2c — Executor             — Agent call + gate check (validate_outputs)
  ### 2d — Report Builder       — Agent call + gate check (validate_report)
  ### 2e — Reviewer-Router      — Agent call + gate check (validate_review_decision)
  ### 2f — Progress report      — Structured status output
  ### 2g — Stop conditions      — Check verdict + iteration cap
## Step 3 — Final summary       — Full experiment summary
## Error handling               — Retry policy + escalation format
## Gotchas                      — 8 critical gotchas
```

---

## Core Design Decisions

### 1. Agent invocation pattern
Each agent is called via the Agent tool with: `"run the <agent-name> agent on projects/<project-name>"`
- Agents self-discover iteration number (planner counts plans, others scan dirs)
- Orchestrator NEVER passes iteration N in the prompt — avoids coupling
- After each agent returns, orchestrator determines N by reading the newest artifact

### 2. Gate checks between steps
After each agent, verify artifacts exist AND pass validation:
```
Agent returns → Check file exists → Run Python validator via Bash → Extract key fields
```
- All validators accept path strings, return summary dicts, raise `*ValidationError` on failure
- Gate checks use `.venv/bin/python3 -c "..."` one-liners

### 3. Iteration number discovery
- After planner: list `artifacts/plans/iteration-*.yaml`, take highest N
- This N is used for all subsequent gate checks in the same loop iteration
- On resume: count completed iterations (dirs with successful manifests)

### 4. Stop conditions (priority order)
1. `reviewer_verdict == "sufficient"` → stop, report final model
2. `current_iteration >= MAX_ITERATIONS (10)` → stop, report best iteration
3. Unrecoverable agent failure → escalate to human

### 5. Retry policy

| Agent | External retries | Rationale |
|-------|-----------------|-----------|
| dataset-analyser | 1 | Deterministic, transient env issue possible |
| planner | 1 | LLM-based, may fail on edge cases |
| coder | 1 | LLM-based, validation may catch fixable issues |
| executor | **0** | Has internal 5-attempt retry. Never retry externally. |
| model-report-builder | 1 | Mostly deterministic scripts |
| reviewer-router | 1 | LLM-based |

### 6. Executor special case
If `manifest.json` shows `status: "failed"`, do NOT retry. The executor already exhausted its internal retry budget. Escalate immediately with `error_class` and `error_summary` from the manifest.

### 7. Resume from prior state
On invocation, the skill checks existing state:
- If `last_verdict == "sufficient"` → report final summary, no new iterations
- If `completed_iterations >= MAX_ITERATIONS` → report final summary
- If `completed_iterations > 0` with `last_verdict == "insufficient"` → enter loop at next iteration
- If `completed_iterations == 0` → fresh start from Step 1

---

## Progress Reporting (M9.3)

### After each iteration:
```
+--------------------------------------------------------------+
| ITERATION <n> COMPLETE                                        |
+--------------------------------------------------------------+
| Metric:    <name> = <value> (delta: <delta>)                  |
| Model:     <model_family>                                     |
| Verdict:   <reviewer_verdict>                                 |
| Route:     <router_decision>                                  |
| Best:      iteration-<best_iteration>                         |
| Elapsed:   <Xm Ys>                                            |
| Progress:  <n> / 10 iterations                                |
+--------------------------------------------------------------+
```

### Final summary:
```
+======================================================================+
| EXPERIMENT COMPLETE                                                    |
+======================================================================+
| Project:          <project-name>                                       |
| Total iterations:  <n>                                                 |
| Stop reason:       <sufficient | max_iterations | failure>             |
| Best iteration:    iteration-<best>                                    |
| Best metric:       <name> = <value>                                    |
| Total elapsed:     <Xh Ym Zs>                                         |
+----------------------------------------------------------------------+
| Iteration History:                                                     |
|   iter-1: <metric>=<value> (<model_family>) [<verdict>/<route>]       |
|   iter-2: <metric>=<value> (<model_family>) [<verdict>/<route>]       |
|   ...                                                                  |
+======================================================================+
```

---

## Error Escalation Format

```
ORCHESTRATOR ESCALATION — HUMAN ACTION REQUIRED
Agent:           <agent-name>
Step:            <2a|2b|2c|2d|2e>
Iteration:       <n>
Project:         <project-name>
What failed:     <missing files or validation error>
Error detail:    <validator error message or manifest error_summary>
Attempts:        <1 or 2>
Suggested action: <specific next step>
```

---

## Gotchas (in SKILL.md body, not references)

1. **Executor has internal retries** — do NOT retry it externally on failure
2. **Agents self-discover iteration N** — never pass it in the prompt
3. **Planner reads run-history.jsonl for routing signals** — orchestrator doesn't encode routing
4. **Gate checks run AFTER Agent tool returns** — it's a blocking call
5. **profile.json is write-once** — only run dataset-analyser if it doesn't exist
6. **run-history.jsonl is append-only** — count lines before/after reviewer to verify
7. **All validators need `.venv/bin/python3`** — the project venv, not system Python
8. **When resuming, don't re-run completed iterations** — pick up from last verdict

---

## Implementation Order

**Phase 1: Create skill files (M9.1 + M9.2 + M9.3 interleaved)**
1. Create directory `.claude/skills/orchestrator/references/`
2. Write `references/gate-checks.md` — exact Bash validation commands per agent
3. Write `references/progress-templates.md` — output format templates
4. Write `SKILL.md` — full skill with loop, gate checks, error handling, progress reporting, gotchas

**Phase 2: Update project docs**
5. Update `.claude/CLAUDE.md` — add orchestrator to skills table, update active milestone

**Phase 3: Smoke test (M9.4)**
6. Run: `run the orchestrator skill on projects/titanic`
7. Verify: detects 3 completed iterations, resumes at iteration 4
8. Verify: progress report printed after iteration 4
9. Verify: loop continues or stops based on verdict
10. Verify: all artifacts, decision-log entries, run-history consistency
11. Verify: final summary on completion

---

## Verification

- **Structural:** SKILL.md under 500 lines, frontmatter name matches directory, description ~800 chars with trigger keywords and exclusions
- **Gate checks:** Each of the 5 validators can be called with the exact Bash one-liners from gate-checks.md (dry-run on existing Titanic artifacts)
- **Resume logic:** Running on Titanic should detect 3 completed iterations and enter at iteration 4, not re-run 1-3
- **Smoke test (M9.4):** Full run from iteration 4 onward, producing valid artifacts at each step
