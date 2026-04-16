# M7 — Reviewer & Action Router Plan

**Status:** Complete
**Created:** 2026-04-15
**Milestone:** M7 (PRD lines 1263–1272)
**Outcome:** System judges iteration outcomes and decides the next loop action.

---

## Design Decisions (agreed with Jose)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent architecture | Single combined `reviewer-router` agent | Simpler, fewer handoffs, model-report.json is the only input |
| Verdict model | Two-tier: `sufficient` / `insufficient` | Inspired by DS-STAR; clean binary gate before routing |
| Routing actions (if insufficient) | `continue` / `rollback` / `pivot` | `continue` = same direction, next step; `rollback` = last direction wrong, revert to prior working approach; `pivot` = change strategy (new model family, technique class) |
| Iter-1 handling | Full review (no special baseline mode) | Apply full rubric even on first iteration |
| Sufficient logic | LLM holistic judgment (no hard thresholds) | New problems have no known target; agent reasons like a senior DS using structured rubric |
| Max iterations | Hardcoded default 10 for now; future orchestrator will make configurable | Hard stop safety net; router must stop when cap hit |
| JSONL writer | Via `src/review/` utility (not agent direct write) | Consistent with M2–M6 pattern; validates before append; testable |
| Planner update | Update Planner in M7 to read `router_decision` | Without this, routing signals are dead letter; Planner already reads run-history but ignores verdict/route fields |
| Plateau detection | Use M6's existing `plateau_signal` field | Keep simple for now, iterate later |
| Rollback semantics | Informational flag to Planner | No actual artifact rollback; Planner handles it in next plan |
| Output artifact | Append to `run-history.jsonl` (Contract 3) | Minimal new artifacts; Planner already reads this |
| Implementation | Claude agent in `.claude/agents/` | Consistent with M2–M6 pattern; flexible for nuanced judgment |

---

## Architecture Overview

```
model-report.json (M6)  ──┐
execution/manifest.json ───┤
run-history.jsonl ─────────┤──► reviewer-router agent ──► append to run-history.jsonl
experiment config ─────────┘         │
                                     ├─► verdict: sufficient → STOP
                                     └─► verdict: insufficient
                                              ├─► route: continue
                                              ├─► route: rollback
                                              └─► route: pivot
```

---

## Minor Milestones

### M7.1 — Review-decision schema and rubric definition

**Deliverables:**
- [ ] Define the `run-history.jsonl` record schema that M7 appends (extends Contract 3)
- [ ] Define the structured rubric the agent uses for judgment
- [ ] Hardcode `MAX_ITERATIONS = 10` default in `src/review/config.py` (orchestrator will override later)

**Review-decision record** (appended to `run-history.jsonl`):
```json
{
  "iteration": 1,
  "timestamp": "2026-04-15T10:00:00Z",
  "status": "completed",
  "plan_summary": "Baseline logistic regression with minimal feature engineering",
  "primary_metric": {"name": "auc_roc", "value": 0.835, "delta": null},
  "model_family": "logistic_regression",
  "reviewer_verdict": "insufficient",
  "reviewer_reasoning": "Model performs reasonably for a baseline but has room for improvement. No risk flags. Feature importance suggests unexploited interactions.",
  "router_decision": "continue",
  "router_reasoning": "First iteration baseline. Natural next step: feature engineering improvements before exploring other model families.",
  "risk_flags_summary": [],
  "best_iteration": 1
}
```

**Structured rubric** (agent evaluates these dimensions):
1. **Risk flags** — Any leakage or severe overfitting? (hard fail if high-severity)
2. **Metric quality** — Is the primary metric reasonable for this task type and dataset complexity?
3. **Improvement trajectory** — Is the metric improving, plateauing, or degrading vs prior iterations?
4. **Strategy exhaustion** — Have multiple model families / technique classes been tried?
5. **Iteration budget** — How many iterations remain before max cap?

**Sufficient criteria** (all must hold):
- No high-severity risk flags
- Metric is reasonable for the problem (LLM judgment)
- Improvement has plateaued OR strategy space is exhausted OR metric meets expectations
- Agent is confident further iteration won't meaningfully improve results

**Files to create/modify:**
- `src/review/schemas.py` — Pydantic models for the review-decision record
- `src/review/validator.py` — validates the appended JSONL record
- Update `.claude/rules/artifact-contracts.md` Contract 3 with the extended fields (`reviewer_reasoning`, `router_reasoning`, `risk_flags_summary`, `best_iteration`)

---

### M7.2 — Reviewer with prior-run comparison logic

**Deliverables:**
- [ ] `src/review/history.py` — utility to load and summarize run-history.jsonl for the agent
- [ ] `src/review/comparator.py` — computes deltas, trends, best-so-far tracking across iterations
- [ ] Agent instructions for prior-run comparison

**Comparison signals computed:**
- Delta vs previous iteration (from `model-report.json.prior_run_comparison`)
- Delta vs best iteration so far (tracked in run-history)
- Improvement trend (improving / plateau / degrading) over last N iterations
- Consecutive stale iterations count (from M6's `plateau_signal`)

**Iteration-1 behavior:**
- No prior-run comparison (fields null)
- Full rubric still applies: check risk flags, judge absolute metric quality
- Verdict is based on: "Is this a reasonable baseline? Any red flags?"

---

### M7.3 — Router schema and allowed actions

**Deliverables:**
- [ ] Formal definition of the three routing actions with trigger conditions
- [ ] Agent instructions for routing logic

**Action definitions:**

| Action | Meaning | When to choose |
|--------|---------|----------------|
| `continue` | Keep current direction, add next step | Metric is improving, current approach has untapped potential (e.g., hyperparameter tuning, more features in same family) |
| `rollback` | Last direction was wrong, revert to prior best | Current iteration degraded metrics vs prior best AND the degradation is due to the approach (not a bug). Tell Planner to base next plan on iteration N (the best one). |
| `pivot` | Change strategy entirely | Plateau detected, or natural progression (e.g., baseline done → try tree-based models), or current technique class exhausted |

**Routing decision includes:**
- `router_decision`: one of `continue`, `rollback`, `pivot`
- `router_reasoning`: 1-3 sentences explaining why
- `best_iteration`: int tracking which iteration had the best primary metric so far

---

### M7.4 — Router decision logic and stop criteria

**Deliverables:**
- [ ] `reviewer-router` agent file in `.claude/agents/`
- [ ] `src/review/writer.py` — validates and appends review-decision record to run-history.jsonl
- [ ] Integration test with Titanic iteration-1 artifacts

**Stop criteria (verdict = `sufficient`):**
1. Max iterations reached → forced stop regardless of metric
2. Agent holistically judges the model is good enough (rubric dimensions satisfied)
3. Plateau detected AND multiple strategy classes already tried

**Decision flow:**
```
1. Read model-report.json
2. Read run-history.jsonl (if exists)
3. Load max_iterations (default 10 from src/review/config.py)
4. Check hard stops:
   a. iteration >= max_iterations → sufficient (forced stop, note in reasoning)
   b. high-severity leakage flag → insufficient (must fix, route=rollback)
5. Apply rubric holistically:
   a. Evaluate 5 rubric dimensions
   b. Determine verdict: sufficient / insufficient
6. If insufficient, determine route:
   a. Metric degraded vs prior best? → rollback
   b. Plateau detected or strategy class exhausted? → pivot
   c. Otherwise → continue
7. Call src/review/writer.py to validate and append record to run-history.jsonl
```

---

### M7.5 — Plateau detection integration

**Deliverables:**
- [ ] Verify M6's `plateau_signal` field is reliably populated
- [ ] `src/review/plateau.py` — thin helper that reads plateau_signal + run-history to confirm plateau
- [ ] Agent instructions reference plateau signal in pivot decision

**Plateau definition (simple, for now):**
- M6 already computes `plateau_signal.detected` and `consecutive_stale_iterations`
- Stale = delta on primary metric < 0.005 (absolute) between consecutive iterations
- Plateau = `consecutive_stale_iterations >= 3`
- Router treats plateau + multiple strategies tried → `sufficient` (stop)
- Router treats plateau + only one strategy tried → `pivot`

---

### M7.6 — Planner agent update for routing signals

**Deliverables:**
- [ ] Update `.claude/agents/planner.md` to read `reviewer_verdict`, `router_decision`, `router_reasoning`, and `best_iteration` from run-history.jsonl
- [ ] Add Planner instructions for each route:
  - `continue` → iterate in the same direction (more features, tune hyperparameters, etc.)
  - `rollback` → base the next plan on `best_iteration`'s approach, try a different variation
  - `pivot` → change model family or strategy class entirely
- [ ] Planner must reference `router_reasoning` when justifying its next plan

**Current Planner gap:**
- Already reads `run-history.jsonl` on iteration > 1
- Already avoids repeating failed approaches
- Does NOT read `reviewer_verdict`, `router_decision`, or `best_iteration`
- Without this update, the router's signals are unused

---

## File Creation Summary

| File | Purpose | Milestone |
|------|---------|-----------|
| `.claude/agents/reviewer-router.md` | Combined agent instructions | M7.4 |
| `src/review/__init__.py` | Package init | M7.1 |
| `src/review/schemas.py` | Pydantic models for review-decision record | M7.1 |
| `src/review/validator.py` | Validates appended JSONL record | M7.1 |
| `src/review/history.py` | Load and summarize run-history.jsonl | M7.2 |
| `src/review/comparator.py` | Deltas, trends, best-so-far tracking | M7.2 |
| `src/review/plateau.py` | Plateau detection helper | M7.5 |
| `src/review/config.py` | Max iterations default (hardcoded 10) | M7.1 |
| `src/review/writer.py` | Validate and append review-decision to run-history.jsonl | M7.4 |
| `tests/test_review.py` | Unit tests for src/review/ | M7.1–M7.5 |
| `tests/test_integration.py` | Integration test with Titanic artifacts | M7.4 |

**Files to modify:**
- `.claude/rules/artifact-contracts.md` — extend Contract 3 with new fields (M7.1)
- `.claude/agents/planner.md` — add routing signal awareness (M7.6)

---

## Execution Order

```
M7.1  Schema + rubric + config
  │
  ├──► M7.2  Prior-run comparison logic
  │
  ├──► M7.3  Router action definitions
  │
  └──► M7.5  Plateau detection helper
         │
         └──► M7.4  Agent + writer + decision logic + integration test
                │
                └──► M7.6  Planner agent update (must follow M7.4 so routing schema is final)
```

M7.1 first → M7.2 + M7.3 + M7.5 in parallel → M7.4 ties it together → M7.6 wires Planner.

---

## Resolved Questions

| Question | Decision |
|----------|----------|
| Where does `max_iterations` live? | Hardcoded default 10 in `src/review/config.py`. Future orchestrator skill will make this configurable per project. |
| Agent writes JSONL directly or via utility? | Via `src/review/writer.py` — consistent with M2–M6 pattern, validates before append. |
| Update Planner now or later? | Now (M7.6). Without it, routing signals are dead letter. |
