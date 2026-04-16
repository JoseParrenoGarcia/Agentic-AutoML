# M8 — Project Memory

## Context

After the iteration-2 smoke test, the memory system is partially functional but has clear gaps:

- **M8.1 (run-history.jsonl)** — already works. `src/review/writer.py` appends validated JSONL records. 2 entries exist for Titanic.
- **M8.2 (decision-log.md)** — empty. Nobody writes to it. The planner reads it but gets nothing.
- **M8.3 (retrieval helper)** — `src/review/history.py` has `load_run_history()` and `summarise_history()`. The reviewer-router already uses them. The planner does NOT — it parses JSONL manually.
- **M8.4 (tests)** — none exist for memory-specific concerns.

The smoke test showed the loop works end-to-end, but the decision-log gap means the planner lacks human-readable narrative context across iterations. The retrieval helper exists but isn't wired into the planner.

---

## Revised M8 Minor Milestones

### M8.1 ✅ — Mark as done
Already delivered in M7. `src/review/writer.py` + `append_review_decision()` + Contract 3 validation.

### M8.2 — Decision-log writer
**What:** Add `append_decision_log()` to `src/review/writer.py`. Called automatically from `append_review_decision()` so every JSONL append also writes a markdown entry.

**Format per entry:**
```markdown
## Iteration N — <model_family>
**Metric:** <primary_metric_name> = <value> (delta: <delta>)
**Verdict:** <reviewer_verdict> | **Route:** <router_decision>
**Summary:** <plan_summary>
**Reasoning:** <reviewer_reasoning (first 200 chars)>
**Risk flags:** <count> (<types>)
```

**Files to modify:**
- `src/review/writer.py` — add `append_decision_log(record, log_path)` function + wire into `append_review_decision()`
- `append_review_decision()` signature gets optional `decision_log_path` param (defaults to `memory/decision-log.md` relative to `history_path`)

**Backfill:** After implementing, run backfill script to populate iterations 1-2 from existing `run-history.jsonl`.

### M8.3 — Planner retrieval integration
**What:** Update planner agent to use `summarise_history()` instead of raw JSONL parsing. This gives the planner a structured summary (best iteration, metric trajectory, families tried, high-severity flags) without manual parsing.

**Files to modify:**
- `.claude/agents/planner.md` — Step 2 "If iteration > 1" section: add a bash snippet calling `summarise_history()` (same pattern the reviewer-router already uses in its Step 2.3)

**Why:** The planner currently reads raw JSONL and has to extract signals manually. With 5+ iterations, this becomes unwieldy. The helper provides a pre-digested summary.

### M8.4 — Tests
**What:** New test file `tests/review/test_memory.py` covering:

1. `append_decision_log()` creates file if missing
2. `append_decision_log()` appends without overwriting
3. Decision-log entry format matches expected markdown structure
4. Consistency: JSONL line count == decision-log `## Iteration` count after N appends
5. Validation failure in `append_review_decision()` does NOT write to decision-log (atomicity)
6. `summarise_history()` returns correct best_iteration across 3+ records
7. `summarise_history()` handles empty history
8. Backfill script produces entries matching live-written entries
9. Round-trip: write N records → load → summarise → verify all families and trajectory
10. Edge case: decision-log.md exists with prior content, new append preserves it

**Files to create:**
- `tests/review/test_memory.py`

### M8.5 — PRD update + CLAUDE.md update
**What:** Update PRD M8 section to mark completions and reflect refined deliverables. Update CLAUDE.md active milestone.

---

## Implementation Order

```
M8.2 (decision-log writer)
  │
  ├──► M8.3 (planner retrieval integration)
  │
  └──► M8.4 (tests)
         │
         └──► M8.5 (PRD + docs update)
               │
               └──► Iteration-3 smoke test
```

M8.2 first because M8.3 and M8.4 both depend on the writer existing. M8.3 and M8.4 can run in parallel. M8.5 after everything is verified.

---

## Critical Files

| File | Action |
|------|--------|
| `src/review/writer.py` | Add `append_decision_log()`, wire into `append_review_decision()` |
| `.claude/agents/planner.md` | Update Step 2 to use `summarise_history()` |
| `.claude/agents/reviewer-router.md` | Add `decision_log_path` to Step 8 writer call |
| `tests/review/test_memory.py` | New — 10 tests |
| `docs/PRD.md` | Update M8 section with completion markers |
| `.claude/CLAUDE.md` | Update active milestone |
| `projects/titanic/memory/decision-log.md` | Backfill iterations 1-2 |

---

## Verification

1. **Unit tests pass:** `pytest tests/review/test_memory.py -v`
2. **Backfill correct:** `decision-log.md` has 2 entries matching `run-history.jsonl`
3. **Iteration-3 smoke test:** Full 5-agent loop. After completion:
   - `run-history.jsonl` has 3 entries
   - `decision-log.md` has 3 entries (auto-written by reviewer-router)
   - Planner used `summarise_history()` output (visible in its reasoning)
   - No human intervention required
