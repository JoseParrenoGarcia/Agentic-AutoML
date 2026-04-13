# M4 Reference Research: Plan-to-Code + Debugger Patterns

Research date: 2026-04-13
Sources: Karpathy autoresearch, AutoKaggle, DS-STAR (arXiv 2509.21825)

---

## 1. Karpathy autoresearch

**Architecture:** Single agent, single file (`train.py`), tight edit-train-evaluate loop.

### Plan-to-Code
- No separate "plan" artifact. The agent thinks, edits `train.py` directly, then runs.
- Code generation is inline: the agent "hacks" `train.py` with an experimental idea.
- Instructions live in `program.md` (human-authored, agent-consumed). The agent never modifies it.
- Constraint: only one file is editable; `prepare.py` (data/eval) is read-only.

### Debugging / Retries
- **No formal debugger agent.** The loop handles failures pragmatically:
  1. Run: `uv run train.py > run.log 2>&1`
  2. Grep for metrics: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
  3. If grep empty → crash. Run `tail -n 50 run.log` to read the stack trace.
  4. Attempt a fix. If it doesn't work after "more than a few attempts," give up.
- **Revert on failure or regression:** `git reset` back to the last known-good commit.
- No retry counter, no error classification. Very lightweight.

### Metrics / Logging
- Append to `results.tsv` with 5 columns: `commit_hash | val_bpb | peak_vram_gb | status (keep/discard/crash) | description`.
- `results.tsv` is untracked by git (ephemeral log, not an artifact).
- Uses 0.000000 for metric and 0.0 for VRAM on crashes.
- Git commit history IS the experiment history (keep = advance branch, discard = reset).

### Artifacts Produced
- Modified `train.py` (the experiment itself)
- `run.log` (stdout/stderr capture)
- `results.tsv` (append-only experiment log)
- Git branch `autoresearch/<tag>` with commit-per-experiment

### Key Takeaways for M4
- **Simplicity criterion**: "All else being equal, simpler is better." A small improvement that adds ugly complexity is not worth it.
- **Git as state machine**: keep = commit stays; discard = `git reset`. Clean rollback with zero custom infra.
- **Redirect all output**: `> run.log 2>&1` — never let output flood context.
- **Fixed evaluation harness**: Separating the eval function from the trainable code prevents the agent from gaming metrics.

---

## 2. AutoKaggle

**Architecture:** Multi-agent (3 specialists + 1 orchestrator). Persistent agents via `SendMessage`. Orchestrator reads only `state.json` + `results.tsv` to stay lean.

### Plan-to-Code
- **Explicit plan phase:** Builder writes a plan → Reviewer approves/revises → Builder codes.
- Flow: `Research → Builder(plan) → Reviewer → [Builder(revise)] → Builder(code+submit) → Reviewer(verify)`
- Plan and code are separate steps (step 6 = plan, step 9 = code). Reviewer gates between them.
- Builder produces per-round artifacts: `R{NN}_plan.md`, `R{NN}_code.py`, `R{NN}_submission.csv`.

### Debugging / Retries
- **No dedicated debugger agent.** The Builder handles code execution.
- If code returns `CV_SCORE=crash`: log crash, skip submission, proceed to logging (Step 11).
- **Crash recovery is state-based**, not retry-based:
  - `state.json` tracks current round, agent IDs, best score.
  - Each phase writes a marker file (`R{NN}_verified`, `R{NN}_plan.md`, etc.).
  - On restart, resume check (`[ -s file ]`) skips completed phases.
  - Atomic writes: write `.tmp` then `mv` — safe against mid-write crashes.
- **Agent re-spawn on failure:** If `SendMessage` gets no response → set agent ID to null → re-spawn.
- **Periodic re-spawn:** Every 15 rounds, re-spawn agents to prevent context overflow.

### Metrics / Logging
- `results.tsv`: append-only, one row per round with CV score, LB score, status.
- `experiments.json`: structured experiment history (richer than TSV).
- `state.json`: mutable state (current round, best scores, agent IDs).
- Submission validation: CV_SCORE must be valid number, better than `best_cv`, within daily submission limit.

### Artifacts Produced
- Per-round: `R{NN}_research.md`, `R{NN}_plan.md`, `R{NN}_code.py`, `R{NN}_submission.csv`, `R{NN}_verified`
- Persistent: `state.json`, `results.tsv`, `experiments.json`, `LEARNINGS.md`
- Git branch with commits per round, plus a PR with updated learnings at end.

### Key Takeaways for M4
- **Reviewer gates code execution.** Plan is reviewed BEFORE any code runs. Catches bad ideas early.
- **File-based communication:** Agents return one-line signals (`DONE`, `APPROVED`, `REVISE: reason`, `CV_SCORE=X`). Never file contents.
- **Marker files for idempotent resume.** Each phase writes a file; on restart, check `[ -s file ]` to skip.
- **Periodic agent re-spawn** prevents context overflow in long-running loops.
- **LEARNINGS.md as cross-run memory.** Accumulated lessons feed into future rounds (e.g., "Ridge stacking on correlated models hurts LB").
- **Fold-1 kill gate:** Run fold-1 before committing to full K-fold. Saves compute on bad ideas.

---

## 3. DS-STAR (arXiv 2509.21825)

**Architecture:** Specialized data science agent for heterogeneous formats and open-ended queries. Google Research, state-of-the-art on DABStep, KramaBench, DA-Code benchmarks.

### Plan-to-Code
- Designed to (1) process and integrate data across diverse heterogeneous formats, and (2) generate comprehensive research reports for open-ended queries.
- Multi-file processing capability — handles real-world workflows requiring exploration of multiple data sources.
- Outperforms baselines especially on "hard-level QA tasks requiring multi-file processing."

### Debugging / Retries (DS-STAR inspired, as adopted in our PRD)
- **Two-stage debugging** (already captured in PRD Section 6.4):
  - **Stage 1 — Syntax and Import Repair:** Parse errors, missing imports, type mismatches. Fast, deterministic fixes. Max 3 retries.
  - **Stage 2 — Logic and Runtime Repair:** Data shape errors, NaN propagation, convergence failures. LLM-assisted diagnosis. Max 2 retries.
- Separate retry counters per error class.
- Total retry cap: 5 attempts per iteration.
- On exhaustion: structured failure artifact + escalate.

### Metrics / Logging
- Generates "high-quality data science reports" as output (preferred over baselines in 88%+ of cases).
- Evaluation across standardized benchmarks with structured scoring.

### Artifacts Produced
- Research reports (for open-ended queries).
- Structured QA answers (for closed-form tasks).
- Processing logs across heterogeneous file formats.

### Key Takeaways for M4
- **Error classification is critical.** Stage 1 (syntax) vs Stage 2 (logic) require fundamentally different repair strategies. Don't treat all errors the same.
- **Bounded retries with separate counters.** Prevents infinite loops while giving each error class fair attempts.
- **Rollback on regression.** If a repair breaks previously-passing code, revert to last known-good state.

---

## Synthesis: Recommendations for M4 Coder + Debugger

### From autoresearch (simplicity)
1. **Git as rollback mechanism.** Commit before each code edit; `git reset` on failure/regression. Zero custom infra.
2. **Redirect all output to log file.** Never let execution output flood agent context.
3. **Fixed evaluation harness.** Separate metric computation from generated code to prevent gaming.
4. **Simplicity criterion for generated code.** Penalize complexity that doesn't improve metrics.

### From AutoKaggle (multi-agent coordination)
5. **Review-before-execute gate.** Validate the plan/code before spending compute on execution.
6. **File-based phase markers for idempotent resume.** Each step writes a marker; restart skips completed steps.
7. **One-line return signals between agents.** Keep inter-agent communication minimal and parseable.
8. **Atomic file writes.** Write `.tmp` then `mv` to prevent corruption on crashes.
9. **Accumulated learnings file.** Cross-iteration memory that feeds future planning.
10. **Fold-1 kill gate.** Fast validation before committing to full training.

### From DS-STAR (debugging rigor)
11. **Two-stage error classification.** Syntax/import (deterministic, fast) vs logic/runtime (LLM-assisted, slower).
12. **Separate retry counters per error class.** Don't burn all retries on one type.
13. **Structured failure artifacts.** On exhaustion, emit a machine-readable failure record for the planner.
14. **Rollback to last known-good.** If repair introduces new failures, revert rather than compound.

### Proposed M4 Artifact Flow
```
iteration-<n>.yaml (from Planner)
    ↓
Coder Agent reads plan + prior code + profile.json
    ↓
runs/iteration-<n>/src/*.py (generated code)
    ↓
[Review gate — optional at M4, required by M6]
    ↓
Executor runs code → run.log, manifest.json
    ↓
On failure → Debugger (Stage 1 or 2, bounded retries)
    ↓         → retry-log.jsonl tracks each attempt
    ↓         → git reset on regression
    ↓
On success → learning_curves.json, predictions, metrics
    ↓
run-history.jsonl append (structured record)
```
