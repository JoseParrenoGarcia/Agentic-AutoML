# Autoresearch Analysis — Planning Prompts & Design Patterns

> Research date: 10 April 2026
> Source: https://github.com/karpathy/autoresearch (70k stars, master branch)

---

## 1. Architecture Summary

**Autoresearch has no explicit planning layer.** This is the most important finding. The entire system is a single Markdown file (`program.md`) that serves as the agent's instruction set — what Claude Code calls a "skill". There are no Python prompt templates, no system messages, no structured plan objects, no planner LLM calls, and no separate orchestration code. The LLM agent (Claude, Codex, etc.) reads `program.md`, reads the codebase, and does all planning implicitly in its reasoning.

The architecture is deliberately minimal: three files (`prepare.py`, `train.py`, `program.md`), one metric (`val_bpb`), one editable file (`train.py`), and an infinite loop. The agent modifies code, runs a 5-minute training experiment, checks the result, keeps or discards via git, logs to `results.tsv`, and repeats. There is no separate "planning phase" vs "execution phase" — every iteration is plan-execute-evaluate in one shot. The agent generates its next experiment idea based on accumulated context (prior results in `results.tsv`, current code state, git history).

The `.gitignore` reveals multi-agent infrastructure exists in practice (`CLAUDE.md`, `AGENTS.md`, `queue/`, `results/` directories are gitignored), suggesting session-specific launcher configs are generated but aren't part of the core repo. The human's role is exclusively to edit `program.md` — the "research org code" — and let the agent run.

---

## 2. The Complete Agent Instruction Prompt (`program.md`)

The entire planning/iteration system is this single document. Quoted in full, section by section:

### 2.1 Preamble

```
# autoresearch

This is an experiment to have the LLM do its own research.
```

### 2.2 Setup Phase

```markdown
## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer.
   If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be
   recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.
```

### 2.3 Experimentation Constraints

```markdown
## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes**
(wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture,
  optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer,
  and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry
about training time — it's always 5 minutes. Everything is fair game: change the architecture, the
optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code
runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should
not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly
complexity is not worth it. Conversely, removing something and getting equal or better results is a
great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the
complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of
hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep.
An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run
the training script as is.
```

### 2.4 Output Format Specification

```markdown
## Output format

Once the script finishes it prints a summary like this:

---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8

Note that the script is configured to always stop after 5 minutes, so depending on the computing
platform of this computer the numbers might look different. You can extract the key metric from the
log file:

grep "^val_bpb:" run.log
```

### 2.5 Logging/Memory System

```markdown
## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas
break in descriptions).

The TSV has a header row and 5 columns:

commit	val_bpb	memory_gb	status	description

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

### 2.6 The Experiment Loop (Core Iteration Prompt)

```markdown
## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or
   let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack
   trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep.
If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like
you're getting stuck in some way, you can rewind but you should probably do this very very sparingly
(if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval
overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and
easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally
broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the
human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?".
The human might be asleep, or gone from a computer and expects you to continue working *indefinitely*
until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers
referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses,
try more radical architectural changes. The loop runs until the human interrupts you, period.
```

---

## 3. Concepts Their System Captures That Our Planner Might NOT

### Already in our planning list (confirmed present in autoresearch):
- ✅ **Memory of prior iterations** — `results.tsv` acts as the experiment memory. Git history provides code-level memory.
- ✅ **Evaluation focus / win conditions** — Single metric (`val_bpb`), lower is better. Binary keep/discard.

### NOT in our planning list but present or implied in autoresearch:

- **Simplicity criterion / complexity cost-benefit** — Autoresearch explicitly weights improvement magnitude against code complexity. "A 0.001 improvement that adds 20 lines of hacky code? Not worth it." This is a decision heuristic our planner lacks. For tabular ML: should we add a complex stacking ensemble for +0.001 AUC? The simplicity criterion would say no.

- **Baseline-first protocol** — The system mandates establishing a baseline before any experimentation. "Your very first run should always be to establish the baseline." Our planner should probably enforce this too.

- **Crash recovery / error classification** — Explicit guidance on distinguishing fixable bugs (typos, imports) from fundamentally broken ideas (OOM = bad idea). Our planner has no crash triage logic.

- **Git-as-rollback mechanism** — Experiments are atomic: commit before running, `git reset` on failure. This gives perfect reproducibility and rollback at zero cost. For our tabular system, we could version experiment artifacts similarly.

- **VRAM/resource as soft constraint** — Memory usage is tracked but not a hard gate. "Some increase is acceptable for meaningful gains." We could apply this pattern to training time, memory, or inference latency.

- **Context window management** — `uv run train.py > run.log 2>&1` explicitly prevents output from flooding the agent's context. Reading results via `grep` extracts only what matters. This is an engineering pattern our system should adopt.

- **The "think harder" escape hatch** — When stuck: "read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes." This is an explicit instruction for the agent to vary exploration strategy when exploitation plateaus.

- **Autonomy / never-stop directive** — The system is designed for overnight autonomous runs (~100 experiments/sleep cycle). The agent must never ask permission. This is a fundamentally different operating mode than interactive planning.

- **Scope constraints as guardrails** — Only one file editable, no new dependencies, fixed evaluation harness. This prevents the agent from "cheating" or going off-rails. Our planner could benefit from explicit scope constraints (e.g., "don't modify the evaluation pipeline").

- **Fixed time budget for comparability** — Every experiment runs identically long, making results directly comparable. For tabular ML, this could mean: fixed cross-validation folds, fixed train/val split, fixed preprocessing pipeline.

---

## 4. Notable Design Patterns

### Pattern 1: "One File, One Metric, One Loop"
The entire system is reduced to the absolute minimum:
- **One editable file** (`train.py`) — scope is crystal clear
- **One metric** (`val_bpb`) — no ambiguity about what "better" means
- **One loop** (edit → run → evaluate → keep/discard → repeat)

**Implication for our planner**: We should also aim for maximal clarity on what the agent can change, what it's optimizing, and what the loop looks like. Our system is more complex (multiple preprocessing/feature/model choices), so we need to be proportionally more explicit.

### Pattern 2: No Structured Plans — LLM Reasoning IS the Planner
There are zero structured plan objects. No YAML plans, no JSON experiment configs, no decision trees. The LLM reads context (code + results history) and decides what to try next using its own reasoning. The "plan" is implicit in the LLM's chain of thought.

**Implication for our planner**: This works for autoresearch because the search space is narrow (one file, one domain). For tabular AutoML with many orthogonal dimensions (features, models, preprocessing, hyperparameters), we likely DO need structured plans to prevent the agent from losing track. But the lesson is: don't over-structure plans for simple decisions.

### Pattern 3: TSV as Append-Only Experiment Memory
`results.tsv` is a flat, append-only log with just 5 columns: commit, metric, memory, status, description. No complex database. No metadata bloat. The agent reads it to understand what's been tried.

**Implication for our planner**: Our `iterations/` directory serves a similar role but is heavier. Consider whether a simple TSV/CSV ledger + git history is sufficient, supplemented by richer artifacts only when needed.

### Pattern 4: Git as Experiment Version Control
Every experiment is a git commit. Successful experiments advance the branch. Failed experiments are `git reset` back. This gives:
- Perfect rollback
- Full diff history
- Atomic experiments
- Zero-cost "undo"

**Implication for our planner**: We should consider whether git-based experiment tracking (rather than copying files to `iterations/N/`) gives us cleaner rollback semantics.

### Pattern 5: Output Parsing via grep, Not Structured Returns
The agent reads results via `grep "^val_bpb:" run.log` rather than a structured API. This is intentionally low-tech and robust.

**Implication for our planner**: Our evaluation outputs should have grep-friendly summary lines in addition to any structured format.

### Pattern 6: Human Programs the Agents, Not the Code
The README states: "you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents." The human's job is meta-programming — designing the agent's instructions, not writing ML code.

**Implication for our system**: This IS what we're doing with `.claude/skills/` and agent files. Autoresearch validates our architectural direction.

### Pattern 7: Anti-Context-Flooding
Explicit instruction to redirect stdout/stderr to a log file and extract only what's needed. The agent never sees raw training output — only the final metrics. This prevents context window waste.

**Implication for our system**: Our profiler outputs, training logs, etc. should have concise summary modes. Never dump raw data into the agent's context.

---

## 5. What Autoresearch Does NOT Have (Gaps Relative to Our System)

| Concept | Autoresearch | Our System Needs |
|---------|-------------|-----------------|
| Structured experiment plans | None — LLM decides implicitly | Yes — tabular ML has too many orthogonal dimensions |
| Hypotheses with rationale | None — just "description" in TSV | Yes — need to know WHY something was tried |
| Feature engineering | N/A (no features) | Critical for tabular data |
| Data profiling | N/A (fixed data) | Critical — drives all decisions |
| Model selection rationale | N/A (one model class) | Yes — need to justify model choices |
| Leakage prevention | N/A (fixed eval harness) | Critical for tabular ML |
| Null imputation strategies | N/A (no nulls in token data) | Critical for real-world tabular |
| High cardinality handling | N/A | Critical |
| Outlier handling | N/A | Important |
| Ensemble / stacking | N/A (single model) | Relevant for tabular |
| Cross-validation strategy | N/A (fixed val set) | Important |
| Multi-metric optimization | No — single metric only | May need (AUC, F1, fairness...) |
| Explicit rollback conditions | Implicit (git reset on no improvement) | Should be explicit in plan |
| Parallelism / multi-agent | Hinted at (.gitignore shows `queue/`) but not in core repo | Possible future extension |

---

## 6. Summary for Our M3 Planning Layer

**Key takeaways to apply:**

1. **Start with a baseline** — Our planner should always produce a baseline experiment as step 0.
2. **Simplicity criterion** — Build a complexity-vs-improvement heuristic into plan evaluation.
3. **Crash triage** — Distinguish "bug in code" from "bad idea" in our error handling.
4. **Context management** — Never dump raw outputs into the agent. Summaries only.
5. **Append-only experiment ledger** — Keep a simple flat log alongside richer artifacts.
6. **Atomic experiments with rollback** — Each iteration should be self-contained and revertable.
7. **"Think harder" directive** — When the planner runs out of ideas, instruct it to combine near-misses, try radically different approaches, or re-examine the data profile.
8. **Scope constraints** — Explicitly tell the planner what it CAN and CANNOT modify.

**What we need that autoresearch doesn't have:**
- Structured plans (autoresearch can get away without them; we can't)
- Data-grounded hypotheses
- Feature engineering plans
- Leakage prevention rules
- Multi-metric awareness
- Explicit rollback conditions in the plan itself
