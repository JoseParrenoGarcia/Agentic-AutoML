---
name: Architecture Decisions
description: Agent-Python bridge contract, src/ layout, profile.json schema design
type: project
---

## Agent-Python Bridge

Agents orchestrate; Python executes deterministically.

Bridge is a Bash tool call from Claude agent:
```bash
.venv/bin/python -m src.analysis.profiler \
  --input projects/titanic/data/raw/train.csv \
  --output projects/titanic/artifacts/data/profile.json \
  --project projects/titanic/project.yaml
```

**Bridge contract (enforced in profiler.py):**
- Explicit interpreter path — never bare `python`
- Non-zero exit on failure
- On failure: writes `{"error": true, "message": "...", "traceback": "..."}` to output path
- Idempotent: same input → identical JSON every time
- Self-creating output directory
- All paths from CLI args — no path discovery inside module

## src/ Layout

Python code that gets executed lives in `src/`, not in `.claude/agents/`.
`.claude/agents/` is for Claude instruction files (.md) only.

```
src/
└── analysis/
    └── profiler.py    # M2.1
```

## profile.json Design

- `inferred_semantic_type` per column — agent-facing semantic layer
- `m2_sections_complete` / `m2_sections_pending` — prevents agents consuming incomplete profile
- No NaN in JSON — custom encoder, `allow_nan=False`
- SHA-256 of source — reproducibility requirement
- Partial in M2.1; extended by M2.2 and M2.3

## ydata-profiling Decision

Custom `profile.json` = agent-facing (controlled schema, LLM-sized).
ydata-profiling = human-facing HTML report (M2.4).
Both coexist — different consumers.

## Test Strategy

Tests depending on Titanic CSV use `@pytest.mark.skipif(not TITANIC_CSV.exists())`.
Tests that just need any CSV use committed fixtures in `tests/fixtures/`.
This ensures CI always has runnable tests without committing large data files.
