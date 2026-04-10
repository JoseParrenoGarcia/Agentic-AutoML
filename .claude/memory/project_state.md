---
name: Project State
description: Current milestone, hybrid roadmap, and key decisions made
type: project
---

Active milestone: M2 — Deterministic Dataset Understanding
Completed: M0.5 (authoring skills), M2.1 (schema and basic stats profiler)
Next: M2.2 — Null, cardinality, outlier, and correlation analysis

**Hybrid roadmap decision:** M0 completed lightly, M1 deferred — will be designed
retrospectively once M2 teaches what the runtime loop actually needs. M1 milestones
remain in PRD as reference.

**Why:** Building M2 first reveals real constraints before over-specifying M1.

**M2.1 deliverables shipped:**
- `src/analysis/profiler.py` — schema inference + basic stats, CLI + importable
- `projects/titanic/project.yaml` — project config stub
- `projects/titanic/data/raw/` — Titanic dataset (gitignored, downloaded via Kaggle CLI)
- `tests/analysis/test_profiler.py` — 14 tests (8 skip in CI, 6 always run)
- `tests/fixtures/` — simple_numeric.csv, mixed_types.csv
- `.github/workflows/ci.yml` — runs pytest on every push/PR
- `.git/hooks/pre-commit` — runs tests before local commits

**PRD changes made:**
- Per-project venv removed → single shared `.venv/` at repo root
- Package list removed from PRD → packages added to requirements.txt as needed
- M1 deferred inside M2 per hybrid approach
- M2.4 updated: ydata-profiling for human-facing HTML report alongside custom profile.json
