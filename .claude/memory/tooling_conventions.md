---
name: Tooling Conventions
description: Tooling choices and conventions established during M2.1
type: project
---

- **Package manager:** pip + requirements.txt (PRD-mandated). Revisit uv at M5 if friction.
- **Virtual environment:** Single `.venv/` at repo root. Activate with `source .venv/bin/activate`.
- **Makefile:** Not added yet. Add at M2.5 when enough common commands exist (install, test, run, lint).
- **CI:** GitHub Actions (`ci.yml`) — runs `pytest tests/ -v` on every push and PR to any branch.
- **Pre-commit hook:** `.git/hooks/pre-commit` — runs pytest locally before commit. Not committed to repo (lives in .git/). GitHub Desktop bypasses this; CI is the safety net.
- **Kaggle CLI:** Installed, credentials at `~/.kaggle/kaggle.json`. Auth via `KAGGLE_API_TOKEN` env var (new CLI format, not legacy kaggle.json username/key).
- **Current requirements.txt packages:** kaggle, pandas, numpy, pyyaml, pytest
