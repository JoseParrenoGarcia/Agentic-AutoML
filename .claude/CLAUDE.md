# Agentic AutoML

A Claude Code–orchestrated system for iterative ML experimentation on tabular data. The core loop: dataset analysis → structured planning → code generation → execution → model evaluation → review → memory → repeat. Produces fully auditable artifacts and experiment history.

**Primary user:** Jose
**Active milestone:** M0 — Documentation & Contracts Foundation (in progress)

---

## Reference Docs

- `docs/PRD.md` — full product requirements, agent specs (9 agents), tech stack, and milestone roadmap

---

## Repo Layout

| Directory | Purpose | Status |
|-----------|---------|--------|
| `docs/` | PRD and planning docs | Active |
| `references/` | Claude Code best-practices, external papers | Pending (M0.5) |
| `knowledge-base/` | Per-project memory, benchmarks, artifacts | Pending (M1) |
| `agents/` | Agent instruction files | Pending (M0.2) |
| `.claude/skills/` | Skill instruction files | Pending (M0.2) |
| `.claude/rules/` | Behavioural guardrails, artifact contracts, ML constraints | Pending (M0.4) |
| `.claude/hooks/` | Automation hooks | Pending (M0.3) |
| `templates/` | Artifact templates: model-report, experiment YAML, etc. | Pending (M0.2) |
| `src/` | Shared Python utilities | Pending (M1) |
| `projects/` | Per-project experiment folders and results | Pending (M1) |

---

## Required MCPs / Plugins

- `context-mode` — context window management (active now)

---

## Rules

Behavioural and maintenance rules live in `.claude/rules/` (auto-loaded each session):

- `authoring.md` — guardrail: load the correct authoring skill before creating/restructuring any agent, skill, hook, rule, or memory file
- `coding-rules.md` — Python style, error handling, audit trail guardrails (pending)
- `artifact-contracts.md` — JSON schema and structure guarantees for all outputs (pending)
- `ml-experiment-constraints.md` — data validation, reproducibility, benchmark tracking (pending)
- `maintenance.md` — what to update when significant changes happen (pending)

---

## Authoring Skills

Skills for correctly building Claude Code primitives. Auto-activate on creation/restructure intent.

| Skill | Trigger | Path |
|-------|---------|------|
| `create-agent` | "create an agent", "new agent", "add an agent to agents/" | `.claude/skills/create-agent/SKILL.md` |
| `create-skill` | "create a skill", "new SKILL.md", "add a skill to .claude/skills/" | `.claude/skills/create-skill/SKILL.md` |
| `create-hook` | "add a hook", "create a hook", "automate on tool use / task completion" | `.claude/skills/create-hook/SKILL.md` |
| `create-rule` | "create a rule", "add a rule to .claude/rules/", "write a behavioral rule" | `.claude/skills/create-rule/SKILL.md` |
