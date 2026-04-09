---
name: Authoring Guardrail
description: Ensures Claude applies the correct authoring skill when creating or significantly restructuring primitives (agents, skills, hooks, rules) in this project.
---

# Authoring Guardrail

When creating or significantly restructuring any of the following primitives in this project, load and apply the corresponding authoring skill **before** producing output. Do not author these from memory or general knowledge alone.

| Task | Authoring Skill |
|------|----------------|
| Creating or restructuring an agent file in `agents/` | `.claude/skills/create-agent/SKILL.md` |
| Creating or restructuring a skill in `.claude/skills/` | `.claude/skills/create-skill/SKILL.md` |
| Adding or modifying a hook in `.claude/settings.json` | `.claude/skills/create-hook/SKILL.md` |
| Creating or restructuring a rule in `.claude/rules/` | `.claude/skills/create-rule/SKILL.md` |

## What "significantly restructuring" means

Load the skill when:
- Creating a new file from scratch
- Rewriting a description, frontmatter, or tool restrictions
- Adding/removing path scoping to a rule
- Redesigning a hook's event type, timeout, or command logic

Do NOT load the skill for:
- Fixing a typo
- Updating a version number
- Correcting a link or reference path

## Why this rule exists

The authoring skills capture the DOs, DON'Ts, anti-patterns, and structural requirements for each primitive type. Skipping them risks:
- Vague agent/skill descriptions that never trigger correctly
- Over-granted tool permissions on agents
- Hooks with no logging or silent failure modes
- Rules too large, too vague, or path-scoped incorrectly
