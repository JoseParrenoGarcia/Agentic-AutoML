# M9 Permissions — Reduce Approval Friction

## Context

During the iteration-4 smoke test, the user had to approve each Agent tool call (5 subagent launches) and some Bash commands manually. The orchestrator is designed to be fully autonomous, so this friction defeats its purpose.

## Root Cause

- `Bash(.venv/bin/python3 -c ":*)` is already allowlisted (line 39 of settings.local.json) — gate checks should auto-approve
- `Bash(wc -l:*)` and `Bash(test -f:*)` patterns are NOT allowlisted — these are used in gate checks
- **The Agent tool is NOT allowlisted at all** — this is the main source of friction (5 approval prompts per iteration)

## Change

Add these patterns to `.claude/settings.local.json` `permissions.allow`:

```
"Agent"
"Bash(wc -l:*)"
```

- `Agent` — allows all subagent launches without approval. Scoped to this project only (settings.local.json).
- `Bash(wc -l:*)` — the reviewer gate check counts lines before/after.
- `Bash(test -f:*)` is probably not needed since the gate checks already use python3 -c, but can add for safety.

## What NOT to add

- No need to add specific agent names (e.g., `Agent(planner)`) — the generic `Agent` pattern covers all.
- No need to add more Bash patterns — the python3 -c wildcard already covers the validators.

## Files to modify

1. `.claude/settings.local.json` — add `Agent` and `Bash(wc -l:*)` to permissions.allow

## Verification

- Re-run the orchestrator on Titanic (iteration 5) and confirm zero manual approvals needed
