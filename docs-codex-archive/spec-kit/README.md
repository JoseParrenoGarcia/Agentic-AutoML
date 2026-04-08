# Spec Kit

This spec kit translates the PRD into build-ready documentation. The goal is to make implementation decisions explicit before the repo fills up with agents, prompts, scripts, and conventions that are hard to unwind later.

## Document Map

- `docs/prd.md`
  The product-level intent, scope, and success criteria.
- `docs/spec-kit/system-architecture.md`
  The end-to-end workflow, component boundaries, artifacts, and proposed repository layout.
- `docs/spec-kit/agent-and-skill-specs.md`
  Detailed contracts for the first wave of agents, deterministic components, and supporting skills.
- `docs/spec-kit/milestones.md`
  Major milestones and PR-sized delivery slices.
- `docs/spec-kit/open-questions.md`
  The key strategic decisions that still need agreement.
- `docs/spec-kit/reference-inventory.md`
  Seed references from the rough notes and how they should be mined for reusable guidance.

## How To Use This Spec Kit

1. Read the PRD to understand the product boundary.
2. Use the system architecture doc to agree on the operating model and file contracts.
3. Use the agent and skill spec doc to decide what gets built first and how components communicate.
4. Use the milestones doc to sequence delivery into reviewable PRs.
5. Keep the open questions doc alive as we refine strategy.

## Spec Kit Principles

- Prefer explicit contracts over implied behavior.
- Keep artifacts machine-readable and human-auditable.
- Make every phase independently testable.
- Build for small iterations, not giant heroic runs.
- Delay optional complexity until the baseline loop is stable.
