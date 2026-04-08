# Milestones

## Planning Principles

- Major milestones should unlock new product capability.
- Small milestones should fit comfortably in a reviewable PR.
- Every milestone should leave the repo in a runnable and testable state.
- We should prefer one strong vertical slice over many half-built subsystems.

## Major Milestone 1: Documentation And Contracts Foundation

Outcome:

- the project has a clear PRD, implementation spec, artifact contracts, and repository structure

PR-sized slices:

1. Add PRD and spec-kit docs.
2. Define artifact templates for project metadata, plans, reviews, and memory logs.
3. Add repository folders and placeholder READMEs for agents, skills, rules, hooks, templates, and references.
4. Add coding and artifact rules docs.

## Major Milestone 2: Single-Project Runtime Skeleton

Outcome:

- a project can be initialized and run through an empty but structured loop

PR-sized slices:

1. Create project bootstrap command and directory template.
2. Add run-iteration skeleton and state loader.
3. Add base config files and environment conventions.
4. Add smoke tests for project creation and iteration folder creation.

## Major Milestone 3: Deterministic Dataset Understanding

Outcome:

- the system can generate a credible dataset profile and EDA summary for a supplied dataset

PR-sized slices:

1. Implement schema and basic stats profiler.
2. Add null, cardinality, outlier, and correlation analysis.
3. Generate `profile.json` and `profile.md`.
4. Add plot generation as optional human-facing output.
5. Add test fixtures for common data-quality cases.

## Major Milestone 4: Planning Layer

Outcome:

- the system can turn project context into a structured first experiment plan

PR-sized slices:

1. Define plan schema and validation.
2. Build initial planner prompt and templates.
3. Add project-memory summary input.
4. Add fixture-based planner tests using stable dataset profiles.

## Major Milestone 5: Plan-To-Code Layer

Outcome:

- the system can translate a plan into executable Python code using templates

PR-sized slices:

1. Define baseline experiment code template.
2. Implement feature step rendering into code.
3. Implement model step rendering into code.
4. Add code-generation smoke tests.

## Major Milestone 6: Execution And Debugging Loop

Outcome:

- the generated experiment can run, fail safely, and attempt bounded self-repair

PR-sized slices:

1. Add run executor and log capture.
2. Add runtime manifest generation.
3. Add bounded debugger loop with retry classification.
4. Add tests for syntax, dependency, and runtime failures.

## Major Milestone 7: Evaluation And Model Reporting

Outcome:

- every run ends with a deep evaluation package, not just one score

PR-sized slices:

1. Add base metric calculators by problem type.
2. Add actual-vs-predicted and segment analysis.
3. Add feature-importance or explainability hooks where supported.
4. Emit `model-report.json` and `model-report.md`.

## Major Milestone 8: Reviewer And Action Router

Outcome:

- the system can judge outcomes and choose the next step in the loop

PR-sized slices:

1. Define reviewer schema and decision rubric.
2. Implement reviewer with prior-run comparison.
3. Define router schema and allowed actions.
4. Implement router decisions and stop criteria.
5. Add local-maxima detection helper.

## Major Milestone 9: Project Memory

Outcome:

- the system learns from prior iterations and avoids repeating weak ideas blindly

PR-sized slices:

1. Add append-only run history.
2. Add decision-log summaries.
3. Add retrieval helper for planner and reviewer inputs.
4. Add tests for historical consistency and retrieval quality.

## Major Milestone 10: Knowledge Base And Reference Ingestion

Outcome:

- reusable tactics and research findings can inform multiple projects

PR-sized slices:

1. Create knowledge-base taxonomy and wiki template.
2. Add reference inventory and source-ingestion conventions.
3. Add a wiki-scribe flow for approved insights.
4. Add retrieval hooks from planner and reviewer.

## Major Milestone 11: Benchmark Validation

Outcome:

- the product proves value on a small set of benchmark projects

PR-sized slices:

1. Select benchmark datasets and target metrics.
2. Run end-to-end baseline experiments.
3. Compare against simple hand-built baselines.
4. Write benchmark retrospectives and gaps.

## Major Milestone 12: Expansion Tracks

Candidate outcomes:

- better research agent
- optional open-source model routing
- richer time-series support
- Obsidian-compatible knowledge graph export
- packaging for internal team adoption

## Recommended Immediate Next Steps

If we want the fastest path to momentum, the next three concrete PRs should be:

1. Documentation and contracts foundation.
2. Project bootstrap plus runtime skeleton.
3. Deterministic dataset-understanding pipeline.
