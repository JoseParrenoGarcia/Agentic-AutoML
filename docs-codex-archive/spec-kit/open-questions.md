# Open Questions

This document captures the highest-value questions to resolve as we refine the product. These are not blockers for drafting, but they will shape the implementation path.

## Product Scope

1. Which problem types are first-class in v1?
   Recommendation: start with tabular regression and classification, then add time series once the loop is stable.
2. What counts as success on the first benchmark?
   Recommendation: require both a competitive metric and high-quality artifacts, not just score.
3. Is Kaggle the only proving ground, or do we also want a generic internal-dataset simulator early?

## Orchestration Choices

4. How tightly should the system depend on Claude Code-specific primitives versus a more portable orchestration core?
5. Should open-source model routing be part of the first architecture, or left as an extension point only?
   Recommendation: design the seam now, implement later.
6. How much autonomy should agents have before user confirmation is required for expensive or risky actions?

## Data And Experiment Design

7. How should multi-table datasets be represented in the project contract?
8. Should the system infer the target variable automatically, require it from the user, or support both?
   Recommendation: support inference suggestions, but require user confirmation when ambiguity exists.
9. What is the minimum acceptable evaluation package per problem type?

## Artifact Contracts

10. Which formats should be canonical?
    Recommendation: YAML for plans and routing decisions, JSON for metric-heavy machine-readable outputs, Markdown for human-facing summaries.
11. How much history should be duplicated in markdown versus stored only in machine-readable logs?
12. Should every iteration be immutable by policy?
    Recommendation: yes, except for explicitly marked repair metadata.

## Knowledge Base

13. Should reusable knowledge and project memory live in separate trees?
    Recommendation: yes, to avoid contaminating generalized guidance with one-off experiments.
14. Do we want the knowledge base to be directly usable in Obsidian from the start?
15. Should wiki entry creation be automatic, approval-based, or hybrid?
    Recommendation: hybrid. Draft automatically, promote with approval.

## Environment And Dependencies

16. Will each project have its own environment and dependencies, or should there be a shared baseline environment?
    Recommendation: shared baseline plus project-specific extension support.
17. What is the acceptable dependency surface for early benchmarks?
18. How should model artifacts be stored and versioned locally?

## Human Workflow

19. Which moments must pause for human review in the first usable version?
20. What is the preferred format for milestone tracking: markdown roadmap, issues, or both?
21. How much narrative explanation should the system provide versus concise operator-style summaries?

## Recommendation For Refinement Sessions

The highest-leverage questions to answer together next are:

1. exact v1 problem types
2. benchmark datasets
3. artifact format decisions
4. human approval checkpoints
5. project/environment layout
