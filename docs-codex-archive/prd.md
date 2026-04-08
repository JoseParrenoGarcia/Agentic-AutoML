# Product Requirements Document

## 1. Document Overview

- Product: Agentic AutoML
- Working title: Data Science Agent
- Status: Draft v0.1
- Source material: Consolidated from rough notes in `PRD for Data Science Agent`
- Purpose: Define the product vision, scope, constraints, and success criteria for an agentic system that can iteratively build, evaluate, and improve machine learning solutions with strong human oversight.

## 2. Product Summary

Agentic AutoML is a Claude Code-centered orchestration system for end-to-end applied machine learning work. It is intended to help a data scientist move from raw or semi-structured project datasets to a well-documented, reproducible, reviewable model candidate by combining:

- deterministic Python data-analysis and evaluation scripts
- structured multi-agent planning and execution loops
- persistent project memory across iterations
- a reusable knowledge base for ML tactics and research findings
- human-readable artifacts that allow review by both LLMs and experienced data scientists

The initial product should not attempt to solve arbitrary open-world data discovery. Instead, it should operate on pre-supplied datasets for known business or competition problems and focus on building a reliable loop for understanding data, proposing experiments, generating code, executing runs, reviewing outcomes, and deciding the next step.

## 3. Problem Statement

Current machine learning work is often slowed by four recurring bottlenecks:

1. A single human data scientist has limited time to explore many plausible modeling directions.
2. Experiment tracking is frequently fragmented across notebooks, ad hoc scripts, local notes, and partially remembered decisions.
3. LLM-driven coding can be productive, but it becomes inconsistent when not grounded in deterministic tooling, clear contracts, and high-quality reference material.
4. Teams can get trapped in local optima, repeatedly refining similar approaches instead of exploring stronger alternatives.

The product aims to reduce these bottlenecks by giving the user a structured system that can work iteratively, document itself, and make explicit decisions about what to try next.

## 4. Vision

Build a practical agentic machine learning system that can:

- understand supplied datasets deeply
- generate and run targeted experiments in small increments
- learn from prior runs through memory and structured review
- surface both strong candidate models and the reasoning behind them
- become a proving ground for future use on internal company data science projects

The long-term vision is an internal AI co-worker for data scientists. The short-term proof point is strong performance and strong documentation on Kaggle-style or similarly well-scoped datasets.

## 5. Product Principles

1. Start narrow, then expand.
   Focus first on predefined datasets and supervised ML workflows instead of open-ended dataset discovery.
2. Small loops beat giant plans.
   Planning, execution, and correction should happen in short, inspectable cycles.
3. Deterministic analysis first, agentic reasoning second.
   Agents should lean on scripts and structured artifacts, not free-form guesses.
4. Every run must leave a trail.
   Decisions, code versions, metrics, and reviewer outputs must be stored in a reusable format.
5. Human review remains first class.
   The system should help a principal or senior data scientist inspect the logic, not hide it.
6. Reuse research deliberately.
   Papers, repos, and internal learnings should be indexed into a knowledge base the system can consult repeatedly.
7. Avoid context bloat.
   Prefer scripts, markdown, JSON, and YAML artifacts over notebooks in the early phases.

## 6. Target Users

### Primary user

A hands-on data scientist or ML practitioner who:

- can define the business or competition objective
- can provide one or more datasets
- wants the system to explore, document, and improve model candidates
- wants to review outputs rather than manually drive every experiment

### Secondary user

A senior reviewer or technical lead who:

- audits model quality and reasoning
- inspects tradeoffs, leakage risks, and failure modes
- wants reproducible evidence instead of opaque agent output

### Future user

An internal data science team using the system as a managed service for repeated ML workflows.

## 7. Jobs To Be Done

When I have a machine learning problem with known datasets, I want the system to understand the data, generate credible modeling strategies, execute them safely, compare results across iterations, and tell me what to do next so that I can reach a strong model faster without losing rigor.

When I revisit a project after several iterations, I want to see what was tried, what worked, what failed, and why so that neither I nor the agents repeat low-value work.

When an approach starts plateauing, I want the system to explicitly challenge the current direction so that we do not get stuck polishing a local maximum.

## 8. In Scope For Initial Versions

- Claude Code-based orchestration using agents, skills, hooks, and rules
- project-scoped workflows operating on user-supplied datasets
- deterministic dataset profiling and reporting
- structured experiment planning for feature engineering and modeling
- code generation into Python scripts
- execution, debugging, and retry loops
- rich evaluation and model audit reporting
- reviewer and action-routing steps for iterative improvement
- persistent project memory and run logs
- a knowledge base or wiki of reusable ML tactics and research notes
- milestone-driven development with small PR-sized units of work

## 9. Out Of Scope For Initial Versions

- autonomous crawling of the open internet for arbitrary datasets
- notebook-first workflows
- full MLOps deployment and production serving
- multi-tenant enterprise features
- highly unstructured multimodal ML use cases
- guaranteed support for every ML paradigm in v1
- a polished graphical UI

## 10. Assumptions

1. Projects begin with one or more known datasets already available locally or via a controlled source.
2. The first meaningful proof point is on Kaggle-like tabular or time-series problems rather than every possible ML modality.
3. Claude Code remains the primary harness even if some tasks later route to external or open-source models.
4. Human users are willing to review structured markdown, JSON, YAML, plots, and run summaries.
5. Deterministic scripts can capture enough of the data-understanding and evaluation workload to ground agent reasoning.

## 11. Core User Workflow

1. User creates or selects a project and provides datasets plus the target objective.
2. Dataset analysis components generate a detailed profile, dataset summary, and candidate target/feature observations.
3. Planner creates an initial experiment plan informed by the dataset profile, project memory, and relevant knowledge-base entries.
4. Code generation converts the current plan into Python scripts and configuration artifacts.
5. Executor runs the code, captures logs, metrics, plots, artifacts, and failures.
6. Debugger repairs execution issues where possible and reruns safely.
7. Model report builder produces a deep evaluation package.
8. Reviewer judges performance changes, failure modes, leakage risk, segment behavior, and signs of plateauing.
9. Action router decides whether to stop, refine, replace a step, add a step, or explore a more divergent strategy.
10. Project memory is updated and the loop continues until stop criteria are met or the human intervenes.

## 12. Functional Requirements

### 12.1 Project setup and orchestration

- The system must support project-scoped execution with isolated inputs, outputs, memory, and artifacts.
- The system must define consistent conventions for agents, skills, rules, hooks, prompts, and generated files.
- The system must support stepwise iteration rather than one-shot end-to-end generation.

### 12.2 Dataset understanding

- The system must generate a deterministic dataset profile including:
  - row and column counts
  - type inference
  - null analysis
  - numerical summaries
  - categorical summaries, including high-cardinality indicators
  - outlier indicators
  - correlation summaries
  - candidate target-variable notes
  - possible leakage or identifier columns
- The system should attempt dataset semantic descriptions at both dataset and column level.
- The system should support user clarification when column meaning is ambiguous.
- The system should generate an EDA report optimized for both LLM consumption and human review.
- The system should support plots as human-facing supplements while ensuring text summaries carry the core meaning.

### 12.3 Feature engineering planning

- The system must propose feature-engineering actions grounded in the dataset profile and project objective.
- The system should reference reusable knowledge-base patterns for scenarios such as:
  - high-cardinality categoricals
  - missing data
  - skewed features
  - outliers
  - class imbalance
  - time-series leakage risks
  - non-normal targets

### 12.4 Model planning and building

- The system must propose model candidates suited to the problem type.
- The system must support structured hyperparameter-search strategies for at least common classical ML approaches in early versions.
- The system should preserve a template-based code structure instead of generating entirely ad hoc scripts each time.
- The system should be able to save trained-model artifacts and run metadata in a standard layout.

### 12.5 Execution and debugging

- The system must execute generated code and capture stdout, stderr, failures, and runtime metadata.
- The system must attempt bounded self-repair for code failures.
- The system must not retry indefinitely; retry policies should be explicit and capped.
- The system should log learning curves and intermediate outputs where relevant.

### 12.6 Model evaluation and reporting

- The system must compute problem-appropriate optimization metrics.
- The system must generate richer evaluation artifacts beyond a single score, including where applicable:
  - actual vs predicted comparisons
  - segment performance
  - feature importance
  - error distribution summaries
  - leakage indicators
  - calibration or drift-like warnings within the project context
- The system should produce a report detailed enough for an experienced data scientist to critique and guide the next step.

### 12.7 Iterative review and routing

- The system must compare new runs to previous runs.
- The system must explain likely causes of improvement or degradation.
- The system must identify diminishing returns and trigger a local-maxima challenge when improvement stalls.
- The system must produce a machine-readable review artifact to feed the next planning cycle.
- The system must support routing decisions such as:
  - stop and finalize
  - replace a step
  - add a step
  - remove a step
  - branch into a more exploratory strategy

### 12.8 Project memory

- The system must maintain a project memory log describing:
  - what was attempted
  - what changed
  - which metrics moved
  - what conclusions were drawn
- The memory should be easy for both agents and humans to query.
- The memory must survive across iterations and project sessions.

### 12.9 Knowledge base and references

- The system must include a structured knowledge base or wiki for reusable ML tactics, findings, and research summaries.
- The system should enforce a standard template or taxonomy for wiki entries.
- The system should capture insights from papers, repos, and project-specific discoveries.
- The system should include a reference index that explains why each source matters and what to extract from it.

### 12.10 Human oversight

- The user must be able to inspect plans, reports, and final recommendations.
- The system should surface uncertainty and unresolved assumptions explicitly.
- The system should support user intervention when semantic clarification or strategic direction is needed.

## 13. Non-Functional Requirements

### 13.1 Reproducibility

- Runs must be reproducible from tracked code, configuration, data references, and environment metadata.

### 13.2 Traceability

- Every major decision must point back to the evidence that informed it.

### 13.3 Modularity

- Agents and skills should have narrow responsibilities and stable interfaces.

### 13.4 Extensibility

- The architecture should allow future routing to open-source models or other providers where sensible.

### 13.5 Cost and context discipline

- The system should minimize unnecessary token usage by relying on structured artifacts and deterministic scripts.

### 13.6 Safety and bounded autonomy

- The system should avoid destructive or unsupported actions without clear guardrails.
- The system should separate exploration from finalization decisions.

## 14. Success Metrics

### Product metrics

- Time from project setup to first credible baseline model
- Number of completed experiment iterations per day with acceptable artifact quality
- Percentage of runs that leave complete traceable artifacts
- Percentage of repeated mistakes avoided due to project memory
- Reviewer satisfaction with clarity of reports and recommendations

### Modeling metrics

- Improvement over a simple baseline on benchmark datasets
- Ability to produce multiple materially distinct experiment branches
- Reduction in failed or non-interpretable experiment runs

### Adoption metrics

- Number of benchmark problems completed end-to-end
- Number of reusable wiki entries and patterns successfully referenced in later projects

## 15. Risks And Failure Modes

1. Over-automation without grounding.
   Agents may generate plausible but weak ML strategies if deterministic analysis and strong templates are missing.
2. Context overload.
   Too much raw text, notebook content, or unstructured research can degrade agent performance.
3. Local-optimum behavior.
   The system may keep refining the same family of ideas unless deliberate divergence is built in.
4. Poor artifact contracts.
   If intermediate outputs are inconsistent, downstream agents will fail or hallucinate.
5. Hidden data leakage.
   Strong metrics may mask invalid modeling choices unless leakage checks are explicit.
6. Tooling sprawl.
   Too many agents or skills too early may create complexity before the core loop is reliable.
7. Premature generalization.
   Supporting every modality or deployment target early may delay useful progress.

## 16. Release Strategy

### Phase 1

Establish a trustworthy single-project loop for dataset understanding, planning, code generation, execution, evaluation, and memory on a narrow class of problems.

### Phase 2

Deepen quality through better evaluation, reviewer intelligence, local-maxima detection, and a stronger knowledge base.

### Phase 3

Broaden adaptability through more model families, optional external model routing, richer research ingestion, and stronger packaging for internal use.

## 17. Open Questions To Resolve

1. Which ML problem classes should be the first hard target: tabular classification, regression, time series, or a combination?
2. What level of automatic model/package installation is acceptable per project?
3. How much human approval should be required before expensive or divergent experiment branches?
4. Should the knowledge base live inside the repo, in an Obsidian-compatible structure, or both?
5. What exact artifact formats should be canonical for plans, reviews, and memory logs?
6. What is the first benchmark dataset set we will use to prove value?
7. When and how should open-source model routing be introduced?

## 18. Initial Recommendation

For the first implementation cycle, optimize for one strong vertical slice:

- predefined project input
- tabular supervised ML
- deterministic dataset profiler
- planner
- plan-to-code
- executor/debugger
- model report
- reviewer
- action router
- project memory

This slice is narrow enough to build in stages and broad enough to validate the core thesis of the product.
