# AutoKaggle Planning Research — Comprehensive Analysis

**Date:** 2026-04-10
**Scope:** All planning-related prompts, workflows, and design patterns from the AutoKaggle ecosystem
**Sources examined:**
- `ShaneZhong/autokaggle` — Claude Code-based autonomous Kaggle competition system with 3 persistent agents
- `multimodal-art-projection/AutoKaggle` — Academic paper repo (AAAI/arXiv) with 5 specialized agents

---

## 1. Architecture Summary

### ShaneZhong/autokaggle ("Operational AutoKaggle")

This system is a **fully autonomous, infinite-loop Kaggle competition runner** orchestrated via a `program.md` file that Claude Code reads and executes. It has **three persistent agents** (Researcher, Builder, Reviewer) spawned once and kept alive for the full competition via `SendMessage`. The orchestrator (`program.md`) is context-lean — it reads only `config.json`, `state.json`, and `experiments.json` each round.

**Planning is done by the Builder Agent** — there is no separate Planner. The Builder combines planning and coding into a single agent with two phases: PHASE 1 (plan) and PHASE 3 (code). The Reviewer Agent challenges the plan before any code runs and independently verifies output afterward. The Researcher Agent provides research briefs (domain research, competition intelligence, model-specific research) that feed into the Builder's planning.

The loop runs **indefinitely** (research → plan → review → [revise] → code → submit → log → repeat) until the deadline or human interruption. Every 10 rounds, the Reviewer runs a full campaign retrospective. Agents are re-spawned every 15 rounds to prevent context overflow.

### multimodal-art-projection/AutoKaggle ("Academic AutoKaggle")

This system is a **6-phase pipeline run** (not an infinite loop). Five agents (Reader, Planner, Developer, Reviewer, Summarizer) collaborate through strictly sequenced phases:

1. Understand Background
2. Preliminary Exploratory Data Analysis
3. Data Cleaning
4. In-depth Exploratory Data Analysis
5. Feature Engineering
6. Model Building, Validation, and Prediction

The **Planner agent** is a dedicated role that produces a structured plan (both Markdown and JSON) for each phase. The Developer then implements the plan. The Reviewer scores both the Planner and Developer (1-5). The Summarizer generates Q&A reports for phase transitions. If quality scores are too low (< 3), the phase is repeated with "experience with suggestion" feedback injected.

Planning is **per-phase, not per-experiment**. The Planner doesn't decide "what model to try next" — it decides "what steps to perform during the Data Cleaning phase."

---

## 2. All Planner-Related Prompts

### 2A. ShaneZhong Builder Agent — Planning Prompts

#### Plan Trigger Message (from Orchestrator)

```
Round {NN} — research done. Write plan.
History: [1-2 sentence summary of what worked/failed in recent rounds]
Direction: [what to try next based on the pattern]
```

With retrospective:
```
Round {NN} — research done. Write plan.
Retro available: rounds/R{NN}_retro.md — read it before planning.
```

#### Experiment Decision Priority Order

> **Priority order:**
> 1. Concrete new technique from this round's scrape (top competitor finding)
> 2. Untried idea from knowledge_files not yet attempted in results.tsv
> 3. Near-miss from a previous round (small gain that could stack with something)
> 4. New model family not yet in the ensemble
> 5. Hyperparameter search on the best existing model
>
> **Rules:**
> - ONE experiment per round. The Reviewer will challenge it before you code.
> - Do not repeat anything marked `discard` unless the approach differs fundamentally.
> - If 3+ consecutive `discard` rows: force a different model family or approach.
> - If `cv_lb_divergence_flag` is true: favour regularisation, simpler features, or simpler blends.
> - Read the `LEARNINGS.md` file at the start and apply its rules throughout.

#### Plan Template (written to `R{NN}_plan.md`)

```markdown
# Round {NN} Plan
Date: YYYY-MM-DD
Metric: {metric} ({metric_direction})
Best CV: X.XXXXXX | Best LB: X.XXXXX
CV/LB divergence flag: true/false | Days to deadline: N

## Situation
[2-3 sentences: current standing, recent momentum, what's been working]

## CV/LB Divergence Assessment
[Current avg gap and trend. Implications if flag is true.]

## Last Round Assessment
[Round {N-1}: tried X, score Y, status Z, taught us W.]

## Hypothesis
[One sentence: "X will improve because Y, expected gain ~+Z."]

## Experiment Specification

**Type:** new_model | feature_engineering | ensemble_change | hyperparameter_tuning | stacking

**Model/approach:** [exact class and import path]

**Key hyperparameters:**
params = { "key": value }

**Features:** [describe feature set]

**CV strategy:** [{cv_strategy}(n_splits={cv_folds}, shuffle=True, random_state=42)]

**Ensemble:** [will this be submitted solo, or blended with prior OOFs? which ones?]

## Execution
- Local CPU or Kaggle GPU kernel: [choose]
- Estimated runtime: ~N hours
- Memory estimate: ~N GB
- OOM risk: low / medium / high

## Success Criterion
[CV_SCORE better than X.XXXXXX (direction: {metric_direction})]
```

#### Reviewer's Six Review Checks

> **CHECK 1 — DIVERSITY & CORRELATION**
> Look at the last 3 rounds in results.tsv. Are 2+ of them the same model family or architectural approach?
>
> **CHECK 2 — FORGOTTEN LEARNINGS**
> Scan your full context for insights from prior rounds and knowledge files that this plan contradicts or ignores.
>
> **CHECK 3 — SEARCH SPACE & EXPLOITATION VS EXPLORATION**
> Consider the competition timeline (days to deadline) and current gap to target.
>
> **CHECK 4 — ENSEMBLE HEALTH**
> Number of genuinely distinct base models, pairwise correlation range, metric gap between weakest and strongest.
>
> **CHECK 5 — BOLDER ALTERNATIVE**
> Name ONE experiment the Builder Agent didn't consider.
>
> **CHECK 6 — COST-BENEFIT ANALYSIS**
> Time cost, expected gain, opportunity cost. Flag if expected gain is negligible AND time > 2 hours.

#### Revise Trigger

```
Round {NN} — reviewer says: REVISE [paste the one-line reason here]. Rewrite plan.
```

One revision cycle per round maximum.

#### Diagnostic Run (Pre-Full Experiment)

> For any experiment expected to take > 1 hour, run a quick diagnostic first.
> - Fold-1 metric: is this competitive?
> - Correlation with best existing OOF: does this add diversity?
> - Hyperparameter sensitivity: does the model respond to key params?
>
> **Kill the experiment if diagnostics show:**
> - Fold-1 metric is much worse than best (> 0.5% relative gap)
> - Correlation with best OOF > 0.998 (no diversity value)
> - Model doesn't respond to any hyperparameter variation

---

### 2B. Academic AutoKaggle — Planner Prompts

#### Main Planner Task Prompt (`PROMPT_PLANNER_TASK`)

```
Please design plan that is clear and specific to each FEATURE for the current
development phase: {phase_name}.
The developer will execute tasks based on your plan.
I will provide you with INFORMATION, RESOURCE CONSTRAINTS, and previous reports and plans.
You can use the following reasoning pattern to design the plan:
1. Break down the task into smaller steps.
2. For each step, ask yourself and answer:
    - "What is the objective of this step?"
    - "What are the essential actions to achieve the objective?"
    - "What features are involved in each action?"
    - "Which tool can be used for each action? What are the parameters of the tool?"
    - "What are the expected output of each action? What is the impact of the action on the data?"
    - "What are the constraints of this step?"
```

#### Main Planner Prompt Body (`PROMPT_PLANNER`)

```
# CONTEXT #
{phases_in_context}
Currently, I am at phase: {phase_name}.

#############
# INFORMATION #
{background_info}

{state_info}

#############
# NOTE #
## PLANNING GUIDELINES ##
1. Limit the plan to a MAXIMUM of FOUR tasks.
2. Provide clear methods and constraints for each task.
3. Focus on critical steps specific to the current phase.
4. Prioritize methods and values mentioned in USER RULES.
5. Offer detailed plans without writing actual code.
6. ONLY focus on tasks relevant to this phase, avoiding those belonging to other phases.

## DATA OUTPUT PREFERENCES ##
1. Prioritize TEXT format (print) for statistical information.
2. Print a description before outputting statistics.
3. Generate images only if text description is inadequate.

## METHODOLOGY REQUIREMENTS ##
1. Provide highly detailed methods, especially for data cleaning.
2. Specify actions for each feature without omissions.

## RESOURCE MANAGEMENT ##
1. Consider runtime and efficiency.
2. Limit generated images to a MAXIMUM of 10 for EDA.
3. Focus on critical visualizations with valuable insights.
```

#### Planner Output Format — Markdown (`PROMPT_PLNNAER_REORGANIZE_IN_MARKDOWN`)

```markdown
## PLAN
### STEP 1
Task: [The specific task to be performed]
Tools, involved features and correct parameters: [tools and params]
Expected output or Impact on data: [expected output or data impact]
Constraints: [Any constraints or considerations]

### STEP 2
...
```

#### Planner Output Format — JSON (`PROMPT_PLNNAER_REORGANIZE_IN_JSON`)

```json
{
    "final_answer": [
        {
            "task": "The specific task to be performed",
            "tools, involved features and correct parameters": ["tools and params"],
            "expected output or impact on data": ["expected output"],
            "constraints": ["constraints"]
        }
    ]
}
```

#### Planner Multi-Turn Flow (from `planner.py`)

The planner runs a **4-round conversation with the LLM**:
1. **Round 0:** System message with role + planning task prompt + state info + user rules + background
2. **Round 1:** Previous plan/report from prior phase + data samples + available tools (RAG-retrieved)
3. **Round 2:** Reorganize into Markdown format
4. **Round 3:** Reorganize into JSON format

This produces both a human-readable markdown plan and a machine-parseable JSON plan.

#### Phase-Specific State Info (from `state.py`)

Each phase gets targeted instructions. Examples:

**Data Cleaning:**
> 1. Address issues identified in the Preliminary EDA phase.
> 2. Handle missing values using appropriate techniques.
> 3. Treat outliers and anomalies.
> 4. Ensure consistency across both datasets.
> 5. Create `cleaned_train.csv` and `cleaned_test.csv`.

**Feature Engineering:**
> 1. Create new features based on insights from the In-depth EDA.
> 2. Transform existing features to improve model performance.
> 3. Handle categorical variables (e.g., encoding).
> 4. Normalize or standardize numerical features if necessary.
> 5. Select the most relevant features for modeling if necessary.
> 6. Create `processed_train.csv` and `processed_test.csv`.

**Model Building:**
> Before training the model:
> 1. For the training set, separate the target column as y.
> 2. Remove the target column and any non-numeric columns.
> Due to computational resource limitations, you are allowed to train a maximum of **three** models.

#### Configurable Rulebook Parameters

The system has a configurable rulebook with default rules per phase:

**Data Cleaning rules:**
- "If the percentage of missing values in a feature is greater than {0.3}, dropped the feature."
- "For numerical features, fill missing values with {median}."
- "For categorical features, fill missing values with {mode}."
- "If you use the IQR method to detect outliers, consider truncating or deleting values that exceed {1.5} IQR."

**Feature Engineering rules:**
- "Apply logarithmic transformation to numerical features with skewness greater than {0.5}"
- "Set the maximum degree for polynomial features. Maximum degree = {2}"
- "For categorical variables with cardinality below {10}, use one-hot encoding. For high cardinality, use target encoding or frequency encoding."
- "Create interaction terms for important features, limiting the max degree of interaction = {2}"

#### Tool-Aware Planning

The Planner receives tool descriptions via RAG retrieval:
```
# AVAILABLE TOOLS #
## TOOL LIST ##
You have access to the following tools: {tool_names}
## DETAILED TOOL DESCRIPTIONS ##
{tools}
```

Tools include: `fill_missing_values`, `remove_columns_with_missing_data`, `detect_and_handle_outliers_zscore`, `detect_and_handle_outliers_iqr`, `convert_data_types`, `remove_duplicates`, `format_datetime`, `one_hot_encode`, `label_encode`, `frequency_encode`, `target_encode`, `correlation_feature_selection`, `variance_feature_selection`, `scale_features`, `perform_pca`, `perform_rfe`, `create_polynomial_features`, `create_feature_combinations`, `train_and_validation_and_select_the_best_model`.

#### Feature Info Tracking

```
# FEATURE INFO #
## TARGET VARIABLE
{target_variable}
## FEATURES BEFORE THIS PHASE
{features_before}
## FEATURES AFTER THIS PHASE
{features_after}
```

#### Experience/Retry Loop

When a phase fails (score < 3), the Planner sees previous experience and reviewer suggestions:

```
## EXPERIENCE {index} ##
<EXPERIENCE>
{experience}
</EXPERIENCE>
<SUGGESTION>
{suggestion}
</SUGGESTION>
<SCORE>
{score}
</SCORE>
```

---

## 3. Concepts Their Planner Captures That Ours Might NOT

### From ShaneZhong/autokaggle (Operational)

- **CV/LB divergence tracking and response:** Explicit `cv_lb_divergence_flag` in state. When true, the planner is instructed to "favour regularisation, simpler features, or simpler blends." This is a real-time overfitting detection mechanism at the campaign level.

- **Submission quota management:** Strategic decisions about when to submit to the public leaderboard. Quota is precious (5/day). The system has rules for early-stage calibration submits vs. late-stage conservation.

- **Ensemble-aware planning:** Every plan considers whether the submission comes from a single model or ensemble. OOF correlation tracking (< 0.995 threshold) determines whether adding a model contributes diversity. Ridge stacking vs. simple weighted blend decision.

- **Fold-1 kill gates:** A pre-commit diagnostic that runs fold-1 only (~15 min) before committing to a full K-fold run. Kill criteria: fold-1 metric much worse than best, correlation with best OOF > 0.998, model doesn't respond to param variation.

- **Stacking checkpoint:** Before planning a new model, check if Ridge stacking on existing OOFs already improves CV. If yes, the next experiment should maximize OOF diversity, not chase individual model score.

- **Kaggle GPU awareness:** Plans explicitly choose between local CPU and Kaggle GPU kernels. Task-specific GPU routing (tree models with GPU support, neural networks → GPU).

- **Execution time and memory estimation:** Plans include estimated runtime, memory estimate, and OOM risk level.

- **Cost-benefit analysis in review:** "Is this the best use of the next N hours?" Reviewer estimates time cost, expected gain, and opportunity cost before approving.

- **Deadline-aware strategy:** Plans reference days-to-deadline. Endgame rules tighten (gains > +0.0002 required within 5 days).

- **Multi-round research pipeline:** Three distinct research types (domain, competition intelligence, model-specific) that feed the planner with different knowledge.

- **Retrospective with pattern detection:** Every 10 rounds, the Reviewer runs a full retrospective: cross-round connections, failure autopsy (wrong params vs. wrong approach vs. never properly tested), diminishing returns detection, "top 5 experiments to try next."

- **Structured experiment registry (experiments.json):** Tracks model families tried, their rounds, best scores, and status (active/exhausted/promising). This is richer than TSV results alone.

- **Crash recovery and atomic file writes:** Every phase output is a named file. On restart, the orchestrator checks existence and skips completed phases.

- **Agent re-spawn for context hygiene:** Agents are re-spawned every 15 rounds to prevent context overflow.

### From multimodal-art-projection/AutoKaggle (Academic)

- **Tool-grounded planning via RAG:** The Planner retrieves tool documentation from a vector store and plans using specific tool functions (e.g., `fill_missing_values(method='auto')`, `detect_and_handle_outliers_iqr`). Plans reference the actual tools the Developer will call.

- **Configurable rulebook with parameterised thresholds:** Missing value drop threshold (30%), outlier detection parameters (3σ or 1.5 IQR), skewness threshold for log transform (0.5), cardinality threshold for encoding choice (10), max polynomial degree (2). These are configurable defaults, not hardcoded.

- **Phase-gated planning:** The Planner cannot plan actions from other phases. During EDA, it "CAN perform detailed univariate analysis" but "CAN NOT modify any feature or modify data." This prevents scope creep.

- **Dual output format (Markdown + JSON):** Plans are produced in both human-readable Markdown and machine-parseable JSON. The JSON is used downstream by the Developer.

- **Unit test integration:** Each phase has unit tests (e.g., `test_cleaned_train_no_missing_values`, `test_processed_test_no_target_column`, `test_submission_validity`). The Developer's code is tested against these. Test failures trigger a locate-fix-merge debug loop.

- **Iterative debug loop for Developer:** When code errors occur, there's a structured 4-attempt loop: locate error → ask for help if stuck → fix → merge. The Developer cannot try more than 4 times.

- **Phase transition summaries via Summarizer:** After each phase, the Summarizer generates 6 targeted questions, answers them using all phase artifacts, and produces a transition report. This report becomes input to the next phase's Planner.

- **Feature tracking across phases:** `FEATURES BEFORE THIS PHASE` and `FEATURES AFTER THIS PHASE` are tracked, so the Planner knows exactly what columns exist at the start of each phase.

- **Image insight integration:** Visualizations from EDA phases are analyzed via `image_to_text.py` and the insights feed into summaries.

- **Score-based retry:** If the Reviewer gives a phase score < 3, the entire phase is repeated. The Planner/Developer receives their previous attempt's experience and reviewer suggestions. Max 3 retries.

---

## 4. Concepts Our Planner Has That Theirs Lack

- **Explicit hypothesis formation with expected impact:** Our schema requires each plan to contain testable hypotheses with `id`, `description`, and `expected_impact`. Neither AutoKaggle system requires hypotheses — they plan actions, not scientific claims.

- **Profile-grounded rationale at feature level:** Our `feature_steps[].rationale` must cite specific profile observations (e.g., "Cabin 77.10% missing per profile.json"). The academic system mentions features but doesn't require profile citations.

- **Leakage prevention as a first-class concern:** Our `leakage_flags.flagged_columns` from `profile.json` triggers hard exclusion in plans. This is a structural guarantee, not a soft guideline. Neither AutoKaggle system has explicit leakage detection in the planning phase.

- **Rollback/stop conditions:** Our schema requires `rollback_or_stop_condition` — what signals the approach is not working. The operational system has fold-1 kill gates (similar purpose, different mechanism), but the academic system has no explicit stop conditions.

- **Evaluation focus per iteration:** Our `evaluation_focus` field forces the planner to articulate what to examine most closely. The operational system has "Success Criterion" (just a metric threshold), and the academic system has no equivalent.

- **Expected win condition:** Our `expected_win_condition` is a metric threshold that confirms success. The operational system's "Success Criterion" is similar.

- **Structured YAML output with schema validation:** Our plans are validated by `src/planning/validator.py` against a strict schema. The academic system produces JSON but doesn't validate against a schema. The operational system produces free Markdown.

- **Mutual information integration:** Our planner receives MI scores per feature from the profile, giving it a quantitative signal ordering. The academic system receives data samples and basic stats but not MI.

- **Near-duplicate feature detection:** Our profile contains `near_duplicate_pairs` (r ≥ 0.98) which the planner uses for redundancy decisions. Neither AutoKaggle system has this.

- **Target validation metadata:** Our profile provides class counts, imbalance ratio, and `is_imbalanced` flag. The academic system mentions "describe the target variable" but doesn't structure it.

- **Risk flags per column:** Our profile generates per-column risk flags (high skew, outlier percentage) that the planner directly consumes. The academic system detects issues during EDA but doesn't pre-compute structured risk flags for the planner.

---

## 5. Key Takeaways for Our M3 Planner

### What to adopt or learn from:

1. **The operational system's "one experiment per round" discipline** is powerful. Each plan is small, testable, and reviewable. Our iteration-level planning is already aligned with this.

2. **Configurable thresholds with sensible defaults** (from the academic system) are worth considering. Instead of hardcoding "drop if > 30% null," let the plan schema or project config specify thresholds. Our planner could reference profile stats and apply rules like "if null_pct > project.missing_value_drop_threshold."

3. **Tool-aware planning** (from the academic system) is relevant if we build a tool library. Our planner currently plans in terms of abstract actions, but if we add validated tools (`fill_missing_values`, `detect_outliers_iqr`), the Coder benefits from plans that reference specific tools.

4. **The review checks structure** from the operational system (diversity, forgotten learnings, search space, ensemble health, bolder alternative, cost-benefit) is an excellent template for our future Reviewer agent.

5. **Feature tracking across iterations** — knowing what features exist before and after each step — is valuable for preventing drift and ensuring consistency.

### What NOT to adopt:

1. **Infinite autonomous loop without human checkpoints.** The operational system runs forever. Our system is designed for human review at key decision points (PRD §2.5, principle 5).

2. **Competition-specific concerns** (submission quota, LB divergence, Kaggle GPU routing) are out of scope for our tabular benchmark focus.

3. **Phase-only planning** from the academic system. Their planner plans "how to do data cleaning" across all features, not "what experiment to try." Our experiment-iteration model is more aligned with hypothesis testing.

4. **Score-based retry loops** — the academic system re-runs phases when quality is low. Our system should produce better plans via better prompts and grounding, not retry poor ones.
