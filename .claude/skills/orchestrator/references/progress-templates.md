# Progress Templates Reference

Templates for structured output at each stage of orchestration.

---

## Iteration Progress Report

Printed after each iteration completes (Step 2f).

### Template
```
+--------------------------------------------------------------+
| ITERATION <n> COMPLETE                                        |
+--------------------------------------------------------------+
| Metric:    <name> = <value> (delta: <delta>)                  |
| Model:     <model_family>                                     |
| Verdict:   <reviewer_verdict>                                 |
| Route:     <router_decision>                                  |
| Best:      iteration-<best_iteration>                         |
| Elapsed:   <Xm Ys> (this iteration)                           |
| Progress:  <n> / 10 iterations                                |
+--------------------------------------------------------------+
```

### Example (Titanic iteration 3)
```
+--------------------------------------------------------------+
| ITERATION 3 COMPLETE                                          |
+--------------------------------------------------------------+
| Metric:    val_auc_roc = 0.8445 (delta: +0.0096)             |
| Model:     RandomForest                                       |
| Verdict:   insufficient                                       |
| Route:     continue                                           |
| Best:      iteration-3                                        |
| Elapsed:   4m 32s (this iteration)                            |
| Progress:  3 / 10 iterations                                  |
+--------------------------------------------------------------+
```

---

## Final Summary

Printed when the loop exits (Step 3).

### Template
```
+======================================================================+
| EXPERIMENT COMPLETE                                                    |
+======================================================================+
| Project:          <project-name>                                       |
| Total iterations:  <n>                                                 |
| Stop reason:       <sufficient | max_iterations | failure>             |
| Best iteration:    iteration-<best>                                    |
| Best metric:       <name> = <value>                                    |
| Total elapsed:     <Xh Ym Zs>                                         |
+----------------------------------------------------------------------+
| Iteration History:                                                     |
|   iter-1: <metric>=<value> (<model>) [<verdict>/<route>]              |
|   iter-2: <metric>=<value> (<model>) [<verdict>/<route>]              |
|   ...                                                                  |
+======================================================================+
```

### Example (Titanic after 5 iterations)
```
+======================================================================+
| EXPERIMENT COMPLETE                                                    |
+======================================================================+
| Project:          titanic                                              |
| Total iterations:  5                                                   |
| Stop reason:       sufficient                                          |
| Best iteration:    iteration-5                                         |
| Best metric:       val_auc_roc = 0.8612                               |
| Total elapsed:     22m 15s                                             |
+----------------------------------------------------------------------+
| Iteration History:                                                     |
|   iter-1: val_auc_roc=0.8349 (LogisticRegression) [insufficient/pivot]|
|   iter-2: val_auc_roc=0.8218 (GradientBoosting) [insufficient/rollbck]|
|   iter-3: val_auc_roc=0.8445 (RandomForest) [insufficient/continue]   |
|   iter-4: val_auc_roc=0.8534 (RandomForest) [insufficient/continue]   |
|   iter-5: val_auc_roc=0.8612 (XGBoost) [sufficient/continue]          |
+======================================================================+
```

---

## Escalation Report

Printed when an agent fails after retry budget exhausted.

### Template
```
+======================================================================+
| ORCHESTRATOR ESCALATION — HUMAN ACTION REQUIRED                       |
+======================================================================+
| Agent:           <agent-name>                                          |
| Step:            <step number and name>                                |
| Iteration:       <n>                                                   |
| Project:         <project-name>                                        |
+----------------------------------------------------------------------+
| What failed:     <missing files or validation error>                   |
| Error detail:    <validator message or manifest error_summary>         |
| Attempts:        <count> (max: <max>)                                  |
+----------------------------------------------------------------------+
| Suggested action: <specific guidance>                                  |
+======================================================================+
```

### Example (executor failure)
```
+======================================================================+
| ORCHESTRATOR ESCALATION — HUMAN ACTION REQUIRED                       |
+======================================================================+
| Agent:           executor                                              |
| Step:            2c — Executor                                         |
| Iteration:       4                                                     |
| Project:         titanic                                               |
+----------------------------------------------------------------------+
| What failed:     execution/manifest.json status=failed                 |
| Error detail:    ImportError: No module named 'xgboost'               |
| Attempts:        1 (max: 1 — executor has internal retries)            |
+----------------------------------------------------------------------+
| Suggested action: Install xgboost in .venv and re-run executor         |
+======================================================================+
```

---

## State Detection Report

Printed at the start of orchestration (Step 0).

### Template
```
+--------------------------------------------------------------+
| ORCHESTRATOR — STATE DETECTION                                |
+--------------------------------------------------------------+
| Project:          <project-name>                              |
| Task type:        <task_type>                                 |
| Profile exists:   <yes/no>                                    |
| Completed iters:  <n>                                         |
| Last verdict:     <verdict or N/A>                            |
| Last route:       <route or N/A>                              |
| Best iteration:   <best or N/A>                               |
| Action:           <starting fresh / resuming at iter N+1 /    |
|                    already complete>                           |
+--------------------------------------------------------------+
```
