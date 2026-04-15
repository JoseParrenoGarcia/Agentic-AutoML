---
name: M4 design decisions
description: Non-obvious design choices made during M4 (Plan-to-Code Layer) that affect M5+ design
type: project
---

Feature engineering uses a two-function contract: `fit_transform(df_train, config) → (df, fitted_params)` and `transform(df, fitted_params, config) → df`. The fitted_params object carries all train-derived statistics to val/test. This replaces the original single-function design.

**Why:** Prevents leakage. Works for any backend — atomic pandas functions, sklearn Pipeline (fitted_params = pipeline object), or PyTorch preprocessing.

**How to apply:** M5 Executor must call fit_transform on train, then transform on val — in that order. The Debugger must not conflate the two functions when patching FE bugs.

---

`train_model` returns the fitted model only — no train_preds. Predictions are computed by evaluate.py.

**Why:** Single responsibility. evaluate.py has X_train and can call model.predict itself.

**How to apply:** Debugger should not expect train_preds from train_model.

---

Directory naming is `iterations/` not `runs/`. coding-rules.md is path-scoped to `iterations/**`.

**Why:** User preference established early in M4.

**How to apply:** M5 Executor and all future agents write to `projects/<project>/iterations/iteration-<n>/`.

---

The smoke test produced a working run: val AUC-ROC = 0.835 on Titanic iteration-1 (win condition > 0.80). All 7 output artifacts written. This is the baseline M5 should be able to reproduce automatically.
