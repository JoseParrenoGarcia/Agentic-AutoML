# Model Report — Iteration 6

**Model:** StackingClassifier (RandomForest + LogisticRegression base, LR meta-learner)  
**Verdict:** improved  
**Primary metric:** val_auc_roc = 0.8560 (+0.0173 vs iteration 5, +0.0115 vs best iteration 3)

---

## Headline Metrics

| Metric | Train | Validation |
|--------|-------|------------|
| AUC-ROC | 0.8935 | 0.8560 |
| Accuracy | 0.8371 | 0.8268 |
| F1 | — | 0.7669 |
| Precision | — | 0.7969 |
| Recall | — | 0.7391 |

---

## Overfitting Check

Train/val AUC-ROC gap: 0.0375 (4.2%) — **severity: low**.  
This is the first iteration to achieve a sub-5% train/val gap, improving from 6.5% (iter 3), 5.5% (iter 4), and 8.6% (iter 5). Learning curve unavailable for StackingClassifier (non-iterative ensemble).

---

## Prior Run Comparison (vs Iteration 5)

| Metric | Iter 5 | Iter 6 | Delta |
|--------|--------|--------|-------|
| val_auc_roc | 0.8387 | 0.8560 | **+0.0173** ✓ |
| val_accuracy | 0.8101 | 0.8268 | +0.0168 ✓ |
| val_f1 | 0.7463 | 0.7669 | +0.0206 ✓ |
| val_precision | 0.7692 | 0.7969 | +0.0277 ✓ |
| val_recall | 0.7246 | 0.7391 | +0.0145 ✓ |

All five tracked metrics improved. The stacking ensemble broke through the 0.845 ceiling that single-family models could not surpass.

---

## Risk Flags

**None.** This is the first iteration with zero risk flags.

---

## Feature Importance (RF base estimator)

1. Title_Mr — 0.2973
2. Sex — 0.2194
3. Fare_log — 0.1047 *(new — log-transform of Fare)*
4. Pclass — 0.0759
5. HasCabin — 0.0672

The Fare_log feature (new in iteration 6) entered the top-3 immediately, confirming that the log-transform better represents the right-skewed Fare distribution for tree-based splits.

---

## Calibration

Brier score: **0.1316** — best across all iterations (iter 3: ~0.155, iter 5: 0.1459). The stacking meta-learner (LR with Platt scaling) produces well-calibrated probabilities.

---

## Error Analysis

Confusion matrix: TP=51, FP=13, FN=18, TN=97 (n=179 validation)  
False negative rate: 18/(51+18) = 26.1% — passengers who survived but were misclassified.  
False positive rate: 13/(13+97) = 11.8% — passengers who did not survive but were misclassified.

---

## Plateau Signal

**Detected** (2 consecutive stale iterations before this one). Iteration 6 broke the plateau with a +0.017 improvement, clearing the 0.845 ceiling that had blocked iterations 3–5.

---

## Recommendation

Iteration 6 is the new best model (AUC 0.8560, all metrics improved, zero risk flags, low overfitting). Overfitting is now below the 5% threshold for the first time. The stacking approach — combining RF's non-linear power with LR's calibrated probabilities — proved effective. If further improvement is desired, consider: (1) adding XGBoost as a third base estimator, (2) tuning RF hyperparameters within the stack, or (3) evaluating sufficiency given the 0.856 AUC on a dataset where published leaderboard scores cluster around 0.85–0.88.
