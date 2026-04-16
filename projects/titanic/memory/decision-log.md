# Decision Log
<!-- Append-only. One section per completed iteration. -->

## Iteration 1 — LogisticRegression
**Metric:** val_auc_roc = 0.8349 (delta: N/A (baseline))  
**Verdict:** insufficient | **Route:** pivot  
**Summary:** Baseline logistic regression model for Titanic survival binary classification with stratified 80/20 split  
**Reasoning:** Iteration 1 establishes a solid baseline (AUC 0.835) with no risk flags, low overfitting, and no leakage. However, only one model family has been tried, 9 iterations remain, and tree-based models typi...  
**Risk flags:** none

## Iteration 2 — GradientBoosting
**Metric:** val_auc_roc = 0.8218 (delta: -0.0131)  
**Verdict:** insufficient | **Route:** rollback  
**Summary:** GradientBoosting (200 estimators, max_depth=4) with engineered features (Title, FamilySize) for Titanic binary classification  
**Reasoning:** Primary metric (AUC-ROC) regressed from 0.835 to 0.822 (-0.013). While secondary metrics (accuracy +0.017, F1 +0.024, recall +0.029) improved slightly, the model exhibits severe overfitting: train AUC...  
**Risk flags:** 1 (high-overfitting)

## Iteration 3 — RandomForest
**Metric:** val_auc_roc = 0.8445 (delta: +0.0096)  
**Verdict:** insufficient | **Route:** continue  
**Summary:** Regularised RandomForestClassifier (200 estimators, max_depth=5) with title-based and family-size features for Titanic binary classification  
**Reasoning:** Iteration 3 is the new best model (AUC 0.8445, +0.0096 vs baseline, +0.0227 vs previous). Regularisation successfully reduced overfitting from 17.2% (high) to 6.5% (medium). No leakage detected. All s...  
**Risk flags:** 1 (medium-overfitting)

