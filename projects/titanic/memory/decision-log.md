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

## Iteration 4 — RandomForest
**Metric:** val_auc_roc = 0.8406 (delta: -0.0039)  
**Verdict:** insufficient | **Route:** pivot  
**Summary:** Refined RandomForest with stronger regularisation (max_depth=4, min_samples_leaf=15, min_samples_split=30, 150 estimators) and Cabin deck-letter feature extraction for Titanic binary classification  
**Reasoning:** Iteration 4 degraded the primary metric (AUC 0.841 vs 0.845, delta -0.004) and all secondary metrics worsened: accuracy -0.045, F1 -0.042, precision -0.077, recall unchanged. The stronger regularisati...  
**Risk flags:** 1 (medium-overfitting)

## Iteration 5 — GradientBoosting
**Metric:** val_auc_roc = 0.8387 (delta: -0.0020)  
**Verdict:** insufficient | **Route:** pivot  
**Summary:** GradientBoostingClassifier with conservative regularisation (lr=0.05, max_depth=3, min_samples_leaf=20, subsample=0.8, early stopping) and lean feature set (Title, FamilySize, HasCabin; dropped Cabin deck-letter features) — hypothesis: surpass iteration 3 AUC 0.8445 with tighter overfitting control.  
**Reasoning:** Iteration 5 GBM (AUC 0.8387) did not beat iteration 3 RandomForest (best: 0.8445, delta -0.0058). Two consecutive iterations have now failed to improve on the best. Medium overfitting remains (8.62% t...  
**Risk flags:** 1 (medium-overfitting)

## Iteration 5 — GradientBoosting
**Metric:** val_auc_roc = 0.8387 (delta: -0.0020)  
**Verdict:** insufficient | **Route:** pivot  
**Summary:** Regularised GradientBoostingClassifier (500 estimators, lr=0.05, max_depth=3, heavy regularisation) with title-based and family-size features, early stopping via n_iter_no_change for Titanic binary classification  
**Reasoning:** Iteration 5 returns GradientBoosting with heavy regularisation (max_depth=3, min_samples_leaf=20, subsample=0.8) but fails to beat the best model. val_auc_roc=0.8387 is -0.0058 below iteration 3 (best...  
**Risk flags:** 1 (medium-overfitting)

## Iteration 5 — GradientBoosting
**Metric:** val_auc_roc = 0.8387 (delta: -0.0020)  
**Verdict:** insufficient | **Route:** rollback  
**Summary:** GradientBoosting with re-tuned hyperparameters (max_depth=3, 300 estimators, lr=0.05) and HasCabin feature for Titanic binary classification  
**Reasoning:** Iteration 5 degrades the primary metric for the second consecutive iteration (iter 3: 0.8445 → iter 4: 0.8406 → iter 5: 0.8387). Medium overfitting persists (8.62% train/val gap). No high-severity fla...  
**Risk flags:** 1 (medium-overfitting)

## Iteration 6 — StackingClassifier
**Metric:** val_auc_roc = 0.8560 (delta: +0.0173)  
**Verdict:** insufficient | **Route:** continue  
**Summary:** StackingClassifier (RF + LR meta) with log-transformed Fare, HasCabin, Title, FamilySize features for Titanic binary classification  
**Reasoning:** Iteration 6 is the new best model (AUC 0.8560, +0.0173 vs previous, +0.0115 vs prior best iter-3). Stacking broke through the 2-iteration plateau. Overfitting is low (4.2% train/val gap, down from 8.6...  
**Risk flags:** none

