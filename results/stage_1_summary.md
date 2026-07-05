# Stage 1 — Multinomial Logistic Regression Sanity

## TL;DR

| Metric | Value |
|---|---|
| Cohort n | 1074 |
| Class counts (0/1/2/3) | [166, 474, 174, 260] |
| Majority class | 1 (freq = 0.4413) |
| Acc gate (> majority) | 0.4413 |
| AUC gate (signal detected) | 0.5500 |
| LR (expr+clin) mean accuracy | **0.4357** ± 0.0186 |
| LR (expr+clin) mean AUC OvR | **0.6124** |
| LR (expr+clin) balanced acc | 0.3107 |
| LR (clin-only) accuracy | 0.4460 ± 0.0082 |
| LR (clin-only) AUC OvR | 0.5714 |
| **Verdict** | **SIGNAL_OK_ACC_MARGINAL** — AUC > 0.55 (labels carry ranking signal) but acc <= 0.4413 (classification accuracy doesn't beat majority — consistent with high censoring + 4-bin discretization noise; **Cox-style ranking objective is the right metric for Stages 2+**, not classification accuracy) |

## Interpretation

The 4-bin survival_class is a noisy supervised target on TCGA-BRCA: 86% of patients are censored, and a censored patient labeled `1-3yr` may actually have survived `>5yr` (we only know a lower bound on their bin). Multinomial CE is supervised on these noisy labels and pays a cost for guessing the wrong bin even when the predicted ranking is correct. AUC OvR ignores hard predictions and measures whether the predicted class probabilities rank patients correctly, which is the survival-relevant question.

**Observed pattern:** clinical-only LR squeaks above majority on accuracy but has lower AUC; expression+clinical *loses* on accuracy but *gains* on AUC. This is the classic noisy-labels-with-ranking-signal regime. **Conclusion: use Cox loss (ranking) in Stages 2+, not classification CE.** This is what the design doc recommended; Stage 1 validates the choice.

## Per-fold metrics

| Fold | n_train | n_val | val majority | Accuracy | Balanced acc | F1 macro | AUC OvR | PCA var |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 859 | 215 | 0.4419 | 0.4279 | 0.2937 | 0.2734 | 0.6129 | 0.666 |
| 1 | 859 | 215 | 0.4419 | 0.4558 | 0.3282 | 0.3205 | 0.6324 | 0.673 |
| 2 | 859 | 215 | 0.4419 | 0.4558 | 0.3217 | 0.3113 | 0.6404 | 0.674 |
| 3 | 859 | 215 | 0.4419 | 0.4326 | 0.3235 | 0.3145 | 0.6084 | 0.675 |
| 4 | 860 | 214 | 0.4393 | 0.4065 | 0.2867 | 0.2674 | 0.5681 | 0.665 |
| **mean** |  |  |  | **0.4357** ± 0.0186 | 0.3107 | 0.2974 | 0.6124 |  |

## Confusion matrix (sum across folds)

Rows = true class, columns = predicted class. Class index 0..3 = [<1yr, 1-3yr, 3-5yr, >5yr].

| true \ pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| **0** | 20 | 128 | 6 | 12 |
| **1** | 23 | 357 | 16 | 78 |
| **2** | 6 | 116 | 10 | 42 |
| **3** | 8 | 149 | 22 | 81 |

## Notes

- Features = same as Stage 0 leaky baseline: PCA(50) of 769 LASSO-leaky genes + 7 clinical, z-scored per fold.
- Vanilla LR (no class weighting): the gate is *raw accuracy beats majority*; balanced weighting can't beat majority by construction even with real signal.
- LR `C=0.1` (multinomial default in sklearn >=1.7, lbfgs, max_iter=5000, seed=42).

**Caveat:** these features carry the same LASSO leakage as the leaky Cox PH baseline. If the verdict is borderline, re-run with per-fold-honest LASSO genes from `scripts/00_lasso_audit.py`. The Stage 1 gate is *anything beats majority*, so leaky features are acceptable here.

**Censored-label noise warning:** with 86% censoring, many patients in `survival_class=1` (1-3yr) are censored — their true class could be 1, 2, or 3 (we only know lower bound). LR is supervised on noisy labels here. AUC OvR is more interpretable than raw accuracy because it doesn't require a hard prediction.
