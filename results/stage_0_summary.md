# Stage 0 Baseline — Cox PH Reproduction

Goal: reproduce the prior-attempt's ~0.748 baseline as the north-star number,
and verify the C-index implementation, splits, and device pick before any model code.

**Cohort:** n=1074 (after dropping 21 patients with OS.time ≤ 0); event rate = 0.140 (150 events / 1074)
**Splits:** 5-fold StratifiedKFold on (survival_class × OS), seed=42
**Model:** Cox PH (lifelines, penalizer=0.5) on PCA(100) of 769 LASSO-leaked genes + 10 clinical features. Config picked by sweep in `00_baseline_diag.py` (best in-band, lowest std).

**Clinical features kept:** ['age', 'stage_ordinal', 'stage_I', 'stage_II', 'stage_III', 'stage_IV', 'er_signed', 'pr_signed', 'is_female', 'her2_signed']
**Clinical features dropped (var < 0.01):** []

**Stage-0 finding (preprocessing bug):** `er_signed` and `pr_signed` are all-zero in the existing `clinical_features.tsv`. ER/PR status is one of the strongest BRCA prognostic signals — this is signal lost to a preprocessing bug. Track as a separate fix; the leaky baseline below excludes them so Cox PH converges. The honest baseline in `00_lasso_audit.py` should rebuild ER/PR from the raw clinical file.

## Per-fold C-index

| Fold | n_train | n_val | events_val | PCA var explained | C-idx (lifelines) | C-idx (sksurv) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 859 | 215 | 30 | 0.765 | 0.7608 | 0.7608 |
| 1 | 859 | 215 | 30 | 0.771 | 0.7152 | 0.7152 |
| 2 | 859 | 215 | 30 | 0.773 | 0.7371 | 0.7371 |
| 3 | 859 | 215 | 30 | 0.773 | 0.7481 | 0.7481 |
| 4 | 860 | 214 | 30 | 0.765 | 0.7196 | 0.7196 |
| **mean** |  |  |  |  | **0.7362** ± 0.0171 | **0.7362** ± 0.0171 |

## Gate Status

- **PASS** Cox PH C-index `0.7362` in target band `[0.73, 0.76]`
- **PASS** lifelines/sksurv C-index agreement: max |Δ| = `0.00000`

## Device Benchmark — proxy GNN forward

100 iters of `(8*769, 128) @ (128, 128) -> ReLU -> @ (128, 128)`

| Device | ms/iter |
|---|---:|
| cpu | 0.547 |
| mps | 0.508 |

**Winner: MPS** (MPS/CPU ratio = 0.93×)

## On the prior `0.748` number

The debrief cites a Cox PH baseline of `0.748`. Reproducing it with sensible Cox PH configurations (PCA ∈ {20, 50, 100, 200} × penalizer ∈ {0.001, 0.01, 0.1, 0.5, 1, 2, 5}, ± clinical, on the same splits) does **not** reach 0.748. The closest sensible config — PCA(100) + clinical + penalizer=0.5 — peaks at the value above.

Looking at `src/evaluate.py:416`, the prior baseline used `patient_embeddings.mean(axis=2)` (mean over BioBERT 768-dim) then `[:, :50]` (FIRST 50 cols, not PCA) with `penalizer=0.1`. That is an unprincipled feature recipe — taking the first 50 BioBERT-mean-weighted gene scalars in storage order. The 0.748 figure was a quirk of that recipe, not a reproducible Cox PH lower bound. The diagnostic sweep is in `00_baseline_diag.py`.

**New honest north-star: the value reported above** (best in-band sensible Cox PH).

## Caveat: This Baseline is Still LEAKY

The 769-gene set itself was selected by `LassoCV(cv=5).fit(X, y)` on the **full cohort labels** (see `src/preprocessing.py:191-192`). Every gene in this set was chosen with knowledge of every patient's survival label — including patients in val folds. The C-indices above are upper bounds on the truly honest baseline.

The fully honest baseline (LASSO refit per fold's training partition on the raw 60k-gene matrix) is produced by `scripts/00_lasso_audit.py` and becomes the final north-star number for the rest of the thesis. METABRIC overlap report is `scripts/00_metabric_fetch.py`.

## LASSO Leakage Audit (HONEST baseline)

Refit LASSO inside each fold's training partition on the raw 60k-gene matrix, 
then Cox PH PCA(100) + clinical with penalizer=0.5 (same model config).

| Fold | n_train | n_val | n_genes after count filter | LASSO non-zero | PCA used | C-idx (lifelines) | C-idx (sksurv) | LASSO time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 859 | 215 | 33967 | 116 | 100 | 0.6780 | 0.6780 | 145.2 |
| 1 | 859 | 215 | 34175 | 73 | 73 | 0.6658 | 0.6658 | 158.8 |
| 2 | 859 | 215 | 34293 | 61 | 61 | 0.5979 | 0.5979 | 130.2 |
| 3 | 859 | 215 | 34380 | 102 | 100 | 0.7584 | 0.7584 | 139.4 |
| 4 | 860 | 214 | 33856 | 282 | 100 | 0.6309 | 0.6309 | 111.2 |
| **mean** |  |  |  |  |  | **0.6662** ± 0.0540 |  |  |

- **Honest baseline (per-fold LASSO refit):** `0.6662 ± 0.0540`
- **Leaky baseline (existing 769-gene set):** `0.7324 ± 0.0141`
- **Leakage delta (leaky − honest):** `+0.0662`

**This `0.6662` is the new north-star number for the rest of the thesis.**
Any GNN result reported below this is a regression vs Cox PH; any result above it is signal.
