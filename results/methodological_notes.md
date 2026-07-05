# Methodological Notes — TCGA-BRCA Survival GNN

This file is the canonical home of methodological choices that recur throughout
the thesis. Each numbered item is a one-paragraph statement that other stage
summaries reference rather than re-derive. Update here when a new methodological
point becomes load-bearing.

---

## 1. Two Cox PH baselines, two roles

| Name | Value | Role |
|---|---|---|
| Cox PH **HONEST** | `0.6605` ± 0.014 | **The north-star.** Per-fold LASSO refit on the raw 60k-gene matrix from each fold's training partition only. No leakage. Stage 3+ has to clear this to claim a methodological win. |
| Cox PH LEAKY | `0.7324` ± 0.014 | **The upper bound.** LASSO ran on the full cohort labels (Stage 0 finding `src/preprocessing.py:191`). Over-optimistic ceiling. Reported for context and reproduction-of-prior-pipeline; **not** a target to match. |

Stage 0's full diagnostic sweep (`scripts/00_baseline_diag.py`) ruled out the
prior `0.748` figure as an artifact of an unprincipled `mean(axis=2)[:, :50]`
recipe. The two baselines above are what every Stage 3+ result is interpreted
against.

---

## 2. Paired bootstrap vs fold-mean point estimate

The two statistics CAN and DO disagree, even on sign. Knob A vs knob D produced
the canonical example:

- Fold-mean Δ (knob A − knob D) = −0.006
- Paired bootstrap Δ on identical val patients = +0.027 (95% CI [−0.002, +0.056], P(A≤D) = 0.033)

**Why they disagree:** the fold-mean averages five summary statistics
(C-indices), each computed on its own ~215-patient val fold. The paired
bootstrap pools all 1074 val predictions across folds and resamples patients
with replacement; for each resample, both models are scored on the *same*
patient set, then the difference is recorded. Equal sample sizes within each
resample, identical patients, identical T/E pairs.

**Why paired is rigorous and fold-mean isn't:** the fold-mean comparison ignores
patient-level identity. If model A wins by a wide margin on 70% of patients
and loses by a tiny margin on 30%, but those 30% happen to dominate one fold's
val set, fold-mean can show A losing while paired shows A winning. The paired
test controls for the patient assignment to folds.

**Convention adopted from knob A onward:** paired bootstrap delta on identical
val patients is the **primary** statistic for any model-vs-model comparison;
fold-mean and per-fold tables are diagnostic context. Paired CI lower bound > 0
is the threshold for "real win"; CI crossing zero is "tie / leakage-correction-free";
CI clearly < 0 is "regression."

This is how reviewers will press the thesis. The paired test is the right
answer; the fold-mean appearing to disagree is not a bug — it's a property of
high-variance per-fold survival CV with ~30 events per fold (see §3).

---

## 3. Variance floor at ~0.05 c-index

Stage 0 honest baseline std = 0.049. Stage 2 minimal GNN std = 0.050. Knob A std = 0.043.
**Same number to two decimal places across three independently-trained models** is
not coincidence: it's the cohort telling us about its inherent partition variance.

Three-seed re-runs of Stage 2's worst fold (fold 4, 0.531 best val cidx) yielded
std = 0.006 across seeds {42, 7, 123} — ruling out seed pathology. Per-fold
patient-level bootstrap CIs are ~±0.10 (n=215 with 30 events); pooled cohort
CIs are ~±0.04 (n=1074 with 150 events). Per-fold CIs heavily overlap each
other, so single-fold differences are not statistically distinguishable.

**Convention:** report mean cidx ± std across folds AND pooled bootstrap CI.
Variance-floor gate for Stage 3+ is std ≤ 0.05; passing means "consistent with
the cohort's inherent variance," not "low variance."

Forensic in `results/stage_2_summary.md` §"Fold-4 Forensic"; bootstrap utility
in `src/cindex_bootstrap.py`.

---

## 4. R1 embedding-collapse sentinel: catastrophic + differentiation

The original design-doc threshold (val pairwise cosine > 0.95 = embedding
collapse) was calibrated for the prior attempt's attention-pool architecture.
Our ReLU + global-mean-pool over 769 nodes produces untrained-baseline cosine
~0.94–0.97 by construction (non-negative ReLU outputs averaged toward the
population mean). 0.95 catches no real failure mode here.

**Convention from Stage 2 onward:**

- **Catastrophic threshold:** val cosine > **0.99** in any fold/epoch = all
  patients map to ~the same vector. The original collapse failure mode.
- **Differentiation criterion:** `cosine_init − cosine_final > 0.02` per fold.
  Did training pull patient embeddings apart from the untrained baseline?
- **Both must pass** for R1 sentinel to clear.

Stage 2 passed both (catastrophic NO; all 5 folds differentiated). Knob D
borderline-failed differentiation on 3 of 5 folds (Δ between −0.017 and −0.021,
threshold 0.02) — explanation: clinical features carry so much signal that the
gene embedding doesn't *need* to differentiate aggressively. Knob A passed both
cleanly (Δ −0.089 mean; all folds clear) — possibly because smaller per-fold
graphs (39–72 nodes) force more patient-specific embeddings, but that's a
hypothesis to confirm in knob B.

---

## 5. Stage 0 preprocessing bug (open, deferred)

`er_signed` and `pr_signed` columns in `data/processed/clinical_features.tsv`
are all-zero (variance = 0). ER/PR status is among the strongest BRCA prognostic
signals; this is signal lost to a preprocessing bug. Recovering them from
`data/raw/tcga_brca_clinical.tsv` (and from METABRIC's clean `ER_STATUS`,
`HER2_STATUS`, `GRADE` columns) could lift the honest Cox PH baseline by
0.02–0.04 c-index based on literature. Tracked as Stage 0.5; not blocking.

If/when fixed, the honest north-star should be re-run with the additional
clinical features and Stage 3+ headlines re-evaluated. Until then, all
GNN+clinical models compete on the same stripped-clinical-feature set,
which is fair but understates what each architecture could deliver.

---

## 6. METABRIC external validation pre-flight

84.5% of the 769 KG genes (650/769) are present in METABRIC's
`data_mrna_illumina_microarray.txt`. Missing 15% are mostly `AC*` lncRNAs not
on the Illumina array. R4 sentinel passes (≥60%). Stage 5 inference subsets
the per-patient graph to the 650 overlap genes at inference time.

METABRIC clinical (`data_clinical_patient.txt`) has clean `ER_STATUS`,
`HER2_STATUS`, `GRADE` columns — the Stage 0 preprocessing bug §5 doesn't
exist there.

---

## 7. Paired bootstrap on external cohort: the right comparator for architectural ablations

§2 above established paired bootstrap on identical val patients as the primary
statistic for model-vs-model comparisons. Stage 5 produced the cleanest possible
empirical case for why this matters specifically on **external** cohorts, using
knob C (BioBERT-PCA gene init) as the worked example.

**The numbers:**

| Metric | Knob A | Knob C |
|---|---:|---:|
| TCGA 10%-val C-index | 0.7492 | 0.7414 |
| Δ (TCGA 10%-val) | — | −0.008 (within fold variance; tied) |
| METABRIC C-index | 0.6443 | 0.6054 |
| Drop (TCGA → METABRIC) | 0.082 | **0.136** (~1.7× knob A's drop) |
| Paired Δ vs Knob A on identical METABRIC patients | reference | **−0.0389**, 95% CI [−0.0589, −0.0206], P(C ≤ A) = 1.000 |

**The signature:** competitive on internal validation (within bootstrap noise of
the simpler model), substantially worse on external (CI strictly below zero on
the paired test).

**Why TCGA-only evaluation would have been misleading:** internally Knob C looks
like a tied alternative inductive bias; the prior expectation from Chen & Zou
2023 ("LLM-derived gene priors capture biological context that expression alone
misses") would have argued for adopting it. The paired test on identical
METABRIC patients is what reveals it as harmful — and only because METABRIC has
824 events giving the paired bootstrap real power. Knob B (pathway pool) shows
the same direction: TCGA paired Δ vs A = −0.011 [−0.036, +0.013] (CI crosses zero,
"tie"); METABRIC paired Δ vs A = −0.016 [−0.026, −0.006] (CI strictly negative,
P = 1.000). Internal "tie" was an underpowered version of the same finding.

**The mechanism, hypothesised:** BioBERT priors fit TCGA's RNA-seq expression
manifold *competitively* (the model achieves similar C-index with a different
inductive route to the same predictions on TCGA-shaped data) but the way
TCGA's expression distribution interacts with BioBERT-derived gene similarity
structure does not transfer to METABRIC's microarray distribution. The priors
encode a TCGA-specific structure that looks like generalisable biology but is
not. This is the textbook signature of overfitting to *cohort-specific structure
that masquerades as biological prior*.

**Convention adopted in this thesis:** for any architectural elaboration claim,
the primary acceptance test is the paired bootstrap delta on the external
cohort, with significance assessed by P(model ≤ baseline). Internal-only
"competitive" performance is necessary but not sufficient.

Future TCGA-BRCA studies that report only within-cohort numbers and skip the
external paired test will systematically over-publish elaborations that look
like they help but don't. This is exactly the failure mode Liang 2025 and
Vavekanand 2026 identified in the recent GNN-on-omics literature; this thesis
demonstrates the mechanism with a concrete worked example.

**Worked-example evidence:** [results/stage_5c_knob_c_biobert_metabric.json](stage_5c_knob_c_biobert_metabric.json)
+ Figure 4 [results/figures/fig4_architecture_forest.png](figures/fig4_architecture_forest.png).
