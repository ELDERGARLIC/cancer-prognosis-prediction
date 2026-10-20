# Methods — Outline (structure only; no prose yet)

Reading order is reader-needs order, not chronology of construction. Each
section depends only on sections above it. A reviewer reading top-to-bottom
should never have to forward-reference.

Target total length: 3000–4000 words across 7 sections. Source material is
`results/methodological_notes.md` and the per-stage summaries (paths
referenced in each section).

---

## 1. Data (≈400 words)

The cohorts and the labels, the gene universe and its limitation, the
clinical features that survived intersection. Establishes everything the
later sections operate on.

- **TCGA-BRCA cohort.** n=1074 after dropping OS.time ≤ 0 (21 patients).
  Event rate 0.140 (150/1074). 4-bin survival_class distribution. 7 clinical
  features (`age, stage_ordinal, stage_I/II/III/IV, is_female`); ER/PR
  preprocessing-bug noted as documented limitation, deferred fix.
- **METABRIC external cohort.** n=1466 after valid-OS + valid-stage filter
  from raw 2509. Event rate 0.561 (824/1466). Clinical features post-
  intersection: 3 (`age, stage_ordinal, is_female`); HER2/grade dropped
  for paired comparability with TCGA's stripped-clinical set.
- **Survival-time encoding.** Both cohorts: days. METABRIC reports
  OS_MONTHS in source; converted to days × 30.4375. TCGA OS.time native days.
- **Gene universe (the leaky-769).** Selected by full-cohort `LassoCV`
  on TCGA `survival_class` (Stage 0 finding `src/preprocessing.py:191`).
  Documented as a Stage-0 limitation up front: knob A's per-fold LASSO
  is a *within-769* refit, not a fully-honest universe rebuild. The
  fully-honest version was attempted (raw-60k LASSO ∩ leaky-769 = 14%
  overlap, 6 edges per fold) and ruled out for sparsity.
- **METABRIC gene-overlap.** 84.5% of leaky-769 in METABRIC's Illumina
  array (650/769; missing mostly AC* lncRNAs). Stage 5 inference subsets
  the per-patient graph to overlap genes.
- **Knowledge graph topology.** STRING PPI ≥700 (high confidence). 49,674
  gene-gene edges over the 769 genes. Symmetric, undirected. Same edge
  set used by all GNN variants; per-fold edge masking subsets it to that
  fold's gene set.

Source: `results/methodological_notes.md` §1, §5, §6.

---

## 2. Baselines (≈500 words)

Three baselines, three roles. The Cox PH HONEST/LEAKY distinction is the
load-bearing setup for everything that follows; introducing it here forces
every later result to be interpreted against the right comparator.

- **Cox PH HONEST (north-star, target = 0.6605 ± 0.014).** Per-fold LASSO
  refit on raw 60k-gene matrix from training partitions only. PCA(100)
  + 7 clinical features, ridge penalizer 0.5. The number any methodological
  win must clear.
- **Cox PH LEAKY (upper bound = 0.7324 ± 0.014).** Same Cox PH formulation,
  but with the legacy preprocessing's full-cohort-LASSO 769-gene universe.
  Reported for context; **not** a target. The +0.072 leakage cost between
  the two baselines is the leakage audit's headline finding.
- **MLP clinical-only (non-linear reference = 0.7122 ± 0.038).** Two-layer
  MLP on the 7 clinical features, Cox loss. Establishes the non-linear
  flat-feature ceiling against which the GNN's gene-graph contribution is
  honestly measured. Without this, the GNN's lift over linear Cox PH
  HONEST conflates "non-linear use of clinical" with "gene-graph signal."
- **The prior `0.748` figure.** Stage 0's diagnostic sweep showed it was
  not reproducible with sensible Cox PH configurations — it was an artifact
  of `mean(axis=2)[:, :50]` over BioBERT-weighted gene scalars. Debunked
  as a methodological prerequisite before any GNN comparisons are made.
- **Why three, not one.** Reviewer-defensive: a single Cox PH baseline
  invites the question "did you pick the easy one?" Three baselines with
  explicit roles (north-star / upper-bound / non-linear-flat-feature
  reference) preempt this and let every GNN result be interpreted against
  the comparator that makes the relevant claim hardest to pass.

Source: `results/stage_0_summary.md`, `results/methodological_notes.md` §1,
`results/stage_3_ref_mlp_clinical.json`.

---

## 3. Architecture (≈500 words)

Knob A as the frozen architecture. Each component has a citation. The
section structure mirrors the per-patient forward pass.

- **Per-patient gene graph.** Nodes = 769 genes (subset to per-fold LASSO
  subset, 39–72 per fold; see §4). Edges = STRING PPI ≥700. Patient-
  specific node features = z-score-normalized expression scalar per gene
  (in_dim=1). Topology shared across patients; features patient-specific.
  Cite Vaida 2025 for patient-as-graph paradigm; Madanipour 2024 for
  inductive design.
- **GraphSAGE backbone.** 2 layers, hidden 128, mean aggregator,
  dropout 0.4 between layers. Cite Hamilton 2017 (SAGE original);
  Madanipour 2024 (inductive justification — required for cold METABRIC
  inference without retraining).
- **Pooling.** Global mean pool over per-fold gene set produces patient
  embedding (R^128). Cite Ling 2022 (over-smoothing avoidance for
  ~50-node fold graphs at depth 2).
- **Clinical late-fusion.** Concatenate 7-d (TCGA-internal) or 3-d
  (METABRIC-compatible) clinical vector with patient embedding before
  MLP head. Cite Gao 2021 (gene-only 0.893 → gene+clinical 0.954
  ablation as design-choice precedent).
- **Head.** 2-layer MLP (Linear-128 → ReLU → Dropout-0.4 → Linear-1)
  produces scalar log-hazard per patient.
- **Loss.** Cox partial likelihood (Breslow tie-handling), `logcumsumexp`
  numerically stable form. Cite Cox 1972, Katzman 2018 (DeepSurv).
- **Training.** Adam optimizer, lr=1e-3, weight_decay=1e-4, batch_size=64,
  30 epochs, best-epoch selection by val C-index. Per-fold seed = 42 + fold.
- **What is NOT in knob A.** No multi-head attention, no aux heads,
  no SMOTE on training, no DeepHit, no GAT, no multi-task heads.
  Each of these was tested-and-discarded in the prior attempt's debrief
  or in our Stage 2/3 design (see §6 Discussion future-work for which).

Source: `src/sage_models.py` (`SAGEClinical`), `scripts/03b_*.py`,
`results/architecture_design.md` §2-§3.

---

## 4. Ablation knobs (≈500 words)

Knobs B and C as ablation rows designed to test specific hypotheses, NOT
"elaborations we tried." Each opens with its testable hypothesis.

- **Knob B (Reactome pathway pooling).** Hypothesis: biological-pathway
  grouping adds discriminative signal beyond global mean pool over
  individual genes. Construction: replace `global_mean_pool` with
  per-Reactome-pathway mean pool followed by single-head attention over
  pathways with uniform-init query (zero-vector → softmax(0) = uniform
  attention before training). Per-fold pathway-membership matrix
  built from KG; R5 sparsity sentinel drops pathways with <3 fold genes
  (design-doc threshold 5 lowered to 3 because per-fold gene sets of 39–72
  genes leave 0–4 pathways at threshold 5; 5–19 at threshold 3). Cite
  Choudhry 2025, Vaida 2025.
- **Knob C (BioBERT gene priors).** Hypothesis: LLM-derived gene priors
  add unique information beyond expression. Construction: per-gene
  feature = `expression_z[patient, gene] * biobert_pca32[gene]` —
  multiplicative, not concatenative; in_dim=32 (vs in_dim=1 for knob A).
  PCA(32) of the 768-d BioBERT embeddings on the overlap-genes universe;
  PCA fit on the gene set only (no patient labels). Cite Chen & Zou 2023
  (GenePT precedent; foreshadow that BioBERT may underperform GenePT — we
  did not run GenePT, future work).
- **What is held fixed.** Both knobs use the same per-fold LASSO gene
  set (knob A's gene selection step), the same edge masking, the same
  clinical late-fusion, the same MLP head, the same training schedule.
  Each knob varies exactly one architectural element.
- **Comparator.** Each knob is compared against knob A on identical val
  patients using paired bootstrap (see §5). Internal-only "competitive"
  performance is not sufficient for adoption; external paired-test win is.
- **Design discipline.** No tuning. The simplest version of each knob
  defends the hypothesis; if the simplest version doesn't lift, more
  attention heads or learnable projections won't reveal a buried effect —
  they'll hide whether the architecture works.

Source: `src/sage_models.py` (`SAGEPathwayClinical`), `scripts/03c_*.py`,
`scripts/06d_*.py`, `scripts/06e_*.py`.

---

## 5. Methodological backbone (≈700 words; the chapter's distinguishing section)

Six methodological choices that recur across every result. Each is given
the precise prose treatment it deserves; subsequent Results sections
forward-reference here rather than re-derive.

- **Stratified 5-fold CV.** Joint stratification on (survival_class × OS
  event), seed=42. Why joint stratification: 80%+ TCGA censoring +
  imbalanced bins → random folds give wildly unequal per-fold censoring,
  breaking the Cox loss's well-conditioning assumption.
- **LASSO leakage audit.** The diagnostic that revealed Cox PH HONEST as
  the correct north-star. Per-fold refit of `LassoCV(cv=5)` on raw 60k
  TCGA matrix from training partitions only; +0.072 average leakage cost
  between full-cohort-LASSO (0.7324) and per-fold-LASSO (0.6605). Result
  (the audit table, 5 folds × leaky/honest cidx) is presented in Results
  §X; the methodology lives here.
- **Bootstrap CI machinery.** `src/cindex_bootstrap.py`. Per-fold
  patient-level bootstrap CI (n_boot=1000, alpha=0.05). Paired bootstrap
  delta on identical resampled patients (n_boot=2000) for model-vs-model
  comparison. Harrell's C via `sksurv.metrics.concordance_index_censored`,
  cross-checked against `lifelines.utils.concordance_index` (max |Δ| <
  10⁻⁵ across 5 folds — confirms tie-handling agreement, important
  because subtle tie differences have produced wrong reported numbers
  in published papers).
- **Paired bootstrap as the primary statistic for model-vs-model comparison.**
  Cross-reference `methodological_notes.md` §2 + §7. Why fold-mean and
  paired-bootstrap can disagree on sign (knob A vs knob D worked example:
  fold-mean Δ = −0.006 but paired Δ = +0.027). Why paired is rigorous and
  fold-mean isn't (fold-mean averages five summary statistics on disjoint
  patient sets; paired controls for patient-fold assignment). Convention:
  paired CI lower bound > 0 = real win; CI crossing zero = tie; CI clearly
  < 0 = regression. This convention is load-bearing; every Results table
  uses it.
- **Variance-floor observation.** Stage 0 honest std = 0.049, Stage 2
  minimal GNN std = 0.050, knob A std = 0.043 — same number across
  three independently trained models. Three-seed re-run of Stage 2
  fold-4 yielded std = 0.006. The cohort-level variance floor at this
  event count (~30 per val fold) is ~0.05 c-index regardless of model.
  Single-fold differences are not statistically distinguishable;
  fold-mean-plus-paired-bootstrap is the meaningful comparator.
- **R1 cosine-collapse sentinel.** Catastrophic threshold: cosine > 0.99
  in any fold/epoch (the prior attempt's failure mode). Differentiation
  criterion: cosine_init − cosine_final > 0.02 per fold. Both must pass.
  Necessary because the prior attempt's collapse was the load-bearing
  failure mode of an earlier architecture; we test for it explicitly
  even though our SAGE backbone is structurally distinct.

Source: `results/methodological_notes.md` §1–§4, §7;
`src/cindex_bootstrap.py`; `scripts/00_lasso_audit.py`.

---

## 6. External validation protocol (≈400 words)

What we do at inference time on METABRIC. Every choice is defended;
omissions are flagged.

- **Per-cohort z-normalization.** Each cohort z-normalized to itself
  (mean 0, std 1 per gene per cohort). Joint normalization is rejected
  because it leaks distribution information and is not available at
  deployment time.
- **Gene-set intersection.** Knob A's per-fold LASSO genes intersected
  with METABRIC's available genes (`Hugo_Symbol` column from
  `data_mrna_illumina_microarray.txt`); 84.5% overlap. The full-TCGA-
  trained model's per-fold gene set (50 genes) is what defines the
  inference graph; non-overlap genes are not inferable.
- **Clinical feature mapping.** TCGA 7-clin → 3-clin (`age,
  stage_ordinal, is_female`) is the cohort-compatible subset. METABRIC
  TUMOR_STAGE → stage_ordinal mapping (z-scored on METABRIC).
  HER2/GRADE/ER status NOT used (METABRIC has them; TCGA's preprocessing
  bug rendered them unavailable on the source cohort). Acknowledge:
  this stripped-clinical comparison understates what each architecture
  could deliver if Stage 0.5 ER/PR fix were applied.
- **Two model variants reported.** (a) Full-TCGA-trained (no holdout;
  10% held out only for best-epoch selection): the headline number.
  (b) Per-fold-trained (5 models, each TCGA fold gets its METABRIC
  inference): consistency check; std=0.004 across folds is the
  robustness signal that goes in Results.
- **Cox PH baseline on METABRIC.** Same 3-clinical, same per-fold-LASSO
  gene set, fit on full TCGA (or per fold), inference on METABRIC. The
  apples-to-apples external comparator.
- **Paired bootstrap on identical METABRIC patients.** All model-vs-model
  comparisons on METABRIC use the same patient set, n_boot=2000.
  Knob A vs Cox PH paired Δ = +0.053 [+0.031, +0.076] is the headline.

Source: `scripts/05_metabric_external.py`, `results/stage_5_summary.md`,
`results/methodological_notes.md` §6.

---

## 7. Computational details (≈200 words)

Reproducibility section. Software versions, hardware, wall-time budgets.

- **Software.** PyTorch 2.11, PyTorch Geometric 2.7, lifelines 0.30,
  scikit-survival 0.27, torchmetrics 1.9, NumPy 2.4, scikit-learn 1.8.
  All version-pinned in `pyproject.toml`.
- **Hardware.** Apple Silicon (M-series, 16-core CPU, 20 GB unified
  memory). Stage 0 device benchmark: dense matmul tied CPU vs MPS
  (0.91× ratio); real PyG forward (sparse message-passing dominates)
  CPU 2× faster than MPS. All training reported here used CPU.
- **Reproducibility.** Seeds: per-fold = 42 + fold; bootstrap = 42 + fold;
  numpy + torch both seeded. The CV splits are saved in
  `data/processed/cv_splits.json` and shared across all stages —
  every result above is reproducible from the same splits.
- **Wall time per stage.** Stage 0 baseline + leakage audit + METABRIC
  fetch ~14 min. Stage 1 (LR sanity) ~5 sec. Stage 2 (minimal SAGE)
  28 min. Stage 3 (knobs D, A, B + clinical-only MLP reference) total
  ~70 min. Stage 5 (METABRIC external for knobs A, B, C) ~12 min.
  Grand total: under 2 hours of compute, fully reproducible from raw
  TCGA + METABRIC + the saved CV splits.

Source: `pyproject.toml`, `results/stage_0_summary.md`, all stage logs.

---

## What's deliberately NOT in Methods

To clarify scope and avoid bloat:

- **Path of construction (chronology).** Lives in retrospectives, not Methods.
  Methods describes the final pipeline; retrospectives describe the path.
- **Stage 1 LR sanity.** A 4-line sanity check (LR beats majority class).
  Lives in a footnote or one-line mention in §2 Baselines, not its own
  subsection.
- **Knob D (clinical fusion ablation).** Was the Stage 3a setup with
  leaky-769 gene set; superseded by Knob A. Mentioned once in §4 as
  "Knob A is Knob D plus per-fold LASSO refit," not as its own knob.
- **Stage 2 R1 sentinel as a separate result.** R1 methodology lives in
  §5; the result (sentinel passed) lives in Results §X.
- **Specific bootstrap n_boot tuning, decile choices for calibration,
  pathway-attention figure layout choices.** Live in figure captions
  + per-stage summaries, not Methods.
