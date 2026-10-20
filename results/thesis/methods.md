# Methods

This chapter describes the data, comparators, model, ablation knobs, and
methodological apparatus used to evaluate a graph neural network for
breast-cancer survival prediction on TCGA-BRCA with external validation on
METABRIC. The presentation is reader-needs ordered: each section depends only
on those above it, and a reader proceeding top-to-bottom should not need to
forward-reference. The narrative of how the design evolved across iterations
is recorded in the supplementary retrospectives; what follows is the
final pipeline.

## 1. Data

### 1.1 Cohorts

The internal cohort is TCGA-BRCA (The Cancer Genome Atlas — Breast Invasive
Carcinoma). Of 1,095 patients with HTSeq read counts and clinical records,
21 were excluded for non-positive overall-survival time (OS.time ≤ 0),
leaving n = 1,074 with an event rate of 0.140 (150 deaths). Survival times
are recorded in days; in addition to the continuous (T, E) representation
used for Cox partial-likelihood loss, each patient was assigned to one of
four survival classes (<1y, 1–3y, 3–5y, >5y) used as the LASSO regression
target during gene selection.

The external cohort is METABRIC (Molecular Taxonomy of Breast Cancer
International Consortium), retrieved from the cBioPortal datahub via Git
LFS. Of 2,509 patients in the source release, 1,466 were retained after
filtering on valid OS_MONTHS and valid TUMOR_STAGE; the event rate is
0.561 (824 deaths), reflecting METABRIC's substantially longer follow-up
(median 118 months vs TCGA's 28). OS_MONTHS was converted to days using
30.4375 days per month for cross-cohort consistency.

### 1.2 Gene universe

A 769-gene candidate set, hereafter the *769-gene universe*, was
inherited from the prior preprocessing pipeline. This set was originally
selected by `LassoCV` on the full 1,074-patient cohort regressing the
four-bin survival_class on log-z-normalized expression, then trimmed to
genes with edges in the constructed STRING knowledge graph. Because this
selection used full-cohort labels, it constitutes label leakage at the
gene-universe level, which the present work documents and partially
corrects (§5.1) but does not fully undo. The gene-universe limitation is
stated up front: knob A's per-fold LASSO refit operates *within* the
769-gene universe rather than on the raw 60k-gene matrix, because
attempting the latter produces per-fold subgraphs of 6 edges or fewer
(§5.1) and is intractable for the GNN. Future work that rebuilds the
STRING knowledge graph per fold would lift this limitation.

The gene-graph topology is the STRING protein-protein-interaction network
at high-confidence threshold (combined score ≥ 700; Szklarczyk et al.,
2023), restricted to the 769-gene universe. This produces 49,674
undirected gene-gene edges. The same edge set is shared across all
patients; per-patient variation enters through node features (expression).

### 1.3 Clinical features

Ten TCGA clinical features were retained for internal-only analyses:
age (z-scored), stage_ordinal (0–1 normalized), one-hot stage indicators
(stage_I/II/III/IV), is_female, and three IHC receptor-status columns
(er_signed, pr_signed, her2_signed). The legacy preprocessed file
provided er_signed and pr_signed as all-zero columns — a preprocessing
artifact inherited from the prior pipeline. The three receptor-status
columns were recovered from the cBioPortal legacy `brca_tcga` patient
file (`ER_STATUS_BY_IHC`, `PR_STATUS_BY_IHC`, `IHC_HER2`) and mapped to
signed integers (Positive → +1, Negative → −1, Indeterminate or
unavailable → 0). Recovery covered 1,094 of 1,095 patients; the
recovered features have variances of 0.68, 0.85, and 0.53 respectively,
all comfortably above the 0.01 low-variance filter that previously
dropped the all-zero columns.

For external-validation comparability, a 3-feature subset (age,
stage_ordinal, is_female) was used in any TCGA → METABRIC comparison. The
restriction is per-cohort common-denominator and is reported as such; full
within-TCGA results use the 10-feature variant.

### 1.4 Splits

Stratified 5-fold cross-validation was used throughout, with stratification
on the joint variable (survival_class × OS event). Joint stratification is
necessary because at TCGA-BRCA's 86% censoring rate and four-bin imbalance,
random folds produce wildly unequal per-fold event distributions, breaking
the well-conditioning of Cox partial-likelihood fits. Splits were generated
once with `seed=42` and saved to `data/processed/cv_splits.json`; every
result reported here uses these same splits, making all comparisons
paired-fold by construction.

### 1.5 Scope

This thesis reports on knob A as the frozen architecture and knobs B and C
as planned ablation rows. Earlier intermediate variants (notably knob D, a
precursor without leakage correction) are documented in the supplementary
retrospectives. Architectures explicitly considered and excluded — graph
attention networks (GAT), heterogeneous graph transformers, retrieval-
augmented variants, multi-omics extensions, multi-task auxiliary heads —
are addressed in the Discussion's future-work section.

## 2. Baselines

Three baselines are reported throughout, each occupying a distinct role.
The triple-baseline structure is itself a methodological choice: a single
Cox PH comparator invites the question of whether the easiest one was
chosen, and the leaky-vs-honest distinction is load-bearing for every
result that follows.

**Cox PH HONEST (north-star, 0.6662 ± 0.054).** The methodologically
defensible Cox proportional-hazards baseline. Per fold, `LassoCV` was
refit on the raw 60,660-gene HTSeq matrix using only that fold's training
patients (with the same `survival_class` regression target as the prior
pipeline), producing a per-fold gene set of 61–282 genes. PCA(100) of the
per-fold-z-normalized expression of those genes plus the 10 clinical
features were fit by `CoxPHFitter` with ridge penalizer 0.5 (Davidson-
Pilon 2019). Ridge was selected by sweeping {0.001, 0.01, 0.1, 0.5, 1, 2, 5}
and choosing the value that minimised cross-fold standard deviation
without sacrificing mean C-index. The honest baseline is the threshold
the present work must clear to claim methodological progress.

**Cox PH LEAKY (upper bound, 0.7362 ± 0.017).** The same Cox PH formulation
applied to the 769-gene leaky universe. Reported as an upper bound because
its gene selection peeked at every patient's label including those in
val folds. The +0.070 c-index gap between LEAKY and HONEST is the leakage-
audit's headline cost (§5.1). Subsequent results are interpreted against
HONEST; LEAKY is reported for context and as a sanity check that the
preprocessing reproduces the prior pipeline's number.

**Prior `0.748` figure debunked.** The legacy pipeline reported a Cox PH
C-index of 0.748. A diagnostic configuration sweep
(`scripts/00_baseline_diag.py`) over n_components ∈ {20, 50, 100, 200} and
penalizer ∈ {0.001, 0.01, 0.1, 0.5, 1, 2, 5}, with and without clinical
features, did not reproduce 0.748 with any sensible configuration. The
prior figure was traced to a non-standard feature construction (mean
across the BioBERT 768-dimensional embedding axis followed by selection
of the first 50 columns in storage order, with `penalizer=0.1`) that
does not correspond to Cox PH on a defensible feature set. The 0.748
number is treated as non-reproducible and excluded from comparisons.

**MLP clinical-only (non-linear flat-feature reference, 0.7158 ± 0.034).**
A two-layer MLP (Linear-128 → ReLU → Dropout-0.4 → Linear-1) on the
10-clinical-feature vector trained with the same Cox partial-likelihood
loss and same optimization schedule as the GNN. This baseline disentangles
"non-linear use of clinical features" from "gene-graph signal": without
it, any GNN lift over linear Cox PH HONEST conflates the two. Reporting
this reference separates the gene-graph contribution from the non-linear-
clinical contribution to the GNN's overall lift.

## 3. Architecture

The model takes a per-patient gene graph as input and outputs a scalar
log-hazard. The architecture, denoted knob A, is held fixed across all
internal and external evaluations; knobs B and C (§4) modify single
elements of it as ablation rows.

**Per-patient gene graph.** Each patient is represented as a graph in
which nodes are genes from a per-fold LASSO subset of the 769-gene
universe (§5.1), edges are the corresponding subset of STRING PPI edges
(§1.2), and node features are scalar z-normalized expression values
(in_dim = 1). The patient-as-graph paradigm with shared topology and
patient-specific node features follows Vaida et al. (2025) and Gao et al.
(2021); the inductive choice — same topology across train, val, and
external inference — is required for cold METABRIC inference without
retraining, following Madanipour et al. (2024).

**Backbone.** Two GraphSAGE convolution layers (Hamilton et al., 2017),
each producing a 128-dimensional hidden representation, with mean
neighborhood aggregation, ReLU activation, and dropout at p = 0.4
between layers. Two layers were selected to provide two-hop neighborhood
reach while avoiding the oversmoothing pattern Ling et al. (2022)
documented in deeper GNNs on small graphs of this scale (39–72 nodes per
fold). Mean aggregation was preferred over attention
because the prior attempt's GAT backbone collapsed at random initialization
across folds; SAGE with mean aggregation is initialization-stable on this
graph size, as confirmed by the embedding-collapse sentinel (§5.4).

**Pooling.** Global mean pool over the per-fold gene set produces a
patient embedding in R^128. The pool is unweighted; pathway-attention
pooling is reserved for knob B (§4) as an ablation row, not adopted in the
default architecture.

**Clinical late-fusion.** The patient embedding is concatenated with the
clinical feature vector (10-dim internally, 3-dim for METABRIC-compatible
comparisons) before the MLP head. Late fusion replicates the design of
Gao et al. (2021), whose ablation showed clinical contributing measurable
lift in TCGA prognosis. Concatenation rather than FiLM-style conditioning
or early fusion was chosen for transparency: the clinical contribution is
isolable and can be measured directly via the clinical-only MLP reference.

**Head and loss.** A two-layer MLP (Linear-128 → ReLU → Dropout-0.4 →
Linear-1) maps the fused vector to a scalar log-hazard. Training optimizes
the Cox partial-likelihood loss (Cox 1972) in its Breslow tie-handled
form, computed via `torch.logcumsumexp` for numerical stability; a single
forward pass produces predicted log-hazards for the entire batch, and the
risk-set sum is computed by sorting batch members by descending T. This
follows the DeepSurv formulation (Katzman et al., 2018).

**Optimization.** Adam optimizer (Kingma and Ba, 2015), learning rate
1e-3, weight decay 1e-4, batch size 64, 30 epochs. The model with the
highest validation C-index across the 30 epochs is retained per fold.
Validation C-index is computed on each fold's held-out 20% partition.
Per-fold seeds are 42 + fold for both NumPy and PyTorch.

## 4. Ablation knobs

Knobs B and C are ablation rows designed to test specific hypotheses
about which architectural elements contribute discriminative signal. Each
knob varies exactly one element relative to knob A; all other choices
(per-fold LASSO gene set, edge set, training schedule, clinical fusion,
MLP head) are held fixed. The hypothesis-prediction-result reporting
structure (Vavekanand and Liang, 2026) frames negative results as
informative rather than as failed attempts.

**Knob B (Reactome pathway pooling).** *Hypothesis:* biological-pathway
grouping of gene-level embeddings adds discriminative signal beyond global
mean pool. *Prior prediction:* positive effect on C-index, with the
additional benefit of clinically-interpretable attention weights, motivated
by Choudhry et al. (2025) and Vaida et al. (2025). *Construction:* after
the two SAGE layers, gene-node embeddings are aggregated per Reactome
pathway by mean over those genes belonging to the pathway (membership
matrix from MSigDB Reactome subset). Single-head dot-product attention
with a uniform-init query (zero vector, so softmax over scores is uniform
before training) weights pathway representations into the patient
embedding. The R5 sparsity sentinel (§5.4) drops pathways with fewer than
three fold genes; the design-doc's threshold of five was lowered because
per-fold gene sets of 39–72 leave 0–4 pathways at threshold five (zero is
degenerate). At threshold three, 5–19 pathways are retained per fold.

**Knob C (BioBERT-derived gene priors).** *Hypothesis:* LLM-derived gene
priors capture biological context that scalar expression alone misses,
adding unique information to per-gene node features. *Prior prediction:*
small positive effect on C-index, motivated by Chen and Zou (2023), who
reported that GenePT (GPT-3.5) gene embeddings outperformed BioBERT and
expression-trained foundation models on gene-gene tasks; BioBERT was
chosen here as the cached available alternative. *Construction:* BioBERT
768-dimensional embeddings (Lee et al., 2020) for each gene in the 769-gene
were projected to 32 dimensions by PCA fit on the gene set only (no
patient labels). The per-gene node feature is the multiplicative
combination `expression_z[patient, gene] × biobert_pca32[gene]` (in_dim =
32), not concatenation; the multiplicative form preserves the convention
that absent expression yields zero contribution. The SAGE backbone
operates on the 32-dimensional input; all other architectural elements
match knob A.

**Comparator.** Each knob's primary acceptance test is paired-bootstrap
delta on identical val patients against knob A, evaluated on both TCGA
internally and METABRIC externally; §5.3 motivates this choice of
statistic, and knob C provides the worked example for why external
paired tests are the rigorous comparator.

## 5. Methodological backbone

### 5.1 LASSO leakage audit

The prior pipeline's gene selection (`src/preprocessing.py:191`) used
`LassoCV(cv=5).fit(X, y)` on the full TCGA cohort, producing a 769-gene
universe whose construction peeked at every patient's `survival_class`
label. The leakage audit quantifies the resulting bias by per-fold refit:
within each of the 5 stratified train partitions (n ≈ 859 each), `LassoCV`
was rerun on the raw 60,660-gene HTSeq matrix after low-count filtering
(min_total_counts ≥ 1000) and per-fold log-z-normalization. The resulting
per-fold gene sets contained 61–282 non-zero-coefficient genes, on which a
fold-matched Cox PH (PCA(100) + 10 clinical, penalizer 0.5) was fit and
evaluated on the held-out val patients. Across folds, the per-fold-honest
Cox PH achieved 0.6662 ± 0.054 (mean C-index ± std across folds), versus
the leaky-cohort Cox PH at 0.7362 ± 0.017 — a +0.070 c-index leakage cost
on average. This audit established Cox PH HONEST = 0.6662 as the
methodological north-star (§2).

A fully-honest GNN counterpart — running per-fold LASSO on the raw 60k
matrix, then rebuilding the STRING knowledge graph for each fold's gene
set — was attempted but found intractable: per-fold-LASSO ∩ 769-gene
overlap was 14% on fold 0 (16 of 116 LASSO genes), leaving only 6 edges
in the masked graph. Knob A therefore operates on the milder within-769
LASSO refit: per-fold `LassoCV` on the 769-gene universe (rather than
60k) selects 39–72 genes per fold, and the STRING edge set is masked to
include only edges where both endpoints lie in the per-fold subset. This
removes the per-fold-selection layer of leakage at the survival prediction
step but does not undo the universe-construction leakage. The gene
universe limitation is restated explicitly in §1.2.

### 5.2 Bootstrap CI machinery and Harrell's-C cross-check

C-index point estimates on a single val fold of 215 patients with ~30
events are uncertain; per-fold patient-level bootstrap CIs (n_boot = 1000,
α = 0.05) on the leaky Cox PH baseline have a mean width of approximately
0.21, demonstrating that single-fold differences between models smaller
than ~0.10 are not statistically distinguishable. Pooled cohort CIs
(n_boot = 2000 over the n = 1074 pooled out-of-fold predictions) have
width ~0.09. The bootstrap utilities are implemented in
`src/cindex_bootstrap.py`.

Harrell's C is computed via `sksurv.metrics.concordance_index_censored`
(Pölsterl 2020), which uses the formal definition with explicit tie
handling. As a cross-check against subtle tie-handling differences that
can produce small numerical discrepancies between implementations, every
reported C-index was independently computed via
`lifelines.utils.concordance_index` (Davidson-Pilon 2019); the maximum
absolute difference across all 5 Cox PH HONEST val folds was below 10⁻⁵,
confirming agreement within numerical precision.

### 5.3 Paired-bootstrap on identical patients: a worked example

Two natural statistics for model-vs-model C-index comparison can disagree
on sign. Knob A versus its leakage-uncorrected precursor (knob D) provides
the canonical worked example. The fold-mean delta is computed by averaging
each model's per-fold C-index across the 5 val folds and subtracting:
Δ_fold-mean(A − D) = −0.006. The paired-bootstrap delta is computed by
pooling all 1,074 val log-hazard predictions (each patient appearing in
exactly one val fold), resampling patients with replacement (n_boot =
2000), scoring both models on the *same* resampled patient set, and
taking the difference: Δ_paired(A − D) = +0.027 with 95% CI [−0.002,
+0.056] and P(A ≤ D) = 0.033.

The two statistics differ on sign. The fold-mean averages five summary
statistics computed on five disjoint patient sets of roughly equal size;
small per-fold imbalances in patient assignment can shift the mean
non-trivially even when one model dominates the other patient-by-patient.
The paired bootstrap controls for patient-fold assignment by always
scoring both models on the identical resampled patient set, with identical
T and E pairs, in equal sample sizes. It eliminates the fold-assignment
nuisance variable that fold-mean leaves uncontrolled.

This thesis adopts the convention that paired-bootstrap delta on identical
val patients is the **primary** statistic for any model-vs-model
comparison; per-fold C-indices and fold-means are reported alongside for
transparency but are not used for inference. A paired-CI lower bound
strictly above zero is the threshold for "real win"; CI crossing zero is
"tie / no measurable difference"; CI clearly below zero is "regression."
The same statistic, computed on identical METABRIC patients (§6), is the
primary test for external claims and is the only test with sufficient
event count (824) to detect small effects (knob B and knob C, §4).

### 5.4 Sentinels and fold-variance forensic

**R1 (embedding-collapse sentinel).** The prior attempt's load-bearing
failure mode was patient-embedding collapse — every patient mapping to
near-identical pooled vectors, so the head sees no signal. Two thresholds
guard against this. *Catastrophic:* val mean pairwise cosine > 0.99 in
any fold or epoch. *Differentiation:* `cosine_init − cosine_final > 0.02`
per fold, where `cosine_init` is computed before any optimization steps.
Both must pass. The pre-training cosine baseline of approximately 0.94–0.97
arises from the architecture's ReLU activations (all node activations
non-negative) and global mean pooling (averaging toward the population
mean), so an absolute threshold of 0.95 from the original design
specification was recalibrated to these two operationalizations. Both
thresholds passed across all five folds for knobs A, B, and C; the prior
attempt's failure mode does not recur in the present architecture.

**R5 (pathway-sparsity sentinel).** Reactome pathway pooling (knob B)
becomes degenerate when per-fold gene sets are too sparse to populate
pathways. Pathways with fewer than three fold genes are dropped from the
pool; a fold is flagged degenerate if fewer than five pathways are
retained. The threshold of three was selected so that knob B has
non-trivial pathway structure on every fold of the present cohort (5–19
retained); raising the threshold to five would render fold 2 degenerate
(zero pathways).

**Fold-variance forensic.** Cross-fold C-index standard deviations of
approximately 0.05 were observed across multiple independently-trained
models, suggesting a cohort-driven floor rather than model variance. A
three-seed re-run of the worst-performing fold yielded a seed-conditioned
std an order of magnitude smaller, ruling out seed-pathology as the
explanation. The variance floor is attributed to per-fold event sparsity
(~30 events per val fold) and is treated as a property of the cohort.
Mean C-index across folds plus paired-bootstrap CI is the meaningful
comparator throughout.

## 6. External validation protocol

The external-validation evaluation applies the same model trained on full
TCGA data (knob A, with per-fold or full-cohort training as specified) to
METABRIC patients with no retraining, refitting, or recalibration on
METABRIC data.

**Cohort feature alignment.** Expression z-normalization is performed
**per cohort** — TCGA expression z-scored using TCGA training-set
statistics, METABRIC expression z-scored using METABRIC's own statistics —
so that distribution information from one cohort does not leak into the
other and the protocol matches a deployment scenario in which only the
target-cohort data is available at inference. Joint normalization is
explicitly rejected as it would constitute a form of test-set leakage at
the distribution level, leaking METABRIC's distributional structure into
the TCGA training pipeline.

**Gene-set intersection.** The 769 leaky-universe genes intersect
METABRIC's `data_mrna_illumina_microarray.txt` at 650 (84.5%); the missing
119 are predominantly AC* lncRNAs not represented on the Illumina array.
At inference time, METABRIC patients are scored using only the per-fold
gene set restricted to the 650-gene overlap, with the corresponding
edge subset.

**Clinical feature alignment.** TCGA's 10-feature clinical vector is
reduced to the 3-feature subset (age, stage_ordinal, is_female) shared
with METABRIC. METABRIC's age is z-scored on METABRIC; TUMOR_STAGE is
mapped to the same ordinal scale and z-scored on METABRIC; sex is uniformly
1 (female) for METABRIC. ER, HER2, and grade columns are present in
METABRIC but were not used because the corresponding TCGA fields are
unavailable in the inherited preprocessing (§1.3); using them on
METABRIC alone would conflate cohort with feature-set effects.

**Two model variants.** The headline external number is from a
full-TCGA-trained knob A model: 90% of TCGA used for training, 10% held
out only for best-epoch selection (no METABRIC data ever influences
training). For consistency assessment, five per-fold-trained knob A
models (each trained on its own 4-fold training split) were also run
through METABRIC inference; the per-fold METABRIC C-indices cluster at
0.634–0.645 with std = 0.004, confirming robustness to the choice of
TCGA training partition.

**Cox PH external comparator.** A Cox PH baseline using the same 3-clinical
feature set, the same per-fold-LASSO gene set, and the same training
patients is fit on full TCGA and applied to METABRIC. This provides the
matched linear comparator for the GNN's external claim.

**Paired test.** All model-vs-model comparisons on METABRIC use
paired-bootstrap delta on identical resampled METABRIC patients
(n_boot = 2000); the n = 1466 / 824-event sample size gives the test
statistical power to detect effects below 0.02 c-index, sufficient to
distinguish knob B and knob C from knob A (§4) — distinctions that are
underpowered on TCGA's 150 events.

## 7. Computational details

**Software.** PyTorch 2.11.0, PyTorch Geometric 2.7.0, lifelines 0.30.0,
scikit-survival 0.27.0, torchmetrics 1.9.0, NumPy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0, on Python 3.13.1. All version constraints are pinned
in `pyproject.toml`. Reactome pathway membership is derived from the
MSigDB Reactome subset packaged with the inherited preprocessing artifacts.

**Hardware.** Apple Silicon (16-core M-series CPU, 20 GB unified memory).
A device benchmark on a representative dense matmul workload showed CPU
and MPS within 10% of each other; a Phase-1 ablation on the actual PyG
forward pass with sparse message passing found CPU approximately twice as
fast as MPS, attributable to MPS kernel-launch overhead dominating the
per-edge compute. All training and inference reported here ran on CPU.

**Reproducibility.** Random seeds are 42 + fold for per-fold operations
(NumPy and PyTorch both seeded), and 42 + fold for bootstrap resamples.
The 5-fold stratified splits are stored in `data/processed/cv_splits.json`
and shared across all stages, making every result reproducible from the
same patient assignments. All model weights, val predictions, and
bootstrap samples are saved per-stage in JSON for downstream re-analysis.
All scripts, configurations, and saved predictions are available at
https://github.com/ELDERGARLIC/cancer-prognosis-prediction; the full
pipeline can be reproduced
end-to-end from `scripts/00_baseline.py` through
`scripts/05_metabric_external.py`.

**Wall-clock per stage on the documented hardware.** The Cox PH leakage
audit (per-fold LASSO refit on the raw 60k-gene matrix) ran for 12.3 min;
the minimal-architecture SAGE on TCGA (5 folds × 30 epochs) for 28.3 min;
the clinical-only MLP reference for 4 s; the leakage-corrupted GNN+clinical
ablation (knob D) for 29.7 min; knob A (per-fold LASSO + clinical) for
8.0 min, faster than knob D because per-fold subgraphs are 30–70× smaller;
knob B (pathway pool) for 8.3 min; the METABRIC external for knobs A, B,
and C combined for 11.6 min. Total compute for the full pipeline is
under 100 minutes on the documented hardware.
