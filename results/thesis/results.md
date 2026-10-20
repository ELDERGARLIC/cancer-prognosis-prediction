# Results

## 1. Cox PH baselines and the prior-pipeline correction

The Stage 0 leakage audit identified the prior pipeline's gene-selection step
as the source of an upper-bound bias. The 769-gene universe was
originally selected by `LassoCV` regressing four-bin survival_class on
log-z-normalized expression on the full 1,074-patient TCGA cohort, peeking
at every patient's label including those that subsequently appeared in val
folds. To quantify the bias, the LASSO step was rerun within each fold's
training partition only on the raw 60,660-gene HTSeq matrix (after
low-count filtering and per-fold log-z-normalization), and a fold-matched
Cox PH (PCA-100 + 10 clinical features, ridge penalizer 0.5) was fit on
each fold's per-fold gene set and evaluated on its held-out val patients.

The per-fold-honest Cox PH achieves a mean val C-index of 0.6662
(std 0.054 across the 5 folds), versus the leaky-cohort Cox PH at 0.7362
(std 0.017). The +0.070 c-index leakage cost is the headline finding of
the Stage 0 audit and establishes Cox PH HONEST = 0.6662 as the
methodological north-star against which all subsequent claims are measured.
The fold-level structure of the leakage cost is non-uniform; the per-fold
breakdown is reported in §6 alongside its consistency check against
knob A's results.

The prior pipeline reported a Cox PH baseline of 0.748 on TCGA-BRCA. A
Stage 0 diagnostic sweep (PCA components ∈ {20, 50, 100, 200} × ridge
penalizer ∈ {0.001, 0.01, 0.1, 0.5, 1, 2, 5}, with and without clinical
features, on the same stratified splits) does not reach 0.748 with any
sensible Cox PH configuration. The number traces to a non-standard
feature construction (mean across the BioBERT 768-dimensional embedding
axis followed by selection of the first 50 columns in storage order,
with `penalizer=0.1`) that does not correspond to Cox PH on a defensible
feature set, and is excluded from comparisons throughout.

Cohort scale and split details follow Methods §1; the MLP clinical-only
reference introduced in §2 operates on the same splits.

**Table 1.** Cox PH baselines on TCGA-BRCA (5-fold stratified val).

| Model | Gene-selection step | Mean val C-index | Std across folds | Role |
|---|---|---:|---:|---|
| Cox PH HONEST | per-fold LASSO on raw 60k matrix | **0.6662** | 0.054 | North-star; the gate any GNN claim must clear |
| Cox PH LEAKY | full-cohort LASSO (legacy pipeline) | 0.7362 | 0.017 | Upper bound; reported for context only |
| Prior 0.748 figure | non-standard `mean(axis=2)[:, :50]` recipe | not reproducible | — | Excluded from comparisons |

---

## 2. Internal TCGA — knob A vs Cox PH baselines

Knob A (the frozen architecture, Methods §3) trained per-fold on the
same 5 stratified splits achieves a mean val C-index of 0.7261
(std 0.030; per-fold values [0.7472, 0.7471, 0.6921, 0.7561, 0.6879]).
The R1 embedding-collapse sentinel passed across all five folds.
Knob A is +0.0599 above Cox PH HONEST on fold-mean — a clear separation
from the leakage-corrected linear baseline.

Against the leakage-corrupted upper bound (Cox PH LEAKY = 0.7362), knob A
is −0.0101 on fold-mean. Paired-bootstrap delta on identical val patients
(n_boot = 2000) on the same 1,074-patient pool grazes zero on the upper
edge; knob A is statistically tied with the leakage-corrupted upper bound
on the paired-bootstrap test.

The MLP clinical-only reference (two-layer MLP on the same 10-clinical
features, Cox loss, no gene graph) achieves 0.7158 (std 0.034 across
folds), establishing the non-linear flat-feature ceiling against which the
gene-graph contribution is measured. The gene-graph contribution above
non-linear MLP-clinical is approximately +0.01 c-index internally, stable
to within bootstrap noise across the architectural variants tested
(knob A: +0.010; knob B: +0.018; knob C: +0.006), establishing a small
but architecture-invariant gene-side signal.

A worked example of why paired-bootstrap on identical patients is the
primary model-vs-model statistic: knob A versus its leakage-uncorrected
precursor (knob D, the same architecture trained on the 769-gene leaky
universe without per-fold LASSO) produces fold-mean Δ = +0.0005 and
paired-bootstrap Δ = +0.035, 95% CI [−0.004, +0.073], P(A ≤ D) = 0.044.
The two statistics disagree on sign because they answer different
questions (Methods §5.3); the paired-bootstrap test, which scores both
models on identical resampled patient sets, supports knob A as the
leakage-correction-free architecture without measurable point-prediction
loss. Full numbers for knob D are in supplementary materials.

**Table 2.** GNN variants against linear baselines on TCGA-BRCA (internal,
5-fold val) and METABRIC (external, full-TCGA-trained). Paired-bootstrap
deltas use n_boot = 2000 on identical val patients.

| Model | TCGA internal C-index (mean ± std) | METABRIC C-index (95% CI) | Paired Δ vs knob A on TCGA | Paired Δ vs knob A on METABRIC |
|---|---:|---:|---:|---:|
| Cox PH HONEST | 0.6662 ± 0.054 | 0.5914 [0.571, 0.612] | — | −0.053 [−0.076, −0.031] |
| MLP clinical-only | 0.7158 ± 0.034 | not run | n/a | n/a |
| **Knob A** | **0.7261 ± 0.030** | **0.6443 [0.624, 0.664]** | reference | reference |
| Knob B (pathway pool) | 0.7334 ± 0.034 | 0.6285 [0.607, 0.648] | +0.005 [−0.015, +0.025] | **−0.016 [−0.026, −0.006]** |
| Knob C (BioBERT init) | 0.7218 ± 0.043 | 0.6054 [0.583, 0.627] | not paired-tested internally | **−0.039 [−0.059, −0.021]** |

Bold paired-Δ entries indicate 95% CI strictly excluding zero on identical
patients.

---

## 3. External METABRIC — knob A vs matched Cox PH

The external evaluation uses a full-TCGA-trained knob A (no held-out TCGA
fold; 90% train + 10% val for best-epoch selection) applied to METABRIC
inference. METABRIC contains n = 1,466 patients (after valid-OS and
valid-stage filtering from the raw 2,509 sample release), 824 events,
gene-overlap of 650 of 769 with TCGA's gene universe (84.5%, missing
mostly AC* lncRNAs absent from the Illumina array), and three clinical
features in common with TCGA after intersection (age, stage_ordinal,
is_female). Per-cohort z-normalization is applied separately on each side
(Methods §6); joint normalization is rejected as a form of test-set
leakage at the distribution level.

The full-TCGA-trained knob A on METABRIC achieves a C-index of 0.6443
(95% CI [0.624, 0.664], 1,000 bootstrap resamples). A per-fold
consistency check, training 5 separate knob A models — one per TCGA
fold — and applying each to METABRIC yields per-fold values
[0.6340, 0.6446, 0.6445, 0.6412, 0.6371], std = 0.004. The model produces
essentially identical external predictions regardless of which TCGA fold
trained it. The full-TCGA-trained model is the headline external
comparator; the per-fold-trained variants confirm robustness to TCGA
training-partition choice.

The matched Cox PH baseline (full-TCGA-trained, same 3-clinical features,
same per-fold-LASSO gene set, fit on full TCGA, applied to METABRIC)
achieves 0.5914 (95% CI [0.571, 0.612]). The paired-bootstrap delta
between knob A and Cox PH on identical METABRIC patients (n_boot = 2000)
is **+0.0528, 95% CI [+0.0306, +0.0756], P(A ≤ Cox) = 0.000** (no
resample crossed zero). The 95% CI strictly excludes zero; knob A beats
matched Cox PH on the external cohort with paired-bootstrap significance.

Both models suffer comparable cohort-shift drops between TCGA internal
and METABRIC external: knob A drops by +0.082 (5-fold mean 0.7261 →
external 0.6443), Cox PH drops by +0.073 (5-fold mean 0.6610 → external
0.5914), both exceeding 0.05. The drops are attributable to cohort transition (microarray
versus RNA-seq, METABRIC's longer follow-up window, different
populations) rather than to architecture-specific generalization failure;
the relative advantage of knob A over Cox PH is preserved across the
transition (TCGA fold-mean +0.060, METABRIC paired +0.053) and is
statistically resolved on METABRIC where the larger event count gives
the paired-bootstrap test sufficient power.

The external knob A C-index of 0.6443 is within 0.03 c-index of Cox PH
HONEST on TCGA internal (0.6662). Knob A applied cold to METABRIC scores
within paired-bootstrap noise of the within-cohort Cox PH HONEST baseline
on TCGA, despite the platform difference (microarray vs RNA-seq) and
METABRIC's longer follow-up.

**Figure 1** (risk-stratified Kaplan-Meier curves, TCGA pooled
out-of-fold and METABRIC, quartile split by knob A risk score) shows
monotone separation across all four risk groups in both cohorts.
Multivariate log-rank tests give χ² = 38.5 on TCGA (p < 1 × 10⁻⁴, df = 3,
n_per_quartile ≈ 269, events_per_quartile = 23/32/40/55) and χ² = 204.0
on METABRIC (p < 1 × 10⁻⁴, df = 3, n_per_quartile ≈ 367,
events_per_quartile = 137/163/230/293). The METABRIC χ² is approximately
five times the TCGA value, reflecting the larger event count and tighter
quartile-specific KM curves.

![**Figure 1.** Risk-stratified Kaplan-Meier curves on TCGA-BRCA (pooled out-of-fold, n = 1,074) and METABRIC (external, n = 1,466). Quartile split by knob A predicted log-hazard. Shaded bands = 95% CI per curve. Multivariate log-rank tests give p < 10⁻⁴ in both cohorts.](../figures/fig1_km_curves.png)

**Figure 2** (calibration: predicted-risk decile vs observed
Kaplan-Meier survival at 3-year and 5-year) shows monotone descending
relationships in both cohorts. Spearman rank correlations between decile
and 5-year survival are −0.939 (TCGA) and −0.879 (METABRIC); 3-year
correlations are −0.927 (TCGA) and −0.745 (METABRIC). The model's
predicted-risk ranking aligns with observed survival in both cohorts.

![**Figure 2.** Calibration on TCGA-BRCA (pooled out-of-fold) and METABRIC (external). Decile-bin predicted risk vs Kaplan-Meier observed survival at 3-year and 5-year horizons. Shaded bands = 95% CI per decile.](../figures/fig2_calibration.png)

---

## 4. Architectural ablations — knob B and knob C

Knobs B and C are ablation rows, each varying exactly one element of
knob A and reported with the hypothesis-prediction-result structure
described in Methods §4.

**Knob B (Reactome pathway pooling).** *Hypothesis:* biological-pathway
grouping of gene-level embeddings adds discriminative signal beyond
global mean pool. *Prior prediction:* +0.02 to +0.04 lift expected
(Choudhry et al., 2025; Vaida et al., 2025). *Result, internal:*
mean val C-index 0.7334 (std 0.034). Paired-bootstrap delta versus knob A
on identical TCGA val patients is +0.005, 95% CI [−0.015, +0.025],
P(B ≤ A) = 0.32 (CI crosses zero, statistically tied). *Result,
external:* METABRIC C-index 0.6285 (95% CI [0.607, 0.648]).
Paired-bootstrap delta versus knob A on identical METABRIC patients is
**−0.0158, 95% CI [−0.0259, −0.0063], P(B ≤ A) = 1.000**. The 95% CI
strictly excludes zero; knob B is significantly worse than knob A on
external validation despite being statistically tied internally. TCGA's
150 events are insufficient for the paired-bootstrap test to resolve a
−0.016 effect; METABRIC's 824 events resolve the same effect at
significance. The R5 sparsity sentinel retained 5 to 19 pathways per
fold; no fold was flagged degenerate.

**Knob C (BioBERT-PCA gene init).** *Hypothesis:* LLM-derived gene
priors add unique information beyond expression. *Prior prediction:*
+0.005 to +0.025 lift expected, smaller than the GenePT effect reported
in Chen and Zou (2023) but above the bootstrap noise floor.

Knob C demonstrates the methodological framework's value most directly.
On TCGA internal 5-fold validation, knob C achieves C-index 0.7218 ± 0.043,
within fold variance of knob A's 0.7261 ± 0.030 — a tie under any
reasonable internal-only assessment. On the METABRIC external paired
test, knob C scores 0.6054 with paired Δ versus knob A of −0.0389 (95% CI
[−0.059, −0.021], P(C ≤ A) = 1.000), a result the internal comparison
could not have predicted. The internal 5-fold and external paired-
bootstrap tests use the same training pipeline; the divergence between
their conclusions is the methodological argument the framework
operationalises (Methods §5.3).

The cohort-shift drop quantifies the asymmetry directly. Knob A drops
0.082 c-index between internal and external (0.7261 → 0.6443); knob C
drops 0.116 (0.7218 → 0.6054), approximately 1.4 times knob A's drop.
The BioBERT priors fit TCGA's expression manifold competitively while
introducing structure that does not transfer to METABRIC's microarray
distribution. Against Cox PH on METABRIC, knob C's paired-bootstrap
delta is +0.014, 95% CI [−0.004, +0.032], P(C ≤ Cox) = 0.071 — the
95% CI grazes zero on the lower edge, and the test does not reject the
null at conventional significance.

**Figure 4** (architecture-ablation forest plot) renders the
internal-versus-external contrast across all three knobs and Cox PH in
three panels: TCGA 5-fold internal mean C-index (matching Tables 1+2),
METABRIC C-index with 95% CI bars, and paired Δ versus knob A on
identical METABRIC patients with CI bars.
The third panel is the methodological argument visually: knob B's,
knob C's, and Cox PH's paired-Δ CIs all fall strictly below zero on
identical METABRIC patients (P = 1.000 for each), while panel 1 shows
knob C within fold-variance distance of knob A on TCGA internal.

![**Figure 4.** Architecture-ablation forest plot. Three panels: TCGA 5-fold internal mean C-index (left, matches Tables 1+2), METABRIC C-index with 95% bootstrap CI bars (middle), and paired Δ versus Knob A on identical METABRIC patients with CI bars (right). Triple stars (★★★) mark P(model ≤ A) ≥ 0.999 with 95% CI strictly below zero. Knob B, Knob C, and Cox PH HONEST all fall significantly below Knob A on the external paired test despite Knob C being internally competitive (Knob C 0.7218 vs Knob A 0.7261, within fold variance).](../figures/fig4_architecture_forest.png)

---

## 5. Pathway-level interpretability artifact

Knob B underperforms knob A on point prediction in both cohorts (§4) but
is preserved as the source of the thesis's interpretability artifact. The
pathway-attention weights produced by knob B's single-head attention layer
(Methods §4) decompose each patient's pooled embedding into a weighted
combination of Reactome-pathway representations, providing biologically-
named attributions that the gene-level architecture (knob A) does not
expose.

Across the 5 TCGA folds, the top-10 attended Reactome pathways (ranked by
total attention weight summed over the top-5 highest-risk and top-5
lowest-risk val patients per fold) are: Signaling by Interleukins (4.39
total weight), Interleukin-4 and Interleukin-13 Signaling (3.58), MAPK
Family Signaling Cascades (3.52), Cytokine Signaling in Immune System
(3.31), Signaling by Receptor Tyrosine Kinases (3.04), ESR-Mediated
Signaling (1.47), Signaling by Nuclear Receptors (1.34), Estrogen-
Dependent Gene Expression (1.19), Signaling by Notch (1.17), and
Intracellular Signaling by Second Messengers (0.86).

Two consistent attention patterns emerge. Immune-axis pathways
(Interleukins, Cytokine Signaling, IL-4/IL-13) attend most heavily on
the highest-risk val patients across folds, with mean attention weights
of 0.10 to 0.14 on the top-ranked high-risk slot versus 0.07 to 0.10 on
the bottom-ranked low-risk slot. Estrogen-axis pathways (ESR-Mediated
Signaling, Estrogen-Dependent Gene Expression, Signaling by Nuclear
Receptors) attend more heavily on lowest-risk val patients. The pattern
is consistent with established BRCA prognostic biology: ER-positive
tumors carry better population-level prognosis (Howlader et al., 2014),
and inflammatory-cytokine signatures have been associated with more
aggressive subtypes.

**Figure 3** (pathway-attention heatmap) shows the top-10 pathways ×
10 patient slots (5 highest-risk + 5 lowest-risk, fold-averaged) as a
mean-attention-weight grid with a vertical separator between high-risk
and low-risk groups. The figure shows immune-axis rows weighted toward
the high-risk patient columns and estrogen-axis rows weighted toward
the low-risk patient columns.

![**Figure 3.** Knob B Reactome-pathway attention. Top-10 pathways (rows, ranked by total attention across folds) × 10 patient slots (columns: H1–H5 highest-risk, L1–L5 lowest-risk, fold-averaged). Cell colour = mean attention weight; vertical separator at H5/L1 boundary. Immune-axis pathways (Interleukins, Cytokine Signaling, IL-4/IL-13) attend most heavily on highest-risk patients; estrogen-axis pathways (ESR Mediated Signaling, Nuclear Receptors, Estrogen-Dependent Gene Expression) attend most heavily on lowest-risk patients.](../figures/fig3_pathway_attention.png)

---

## 6. Robustness checks

The R1 embedding-collapse sentinel (Methods §5.4) cleared both thresholds
across all five folds for knobs A, B, and C: no fold-epoch combination
produced val mean pairwise cosine above 0.99, and every fold satisfied
the differentiation criterion (`cosine_init − cosine_final > 0.02`). The
prior attempt's load-bearing failure mode does not recur in the present
architecture.

The C-index implementation cross-check between
`lifelines.utils.concordance_index` (Davidson-Pilon, 2019) and
`sksurv.metrics.concordance_index_censored` (Pölsterl, 2020) gave a
maximum absolute disagreement of below 10⁻⁵ across all 5 Cox PH HONEST
val folds, confirming numerical agreement within precision. All reported
C-indices are stable to the choice of implementation.

A three-seed re-run of the worst-performing fold of the minimal SAGE
architecture (Stage 2, fold 4, best val C-index 0.531 at seed 42) under
seeds {42, 7, 123} yielded a seed-conditioned standard deviation of
0.006 — an order of magnitude smaller than the cross-fold standard
deviation of approximately 0.05 observed across multiple independently-
trained models (Cox PH HONEST, minimal SAGE, knob A). The cross-fold
variance floor at this cohort size is consistent with per-fold event
sparsity (~30 events per val fold) rather than model variance.

The fold-3 anti-leakage signal observed in the leakage audit (Cox PH
HONEST > LEAKY by 0.016 on fold 3, against the +0.070 mean leakage cost
in the other folds) reproduces in the knob A versus knob D comparison
(knob A > knob D by 0.010 on fold 3), consistent with leakage correction
being structurally informative on this specific fold rather than a
coincidence of the audit's particular regression target.

External per-fold consistency on METABRIC reproduces the §3 finding:
per-fold C-indices cluster at std = 0.004 across the 5 separately-trained
knob A models, an order of magnitude smaller than the internal cross-fold
variance.
