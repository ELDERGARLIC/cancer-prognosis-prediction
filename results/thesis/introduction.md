# Introduction

## 1. Background and motivation

Breast cancer prognosis from gene expression has become a standard
application of graph neural networks, but the field's evaluation of GNN
claims has not kept pace with the architectures it evaluates. This
thesis is a methodological response: it builds a leakage-corrected
graph neural network for breast-cancer survival prediction on TCGA-BRCA
and evaluates it under a framework — per-fold leakage correction during
training plus paired-bootstrap on identical external patients during
evaluation — that resolves architectural comparisons the field's current
within-cohort assessment cannot.

Breast cancer is heterogeneous at the molecular level. The intrinsic
molecular subtypes (luminal A, luminal B, HER2-enriched, basal-like)
identified through hierarchical clustering of expression profiles
(Sørlie et al., 2001) are prognostically distinct in ways that
clinical staging alone does not capture. Hormone-receptor status (ER
and PR) and HER2 amplification status are routinely measured and carry
substantial prognostic information; SEER population data show ER-positive
disease consistently associated with longer survival than ER-negative
disease at matched stage (Howlader et al., 2014). Tumor-Node-Metastasis
staging captures the spread of disease at diagnosis but does not capture
the molecular heterogeneity that contributes to outcome divergence
within stage strata. Computational models that integrate transcriptomic
information with clinical staging therefore have a clear motivation:
better stratification of within-stage prognostic risk, with downstream
implications for treatment intensity and follow-up frequency.

Survival analysis as a discipline is built around the Cox proportional-
hazards model (Cox, 1972), which estimates hazard ratios from a
linear combination of covariates without requiring the baseline hazard
to be specified. Time-to-event prediction with right-censoring — patients
whose follow-up ends before an event is observed — is the structural
property that distinguishes survival data from standard regression
or classification, and Cox PH handles censoring through the partial
likelihood. The concordance index introduced by Harrell et al. (1982),
which measures the fraction of patient pairs whose predicted-risk
ordering agrees with their observed event ordering, is the standard
performance metric in survival prediction and the metric this thesis
reports throughout.

Computational models for survival prediction from high-dimensional
gene-expression data face two challenges that linear Cox PH alone does
not address: dimensionality (a typical RNA-seq matrix has tens of
thousands of genes against hundreds to thousands of patients) and
non-linearity (gene effects on outcome interact in ways linear models
cannot capture). The standard solutions are gene selection — most
commonly via the LASSO regression introduced by Tibshirani (1996), which
shrinks irrelevant coefficients to zero — and non-linear extensions of
the Cox PH framework. Katzman et al. (2018) introduced DeepSurv, the
first widely-adopted deep-learning Cox PH variant, which preserves the
partial-likelihood loss while replacing the linear hazard combination
with a neural network. The architecture this thesis builds is a graph-
based DeepSurv variant in this lineage.

Graph neural networks have become the field's preferred relational
architecture for cancer prognosis because gene-gene interactions are
naturally graph-structured: protein-protein interaction networks,
gene-pathway memberships, and co-expression correlations all define
edges over a shared gene-node universe. The GraphSAGE inductive variant
(Hamilton et al., 2017) — which learns aggregation functions over local
neighborhoods rather than memorizing positions in a fixed graph — is the
standard backbone choice for cohort-spanning prognosis tasks where
inference on new patients without retraining is required (Madanipour et
al., 2024). The patient-as-graph paradigm in which each patient is
represented as a graph with shared topology and patient-specific node
features (Vaida et al., 2025) makes the architecture compatible with
external-cohort inference whenever the source-cohort gene universe
intersects with the target-cohort assay coverage. The architecture's
known failure modes on small per-fold gene subgraphs — particularly
oversmoothing at depth (Ling et al., 2022) — constrain the design space
to shallow models, with two-layer GraphSAGE the consensus default.

Graph neural networks are now widely applied to cancer prognosis on
TCGA-BRCA and adjacent cohorts. The question is no longer whether to
use GNNs but how to evaluate the claims they generate. The next section
examines two specific aspects of how the field currently does that
evaluation.

---

## 2. The field's problem

The proliferation of GNN architectures for cancer prognosis has outpaced
the methodological scrutiny applied to their comparisons. Two recent
reviews of cancer-AI evaluation practice argue that external validation
should be a default expectation rather than an optional addition; both
observe that within-cohort lifts have been treated as sufficient
evidence of architectural superiority more often than they should be
(Liang, 2025; Vavekanand and Liang, 2026). This thesis works on two
specific gaps the reviews identify: gene-selection leakage at the
training stage, and lift-attribution conflation at the comparison stage.

The first gap is methodological. When a gene-selection step (LASSO,
recursive feature elimination, mutual information, or similar) is fit
on the full source cohort and cross-validation folds are constructed
afterwards for evaluation, every fold's validation patients have already
participated in the gene selection. The genes retained are partly chosen
because they correlate with those validation patients' labels. Cox PH
and GNN models built on such gene sets inherit this leakage; the
resulting C-indices are upper bounds on what the same architecture
would deliver if gene selection were re-fit per fold. The standard
correction is per-fold refit of the selection step (Tibshirani, 1996,
in the LASSO case): each fold's training partition fits its own gene
set without seeing its validation patients. The correction is not always
applied. This thesis quantifies the leakage cost on the inherited
TCGA-BRCA pipeline at +0.070 C-index between leaky and per-fold-honest
Cox PH baselines (Methods §5.1, Results §1), with the corrected
baseline becoming the methodological north-star against which all
subsequent claims are measured.

The second gap is interpretive. When a GNN reports a headline C-index
lift over a Cox PH baseline, the lift is typically attributed to the
GNN's relational inductive bias — the gene graph itself. But Cox PH
operates on a small set of clinical features and a low-dimensional
projection of expression, while the GNN has access to the same clinical
features through its head and adds non-linear capacity. A headline
GNN-vs-Cox lift therefore conflates two distinct contributions:
gene-graph contribution from the relational architecture, and
non-linear-flat-feature contribution from having a more flexible head
on the same clinical features. Decomposing the two requires a non-linear
MLP reference operating on the same clinical features without the gene
graph; this reference is rarely reported. Where clinical-fusion ablation
has been conducted with appropriate controls — Gao et al. (2021)'s
gene-only versus gene-plus-clinical comparison being one of the few
clean examples — the clinical contribution has been substantial. The
absence of the MLP-clinical reference in most GNN-on-omics papers means
that headline lifts conflate two contributions of very different
magnitudes.

External validation, when present, often uses platform-similar cohorts
that do not stress-test cross-platform transfer. TCGA-BRCA paired with
METABRIC (Curtis et al., 2012) is one of the few publicly available
source-target combinations that does: TCGA is RNA-seq, METABRIC is
Illumina microarray, the platforms differ in dynamic range and probe
coverage, and METABRIC's longer follow-up window yields a substantially
higher event count (824 events on 1,466 patients) than TCGA's (150 on
1,074). The pair gives paired-bootstrap tests sufficient statistical
power to resolve small architectural effects that within-TCGA
comparisons cannot. The cohorts are not novel as source-target choices;
what is missing is the methodological framework that uses them
rigorously.

---

## 3. Thesis position and structure

This thesis builds a leakage-corrected GraphSAGE for TCGA-BRCA prognosis,
evaluates it on METABRIC with paired-bootstrap on identical external
patients, and decomposes the resulting lift against a non-linear MLP-
clinical reference. The methodological framework comprises per-fold
leakage correction during training and paired-bootstrap on identical
external patients during evaluation. The architecture and preprocessing
build on an inherited TCGA-BRCA pipeline; the inheritance's leakage
characteristics are audited and corrected per fold.

The thesis has three operational components. The model is a two-layer
GraphSAGE on per-patient gene graphs with STRING protein-protein-
interaction edges, per-fold LASSO gene selection within the inherited
universe, clinical late-fusion, an MLP head, and Cox partial-likelihood
loss; the architecture is held minimal as a deliberate discipline so
that ablation results are interpretable. The comparator — paired-
bootstrap on identical external patients — is the framework's primary
methodological move: by scoring two models on the same resampled
patient set with identical (T, E) pairs in equal sample sizes, the test
eliminates the patient-fold-assignment variance that fold-mean
averaging leaves uncontrolled, and resolves architectural comparisons
that internal validation alone leaves underpowered. The decomposition
introduces a non-linear MLP-clinical-only reference: a two-layer MLP
on the same clinical features with the same Cox loss, no gene graph,
which establishes the flat-feature ceiling against which the gene-graph
contribution can be honestly measured. The inductive paradigm follows
Madanipour et al. (2024), which provides the cohort-spanning inference
property the framework requires.

The framework produces two coordinate findings. The first is that a
leakage-corrected GraphSAGE matches the strongest available Cox PH
baseline on TCGA-BRCA internally and beats matched Cox PH on external
METABRIC validation with paired-bootstrap significance (Δ = +0.053,
95% CI [+0.031, +0.076], P < 0.001 over 2,000 resamples on n = 1,466
patients with 824 events). The lift is measurable but moderate: against the leakage-corrected
Cox PH baseline, the headline +0.060 C-index lift is genuine, with
approximately +0.010 attributable to the gene graph itself and the
remainder to the non-linear MLP head's use of clinical features. The
+0.010 gene-graph contribution is small but stable to within bootstrap
noise across all three GNN variants tested.

The second finding is that two architectural elaborations from the
field's recent design space — Reactome pathway pooling (Choudhry et
al., 2025) and BioBERT-PCA gene priors (Lee et al., 2020; Chen and Zou,
2023) — significantly underperform the minimal architecture on external
validation despite competitive internal performance, with paired
deltas versus the minimal architecture of −0.016 and −0.039 respectively,
both with 95% CIs strictly below zero on identical METABRIC patients.
Internal-only assessment treats both elaborations as competitive; the
external paired-bootstrap test resolves them as significantly worse.
The finding generalizes: added complexity in this design space requires
external paired-bootstrap testing to verify, and within-cohort numbers
alone are insufficient to support architectural claims at this cohort
size and event count.

The methodological framework is the thesis's contribution. The model
and the decomposition are necessary supports for the comparator to do
its work; the comparator is what transforms within-cohort architectural
comparisons from inflated point estimates into resolvable findings.
Both findings — the positive Finding 1 and the negative Finding 2 —
are joint outputs of the same framework, demonstrations of what the
framework permits and what it refuses. The framework is reusable
beyond the present task: per-fold leakage correction at the gene-
selection step plus paired-bootstrap on identical target-cohort
patients applies to any prognosis study where a source cohort and a
sufficiently-evented target cohort exist.

This thesis asks what a leakage-corrected GNN's lift over Cox PH on
TCGA-BRCA prognosis actually contains, and which architectural
elaborations from the field's recent design space survive external
paired-bootstrap testing. The remainder of the thesis is organised as
follows.

---

## 4. Roadmap

**Methods** establishes the data, the three baselines, the architecture,
the ablation knobs, the methodological apparatus (LASSO leakage audit,
bootstrap CI machinery, paired-bootstrap-vs-fold-mean worked example,
embedding-collapse and pathway-sparsity sentinels), the external
validation protocol, and the computational details.

**Results** presents the Cox PH baselines and the prior-pipeline
correction; the within-TCGA knob-A-versus-Cox-PH comparison; the
external METABRIC paired-bootstrap result with risk-stratified Kaplan-
Meier curves and decile-binned calibration; the architectural-ablation
forest plot showing knob B and knob C significantly underperforming
externally; the pathway-attention interpretability artifact; and the
robustness checks.

**Discussion** restates the two coordinate findings, argues for
external paired-bootstrap as the load-bearing comparator with knob C as
the worked example, presents the small-but-stable +0.010 gene-graph
contribution as a finding in its own right, documents the three
negative findings with mechanism and generalization, names three
limitations, lists six future-work items, and closes with the
framework restated as the thesis's contribution.

**Conclusion** summarises in a single tight paragraph: what was done,
what was found, what comes next. The framework's reusability and the
boundedness of the gene-graph signal are the chapter's two final
emphases.
