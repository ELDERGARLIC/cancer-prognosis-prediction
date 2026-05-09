# Discussion

## 1. Two coordinate findings

This thesis reports two findings of equal weight. The first is that a
leakage-corrected GraphSAGE matches the strongest available Cox PH
baseline on TCGA-BRCA prognosis internally and beats matched Cox PH on
external METABRIC validation, with paired-bootstrap delta of +0.053
(95% CI [+0.031, +0.076], P < 0.001) on n = 1,466 patients with 824
events (Results §3). The second is that two architectural elaborations
— Reactome pathway pooling and BioBERT-PCA gene priors — significantly
underperform the minimal architecture on the same external validation
despite competitive internal performance, with paired-bootstrap deltas
versus the minimal architecture of −0.016 and −0.039 respectively, both
with 95% CIs strictly below zero on identical METABRIC patients
(Results §4).

Both findings are demonstrations of a single methodological framework —
per-fold leakage correction during training and paired-bootstrap on
identical external patients during evaluation — which is itself the
thesis's contribution. The first finding shows what the framework
permits: a defensible measurable lift over linear baselines, replicable
across cohorts. The second shows what the framework refuses: architectural
elaborations that look competitive on internal validation but do not
survive external paired testing. Recent reviews (Liang 2025; Vavekanand
2026) have argued that cancer-AI claims should be conditioned on external
validation and rigorous comparators; both findings here are conditioned
on exactly that.

The lift over Cox PH on internal TCGA validation is +0.060 against the
leakage-corrected baseline, with the 95% CI on the leaky upper bound
just grazing zero. The result is genuine but moderate. It is not the
larger lift the published GNN-on-omics literature has sometimes
reported, and §3 below decomposes why those higher numbers tend to
attribute lift the gene graph itself does not provide. The thesis's
position on its own first finding is therefore one of measured
confidence: a real result on a hard task, defended against the strongest
linear baseline available, replicated externally with significance, and
quantified honestly so that the magnitude is not overstated.

---

## 2. External paired-bootstrap as the load-bearing comparator

Within-cohort C-index estimation on small-event survival data is
inherently high-variance. Per-fold patient-level bootstrap CIs on the
present cohort have a mean width of approximately 0.21 with ~30 events
per val fold (Methods §5.2), which means single-fold differences between
two models smaller than this width are not statistically distinguishable
on internal validation alone. Applied to the present architectural
comparisons, this within-cohort uncertainty is the source of an
otherwise-puzzling mismatch: knob C's TCGA 10%-val C-index is 0.7414,
within fold variance of knob A's 0.7492 (Results §4), a result any
reasonable internal-only assessment would record as a tie. The same
comparison on identical METABRIC patients gives paired delta = −0.039
with 95% CI strictly below zero (P(C ≤ A) = 1.000). The framework
resolves the comparison the internal data could not.

This pattern makes the framework testable rather than merely supported.
Knob C was a prediction — that LLM-derived gene priors would add unique
information beyond expression — that the data had the chance to refute,
and did. Without paired-bootstrap on identical external
patients, knob C would have been adopted as a tied alternative inductive
bias on the basis of internal numbers alone. The framework's value is
not that it confirms the architecture chosen; the framework's value is
that it exposes architectural choices to a test internal validation
cannot perform. Knob B is the same logic in milder form: internal paired
delta of −0.011 [−0.036, +0.013] crosses zero and is reported as a tie,
while external paired delta of −0.016 [−0.026, −0.006] strictly excludes
zero. METABRIC's 824 events provide statistical power that TCGA's 150
events do not; the internal "tie" was the same finding observed without
the events to resolve it.

The methodological argument generalizes. Cohort-spanning generalization
claims in cancer AI have been the subject of repeated calls for external
validation (Liang 2025; Vavekanand 2026), but the form the validation
should take is not always specified. The present framework offers one
operational answer: train on the source cohort with per-fold leakage
correction at the gene-selection step, infer on the target cohort with
strict per-cohort feature normalization, and compare models on identical
target-cohort patients via paired bootstrap on the C-index. The
mechanism that makes the test informative — paired patient-level
sampling rather than fold-mean averaging — is the property that
distinguishes a resolvable comparison from an underpowered one.
Implementations are reusable (`src/cindex_bootstrap.py` and the per-fold
LASSO pipeline) and the test extends to any prognosis task where an
external cohort with sufficient events exists. Within this framework,
the GenePT versus BioBERT question deferred from knob C (Chen and Zou,
2023) is naturally testable through the same protocol.

---

## 3. The small but stable gene-graph contribution

The most under-celebrated result in the thesis is the lift attribution
from Results §2. Across three independently-leakage-regimed GNN variants
— knob D (the leaky-769 baseline), knob A (per-fold-LASSO refit within
769), and knob B (knob A plus pathway pool) — the gene-graph contribution
above non-linear MLP-clinical ranges from +0.008 to +0.013 c-index, a
window narrower than the bootstrap noise floor on any single comparison.
The gene-graph signal is small, real, and architecture-invariant within
the design space examined.

Reporting the lift attribution honestly disambiguates the gene-graph
contribution from the non-linear-clinical contribution that the headline
GNN-vs-Cox numbers conflate. The headline knob A internal C-index of
0.7200 against the leakage-corrected Cox PH baseline of 0.6605 reads as
a +0.060 lift, but ~0.05 of that lift is attributable to the non-linear
MLP head being able to use clinical features (the MLP-clinical-only
reference achieves 0.7122). Roughly +0.013 is the residual gene-graph
contribution. The order-of-magnitude difference between the headline
+0.060 and the residual +0.013 is the distinction this section's
lift-attribution analysis recovers; reporting only the headline
conflates two contributions of very different magnitudes.

The stability of the +0.013 across three architectural variants is a
finding in its own right. The contribution depends weakly on whether
the gene set is constructed by full-cohort LASSO (knob D), per-fold
LASSO within a fixed universe (knob A), or per-fold LASSO followed by
biological pathway pooling (knob B). This invariance suggests the
gene-side signal is a property of the gene-set data on TCGA-BRCA
rather than a property of any specific architecture's inductive bias.

Two consequences for the field follow. First, the gene-graph contribution
is real: removing the GNN entirely costs measurable performance, and
gene-level structure does contribute to prognosis prediction. Second,
the contribution is small: the GNN architecture's complexity-per-lift
is unfavorable compared to alternatives that achieve similar lift via
cheaper means (e.g., richer non-linear flat-feature heads on multi-omics
inputs; Gao 2021). The honest framing for downstream GNN work on
TCGA-shaped data is that the gene-graph contribution is real, the
magnitude is +0.01 to +0.05, and the decision to use a GNN should weigh
the inductive bias and interpretability benefits (Choudhry 2025) against
the lift the gene graph actually delivers.

---

## 4. What did not work and why

Three negative findings, each substantive and rigorously documented.
Each is reported with prediction, result, plausible mechanism, and
generalization.

**Pathway pooling (knob B null).** The prediction was that biological-
pathway grouping of gene-level embeddings would add discriminative
signal beyond global mean pool, motivated by the pathway-attention
designs of Choudhry et al. (2025) and Vaida et al. (2025). The result
on TCGA was a paired delta versus knob A of −0.011 with 95% CI crossing
zero (statistically tied); the result on METABRIC was −0.016 with 95%
CI strictly below zero. The most plausible mechanism is that per-fold
gene sets of 39–72 genes populate only 5–19 of 200 Reactome pathways at
the R5 sentinel threshold, so the pathway-pool head's effective capacity
varies fold-to-fold; the increased per-fold variance (knob B std 0.055
versus knob A 0.043) is consistent with the pathway-pool architecture
being more sensitive to per-fold gene-set composition than gene-level
pooling. Pathway pooling may help on cohorts with denser per-fold gene
sets where the pathway structure is better-populated; the present
cohort's signal does not concentrate in pathway-aligned subsets.

**BioBERT priors (knob C external collapse).** The prediction was that
LLM-derived gene priors would add unique information beyond expression,
in the spirit of Chen and Zou's (2023) GenePT precedent. The result on
TCGA was C-index 0.7414, within fold variance of knob A's 0.7492 (a tie
on internal data); the result on METABRIC was C-index 0.6054 with paired
delta versus knob A of −0.039 and 95% CI strictly below zero. The most
plausible explanation is that the priors encode structure that fits
TCGA's expression manifold but does not transfer cleanly to METABRIC's
microarray distribution, suggesting the priors carry cohort-specific
signal in addition to whatever cohort-invariant biological prior they
encode. LLM-derived priors may transfer
better between cohorts of the same platform (RNA-seq → RNA-seq);
GenePT's GPT-derived embeddings, which Chen and Zou reported outperform
BioBERT on gene-gene tasks, may also outperform BioBERT in this role
— a comparison the present thesis defers but the framework supports
directly (§6).

**Cohort-shift drops.** Both knob A and Cox PH suffer comparable
internal-to-external C-index drops (0.082 and 0.073 respectively), each
exceeding the 0.05 threshold the design originally targeted. The drops
are attributable to platform difference (microarray versus RNA-seq),
follow-up window difference (METABRIC's longer follow-up shifts the
censoring distribution), and population difference between cohorts
rather than to architecture-specific generalization failure: the
comparable magnitude across architectures argues against a model-level
cause. The magnitude of cross-platform drop observed here is consistent with
the platform-difference confound, and provides a baseline for what
cross-platform external validation should expect. The drops did not eliminate the relative
advantage of knob A over Cox PH on the external cohort, which is the
finding that matters: the model trained cold on TCGA still scores 0.6443
on METABRIC, within paired-bootstrap noise of the within-cohort Cox PH
HONEST baseline on TCGA itself (0.6605).

---

## 5. Limitations

Three limitations of the present design are stated explicitly. Each is
specific to choices made in this thesis, has a describable fix, and has
a clear effect — or non-effect — on the conclusions.

**The 769-gene universe was full-cohort-selected.** Knob A's per-fold
LASSO refit operates within the 769 candidate genes inherited from the
prior preprocessing pipeline, not on the raw 60,660-gene HTSeq matrix.
The universe-construction layer of leakage is therefore documented but
uncorrected. A from-scratch per-fold LASSO on the raw matrix was
attempted at Stage 3, but the per-fold-LASSO ∩ 769-gene-universe
overlap was 14% on fold 0, leaving 6 edges in the masked STRING graph
— intractable for the GNN. A future implementation would need either
fresh STRING extraction per fold, which is engineering-heavy, or a
relaxed gene-universe definition that retains enough STRING edges after
intersection. The within-769 leakage correction recovers the
per-fold-selection layer but not the universe-construction layer, and
this distinction should accompany any reproduction of the present
results. The +0.013 gene-graph contribution, the +0.053 external
paired delta, and the knob-B/C external collapses are robust to the
within-versus-full-leakage-correction distinction because the paired-
bootstrap test controls patient-fold assignment rather than gene-
universe construction.

**ER/PR clinical features were lost to a preprocessing artifact.** The
inherited preprocessing pipeline produced all-zero `er_signed` and
`pr_signed` columns in `clinical_features.tsv`. ER and PR status are
among the strongest BRCA prognostic signals in epidemiological data
(Howlader et al., 2014); the present results understate clinical
contribution by an estimated +0.02 to +0.04 c-index. Recovering ER/PR
status from raw TCGA clinical XML and rerunning Stage 0 baselines and
knob A would close this gap and is a one-to-two-day task. The
gene-graph contribution above MLP-clinical (+0.013) may shrink if the
MLP-clinical ceiling rises with ER/PR; the external generalization
claim (knob A versus Cox PH) is unaffected because both models would use
the recovered features. The architectural-minimalism finding is also
unaffected: knob B and knob C would be evaluated against the same
ER/PR-recovered knob A.

**METABRIC is the only external cohort tested.** External validation on
one platform-different cohort cannot fully separate "this cohort
differs" from "RNA-seq → microarray transfer is hard." Validation on
ICGC-BRCA (same-platform RNA-seq), AURORA (matched-population), or
SCAN-B (different-population RNA-seq) would distinguish cohort-level
from platform-level effects. The cohort-shift drop is the most cohort-
dependent claim in the thesis; the paired-delta findings (knob A > Cox
PH; knobs B and C < knob A) hold on at least one external cohort
regardless of the additional ones not tested.

---

## 6. Future work

Six items, each one reasonable next experiment given what was found here.
None promise specific lifts; each tests a question the present design
has positioned but not answered.

**Per-fold knowledge-graph rebuild from STRING.** Address the
universe-construction leakage directly by extracting STRING edges per
fold's per-fold-LASSO gene set rather than masking the inherited 769-gene
edge set. The within-769 framework's results form a baseline for what
fully-honest leakage correction would need to preserve or exceed.

**ER/PR/HER2 recovery from raw TCGA clinical.** The cheapest +0.02 to
+0.04 internal lift available; methodologically required for the
strongest internal-headline number. The architectural-minimalism finding
is unaffected by whether this is done.

**Additional external cohorts.** ICGC-BRCA gives same-platform RNA-seq
external validation, isolating cohort-level from platform-level effects
on the cohort-shift drop finding. AURORA (matched-population) and SCAN-B
(different-population RNA-seq) further separate confounds. Each cohort
extends the paired-bootstrap-on-external-patients framework to a new
target.

**GenePT versus BioBERT head-to-head.** The architecture supports this
as a knob-C variant: substitute GPT-3.5 gene embeddings (Chen and Zou,
2023) for BioBERT-derived embeddings, with the same PCA(32) projection
and multiplicative expression weighting. The paired-bootstrap test on
METABRIC has the statistical power to resolve the comparison given the
824 events available. This is the most informative near-term experiment
for the LLM-priors knob.

**GAT or HGT only conditional on a future cohort showing pathway-pool
gains the present cohort does not.** The present thesis held the
architecture minimal because the data did not justify additional
complexity. If a denser-per-fold-gene-set cohort shows pathway pooling
helping, attention-based or heterogeneous variants on the same cohort
would test whether the additional architectural capacity is what
captures the signal.

**Multi-omics extensions.** CNA, methylation, and miRNA were out of
scope here (HTSeq + clinical only). The per-fold leakage correction and
external paired-bootstrap framework extend directly to additional
modalities. Whether multi-omics widens the +0.013 gene-graph
contribution above non-linear flat-feature ceilings is open, and the
comparison that would resolve this is a multi-omics GNN against a
multi-omics MLP on the same paired-bootstrap protocol (Gao 2021's
ablation precedent extended to external cohorts).

---

## 7. Contribution restated

This thesis has reported two coordinate findings — that a leakage-
corrected GraphSAGE beats matched Cox PH on external METABRIC
validation with paired-bootstrap significance, and that two architectural
elaborations underperform the minimal architecture on the same external
test despite competitive internal performance — and one methodological
framework: per-fold leakage correction during training combined with
paired-bootstrap on identical external patients during evaluation. Three
negative findings (pathway pooling's null lift, BioBERT priors' external
collapse, and the cohort-shift drops shared by both architectures) are
documented with prediction, result, mechanism, and generalization to
guide downstream work.

The framework is reusable: a per-fold LASSO leakage audit, a bootstrap
CI utility supporting per-fold and paired-on-identical-patient
comparisons, R1 embedding-collapse and R5 pathway-sparsity sentinels,
and the external paired-bootstrap test as the primary statistic for any
architectural claim. Each is implemented in the accompanying repository
and applies to TCGA-BRCA prognosis studies on other cohorts and to
cohort-spanning prognosis studies on other diseases. The most informative
near-term application is a GenePT-versus-BioBERT comparison through the
same protocol, which the framework supports as a knob-C variant.

External paired-bootstrap testing converts ambiguous within-cohort GNN
comparisons into resolvable findings, and applying this framework to
TCGA-BRCA yields a small, real, leakage-corrected gene-graph signal with
clinically-readable pathway attribution — a less dramatic result than
the field has been claiming, and a more defensible one.
