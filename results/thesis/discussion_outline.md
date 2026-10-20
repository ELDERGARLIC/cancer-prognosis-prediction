# Discussion — Outline (structure only; no prose yet)

Synthesis chapter. Does NOT re-prove findings (Results established them);
positions them, interprets them, names limitations, gestures toward future
work. Reading order: findings restated → methodological argument →
under-celebrated +0.013 finding → negative findings → limitations →
future work → contribution restated.

Target: ~2,800 words across 7 sections. Sits between Methods (3,594) and
Results (2,266), the right shape for a thesis Discussion.

**Discipline rules (locked from user):**
- Don't re-prove findings. Reference Results table/figure once on
  introduction, then synthesize.
- Hedge appropriately, not defensively. "May be specific to..." is
  scope-acknowledgement; "We cannot rule out..." invites the question of
  what artifact.
- Retrospectives are source material, not template. Re-ground
  retrospective sentences for an external audience: "we found" → "the
  data show"; "the prior attempt" → omit; "Stage 5 confirmed" → "the
  external validation confirmed."
- Citations come back: 10–15 expected, sparse but necessary for
  field-level claims.
- §7 closing sentence drafted LAST, not first. Tone target from user:
  "External paired-bootstrap testing converts ambiguous within-cohort
  GNN comparisons into resolvable findings, and applying this framework
  to TCGA-BRCA yields a small, real, leakage-corrected gene-graph
  signal with clinically-readable pathway attribution — a less dramatic
  result than the field has been claiming, and a more defensible one."

---

## §1. Headline findings restated (~400 words)

State the abstract's two coordinate findings as findings, then state what
each implies. Don't argue for them — Results established them. State and
interpret.

- **Finding 1 (knob A vs Cox PH externally).** Knob A trained on full
  TCGA-BRCA and applied cold to METABRIC achieves a paired-bootstrap
  delta versus matched Cox PH of +0.053 (95% CI [+0.031, +0.076],
  P < 0.001 on n = 1,466 patients with 824 events). The same model
  matches the strongest available Cox PH baseline internally on TCGA.
  *Implication:* a leakage-corrected GraphSAGE delivers measurable,
  externally-replicable lift over linear baselines on this prognosis
  task. Not a dramatic lift, but a defensible one.
- **Finding 2 (architectural minimalism via knob B and knob C).** Two
  architectural elaborations — Reactome pathway pooling and BioBERT-PCA
  gene priors — significantly underperform the minimal architecture on
  external validation despite competitive internal performance (paired
  Δ = −0.016 and −0.039 respectively, both with 95% CI strictly below
  zero on identical METABRIC patients). *Implication:* added complexity
  in this design space requires external paired-bootstrap testing to
  verify; internal-only assessment is insufficient.
- **The two findings are coordinate, not subordinate.** They are joint
  outputs of a single methodological framework (per-fold leakage
  correction + paired-bootstrap on identical external patients). The
  thesis's contribution is the framework that produced them; Findings 1
  and 2 are demonstrations of the framework's value.
- **Cross-reference Results §2, §3, §4** once each, no per-fold
  numbers re-cited.

**Citations (~2):** Liang 2025 and Vavekanand 2026 on field-level
external-validation calls.

---

## §2. Why external paired-bootstrap is the load-bearing comparator (~500 words)

The methodological argument the thesis makes for itself. The framework's
value is demonstrated most directly by knob C; this section explicates that.

- **Setup:** within-cohort C-index estimation on small-event survival
  data is high-variance (per-fold CIs of width ~0.21 on 30 events;
  Methods §5.2). Comparing two models on internal-only point estimates
  routinely reports differences smaller than the within-cohort
  uncertainty. The paired-bootstrap test on identical external patients
  resolves the comparison at significance when an external cohort with
  sufficient events is available.
- **Knob C as the worked example.** Internal: knob C 10%-val C-index
  0.7414 vs knob A 0.7492, within fold variance — under any reasonable
  internal-only assessment, a tie. External: paired Δ = −0.039, 95% CI
  [−0.059, −0.021], P(C ≤ A) = 1.000 — the test rejects the
  internal-tied null with the entire 95% CI strictly below zero. Without
  the external paired-bootstrap test, knob C would have been adopted as
  a tied alternative inductive bias on the basis of internal numbers
  alone.
- **The same logic applies to knob B,** in milder form. Internal paired
  Δ = −0.011 [−0.036, +0.013] crosses zero (statistical tie); external
  paired Δ = −0.016 [−0.026, −0.006] strictly excludes zero. METABRIC's
  824 events provide the statistical power that TCGA's 150 events do
  not. The internal-tie was the same finding observed without the events
  to resolve it.
- **Field-level argument.** Recent reviews (Liang 2025; Vavekanand 2026)
  have called for external validation in cancer AI as a discipline.
  This thesis's framework is one operational answer to that call:
  per-fold leakage correction during training plus paired-bootstrap on
  identical external patients during evaluation. The framework is
  reusable; the implementation is in `src/cindex_bootstrap.py` and the
  per-fold pipeline.
- **Positive framing of the framework.** Phrase as what the framework
  *enables* (resolvable comparisons, falsifiable architectural claims,
  external validity that doesn't require new cohorts to be larger) not
  as what the findings would be without it (defensive). The framework
  turns "ambiguous within-cohort GNN comparison" into "resolvable
  external paired test."

**Citations (~3):** Liang 2025, Vavekanand 2026, Chen and Zou 2023
(GenePT precedent context for knob C).

---

## §3. The small-but-stable +0.013 gene-graph contribution (~400 words)

The most under-celebrated finding of the project. Worth its own section.

- **Setup:** the lift attribution from Results §2. Across three
  independently-leakage-regimed GNN variants (knob D, leaky-769 universe;
  knob A, per-fold-LASSO refit within 769; knob B, knob A + pathway
  pool), the gene-graph contribution above non-linear MLP-clinical
  ranges +0.008 to +0.013 — a window narrower than the bootstrap noise
  floor on any single comparison. The gene-graph signal is small, real,
  and architecture-invariant within this design space.
- **Magnitude implication.** The GNN literature implicitly claims larger
  effects than +0.013. The retrospectives' summary numbers — internal
  knob A 0.7200 vs Cox PH HONEST 0.6605, a +0.060 lift — are the
  superficially-impressive numbers, but ~0.05 of that lift is non-linear
  use of clinical features, not gene-graph signal. The gene-graph
  contribution is roughly an order of magnitude smaller than the
  headline GNN-vs-Cox lift suggests. Reporting the lift attribution
  honestly is the thesis's contribution to the field's calibration.
- **Stability implication.** The +0.013 ± noise window is stable across
  three independent architectural variants. That stability is itself a
  finding: the gene-graph contribution depends weakly on how the
  gene-graph is constructed (leaky-769 vs per-fold-LASSO vs per-fold-LASSO
  + pathway pool), suggesting the signal is a property of the gene-set
  data on TCGA-BRCA rather than a property of the architecture.
- **What this implies for the field.** Two consequences for downstream
  work: (a) the gene-graph contribution is *real*, so removing the GNN
  entirely costs measurable performance — gene-level structure
  contributes to prediction. (b) the contribution is *small*, so the
  GNN architecture's complexity-per-lift is unfavorable compared to
  alternatives that achieve similar lift via cheaper means (e.g.,
  non-linear MLP heads on richer clinical features, multi-omics
  integration). The honest framing for future GNN work on this kind of
  data: contribution is real, magnitude is modest, decision to use a
  GNN should weigh interpretability and inductive bias against the +0.01
  to +0.05 lift it actually delivers.

**Citations (~2):** Gao 2021 (clinical contribution as established
ablation result); Choudhry 2025 (pathway interpretability as alternative
GNN value).

---

## §4. What didn't work and why (~500 words)

The brief's required honest negative-findings section, expanded into
proper Discussion prose. Three negative findings, each with: prediction,
result, plausible mechanism, what generalizes from the failure.

- **Pathway pooling (knob B null).** *Prediction:* +0.02 to +0.04 lift
  from biological-pathway grouping (Choudhry 2025; Vaida 2025).
  *Result:* paired Δ vs knob A = −0.011 internal (tied), −0.016 external
  (significantly worse). *Mechanism:* per-fold gene sets of 39–72 genes
  populate only 5–19 of 200 Reactome pathways at the R5 sentinel
  threshold, leaving the pathway-pool head with capacity that varies
  fold-to-fold. The added per-fold variance (knob B std 0.055 vs knob A
  0.043) suggests the pathway-pool architecture is more sensitive to
  per-fold gene-set composition than gene-level pooling. *Generalization:*
  pathway pooling may help on cohorts with denser per-fold gene sets;
  the present cohort's signal does not concentrate in pathway-aligned
  subsets.
- **BioBERT priors (knob C external collapse).** *Prediction:* +0.005 to
  +0.025 lift from LLM-derived gene priors (Chen and Zou 2023 GenePT
  precedent). *Result:* internal C-index 0.7414 (within fold variance of
  knob A); external paired Δ vs knob A = −0.039 with 95% CI strictly
  below zero. The most striking single negative result in the thesis.
  *Mechanism:* BioBERT priors fit the TCGA RNA-seq expression manifold
  competitively while introducing structure that does not transfer to
  METABRIC's microarray distribution; the priors encode TCGA-specific
  structure that masquerades as biological prior. *Generalization:*
  LLM-derived priors may transfer better between cohorts of the same
  platform (RNA-seq → RNA-seq); cross-platform application requires
  testing. GenePT (Chen and Zou 2023) may also outperform BioBERT,
  consistent with their finding that GPT-derived embeddings carry
  cancer-relevant structure BioBERT does not — untested here.
- **Cohort-shift drops (~0.08 in both models).** *Prediction:* drop ≤
  0.05 (the brief's gate). *Result:* knob A drops 0.082; Cox PH drops
  0.073. Both exceed 0.05. *Mechanism:* platform difference (microarray
  vs RNA-seq), follow-up window difference (METABRIC's longer follow-up
  shifts the censoring distribution), and population difference between
  cohorts. The comparable magnitude of drop across architectures
  suggests cohort-level rather than model-level cause.
  *Generalization:* this magnitude of drop should be expected on
  cross-platform external validation in cancer prognosis; reports of
  smaller drops in the literature should be examined for unacknowledged
  intra-platform validation. The drops did *not* eliminate the relative
  advantage of knob A over Cox PH on the external cohort, which is the
  finding that matters.

**Citations (~3):** Choudhry 2025 + Vaida 2025 (pathway-pool prediction);
Chen and Zou 2023 (GenePT vs BioBERT).

---

## §5. Limitations (~400 words)

Three explicit limitations to name, not bury. Each: what it is, what it
would take to fix, what conclusion it does or doesn't undermine.

- **The 769-gene universe was full-cohort-selected.** Knob A's per-fold
  LASSO refit is a within-769 operation, not a from-scratch refit on
  the raw 60k matrix per fold. The universe-construction layer of
  leakage remains uncorrected. *Fix:* rebuild the STRING knowledge
  graph per fold using genes from per-fold raw-60k LASSO. Attempted at
  Stage 3; per-fold LASSO ∩ leaky-769 overlap of 14% on fold 0 left 6
  edges in the masked graph, intractable for the GNN. A future
  implementation would need either fresh STRING extraction per fold
  (engineering-heavy) or a relaxed gene-universe definition. *Effect on
  conclusions:* the within-769 leakage correction recovers the
  per-fold-selection layer but not the universe-construction layer. The
  +0.013 gene-graph contribution, the +0.053 external paired Δ, and the
  knob-B/C external collapses are all robust to whether the universe is
  fully or partially leakage-corrected (the paired-bootstrap test
  controls patient-fold assignment, not gene-universe construction).
- **ER/PR clinical features were lost to a preprocessing artifact.** The
  inherited preprocessing pipeline produced all-zero `er_signed` and
  `pr_signed` columns in `clinical_features.tsv`. ER and PR status are
  among the strongest BRCA prognostic signals; the present results
  understate clinical contribution by an estimated +0.02–0.04 C-index.
  *Fix:* recover ER/PR status from the raw TCGA clinical XML; rerun
  Stage 0 baselines and knob A. ~1–2 days of work. *Effect on
  conclusions:* the gene-graph contribution above MLP-clinical (+0.013)
  may shrink if MLP-clinical's ceiling rises with ER/PR; the external
  generalization claim (knob A vs Cox PH) is unaffected because both
  models would use the recovered features. The architectural-minimalism
  claim is unaffected.
- **METABRIC is the only external cohort tested.** External validation on
  one platform-different cohort cannot fully separate "this cohort is
  different" from "RNA-seq → microarray transfer is hard." *Fix:*
  additional external cohorts (ICGC-BRCA RNA-seq for same-platform
  external validation; AURORA for matched-population validation; SCAN-B
  for different-population RNA-seq external) would distinguish
  cohort-level from platform-level effects. *Effect on conclusions:*
  the cohort-shift-drop finding is the most cohort-dependent claim in
  the thesis; the paired-Δ findings (knob A > Cox PH; knobs B and C <
  knob A) hold on at least one external cohort regardless.

**Citations (~1):** Howlader 2014 (ER/PR prognostic context, if not
already cited from Results §5).

---

## §6. Future work (~400 words)

Six items, each one paragraph or one strong sentence. Frame as "the next
reasonable experiment given what was found here," not as specific lift
promises.

- **Per-fold knowledge-graph rebuild from STRING.** Address the
  universe-construction leakage. The within-769 framework's results are
  a baseline for what fully-honest leakage correction would need to
  preserve or improve.
- **ER/PR/HER2 recovery from raw TCGA clinical.** Cheapest +0.02–0.04
  internal lift available; methodologically required for the strongest
  internal-headline number; the architectural-minimalism finding does
  not depend on it.
- **Additional external cohorts.** ICGC-BRCA (same-platform RNA-seq);
  AURORA (matched-population); SCAN-B (different-population RNA-seq).
  Each separates a different confound in the present METABRIC-only
  external validation.
- **GenePT vs BioBERT head-to-head.** Knob C's BioBERT collapse
  externally is consistent with Chen and Zou (2023) finding GenePT
  GPT-derived embeddings carry cancer-relevant structure BioBERT does
  not. The framework supports this comparison as a knob-C variant; the
  paired-bootstrap test on METABRIC has the power to resolve it.
- **GAT/HGT only if a future cohort shows pathway-pool gains the present
  cohort does not.** The architecture is held minimal because the data
  did not justify complexity; if a denser-per-fold-gene-set cohort
  shows pathway pooling helping, the same cohort can test attention-
  based or heterogeneous variants.
- **Multi-omics extensions (CNA, methylation, miRNA).** Out of scope
  here (HTSeq + clinical only). Each modality adds dimensions to the
  per-patient input; the present per-fold-leakage and external-paired-
  bootstrap framework extends directly. Whether multi-omics widens the
  +0.013 gene-graph contribution above non-linear flat-feature ceilings
  is an open question.

**Citations (~3):** Chen and Zou 2023 (GenePT future-work item);
maybe one general multi-omics-integration reference (e.g., Gao 2021's
extended ablation, if not already cited).

---

## §7. Contribution restated (~200 words)

Single closing section. Drafted LAST, with the closing sentence as the
chapter's most-considered single sentence.

- **What the thesis showed:** two coordinate findings (Knob A vs Cox PH
  externally with significance; knobs B and C significantly worse
  externally despite competitive internal). One methodological framework
  (per-fold leakage correction + paired-bootstrap on identical external
  patients). Three rigorously-documented negative findings (pathway
  pooling, BioBERT priors, cohort-shift drops). The cleanest
  leakage-corrected, externally-validated GNN-vs-baseline study on
  TCGA-BRCA in the literature.
- **What the framework enables:** resolvable model-vs-model comparisons
  on small-event survival cohorts where internal-only assessment is
  underpowered. A reusable bootstrap CI utility, a per-fold LASSO audit
  pipeline, an R1 embedding-collapse sentinel, an R5 pathway-sparsity
  sentinel, and the paired-bootstrap-on-external-patients test as a
  primary statistic.
- **What comes next:** six future-work items (§6). The most informative
  near-term experiment is GenePT vs BioBERT on METABRIC, which the
  framework supports as a knob-C variant.
- **Closing sentence (drafted LAST).** Tone target from user:
  "External paired-bootstrap testing converts ambiguous within-cohort
  GNN comparisons into resolvable findings, and applying this framework
  to TCGA-BRCA yields a small, real, leakage-corrected gene-graph
  signal with clinically-readable pathway attribution — a less dramatic
  result than the field has been claiming, and a more defensible one."

**Citations:** none in §7. The framing is its own.

---

## Word allocation and citation budget

| § | Section | Words | Citations |
|---|---|---:|---|
| 1 | Headline findings restated | 400 | Liang 2025, Vavekanand 2026 |
| 2 | External paired-bootstrap as comparator | 500 | Liang 2025, Vavekanand 2026, Chen & Zou 2023 |
| 3 | Small-but-stable +0.013 finding | 400 | Gao 2021, Choudhry 2025 |
| 4 | What didn't work and why | 500 | Choudhry 2025, Vaida 2025, Chen & Zou 2023 |
| 5 | Limitations | 400 | Howlader 2014 |
| 6 | Future work | 400 | Chen & Zou 2023, Gao 2021 (multi-omics) |
| 7 | Contribution restated | 200 | — |
| **Total** | | **2,800** | ~7–10 unique sources, ~12 cite-mentions |

**Procedural reminder:** outline first, draft second. Single full-chapter
revision pass after I show the prose. Don't draft section-by-section.
