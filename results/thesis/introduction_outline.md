# Introduction — Outline (structure only; no prose yet)

Reading order:
- §1: Background. ~600 words. Citation-dense.
- §2: The field's problem. ~600 words. Citation-medium.
- §3: Thesis position and structure. ~600 words. Citation-light.
  Research-question sentence drafted LAST, placeholder slot in outline.
- §4: Roadmap. ~200 words. Zero citations.

Target total: ~2,000 words. Citation budget: ~20–25 mentions across
~15–18 unique sources. Second-densest chapter after Methods.

**Discipline rules locked from user:**
- Spend more on field-problem + thesis-position + roadmap (1,400 words)
  than on background (600 words). Reverses the failure mode where
  Introduction is a textbook chapter with a thesis tacked on.
- Research-question sentence drafted last, derived from what the rest
  of §3 has earned. Outline marks the slot as `[placeholder, draft last]`.
- Contribution claims down-translated from Discussion §7. Match the
  chapter we already wrote; do not preview stronger claims than
  Discussion delivers.
- No new findings in Introduction. Every claim is either cited
  (background, prior work) or pre-announced as a forward-reference to
  Methods / Results / Discussion.
- Citations cluster: §1 dense, §2 medium, §3 light, §4 zero. Don't pad
  to hit upper bound; don't strip below 20.

**Three structural calls answered (from user feedback):**
1. Preview both findings in §3 (option a) — locked.
2. ~150 words on BRCA biology in §1, no more — locked.
3. One-line acknowledgement of inherited preprocessing in §3 (not §1) —
   locked.

---

## §1 Background and motivation (~600 words; ~9 citations)

The minimum domain context a non-oncology reader needs to understand
why a methodological thesis on TCGA-BRCA prognosis matters. Heavy on
foundational citations; light on the specific GNN-on-cancer literature
(that's §2's territory).

**Sub-structure:**

- **Breast cancer prognosis as a clinical problem (~150 words).**
  Heterogeneity at the molecular level (intrinsic subtypes — luminal,
  HER2-enriched, basal-like) and clinical staging captures only part
  of the prognostic variation. Receptor status (ER, PR, HER2) and
  TNM stage are the standard prognostic covariates; molecular subtype
  added via expression assays (Pam50; not used here directly). The
  clinical motivation: better prognosis prediction informs treatment
  intensity and follow-up frequency. **Cap at ~150 words; this is
  background, not a clinical review.**
  - Citations: Sørlie et al. 2001 (intrinsic subtypes), Howlader et al.
    2014 (SEER population statistics on ER/PR prognosis), one BRCA
    epidemiology / staging reference.

- **Survival analysis and the Cox proportional-hazards model (~120 words).**
  Time-to-event prediction with right-censoring; Cox PH as the
  field-standard linear baseline; partial-likelihood loss; concordance
  index (Harrell's C) as the primary metric. Brief framing only.
  - Citations: Cox 1972 (proportional hazards), Harrell 1982 (C-index).

- **Computational models for prognosis from gene expression (~120 words).**
  Why expression data complements clinical staging: molecular
  heterogeneity is finer-grained than histopathology can resolve.
  High-dimensional inputs (~20k–60k genes) require regularization or
  feature selection. LASSO regression for gene selection; deep
  survival models (DeepSurv) for non-linear feature combination.
  - Citations: Tibshirani 1996 (LASSO original), Katzman et al. 2018
    (DeepSurv).

- **Graph neural networks as the field's preferred relational architecture
  (~150 words).** Why GNNs were brought into cancer prognosis: gene
  interactions are naturally graph-structured (PPI networks, pathway
  membership), and GNNs let architectural inductive bias encode
  biological prior. The patient-as-graph paradigm: each patient
  represented as a graph with shared topology and patient-specific
  node features. Inductive (vs transductive) GNNs are required for
  cohort-spanning inference without retraining. End the section with
  the framing line: GNNs are now widely applied to cancer prognosis
  on TCGA-BRCA and adjacent cohorts.
  - Citations: Hamilton et al. 2017 (GraphSAGE), Vaida et al. 2025
    (patient-as-graph for cancer), Madanipour et al. 2024 (inductive
    paradigm), Ling et al. 2022 (oversmoothing on small graphs).

- **Closing transition (~60 words).** Set up §2: the question is no
  longer whether to use GNNs on cancer prognosis but how to evaluate
  the claims they generate. The next section names two specific gaps
  in how the field currently does that evaluation.

---

## §2 The field's problem (~600 words; ~7 citations)

Two specific gaps in how GNN-on-omics claims are currently evaluated.
Diplomatic in tone — observe gaps as patterns in the field, not as
indictments of specific authors. Cite the recent reviews that have
already named the issues at field level.

**Sub-structure:**

- **Opening (~80 words).** The proliferation of GNN architectures for
  cancer prognosis has outpaced the methodological scrutiny applied
  to their comparisons. Two recent reviews have argued that
  external validation should be a default expectation rather than an
  optional addition; both observe that within-cohort lifts have been
  treated as sufficient evidence of architectural superiority more
  often than they should be.
  - Citations: Liang 2025, Vavekanand 2026.

- **Gap 1: within-cohort gene selection + within-cohort evaluation
  inflates apparent gains (~200 words).** The mechanism: when a gene
  selection step (LASSO, RFE, mutual information, etc.) is fit on the
  full cohort and then folds are constructed for evaluation, every
  fold's "validation" patients have already participated in the gene
  selection. The selected genes are partly chosen because they
  correlate with those validation patients' labels. Cox PH and GNNs
  built on such gene sets inherit this leakage; the resulting C-index
  numbers are upper bounds on what the same architecture would deliver
  with leakage-corrected gene selection. Per-fold refit of the
  selection step is the standard correction; it is not always applied.
  Brief mention: this thesis quantifies the leakage cost on the
  inherited TCGA-BRCA pipeline at +0.072 C-index (Methods §5.1).
  - Citations: Tibshirani 1996 (LASSO), Liang 2025 (review citing the
    inflation pattern).

- **Gap 2: lift attribution is rarely decomposed (~200 words).** The
  mechanism: a GNN reports a headline C-index lift over a Cox PH
  baseline and the lift is attributed to the GNN's architectural
  inductive bias. But Cox PH operates on a small set of clinical
  features and a low-dimensional projection of expression; the GNN
  has access to the same clinical features through its head, with
  added non-linear capacity. A headline GNN-vs-Cox lift therefore
  conflates (a) gene-graph contribution from the GNN's relational
  inductive bias with (b) non-linear-flat-feature contribution from
  having a more flexible head. Decomposing the two requires an MLP
  reference operating on the same clinical features without the gene
  graph; this reference is rarely reported. The decomposition matters
  because the two contributions have very different magnitudes on
  TCGA-shaped data, and reporting only the headline conflates them.
  - Citations: Gao et al. 2021 (clinical-fusion ablation as design-
    choice precedent), Choudhry et al. 2025 (representative recent
    GNN-on-cancer paper; cited as field-level pattern, not as
    indictment).

- **Why TCGA-BRCA + METABRIC is the right setting to address these
  gaps (~80 words).** TCGA-BRCA is the canonical training cohort with
  RNA-seq expression and complete clinical annotation. METABRIC is
  publicly available, microarray-based (which stresses cross-platform
  transfer), with a large event count (824 events on 1,466 patients)
  that gives paired-bootstrap tests sufficient statistical power to
  resolve small architectural effects. The pair is not novel as
  source-target cohorts; what is missing is the methodological
  framework that uses them rigorously.
  - Citations: Curtis et al. 2012 (METABRIC original).

- **Closing (~40 words).** Set up §3: the next section states what
  this thesis does about both gaps.

---

## §3 Thesis position and structure (~600 words; ~4 citations)

What this thesis does. Three components, two coordinate findings, one
methodological framework. Research-question sentence as a slot, drafted
last after the rest of §3 is written.

**Sub-structure:**

- **Opening (~60 words).** This thesis builds a leakage-corrected
  GraphSAGE for TCGA-BRCA prognosis, evaluates it on METABRIC with
  paired-bootstrap on identical patients, and decomposes the lift
  against a non-linear MLP-clinical reference. The methodological
  framework comprises per-fold leakage correction during training
  and paired-bootstrap on identical external patients during
  evaluation. One-line acknowledgement of inherited preprocessing
  (the 769-gene universe carries forward; its leakage characteristics
  are audited and corrected per fold).

- **Three architectural / methodological components (~180 words).**
  - **(i) The model.** Two-layer GraphSAGE on per-patient gene graphs
    with STRING PPI edges; per-fold LASSO gene selection within the
    inherited universe; clinical late-fusion; MLP head; Cox partial-
    likelihood loss. The architecture is held minimal — a deliberate
    discipline so that ablation results are interpretable.
  - **(ii) The comparator.** Paired-bootstrap on identical external
    patients (n = 1,466 METABRIC), with knob A trained on full TCGA
    and Cox PH baseline trained on the same data and applied to the
    same external cohort. The paired test on identical patients
    eliminates the patient-fold-assignment variance that fold-mean
    averaging leaves uncontrolled.
  - **(iii) The decomposition.** A non-linear MLP-clinical-only
    reference establishes the flat-feature ceiling against which the
    gene-graph contribution is honestly measured. The +0.013 c-index
    contribution above MLP-clinical is the actual gene-graph signal;
    larger headline numbers conflate this with non-linear-clinical
    contribution.
  - Citations: Madanipour et al. 2024 (inductive paradigm justification).

- **Two coordinate findings, previewed (~140 words).** Stating both
  findings up front lets the reader navigate the body chapters with
  the synthesis already in mind.
  - **Finding 1.** A leakage-corrected GraphSAGE matches the strongest
    available Cox PH baseline on TCGA-BRCA internally and beats matched
    Cox PH on external METABRIC validation with paired-bootstrap
    significance (Δ = +0.053, 95% CI [+0.031, +0.076], P < 0.001 over
    2,000 resamples).
  - **Finding 2.** Two architectural elaborations — Reactome pathway
    pooling and BioBERT-PCA gene priors — significantly underperform
    the minimal architecture on external validation despite competitive
    internal performance (paired Δ = −0.016 and −0.039 respectively,
    both with 95% CIs strictly below zero on identical METABRIC
    patients).
  - Citations: Lee et al. 2020 (BioBERT precedent, foreshadows knob C),
    Chen and Zou 2023 (GenePT precedent for the LLM-prior hypothesis).

- **The methodological framework as the thesis's contribution (~80 words).**
  The framework is what produced both findings. Per-fold leakage
  correction during training plus paired-bootstrap on identical
  external patients during evaluation is reusable across prognosis
  tasks; the framework's value is demonstrated by what it permits
  (Finding 1) and what it refuses (Finding 2).
  - No new citations.

- **Boundedness of contribution (~80 words).** Down-translated from
  Discussion §7. The framework applies to small-event survival CV
  with one external cohort available; the gene-graph contribution
  identified is small but real and architecture-invariant within the
  design space examined; the present implementation operates within
  the inherited gene universe, not on a from-scratch leakage
  correction (Discussion §5 names this as the principal limitation).
  Hedge to match the Discussion's tone — claim what was done, not
  more.

- **Research question (~30 words).** `[placeholder slot — drafted last]`
  Anchor: derived from the case built in §1 and §2; states what the
  thesis asks, not what it claims. Tone target: a question the reader
  would arrive at independently from §1 + §2. Match Discussion §7's
  framing: not "can a GNN beat Cox PH?" but "what does a leakage-
  corrected, externally-paired-bootstrap-tested GNN-vs-Cox comparison
  on TCGA-BRCA reveal about the architectural choices the field has
  been making?" — or a tighter version of the same.

- **Closing transition (~30 words).** Set up §4: the next section
  describes how the chapters are organised.

---

## §4 Roadmap of the chapters (~200 words; 0 citations)

What each chapter does. Brief, declarative, no synthesis.

**Sub-structure:**

- **Methods (~50 words).** Establishes the data (TCGA + METABRIC,
  cohort scale, gene universe), the three baselines (Cox PH HONEST,
  Cox PH LEAKY, MLP clinical-only), the architecture (knob A as
  frozen design), the ablation knobs (B and C as hypothesis-tests),
  the methodological backbone (leakage audit, bootstrap CI machinery,
  paired-bootstrap-vs-fold-mean worked example, sentinels), the
  external validation protocol, and the computational details.

- **Results (~50 words).** Presents the leakage audit's +0.072 cost
  finding and the prior 0.748 figure debunk; the within-TCGA knob A
  vs Cox PH comparison; the external METABRIC paired-bootstrap result
  (Δ = +0.053, P < 0.001) with KM and calibration figures; the
  architectural ablations (knob B and knob C significantly worse
  externally) with the architecture-ablation forest plot; the
  pathway-attention interpretability artifact; and robustness checks.

- **Discussion (~50 words).** Restates the two coordinate findings,
  argues for external paired-bootstrap as the load-bearing comparator
  via the knob C worked example, presents the +0.013 gene-graph
  contribution as a small but stable architecture-invariant finding,
  documents the three negative findings with mechanism and
  generalization, names three limitations, lists six future-work items,
  and closes with the framework restated as the contribution.

- **Conclusion (~50 words).** Single tight paragraph (Conclusion is
  written separately; this slot just notes its existence and that it
  contains the thesis's closing sentence).

---

## Citation budget summary

| § | Section | Words | Citations | Source plan |
|---|---|---:|---:|---|
| 1 | Background | 600 | ~9 | Sørlie 2001, Howlader 2014, Cox 1972, Harrell 1982, Tibshirani 1996, Katzman 2018, Hamilton 2017, Vaida 2025, Madanipour 2024, Ling 2022 (≤10) |
| 2 | Field's problem | 600 | ~7 | Liang 2025, Vavekanand 2026, Tibshirani 1996, Gao 2021, Choudhry 2025, Curtis 2012 (Curtis = METABRIC source) |
| 3 | Thesis position | 600 | ~4 | Madanipour 2024, Lee 2020, Chen and Zou 2023, possibly one more |
| 4 | Roadmap | 200 | 0 | — |
| **Total** | | **2,000** | **~20** | ~15–17 unique sources |

---

## Two open questions for the user before drafting

1. **§1 citations cap at 10.** I've outlined 10 — that's at the upper
   bound of the budget. If you'd prefer §1 lighter (say, 7), Tibshirani
   1996 and Katzman 2018 could move into §2 (where they appear anyway
   in the form of LASSO and DeepSurv references) and Sørlie 2001 could
   be cut entirely (the BRCA-subtypes background might survive without
   it). I lean toward keeping at 10 — the §1 budget is "citation-dense"
   per spec.

2. **§3's "Boundedness of contribution" sub-section (~80 words).** This
   sub-section pre-emptively states what the thesis is *not* claiming,
   matching Discussion §7's bounded tone. The risk is that introducing
   the limitation framing in Introduction creates a hedge-heavy first
   chapter that primes the reader to expect a weak thesis. The
   alternative is to defer all boundedness language to Discussion §5.
   I lean toward keeping it — the Discussion will land more cleanly
   if Introduction has already set the bounded-confidence tone — but
   it's a defensible call either way.

**Procedural reminder:** outline first, draft second. Single full-chapter
revision pass after I show the prose. Don't draft section-by-section.
