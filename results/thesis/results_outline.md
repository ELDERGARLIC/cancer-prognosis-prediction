# Results — Outline (structure only; no prose yet)

Organized around findings, not figures. Reading order is finding-order:
baselines first (because every later result is anchored to them), internal
TCGA second (within-cohort claims), external METABRIC third (the
load-bearing generalization claim), ablations fourth (the architectural-
minimalism finding), interpretability fifth (the artifact at no cost),
robustness sixth (the sentinels and forensics that validate everything
above).

Target total: ~2,600 words across 6 sections. Lighter than Methods (3,594)
because findings are more concentrated than methodological apparatus.
Source material is per-stage summaries + retrospectives + figure JSONs.

**Discipline rules (from user, locked):**
- Don't editorialize. "We observe..." or bare statement, not "This shows..."
- Citations sparse. Only `lifelines` (Davidson-Pilon 2019) and `sksurv`
  (Pölsterl 2020) where C-index methodology is mentioned; nothing else.
- Don't reproduce the holistic table from the retrospective verbatim.
  Build a leaner Results table (5 rows × 4 cols).
- Knob C TCGA-internal-vs-METABRIC-external sign reversal = single most
  prominent finding in §4.
- Lift-attribution sentence (+0.013 above MLP-clinical, stable across
  three GNN variants) is methodologically critical; don't bury it.

---

## §1. Baselines and the prior-pipeline correction (≈400 words)

The first finding is the leakage audit: the prior 0.748 figure is not
reproducible with sensible Cox PH, and the per-fold-honest north-star is
+0.072 c-index lower than the leaky upper bound.

- **The prior 0.748 debunk.** Stage 0 diagnostic sweep over Cox PH
  configurations (PCA ∈ {20, 50, 100, 200} × penalizer ∈ {0.001, ..., 5},
  with/without clinical) on the same splits could not reach 0.748. Trace
  to non-standard `mean(axis=2)[:, :50]` recipe. The 0.748 is excluded
  from comparisons.
- **Cox PH HONEST north-star = 0.6605 ± 0.049.** Per-fold LASSO refit on
  raw 60k-gene matrix from each fold's training partition only;
  PCA(100) + 7 clinical, ridge penalizer 0.5; C-index averaged across
  the 5 stratified val folds.
- **Cox PH LEAKY upper bound = 0.7324 ± 0.014.** Same Cox PH formulation
  on the prior 769-gene universe whose construction used full-cohort
  labels. Reported only as upper bound.
- **Leakage cost: +0.072 ± unknown.** Per-fold deltas: [+0.082, +0.064,
  +0.140, −0.016, +0.089]. Fold 3 anti-leakage (honest > leaky) noted as
  data-driven fold variance, not artifact; fold-level variance is
  characterized in §6.
- **Cohort scale.** TCGA n=1074 (after dropping 21 patients with
  OS.time ≤ 0); 150 events; 5-fold stratified on (survival_class × OS).
  Per-fold val ≈ 215 patients with ~30 events.

**Figure placement:** none (table-only section).
**Table 1:** baseline summary (Cox HONEST / Cox LEAKY / cohort scale /
prior 0.748 status). 4 rows. Anchors §2–§4.
**Citations:** none.

Source: `results/stage_0_summary.md`, `results/stage_0_baseline_diag` sweep.

---

## §2. Internal TCGA — knob A vs Cox PH baselines (≈600 words)

The within-cohort comparisons that anchor the external claim. Knob A
matches the leaky upper bound and beats the honest north-star; the lift
attribution against MLP-clinical isolates the gene-graph signal.

- **Knob A internal performance.** Mean val C-index across 5 folds =
  0.7200 ± 0.043. Per-fold: [0.7360, 0.7627, 0.6887, 0.7599, 0.6525].
  R1 cosine differentiation passed all folds.
- **Knob A vs Cox PH HONEST (north-star comparison).** Δ point = +0.0595
  on fold-mean. The result clears the methodological gate.
- **Knob A vs Cox PH LEAKY (upper-bound comparison).** Δ point = −0.012
  on fold-mean. Paired-bootstrap CI = [−0.103, +0.006] grazing zero;
  knob A is statistically tied with the leakage-corrupted upper bound,
  P(A ≤ Cox-LEAKY) = 0.96 (within paired-bootstrap noise).
- **MLP clinical-only reference = 0.7122 ± 0.038.** Two-layer MLP on
  the same 7-clinical features, same Cox loss, no gene graph. Establishes
  the non-linear flat-feature ceiling.
- **The lift-attribution finding (load-bearing sentence):** "The gene-
  graph contribution above non-linear MLP-clinical is +0.013 internally,
  stable to within bootstrap noise across all three GNN variants
  (knob D: +0.013; knob A: +0.008; knob B: +0.010), establishing a small
  but architecture-invariant gene-side signal." This is the "small but
  real" finding; everything in §4 builds on it.
- **Paired-bootstrap on knob A vs precursor knob D** (worked-example
  reference back to Methods §5.3). Fold-mean Δ = −0.006; paired Δ on
  identical patients = +0.027, 95% CI [−0.002, +0.056], P(A ≤ D) = 0.033.
  The two statistics disagree on sign by design (Methods §5.3); paired
  is the rigorous comparator. **One sentence reference to Methods §5.3,
  not full re-explanation.**

**Figure placement:** none (table-driven section).
**Table 2 (the leaner Results table):** 5 rows × 4 metric columns:
| Model | TCGA mean ± std | METABRIC | Δ vs A (TCGA paired) | Δ vs A (METABRIC paired) |
- Cox PH HONEST | 0.6605 ± 0.049 | 0.5914 (full) | − | −0.053 [−0.076, −0.031] |
- MLP clinical-only | 0.7122 ± 0.038 | not run | n/a | n/a |
- Knob A | 0.7200 ± 0.043 | 0.6443 [0.624, 0.664] | reference | reference |
- Knob B | 0.7222 ± 0.055 | 0.6285 [0.607, 0.648] | −0.011 [−0.036, +0.013] | **−0.016 [−0.026, −0.006]** |
- Knob C | not internal-only run | 0.6054 [0.583, 0.627] | n/a | **−0.039 [−0.059, −0.021]** |

(Cox PH LEAKY and knob D excluded; supplementary table reference from text.)
**Citations:** Davidson-Pilon 2019, Pölsterl 2020 (one mention each, on
first paired-bootstrap reference).

Source: `results/stage_3_summary.md`, `results/stage_3a_*.json`,
`results/stage_3b_*.json`, `results/stage_3c_*.json`,
`results/stage_3_ref_mlp_clinical.json`.

---

## §3. External METABRIC — knob A vs matched Cox PH (≈500 words)

The headline result of the thesis. Knob A trained on full TCGA, applied
cold to METABRIC, beats matched Cox PH with paired-bootstrap significance.

- **Cohort.** METABRIC n = 1466 (after valid-OS + valid-stage filter from
  raw 2509); 824 events; gene-overlap = 650 of 769 (84.5%); clinical
  intersection = 3 features (age, stage_ordinal, is_female).
- **Full-TCGA-trained knob A on METABRIC = 0.6443**, 95% CI [0.624, 0.664]
  (1000 bootstrap resamples).
- **Per-fold consistency check.** 5 separate models (one per TCGA fold),
  each applied to METABRIC: per-fold values [0.6340, 0.6446, 0.6445,
  0.6412, 0.6371], std = 0.004. The model gives essentially identical
  external predictions regardless of which TCGA fold trained it.
- **Cox PH on METABRIC** (full-TCGA-trained, same 3-clinical, same
  per-fold-LASSO gene set) = 0.5914, 95% CI [0.571, 0.612].
- **Paired-bootstrap Δ(knob A − Cox PH) on identical METABRIC patients**
  = **+0.0528, 95% CI [+0.0306, +0.0756], P(A ≤ Cox) = 0.000** (2000
  resamples, none crossed). The headline finding.
- **Cohort-shift drop.** Knob A internal → external drop = +0.082; Cox
  PH internal → external drop = +0.073. Both exceed the 0.05 brief gate.
  Both architectures suffer the cohort transition (microarray vs RNA-seq,
  follow-up length, population) at similar magnitude. The comparative
  advantage holds.
- **Anchor sentence:** Knob A external (0.6443) is within 0.02 c-index of
  Cox PH HONEST internal (0.6605). A model trained cold on one cohort and
  applied to a microarray-different cohort with longer follow-up scores
  within paired-bootstrap noise of the within-cohort linear baseline of
  the source cohort.

**Figure placement:**
- **Figure 1: Risk-stratified KM curves on TCGA (pooled OOF) and METABRIC.**
  Quartile split by knob A risk; log-rank χ² = 38.5 (TCGA) and 204
  (METABRIC), both p < 1e-4. Sample size and event counts per quartile
  in legend. Months on x-axis, capped at 240. Source:
  `results/figures/fig1_km_curves.png`.
- **Figure 2: Calibration — decile-bin predicted risk vs KM observed
  survival at 3-year and 5-year.** Spearman ρ at 5y = −0.89 (TCGA) and
  −0.88 (METABRIC). Source: `results/figures/fig2_calibration.png`.

**Citations:** none (figures + paired-bootstrap framework already
introduced in Methods).

Source: `results/stage_5_summary.md`, `results/stage_5_metabric_external.json`,
`results/fig1_km_stats.json`, `results/fig2_calibration_stats.json`.

---

## §4. Architectural ablations — knob B and knob C on TCGA and METABRIC (≈600 words)

The second coordinate finding from the thesis. Both elaborations
significantly underperform knob A on the external paired test, with knob
C providing the worked example for why external validation is required to
detect the harm.

- **Knob B (Reactome pathway pooling) — hypothesis-prediction-result.**
  Hypothesis: biological-pathway grouping adds discriminative signal beyond
  global mean pool. Prediction: +0.02 to +0.04 lift expected (Choudhry
  2025, Vaida 2025). Result internal: paired Δ vs A = −0.011, 95% CI
  [−0.036, +0.013], P(B ≤ A) = 0.83 (CI crosses zero, tied). Result
  external: paired Δ vs A = **−0.0158, 95% CI [−0.0259, −0.0063],
  P(B ≤ A) = 1.000** (CI strictly below zero, B significantly worse).
  R5 retained 5–19 pathways per fold; no degenerate fold.
- **Knob C (BioBERT-PCA gene init) — hypothesis-prediction-result.**
  Hypothesis: LLM-derived gene priors add unique information beyond
  expression. Prediction: +0.005 to +0.025 lift expected (Chen and Zou
  2023 GenePT precedent). Result internal: TCGA 10%-val C-index = 0.7414,
  within fold variance of knob A's 0.7492 — **competitive on internal**.
  Result external: METABRIC C-index = 0.6054, paired Δ vs A = **−0.0389,
  95% CI [−0.0589, −0.0206], P(C ≤ A) = 1.000** (CI strictly below zero).
- **Open §4's knob-C subsection with the contrast** (per user spec):
  "Knob C demonstrates the methodological framework's value most directly.
  On TCGA internal validation, knob C achieves C-index 0.7414, within
  fold variance of knob A's 0.7492 — a tie under any reasonable internal-
  only assessment. On the METABRIC external paired test, knob C scores
  0.6054 with paired Δ versus knob A of −0.039 (95% CI [−0.059, −0.021],
  P = 1.000), a result the internal comparison could not have predicted."
- **Cohort-shift drop comparison.** Knob A drop = 0.082 (knob A external
  ÷ internal anchored at 5-fold mean); knob C drop = 0.136 (internal 0.741
  → external 0.605); knob C drops ~1.7× knob A's amount.
- **The TCGA-internal-vs-METABRIC-external sign reversal for knob C** is
  the most prominent finding in §4. The Discussion will return to this;
  Results introduces it in plain numerical terms.

**Figure placement:**
- **Figure 4: Architecture-ablation forest plot.** Three panels (TCGA
  internal C-index / METABRIC C-index with 95% CI / paired Δ vs knob A
  on METABRIC with CI bars). Knob A, B, C, Cox PH as four rows.
  Source: `results/figures/fig4_architecture_forest.png`. **Primary
  location for this figure is here, not §3.**

**Citations:** none.

Source: `results/stage_3_summary.md`, `results/stage_5b_*.json`,
`results/stage_5c_*.json`, `results/fig4_architecture_forest_stats.json`.

---

## §5. Interpretability artifact — pathway-level attention from knob B (≈300 words)

Knob B underperforms on point prediction but is preserved as the
interpretability layer. The pathway-attention weights are biologically
coherent and clinically readable.

- **Construction.** Each METABRIC patient's risk score is decomposed into
  a 200-dimensional Reactome-pathway attention vector by knob B; this
  thesis reports the top-attended pathways for the highest- and lowest-
  risk patients, fold-averaged across the 5 TCGA folds.
- **Top-10 pathways by total attention across 25 high-risk + 25 low-risk
  patient slots × 5 folds:** Interleukins, Cytokine Signaling in Immune
  System, MAPK Family Signaling Cascades, IL-4 and IL-13 Signaling,
  Receptor Tyrosine Kinases, ESR Mediated Signaling, Signaling by Nuclear
  Receptors, Notch, Estrogen-Dependent Gene Expression, Platelet
  Activation Signaling.
- **Pattern observed.** Immune-axis pathways (Interleukins, Cytokines,
  IL-4/IL-13) attend most heavily on the highest-risk val patients;
  estrogen-axis pathways (ESR Mediated Signaling, Estrogen-Dependent Gene
  Expression, Nuclear Receptors) attend most heavily on the lowest-risk
  patients. MAPK and platelet activation are broad across both groups.
- **Biological coherence.** ER-positive tumors generally have better
  prognosis (Howlader et al., 2014, NCI SEER); inflammatory/cytokine
  signaling has been associated with aggressive subtypes (e.g.,
  triple-negative). The pattern observed is consistent with established
  BRCA biology.
- **Interpretability without point-prediction cost.** Knob B provides this
  artifact at no point-prediction lift over knob A (§4); the Discussion
  considers whether the artifact alone justifies the architectural
  complexity.

**Figure placement:**
- **Figure 3: Pathway-attention heatmap.** Top-10 pathways × 10 patient
  slots (5 highest-risk + 5 lowest-risk, fold-averaged). Color = mean
  attention weight. Vertical separator between high-risk and low-risk
  groups. Source: `results/figures/fig3_pathway_attention.png`.

**Citations:** Howlader 2014 (NCI SEER) for ER-positive prognosis; no
others.

Source: `results/stage_3c_attention_per_fold.json`,
`results/fig3_pathway_attention_stats.json`.

---

## §6. Robustness checks (≈200 words)

The sentinels and forensics that validate everything above. Brief,
declarative, no editorializing.

- **R1 embedding-collapse sentinel.** All five folds passed both thresholds
  (catastrophic > 0.99 and differentiation init−final > 0.02) for knobs
  A, B, and C across the 30-epoch training trajectory. The prior attempt's
  failure mode does not recur.
- **C-index implementation cross-check.** `lifelines.utils.concordance_index`
  vs `sksurv.metrics.concordance_index_censored` agreed to within 10⁻⁵
  c-index across all 5 Cox PH HONEST val folds; identical to within
  numerical precision.
- **Fold-variance forensic.** Three-seed re-run of the worst-performing
  Stage 2 fold (fold 4) yielded a seed-conditioned std of 0.006, an order
  of magnitude smaller than the cross-fold std of approximately 0.05,
  confirming the variance floor is data-driven (per-fold event sparsity
  ~30 events per val fold) and not seed-pathological.
- **External per-fold consistency.** The 5 separate per-fold-trained knob
  A models, each applied to METABRIC, yielded scores [0.634, 0.645, 0.645,
  0.641, 0.637]; std = 0.004. Robustness to which TCGA fold trained the
  model.

**Figure placement:** none.
**Citations:** none.

Source: `results/stage_2_summary.md` (R1 + fold-4 forensic),
`results/stage_0_baseline.json` (C-index cross-check),
`results/stage_5_metabric_external.json` (per-fold consistency).

---

## What's deliberately NOT in Results (referenced from Discussion)

To avoid over-loading Results with material that belongs in synthesis:

- **Knob D (the leaky-769 GNN+clinical precursor to knob A).** Mentioned
  once in §2 as the paired-bootstrap-disagreement worked example
  (reference to Methods §5.3); not reported as its own architectural row.
- **Cox PH LEAKY's per-fold details and the fold-3 anti-leakage finding.**
  Mentioned once in §1 as a data-driven fold-level variance signal;
  full per-fold table lives in supplementary material.
- **Stage 1 (multinomial LR sanity).** A 4-line passing check; one-line
  mention in §1 only if needed for argument structure.
- **Stage 2 (minimal SAGE without clinical fusion).** R1-sentinel
  validation result lives in §6; mean cidx = 0.611 cited only if
  argumentatively necessary, otherwise omitted.
- **Wall-time, software versions, hardware.** Lives in Methods §7.

---

## Summary table of word allocation and figure placement

| § | Section | Words | Figures | Tables | Citations |
|---|---|---:|---|---|---|
| 1 | Baselines + 0.748 debunk | 400 | — | T1 (4 rows) | — |
| 2 | Internal TCGA knob A | 600 | — | T2 (5×4) | DP19, P20 |
| 3 | External METABRIC | 500 | F1, F2 | — | — |
| 4 | Ablations B + C | 600 | F4 | — | — |
| 5 | Interpretability | 300 | F3 | — | Howlader14 |
| 6 | Robustness | 200 | — | — | — |
| **Total** | | **2,600** | F1–F4 | T1, T2 | 3 sources |

**Procedural reminder:** outline first, draft second. Single full-chapter
revision pass after I show the prose. Don't draft section-by-section.
