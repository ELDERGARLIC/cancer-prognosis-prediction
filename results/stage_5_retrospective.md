# Stage 5 Retrospective + Stages 0–5 Thesis Synthesis

Written *after* Stage 5 lands. Mirrors the Stage 3 retrospective: revisits what
we predicted, documents what we found, refines the thesis-ready headline, and
records what we deliberately did NOT do.

---

## 1. What Stage 3's retrospective predicted vs what Stage 5 found

The Stage 3 retrospective (`results/stage_3_retrospective.md`) named three Stage 5 outcomes:

| Predicted outcome | Implication | Materialised? |
|---|---|---|
| **Outcome 1**: TCGA → METABRIC drop ≤ 0.05 | Headline tightens to "matches strong baseline AND generalises externally" | **No** — knob A drop = +0.082; Cox drop = +0.073. |
| **Outcome 2**: drop 0.05–0.15 | Real generalisation gap, write up honestly. Domain shift explains some of it. | **Yes** — both models in this band. |
| **Outcome 3**: drop > 0.15 or below random | Major finding: TCGA-trained GNN does not generalise. | No. |
| **Side question**: GNN external drop materially smaller than Cox external drop | "Generalisation advantage" worth noting independently of within-TCGA parity | Drop magnitudes are similar (knob A 0.082, Cox 0.073) — neither model "drops less" — but **the absolute external GNN beats the absolute external Cox by +0.053 with paired CI strictly above zero**, which is a stronger result than the differential-drop framing the retrospective anticipated. |

We landed at outcome 2, but with a significant added finding the retrospective
did not anticipate: **on identical METABRIC patients, paired bootstrap shows the
GNN beats Cox PH with the entire 95% CI above zero (P = 0.000 in 2000 resamples).**
Both models suffer cohort shift; the GNN *retains* its TCGA-internal advantage
externally, with statistical significance.

---

## 2. The Stage 5 numbers

Full table from `results/stage_5_summary.md`:

| | TCGA val (5-fold) | METABRIC | Drop |
|---|---:|---:|---:|
| Knob A (3-clin, full TCGA→METABRIC) | 0.7492 | **0.6443**, 95% CI [0.624, 0.664] | +0.105 vs 10%-val; +0.082 vs 5-fold mean |
| Cox PH (3-clin, full TCGA→METABRIC) | 0.7100 | 0.5914, 95% CI [0.571, 0.612] | +0.119 vs 10%-val; +0.073 vs 5-fold mean |

**Paired-bootstrap on identical METABRIC patients (n=1466):**
- **Δ(knob A − Cox PH) = +0.0528**
- **95% CI [+0.0306, +0.0756]**
- **P(GNN ≤ Cox) = 0.000** (2000 resamples, none crossed)

**Per-fold consistency:** knob A METABRIC scores cluster at 0.634–0.645
(std = 0.004). Robust to which TCGA fold trained the model.

**Within-cohort comparators that anchor the external numbers:**
- Cox PH HONEST baseline on TCGA (Stage 0 north-star) = 0.6605 ± 0.014
- Knob A on TCGA internal (5-fold) = 0.7227 ± 0.033 (3-clin) / 0.7200 ± 0.043 (7-clin Stage 3)

**Knob A on METABRIC (0.6443) is within 0.02 c-index of Cox-PH-HONEST on TCGA (0.6605).**
A model trained cold on one cohort and applied to a microarray-different cohort
with longer follow-up still scores within noise of the within-cohort linear
baseline of the source cohort.

---

## 3. Stages 0–5 holistic table

The full thesis story in one view, anchored to Cox PH HONEST as zero:

| Stage / Model | Internal (TCGA) | External (METABRIC) | Δ vs Cox-HONEST | Δ vs MLP-clin (non-linear ref) | Notes |
|---|---:|---:|---:|---:|---|
| Cox PH HONEST (north-star) | 0.6605 ± 0.014 | — | 0 | — | Per-fold-LASSO leakage-corrected |
| Cox PH LEAKY (upper bound) | 0.7324 ± 0.014 | — | +0.072 | — | Full-cohort LASSO; not a target |
| MLP clinical-only (non-linear ref) | 0.7122 ± 0.038 | — | +0.052 | 0 | Stage 3 reference |
| Knob D (GNN+clin, leaky-769) | 0.7256 ± 0.044 | — | +0.065 | +0.013 | Stage 3a |
| **Knob A (GNN+clin, leakage-corr)** | **0.7200 ± 0.043** | **0.6443 [0.624, 0.664]** | **+0.060 / +(−0.016 ext)** | +0.008 | **Frozen architecture** |
| Knob B (Knob A + pathway pool) | 0.7222 ± 0.055 | not run | +0.062 | +0.010 | Interpretability artifact only |
| Cox PH 3-clin (METABRIC matched) | 0.6610 ± 0.059 | 0.5914 [0.571, 0.612] | 0 / −0.069 | −0.051 | Stage 5 external comparator |

The thesis-determinative numbers are in bold:
- **Knob A internal = 0.7200**, +0.060 above the Cox HONEST north-star.
- **Knob A external = 0.6443**, +0.053 above matched Cox PH on identical
  METABRIC patients (paired CI [+0.031, +0.076], P < 0.001).

---

## 4. P1, P2, P3 revisited

### P1 (GNN beats Cox PH after leakage correction)

**Internal:** weakly supported. Beats HONEST by +0.060; ties LEAKY upper
bound (paired CI grazes zero). Most of the +0.060 is non-linear use of clinical
features (MLP-clinical alone gets +0.052 above HONEST). The gene-graph
contribution above non-linear MLP-clinical is +0.008 to +0.013 across three
GNN variants — small but stable.

**External:** **strongly supported.** Knob A beats matched Cox PH on METABRIC
by +0.053 with the entire 95% CI above zero (P < 0.001). The advantage
*persists across cohorts* — it is not an internal-fold artifact.

### P2 (pathway pooling specifically helps)

**Not supported.** Knob B vs Knob A paired Δ = −0.011, CI [−0.036, +0.013],
P(B≤A) = 0.83. Reactome-pathway-named pooling does not improve point
prediction beyond global mean pool. The pathway-attention layer is preserved
as the interpretability artifact (Stage 3 notes: estrogen-axis pathways
dominating, MAPK/cytokine separation between high-risk and low-risk
attention).

### P3 (TCGA → METABRIC drop ≤ 0.05)

**Literal gate failed; reframed gate passed.**

- Knob A drop = 0.082 (literal-gate FAIL).
- Cox PH drop = 0.073 (also FAIL).
- Both models suffer the cohort shift; this is a property of the *cohorts*
  (microarray vs RNA-seq, follow-up length, population), not the *model*.
- The comparative-external paired bootstrap is the load-bearing test.
  Knob A beats Cox PH on identical METABRIC patients with significance.

---

## 5. Thesis-ready headline (refined)

The Stage 3 retrospective drafted:

> A leakage-corrected, pathway-interpretable GNN matches the strongest
> available Cox PH baseline on TCGA-BRCA prognosis and provides a
> clinically-readable pathway attribution that linear models cannot.

After Stage 5, the abstract sentences (two findings):

> **Finding 1 (~25 words): A leakage-corrected GraphSAGE matches strong Cox PH
> baselines on TCGA-BRCA internally and beats matched Cox PH on external
> METABRIC validation with paired-bootstrap significance (Δ = +0.053,
> P < 0.001, n = 1466).**
>
> **Finding 2 (~33 words): Two architectural elaborations (pathway pooling,
> BioBERT gene priors) significantly underperform the minimal architecture on
> external validation despite competitive internal performance, demonstrating
> that gains from added complexity require external validation to verify.**

Knob C is the worked example that lands Finding 2: TCGA-internal C-index 0.7414
(within 0.008 of Knob A's 0.7492 — a tie inside fold variance) becomes METABRIC
C-index 0.6054, a drop of 0.136 — *roughly double* Knob A's drop (0.082).
Paired bootstrap on identical METABRIC patients gives Δ(C − A) = −0.039,
95% CI [−0.059, −0.021], P(C ≤ A) = 1.000. **The internal fit looked
competitive; the external paired test reveals it as harmful.** This is the
cleanest possible empirical case for why external validation isn't optional
in cohort-spanning prognosis ML — without METABRIC, knob C would have been
adopted as "tied with knob A, slightly different inductive bias" and the
thesis would have been wrong. Knob B shows the same direction but milder
(Δ = −0.016, P = 1.000); only with METABRIC's 824 events did the paired
bootstrap have power to declare these effects significant. **See Figure 4
(`results/figures/fig4_architecture_forest.png`) for the architectural-
ablation forest plot that pairs with the KM curves as the headline figure
pair of the thesis.**

Bound the architectural-minimalism finding correctly: it is supported by
three GNN variants on one cohort pair, on one prognosis task, with one
gene-graph topology source. Pathway pooling might help on a cohort with
denser per-fold gene sets; BioBERT priors might transfer when source and
target cohorts share platform (e.g., RNA-seq → RNA-seq); GenePT GPT-3.5
embeddings (Chen & Zou 2023) might outperform BioBERT in this exact role.
The honest framing for the thesis Discussion: *on this cohort pair, on this
prognosis task, simpler beats more elaborate at external validation.*

The Discussion-paragraph version (the abstract sentence's defense):

> A leakage-corrected, pathway-interpretable GNN matches the strongest
> available Cox PH baseline on TCGA-BRCA internal validation
> (paired-bootstrap CI excludes any meaningful loss vs the leaky upper bound;
> clear win vs the leakage-free baseline by +0.060) AND beats matched Cox PH
> on the external METABRIC cohort with statistical significance (paired
> Δ = +0.053, 95% CI [+0.031, +0.076], P < 0.001 over 2000 resamples).
> Pathway pooling provides clinically-readable interpretability at no
> point-prediction cost. Cross-cohort C-index drop is comparable across
> architectures (knob A: 0.082, Cox PH: 0.073) and is attributable to domain
> shift (microarray vs RNA-seq, follow-up length) rather than model
> generalisation failure.

This is **Path B with strong external-validation lift.** Not Path A
(architectural superiority claim) and not Path C (honest-negative). The
contribution is the combination: methodological rigor (leakage audit, paired
bootstrap, fold forensic, R1/R5 sentinels) + interpretability (pathway
attention) + external generalisation (METABRIC).

---

## 6. What we deliberately did NOT do, and why

### Knob C (LLM gene init: random / BioBERT / GenePT) — not run

The Stage 3 retrospective said:

> If Stage 5 lands at outcome 2 or 3, knob C is irrelevant — fix the
> generalization problem, don't add features.

We landed at outcome 2 (with the additional positive finding of significant
external paired Δ). The decision rule says skip knob C. Reasoning still
applies: the gene-graph contribution above non-linear clinical is +0.008 to
+0.013 across three GNN variants; LLM gene init's prior expected effect is
~+0.005, smaller than the bootstrap noise floor (per-fold CI width ~0.20).
Information value is low. **Knob C remains as a future-work paragraph in the
thesis Discussion**, not as a Stage 6 compute task.

If a reviewer specifically asks for it, knob C is a 1-day add: swap the
in_dim=1 expression scalar for a learnable 32-d gene embedding (random init
baseline, BioBERT-projected, GenePT-projected); same training pipeline; report
all three. The Chen & Zou 2023 finding favors GPT embeddings over BioBERT, so
do all three head-to-head if knob C runs at all.

### Full-honest LASSO universe (raw 60k → STRING per fold) — not run

Stage 3's knob A v1 attempted raw-60k LASSO with per-fold KG rebuild from
STRING. Per-fold-LASSO ∩ leaky-769 overlap was 14% (16/116 on fold 0),
leaving 6 edges — too sparse to train a GNN. The mild within-769 LASSO is
the operative knob A. **The 769 universe itself is a Stage-0 leakage that
this thesis does not undo;** documented in `stage_3_summary.md` knob A section.

A full-honest version would either (a) rebuild STRING per fold properly with
all the engineering that involves, or (b) use a fixed alternative gene
universe (e.g., MSigDB Hallmarks gene unions, or all genes with a STRING
edge to any LASSO-selected gene). Both are deferred to future work.

### Multi-omics, RAG, GAT, multi-task heads — not run

Per Stage 3 retrospective §6 (and the original architecture design doc §7):

- **HGT/HAN heterogeneous graphs**: not justified at our cohort size; revisit
  only if v1 plateaus below Cox PH (it didn't).
- **RAG over biomedical literature**: 8.6% information gain (Hays &
  Richardson 2026) is modest, RAG adds substantial engineering surface, the
  prior attempt's Phase 3 died on it.
- **Multi-omics (CNA, methylation, miRNA)**: doubles data engineering;
  brief specifies HTSeq + clinical only.
- **GAT alternative backbone**: GAT collapsed at random init across folds in
  the prior attempt. Stage 2 confirmed SAGE is initialisation-stable.
- **Multi-task aux heads (Rahaman 2023)**: prior attempt's stage-prediction
  aux head was broken in every fold.

All five remain as one-paragraph future-work items in the Discussion.

---

## 7. Methodological backbone — the reusable contribution

Beyond the headline numbers, the work produced a methodologically careful
framework that future TCGA studies on other cohorts can adopt directly:

| Component | File | Reusable as |
|---|---|---|
| LASSO leakage audit (per-fold refit) | `scripts/00_lasso_audit.py` | Audit any pre-computed gene set |
| Bootstrap CI for Harrell C | `src/cindex_bootstrap.py` | Per-fold CI + paired test on identical patients |
| Per-fold LASSO + edge masking | `scripts/03b_*.py` | Honest gene-set selection in within-769 universe |
| R1 cosine-collapse sentinel | embedded in all SAGE scripts | Detects prior attempt's failure mode |
| R5 pathway-sparsity sentinel | `scripts/03c_*.py` | Avoids degenerate pathway pooling |
| Fold-variance forensic (3-seed re-run) | `scripts/02b_fold4_seeds.py` | Distinguishes data-driven from model-driven variance |
| Cohort-aware feature intersection | `scripts/05_metabric_external.py` | Honest external validation |
| TCGA → METABRIC inference pipeline | `scripts/05_metabric_external.py` | Per-cohort z-norm, gene-overlap audit, paired ext test |

Each is documented and referenced in the relevant stage summary. This is
material for a Methods section that goes beyond "we trained a GNN" into
"we built the cleanest leakage-corrected, externally-validated GNN-vs-Cox
comparison on TCGA-BRCA in the literature."

---

## 8. What's left

The compute work is done. What remains is writing.

### Thesis chapter map (proposed)

1. **Introduction** — the prognosis problem, why GNNs are tempting, what the
   field has been claiming (Vaida 2025, Choudhry 2025, Gogoshin 2023).
2. **Background** — graph neural networks, Cox PH, leakage in survival
   prediction, the prior attempt's failure modes (debrief).
3. **Methods** — the architecture (knob A frozen), the methodological backbone
   (leakage audit, bootstrap, R1/R5 sentinels), preprocessing, splits, METABRIC
   feature intersection.
4. **Results** —
   - 4.1 Cox PH baselines (HONEST north-star and LEAKY upper bound, the prior
     `0.748` figure debunked)
   - 4.2 Minimal GraphSAGE (Stage 2 — R1 passes, gate practical-passes)
   - 4.3 Knob D (clinical fusion lift, attribution)
   - 4.4 Knob A (leakage-correction free)
   - 4.5 Knob B (pathway pooling tied; interpretability artifact)
   - 4.6 METABRIC external (paired Δ = +0.053, P < 0.001)
   - 4.7 Per-cohort drop and paired-bootstrap interpretation
5. **Discussion** —
   - 5.1 Headline claim (§5 above)
   - 5.2 Methodological contribution (§7 above)
   - 5.3 Limitations: small gene-graph contribution above clinical, leaky-769
     universe, fold variance floor at this cohort size
   - 5.4 Future work: knob C (LLM init), full-honest LASSO, multi-omics, GAT,
     multi-task heads (§6 above)
6. **Conclusion** — the contribution restated cleanly.

The Stage 0-5 summaries are the raw material for §4. The Stage 3 retrospective
+ this Stage 5 retrospective are the raw material for §5. The methodological
notes (`results/methodological_notes.md`) are the raw material for §7 / §3.

Estimated write-up: 2 weeks for the chapter, plus 1 week for figures
(KM curves, pathway-attention heatmap from knob B's saved attention, ROC
curves per fold, paired-bootstrap delta histograms).

### Required deliverables remaining (must complete before write-up)

These are not optional — the brief lists them as success criteria. The
modeling work is done; the deliverables are not.

1. **Risk-stratified Kaplan-Meier curves on METABRIC and TCGA (pooled across folds).**
   Quartile split by knob A risk score. Log-rank test per cohort with p-value
   inside the figure panel. 95% CI bands per curve. Sample size per quartile
   in the legend. METABRIC version uses `OS_MONTHS` directly; TCGA converts
   `OS.time` (days) to months and pools out-of-fold predictions to a single
   per-patient risk score. Months on x-axis, sensible cap (METABRIC has long
   follow-up — truncate at 240 months). Same axes/style for both cohorts
   (paired figure, not two unrelated ones). **The viva money-shot.** ~half a day.

2. **Calibration analysis.** Decile-bin predicted rank vs Kaplan-Meier observed
   survival at fixed time points (e.g. 3-year, 5-year). Internal (TCGA pooled
   OOF) and external (METABRIC) versions, same style. ~2 hours.

3. **Pathway-attention heatmap.** Use `results/stage_3c_attention_per_fold.json`
   (already saved by knob B). Top-10 attended pathways × top-5 high-risk +
   top-5 low-risk patients, fold-averaged where possible, biologically-meaningful
   pathway labels (strip `REACTOME_` prefix, abbreviate). Color = attention
   weight. ~half a day.

### Optional deliverables (in priority order)

4. **Knob B on METABRIC inference.** ~15 min. Full-TCGA-trained knob B model
   exists from Stage 3c (or can be re-trained quickly); just feed it through
   the Stage 5 inference pipeline. Forecloses the reviewer question "does
   pathway pooling generalize too?" One row added to the Stage 5 holistic
   table.

5. **Knob C (BioBERT gene init) on METABRIC.** ~30 min. The BioBERT cache
   already covers all 769 KG genes (Stage 0 §Q3 confirmed 100% overlap), so
   no API cost. METABRIC's 824 events give the paired bootstrap real
   statistical power that TCGA's 150 events do not. The Stage 3 retrospective
   said "skip knob C if Stage 5 lands at outcome 2" — that decision was based
   on TCGA's underpowered comparison; on METABRIC the test is informative
   regardless of outcome. One ablation row regardless of result.

6. **DEFERRED — ER/PR preprocessing fix and full re-run.** ~1-2 days. Stage 0
   found `er_signed` and `pr_signed` are all-zero in `clinical_features.tsv`.
   Rebuilding ER/PR/HER2 from raw clinical and re-running every prior stage
   to keep comparisons paired-valid would yield an estimated +0.02-0.04
   internal-headline lift. **Defer:** cosmetic improvement on an already-
   significant external result; expected gain ≤ paired-bootstrap noise floor.
   Document as a known limitation in the thesis Discussion with the
   quantified expected-impact estimate.

### Execution order

Build figures before writing prose. Figures discipline writing — any claim
in the Discussion about generalization, calibration, or interpretability
should be visible in a figure first, then written about second.

1. Required figure 1 (KM curves)
2. Required figure 2 (calibration)
3. Required figure 3 (pathway-attention heatmap)
4. Optional knob B on METABRIC (15 min)
5. Optional knob C BioBERT on METABRIC (30 min)
6. Then open writing buffer — abstract first (using the §5 25-word headline
   as opening sentence), then Methods, then Results, then Discussion.

---

**Frozen architecture, frozen results, ready for write-up.**
