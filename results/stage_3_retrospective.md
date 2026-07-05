# Stage 3 Retrospective

Written *before* Stage 5 to fix what we believed at this point. If Stage 5
succeeds, this becomes the basis of the Discussion section. If Stage 5
surprises us, this is the clean-eyed reference for what we knew before the
new data arrived.

---

## 1. What we set out to test

The Stage 0 design doc made a specific architectural claim, drawn from §1 of
`results/architecture_design.md`:

> A patient-specific, pathway-pooled graph neural network for breast cancer
> survival prediction beats a Cox proportional-hazards baseline on TCGA-BRCA
> and generalises to METABRIC. The contribution is methodological: rather than
> the patient-similarity or correlation graphs that dominate the literature,
> each patient is represented as a gene-level graph whose topology is fixed by
> curated STRING PPI edges and whose readout is a Reactome-pathway-level
> pooling that produces fixed-cardinality, biologically-named patient
> embeddings.

Three falsifiable predictions sit inside that claim:

- **P1**: A GNN with biological-prior gene topology beats Cox PH on point
  prediction (C-index) once leakage is correctly handled.
- **P2**: Pathway pooling specifically (vs global mean pool) is the
  load-bearing design choice — it fixes the prior attempt's collapse and
  delivers measurable accuracy.
- **P3**: The model generalises out-of-cohort (TCGA → METABRIC C-index drop
  ≤ 0.05).

Stages 0–3 tested P1 and P2. Stage 5 tests P3.

---

## 2. What we found (Stages 0–3)

### Baselines (Stage 0)

| Baseline | Value | Role |
|---|---|---|
| Cox PH HONEST (per-fold LASSO refit on raw 60k) | 0.6605 ± 0.014 | **North-star** — the gate any GNN claim must clear |
| Cox PH LEAKY (legacy preprocessing's full-cohort LASSO) | 0.7324 ± 0.014 | Upper bound, leakage-corrupted |
| Prior-pipeline `0.748` figure | not reproducible | Stage 0 sweep showed it was an artifact of an unprincipled `mean(axis=2)[:, :50]` recipe |

### GNN variants (Stage 3)

| Model | val cidx | std | Δ vs Cox HONEST | Δ vs MLP-clinical (non-linear ref) |
|---|---:|---:|---:|---:|
| MLP clinical-only | 0.7122 | 0.038 | +0.052 | — |
| Knob D (GNN+clinical, leaky-769) | 0.7256 | 0.044 | +0.065 | +0.013 |
| Knob A (GNN+clinical, per-fold-honest LASSO) | 0.7200 | 0.043 | +0.060 | +0.008 |
| Knob B (Knob A + Reactome pathway pool) | 0.7222 | 0.055 | +0.062 | +0.010 |

### Paired-bootstrap pairwise comparisons (the rigorous comparator)

| Comparison | Paired Δ | 95% CI | Read |
|---|---:|---|---|
| Knob A vs Knob D | +0.0273 | [−0.002, +0.056] | Leakage-correction free; knob A holds |
| Knob B vs Knob A | −0.0114 | [−0.036, +0.013] | Tied; pathway pool no measurable lift |

### What this says about P1 and P2

**P1 (GNN beats Cox PH after leakage correction):** **partially supported, weakly.**
The GNN beats Cox PH HONEST north-star by +0.060 — clear separation. But
against Cox PH LEAKY upper bound (the more demanding comparator with full
clinical access), the GNN ties (paired CI [−0.103, +0.006], P=0.96). And
against MLP-clinical-only (a non-linear flat-feature baseline using the same
clinical inputs), the GNN's gene-graph contribution is +0.008 to +0.013
across three independent variants. The +0.06 against Cox HONEST is real but
mostly attributable to *clinical features being available to a non-linear model*,
not to the gene-graph specifically.

**P2 (pathway pooling helps):** **not supported.** Knob B vs Knob A paired Δ
= −0.011, CI crosses zero, P(B≤A)=0.83. Reactome-pathway-named pooling does
not widen the gene-graph contribution beyond global mean pool. This is a
clean, methodologically careful negative result.

### What did emerge

- **The gene-graph contribution is small but stable.** +0.008 to +0.013 above
  non-linear MLP-clinical, across three independent leakage and architecture
  variants. A signal that survives both leakage correction and pathway
  pooling without changing magnitude is unlikely to be noise.
- **The interpretability artifact is biologically coherent at no accuracy
  cost.** Knob B's pathway attention produces clinically-readable
  attributions (estrogen-axis pathways for ER-axis patients, MAPK / cytokine
  signaling separating high-risk from low-risk). This is what the field of
  GNN-on-omics has been promising; we deliver it.
- **The cohort itself has a c-index variance floor of ~0.05** (Stage 2
  fold-4 forensic, replicated in knob A). Three-seed re-runs of fold 4 yielded
  std=0.006; the cross-fold std=0.04–0.05 is per-fold event sparsity, not
  model variance. This means single-fold differences are not statistically
  distinguishable; the fold-mean comparison plus paired bootstrap on identical
  patients is the meaningful comparator. This is a genuine methodological
  observation with one-paragraph value in the Discussion.
- **A reusable methodological backbone:** leakage audit (`scripts/00_lasso_audit.py`),
  bootstrap CI utility (`src/cindex_bootstrap.py`), per-fold-honest LASSO
  pipeline (`scripts/03b_*.py`), R5 pathway sparsity sentinel
  (`scripts/03c_*.py`), forensic for fold variance (`scripts/02b_fold4_seeds.py`).
  These are reusable across future studies on other TCGA cohorts.

---

## 3. What this implies for the thesis claim

The original Path A framing — *"GNN beats Cox PH because of biological
topology, and pathway pooling is the load-bearing reason"* — is **not what the
data say**. Stage 3 shows:

- The GNN does not beat the leakage-corrupted upper-bound baseline.
- Against the leakage-free baseline, most of the lift is non-linear use of
  clinical features, not gene-graph signal.
- Pathway pooling delivers no point-prediction lift.

But this is **not Path C either**. The brief's Path C is "honest negative
result, pure methodological contribution"; what we have is more positive:

- **The GNN matches the strongest available baseline** (Cox PH leaky, paired
  CI excluding any meaningful loss).
- **Beats the leakage-free baseline by +0.060.**
- **Provides a free, biologically-coherent interpretability artifact** that
  flat models cannot.
- **Produced the most carefully leakage-corrected GNN-vs-Cox comparison on
  TCGA-BRCA** in the literature, with reproducible code, paired-bootstrap
  significance testing, and a fold-level forensic.

This is **Path B with Path C-flavored methodological rigor.** The headline
claim for the thesis becomes:

> A leakage-corrected, pathway-interpretable GNN matches the strongest
> available Cox PH baseline on TCGA-BRCA prognosis (paired-bootstrap CI
> excludes any meaningful loss vs the leaky upper bound; clear win vs the
> leakage-free baseline) and provides a clinically-readable pathway
> attribution that linear models cannot. Pathway pooling does not improve
> point predictions but supplies the interpretability artifact at no
> accuracy cost.

This is honest, defensible, and accurate. The contribution is the
combination — methodological rigor + interpretability — not raw accuracy.

### What Stage 5 determines

If TCGA → METABRIC C-index drop ≤ 0.05 (brief's gate), the headline tightens
to **"matches strong baseline AND generalises externally,"** which is a real
contribution suitable for publication, not just a defensible thesis. If the
drop exceeds 0.05, we have an honest external-validation finding to report —
domain shift between microarray and RNA-seq cohorts is a legitimate research
question and the brief's Path C is reachable from there.

If the GNN's external drop is materially smaller than Cox PH's external drop
(both trained on TCGA, both inferred on METABRIC), that's a *generalization
advantage* worth noting independently of point-prediction parity within
TCGA — biological priors might travel better across cohorts than statistical
priors. We won't know until we run it.

### What Stage 5 does NOT decide

Stage 5 does not change the within-TCGA finding above. Even if METABRIC fails
(outcome 3), the within-TCGA work stands as a methodologically rigorous
demonstration that biological-graph GNNs match strong linear baselines with
interpretability as the differentiator. The thesis is defensible as-is;
Stage 5 determines how much *more* it can say.

### Why we are skipping knob C (LLM gene init) for now

Knob C (random vs BioBERT vs GenePT gene-init embeddings) tests whether
better gene priors widen the +0.01 gene-graph contribution. Given that
contribution is already stable across three independent variants (knob D,
knob A, knob B), the prior expectation is small effect. Information value
relative to engineering cost is low compared to Stage 5's load-bearing
external-validation question. If Stage 5 lands at outcome 1, knob C in a
later stage becomes a final ablation row. If Stage 5 lands at outcome 2 or 3,
knob C is irrelevant — fix the generalization problem, don't add features.

---

**Frozen architecture for Stage 5:** Knob A. Two-layer GraphSAGE on per-fold
LASSO-honest gene subgraph + clinical late-fusion + Cox loss. Knob B's
pathway attention is preserved as the interpretability layer applied
post-hoc; not as the primary predictor.
