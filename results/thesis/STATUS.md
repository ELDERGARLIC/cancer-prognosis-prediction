# Thesis Submission Status

Generated after assembly tasks 1–3 (document assembly, bibliography parsing,
code release prep) completed. The thesis is ready for supervisor read; this
file lists what was assembled, what placeholders remain, and what the next
human step is.

---

## What was assembled

### Submission-ready documents

| Artifact | Path | Size |
|---|---|---|
| Full thesis PDF (32 pages, 4 figures embedded) | `results/thesis/thesis_full.pdf` | 2.5 MB |
| Full thesis Markdown (single document for ongoing edits) | `results/thesis/thesis_full.md` | ~80 KB |
| Intermediate HTML (pandoc → weasyprint pipeline) | `results/thesis/thesis_full.html` | ~91 KB |
| Bibliography (Vancouver style, human-readable) | `results/thesis/bibliography.md` | — |
| Bibliography (BibTeX for pandoc integration) | `results/thesis/references.bib` | — |
| Citations audit (per-source location + completeness) | `results/thesis/citations_audit.md` | — |
| Repository README (stage → script mapping + reproducibility) | `README.md` (project root) | — |

### Six chapter files (locked)

- `results/thesis/abstract.md` — 201 words
- `results/thesis/introduction.md` — 1,966 words, 4 sections, 17 unique citations
- `results/thesis/methods.md` — 3,594 words, 7 sections, citation-dense
- `results/thesis/results.md` — 2,266 words, 6 sections, 4 figures embedded
- `results/thesis/discussion.md` — 2,541 words, 7 sections
- `results/thesis/conclusion.md` — 133 words, 3 sentences

**Total prose: 10,701 words across the six chapters.**

### Four figures (publication-ready, embedded in PDF)

- `results/figures/fig1_km_curves.png` — risk-stratified KM curves on TCGA + METABRIC
- `results/figures/fig2_calibration.png` — decile-bin calibration on both cohorts
- `results/figures/fig3_pathway_attention.png` — Reactome attention heatmap
- `results/figures/fig4_architecture_forest.png` — architecture-ablation forest plot

### Methodological backbone (referenced from Methods §5 and §7)

- `src/cindex_bootstrap.py` — bootstrap CI utility (per-fold + paired-on-identical-patients)
- `scripts/00_lasso_audit.py` — leakage audit pipeline
- `scripts/03b_*.py` — per-fold-honest LASSO + edge-masking pipeline
- `scripts/05_metabric_external.py` — external validation protocol
- `data/processed/cv_splits.json` — stratified 5-fold splits (seed=42), shared across all stages

### Reproducibility infrastructure

- `pyproject.toml` — verified consistent with Methods §7 version pins (PyTorch 2.11.0,
  PyTorch Geometric 2.7.0, lifelines 0.30.0, scikit-survival 0.27.0, torchmetrics 1.9.0,
  NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0; Python 3.13.1)
- `poetry.lock` — committed for byte-for-byte resolution

---

## Placeholders remaining

These are items the assembly pipeline cannot resolve without supervisor or
student input. Address each before final submission.

### Title-page metadata

- **Date.** Currently set to "2026" as a placeholder in `scripts/07_assemble_thesis.py`
  (constant `DATE`). Update to actual submission month + year (e.g., "May 2026") and
  re-run `poetry run python scripts/07_assemble_thesis.py` to regenerate
  `thesis_full.pdf`.
- **Title.** Currently:
  *"A Leakage-Corrected, Externally-Validated Graph Neural Network for Breast
  Cancer Prognosis"* with subtitle *"Per-fold leakage correction and
  paired-bootstrap on identical external patients as a methodological
  framework for GNN claims on TCGA-BRCA"*. If the supervisor or department
  prefers a different title, update the `TITLE` and `SUBTITLE` constants in
  the assembly script and regenerate.

### Repository URL

- ✅ **Resolved.** The Methods §7 placeholder and the README footer were
  replaced with `https://github.com/ELDERGARLIC/cancer-prognosis-prediction`
  in the post-assembly fix pass. Both files now point to the public
  repository directly.

### Bibliography entries flagged METADATA INCOMPLETE

7 of 21 bibliography entries have placeholder metadata that requires supervisor
verification before submission. From `citations_audit.md`:

- **Choudhry et al. 2025** (pathway-attention pooling design precedent)
- **Gao et al. 2021** (clinical-fusion ablation)
- **Liang 2025** (review on external validation)
- **Ling et al. 2022** (oversmoothing in deeper GNNs)
- **Madanipour et al. 2024** (inductive GraphSAGE on cancer prognosis)
- **Vaida et al. 2025** (patient-as-graph paradigm)
- **Vavekanand and Liang 2026** (review on cancer-AI evaluation practice)

For each: verify exact title, full author list, journal, volume, issue, pages,
and DOI. Update the `REFERENCES` list in `scripts/08_bibliography.py` and re-run
to regenerate `bibliography.md` and `references.bib`.

### Supplementary materials (deferred per user instruction)

The following are explicitly NOT yet included in supplementary materials,
pending the supervisor's preference on what the department expects:

- **Knob D full per-fold table.** Referenced from Results §2 as
  "supplementary materials." Source data is in
  `results/stage_3a_sage_clinical.json`.
- **Per-stage retrospective documents.** `results/stage_3_retrospective.md`
  and `results/stage_5_retrospective.md` were used as Discussion source
  material; they are methodological evidence and could be included as
  supplementary appendices if the department's norm is to include them.
- **Methodological notes.** `results/methodological_notes.md` is the
  canonical source for Methods §5; if Methods is reviewed, the supervisor
  may want this as backup reference.
- **Per-stage summary documents** (`results/stage_*_summary.md`) — fine-grained
  per-stage details that did not make the main chapters.

Decision on each pending supervisor input.

---

## Things explicitly NOT done in this assembly pass

These were ruled out of scope by the user's instruction:

- **Knob D supplementary table.** Deferred until supervisor weighs in on
  supplementary content.
- **Per-stage retrospective inclusion in supplementary.** Same.
- **Any new model runs.** No GenePT, MedGemma, AlphaGenome, or ER/PR re-run.
  The thesis is locked; new experiments are post-defense follow-up.

---

## Next human step

**Send `results/thesis/thesis_full.pdf` to the supervisor** (Doç. Dr. Özgür Gümüş)
for the integrated read-through. Optionally also send the bibliography file
(`results/thesis/bibliography.md`) and the citations audit
(`results/thesis/citations_audit.md`) so the supervisor can flag entries that
need metadata verification or citation-style adjustments.

After the supervisor's read:
1. Resolve title-page metadata (date, optional title adjustment).
2. Resolve repository URL placeholder.
3. Verify the 7 flagged bibliography entries (or have the supervisor request
   specific corrections).
4. Decide which retrospectives + summaries to include as supplementary.
5. Re-run `scripts/07_assemble_thesis.py` and `scripts/08_bibliography.py`
   to regenerate the PDF and bibliography after each round of changes.

The iterative-review-with-Claude phase has reached its natural ceiling. Any
further structural or substantive changes are conditional on supervisor
feedback. The architecture is frozen, the results are frozen, the prose is
locked; what remains is the supervisor read and any specific revisions they
request.

---

## Word count summary

| Chapter | Words |
|---|---:|
| Abstract | 201 |
| Introduction | 1,966 |
| Methods | 3,594 |
| Results | 2,266 |
| Discussion | 2,541 |
| Conclusion | 133 |
| **Total prose** | **10,701** |

Plus four figures, two main tables (T1, T2), one supplementary table reference (T S1
for knob D), and ~22 unique bibliography entries.
