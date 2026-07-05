# A Leakage-Corrected, Externally-Validated GraphSAGE for Breast Cancer Prognosis

This repository contains the implementation accompanying an MSc thesis at
Ege University (Department of Computer Engineering). The thesis builds a
leakage-corrected GraphSAGE for TCGA-BRCA prognosis, evaluates it on
METABRIC with paired-bootstrap on identical external patients, and
decomposes the resulting lift against a non-linear MLP-clinical reference.

**Headline finding.** Knob A (the frozen architecture) beats matched Cox PH
on external METABRIC validation with paired-bootstrap delta = +0.053
(95% CI [+0.031, +0.076], P < 0.001 over 2,000 resamples on n = 1,466
patients with 824 events). Two architectural elaborations — Reactome
pathway pooling and BioBERT-PCA gene priors — significantly underperform
the minimal architecture on the same external test despite competitive
internal performance.

For the full thesis, see [`results/thesis/thesis_full.pdf`](results/thesis/thesis_full.pdf).
For methodological detail, see [`results/thesis/methods.md`](results/thesis/methods.md).

---

## Stage → script mapping (from thesis Methods §7)

The pipeline runs sequentially across six stages. Each stage's deliverables
are saved in `results/` as JSON; figures are at `results/figures/`.

| Stage | Description | Script(s) | Wall-clock |
|---|---|---|---|
| 0 | Cox PH baselines + LASSO leakage audit + METABRIC fetch | `scripts/00_baseline.py`, `scripts/00_baseline_diag.py`, `scripts/00_lasso_audit.py`, `scripts/00_metabric_fetch.py` | ~14 min |
| 0.5 | Recover ER/PR IHC status from raw TCGA clinical and patch `clinical_features.tsv` (fixes all-zero `er_signed`/`pr_signed` columns found in Stage 0) | `scripts/00_5_recover_er_pr.py` | not recorded |
| 1 | Multinomial logistic-regression sanity check | `scripts/01_logreg.py` | ~5 sec |
| 2 | Minimal SAGE + R1 embedding-collapse sentinel forensic | `scripts/02_sage_minimal.py` | ~28 min |
| 3a | Knob D — GNN+clinical with leaky-769 gene set | `scripts/03a_sage_clinical.py` | ~30 min |
| 3 ref | MLP-clinical-only non-linear flat-feature reference | `scripts/03_ref_mlp_clinical_only.py` | ~5 sec |
| 3b | Knob A — per-fold-honest LASSO + clinical late-fusion | `scripts/03b_sage_clinical_lasso_honest.py` | ~10 min |
| 3c | Knob B — Reactome pathway pooling on top of knob A | `scripts/03c_sage_pathway_clinical.py` | ~15 min |
| 5 | Knob A on METABRIC — full-TCGA train + external paired bootstrap | `scripts/05_metabric_external.py` | ~12 min |
| 6 | Figure scripts (KM, calibration, pathway attention, forest plot) | `scripts/06_*.py` | ~2 min total |
| 6d | Knob B on METABRIC | `scripts/06d_knob_b_metabric.py` | ~3 min |
| 6e | Knob C — BioBERT-PCA gene priors on METABRIC | `scripts/06e_knob_c_biobert_metabric.py` | ~3 min |
| 6g | Knob C 5-fold internal cross-validation on TCGA-BRCA (extends Table 2/Results §4 with mean±std for Knob C, matching Knobs A/B) | `scripts/06g_knob_c_5fold_internal.py` | ~10 min |
| 7 | Thesis assembly (concat chapters → PDF) | `scripts/07_assemble_thesis.py` | <30 sec |
| 8 | Bibliography parser | `scripts/08_bibliography.py` | <5 sec |

**Total compute: under 2 hours end-to-end on the documented hardware**
(M-series Apple Silicon, 16-core CPU, 20 GB unified memory).

---

## Data prerequisites

| Dataset | Location | Source | Size |
|---|---|---|---|
| TCGA-BRCA HTSeq counts | `data/raw/tcga_brca_htseq_counts.tsv` | NCI Genomic Data Commons (`gdc-client`) | 60,660 genes × 1,095 patients |
| TCGA-BRCA clinical | `data/raw/tcga_brca_clinical.tsv` | NCI Genomic Data Commons | 1,095 patients × clinical fields |
| METABRIC mRNA + clinical | `data/external/brca_metabric/` | cBioPortal datahub LFS, fetched by `scripts/00_metabric_fetch.py` | 24,368 genes × 1,980 samples |
| BioBERT gene-embedding cache | `data/embeddings/gene_embeddings.npy` + `gene_names.json` | Pre-computed via `dmis-lab/biobert-base-cased-v1.2`; covers all 769 KG genes | 1,500 genes × 768 dim |
| STRING PPI knowledge graph | `data/processed/kg_edges.pt` + `kg_metadata.json` | STRING v11.5, combined-score ≥ 700 | 49,674 gene-gene edges over 769 genes |
| 5-fold stratified CV splits | `data/processed/cv_splits.json` | Built by Stage 0; stratified on (survival_class × OS event), seed 42 | 5 folds × ~215 val patients |

**Inherited preprocessing.** The 769-gene candidate set in `data/processed/`
was selected by full-cohort LASSO in earlier work; this repository documents
the leakage characteristics of that selection (Methods §1.2, §5.1) and
applies a per-fold-honest correction within the inherited universe. A
fully-honest universe rebuild from raw 60k via per-fold STRING extraction
remains future work (Discussion §6).

---

## Reproducing the environment

This project uses [Poetry](https://python-poetry.org/) for dependency
management. Version pins below match the exact installed versions reported
in Methods §7 of the thesis.

```bash
# Install Poetry (once)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies into a project-local venv
poetry install

# Run any stage script
poetry run python scripts/00_baseline.py
poetry run python scripts/05_metabric_external.py
```

**Pinned versions (Methods §7):**

- Python 3.13.1
- PyTorch 2.11.0
- PyTorch Geometric 2.7.0
- lifelines 0.30.0
- scikit-survival 0.27.0
- torchmetrics 1.9.0
- NumPy 2.4.4
- pandas 3.0.2
- scikit-learn 1.8.0

Constraints in `pyproject.toml` are minimum-pinned (`>=`); `poetry install`
on a fresh machine will resolve to versions equal to or newer than these.
For exact byte-for-byte reproducibility, `poetry.lock` (committed) records
the resolved dependency graph at the time of thesis submission.

---

## Repository layout

```
configs/                   # YAML configuration (mostly legacy from earlier work)
data/
  raw/                     # TCGA + METABRIC source files
  processed/               # gene universe, splits, KG, clinical features
  embeddings/              # BioBERT cache
  external/                # METABRIC raw download
results/
  figures/                 # 4 thesis figures (KM, calibration, pathway, forest)
  thesis/                  # all six chapter files + bibliography + assembled PDF
  *.json                   # per-stage saved predictions and statistics
  *.md                     # per-stage summaries and retrospectives
scripts/                   # numbered stage scripts, see the Stage → script mapping table above
src/                       # library modules (sage_models, cindex_bootstrap, etc.)
pyproject.toml             # poetry dependency manifest
```

---

## Where to start reading

For the methodology and the framework's reusable contribution:
[`results/thesis/methods.md`](results/thesis/methods.md), particularly
§5 (Methodological backbone) and §6 (External validation protocol).

For the headline findings: [`results/thesis/results.md`](results/thesis/results.md)
§3 (External METABRIC) and §4 (Architectural ablations).

For the synthesis and what comes next:
[`results/thesis/discussion.md`](results/thesis/discussion.md).

---

## Citation

If you use the framework or any of the released scripts, please cite the
underlying thesis. Bibliography entries for thesis sources are in
[`results/thesis/references.bib`](results/thesis/references.bib).

---

**Repository URL:** https://github.com/ELDERGARLIC/cancer-prognosis-prediction
**Supervisor:** Doç. Dr. Özgür Gümüş, Ege University
**Author:** Mahdi Sarhangi
