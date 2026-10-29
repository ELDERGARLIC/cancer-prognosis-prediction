# Breast cancer prognosis from gene expression

This repository holds the code of an MSc thesis at Ege University (Department of Computer
Engineering) on predicting overall survival in breast cancer from tumour gene expression and
clinical variables. The models are developed on the TCGA-BRCA cohort; METABRIC is kept as an
independent cohort for the end. The work is written as Jupyter notebooks that a reader follows
in order, one block at a time: each block says what it does, why, and what to look at in its
output, and every number in the text is rendered from the variables that hold it.

## Research question and current stage

The thesis asks whether a model of tumour gene expression, with or without a gene graph, predicts
overall survival better than the clinical variables alone, and whether it holds on an external
cohort. The project is being rebuilt from the start after an earlier version (kept on the
`depricated` branch) failed its defence; the question, title and data sources are unchanged.

Five notebooks exist. Notebook 1 builds the data; notebooks 2 and 3 are the comparison study of
four gene selection methods under nested cross-validation (they found the methods tied and are
kept as evidence, unchanged); notebook 4 builds the STRING gene graph; notebook 5 implements the
procedure agreed with the supervisors on 22 October 2026: elastic-net Cox selects the genes inside
every training part, a GraphSAGE model is trained on the selected genes, both tuned by an inner
five-fold loop and scored by an outer five-fold loop over every TCGA patient, with a clinical-only
Cox and the selector itself as reference rows, and a final model trained on all patients and
frozen. METABRIC has not been scored yet; the scoring cell is in notebook 5 and switched off until
the procedure is confirmed. The numbers, with their intervals, are in notebook 5.

## The notebooks, in reading order

| Notebook | What it does | Writes to |
|---|---|---|
| `notebooks/01_data.ipynb` | Downloads the sources, builds the TCGA-BRCA and METABRIC cohorts, defines overall survival as time and event, builds the candidate gene pool and writes one table per cohort. | `data/` (ignored by git) and `results/01_data/` |
| `notebooks/02_gene_selection_runs.ipynb` | The gene selection study: a shuffled-outcome check, four selection methods compared under nested cross-validation on the development set, and a leakage check. Walks once through the procedure with the library's models on one fold. Reads the saved runs by default. | `results/02_gene_selection_runs/` |
| `notebooks/03_gene_selection_results.ipynb` | Reads the saved runs and shows the results: tables, figure, paired comparisons, and what the results do and do not support. No modelling code. | `results/03_gene_selection_results/` |
| `notebooks/04_gene_graph.ipynb` | Downloads STRING 12.5 with a record, maps proteins to the candidate gene pool and saves the gene-level edge list at combined score 700 or more. | `results/04_gene_graph/` |
| `notebooks/05_pipeline.ipynb` | The procedure of 22 October: stage A chooses the selector's settings and stage B the graph model's settings on the same inner folds, stage C fits one model per outer fold and scores its test part once beside the reference rows, part 2 trains and freezes the final model. Reads the saved fold files by default. | `results/05_pipeline/` |

The results folders hold aggregate tables and figures only (CSV, JSON, PNG). The saved runs of an
earlier version of the study, made on the previous cohort definition, are kept in
`results/02_gene_selection_runs/first_run/` and shown beside the current one in notebook 3.

## Folders

| Path | Contents |
|---|---|
| `notebooks/` | The three notebooks, numbered in reading order. |
| `results/<notebook name>/` | Aggregate outputs of each notebook, committed with the notebook. |
| `data/` | Downloaded sources and patient tables. Created by notebook 1, never committed. |
| `pyproject.toml`, `uv.lock` | The pinned Python environment. |

## Setup

The environment needs [uv](https://docs.astral.sh/uv/) and Python 3.12; uv installs the
interpreter if it is missing. Every package is pinned in `uv.lock`, including PyTorch (the CPU
build), PyTorch Geometric and torchsurv for the graph model; no GPU is needed.

    uv sync

To read a notebook, start Jupyter and open it from `notebooks/`:

    uv run jupyter lab

## Running a notebook

Each notebook runs from a fresh kernel with one command and keeps its outputs in the file. The
last option leaves run times out, so a repeated run does not change the file.

    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.record_timing=False notebooks/01_data.ipynb

Which steps need data:

- Notebook 1 downloads about 5 GB into `data/` on its first run (about six minutes on a fast
  connection) and records the version and checksum of every file. Later runs reuse the files and
  take about two minutes.
- Notebook 2 needs the development set written by notebook 1. By default it reads the saved runs
  in `results/02_gene_selection_runs/` and finishes in about two minutes, most of it the one-fold
  demonstration and the check of the reference implementation against the library. Computing a
  run is a deliberate choice: the switch `READ_SAVED_RUNS` in its settings cell turns it on, only
  runs whose saved file is missing are computed, and a full recompute takes about two and a half
  hours on four cores.
- Notebook 3 needs only the saved runs and two aggregate tables of notebook 1. It takes seconds.
- Notebook 4 downloads two STRING files (about 65 MB) on its first run and takes about a minute.
- Notebook 5 needs the cohort table of notebook 1 and the edge list of notebook 4. By default it
  reads the saved fold files in `results/05_pipeline/` and finishes in under a minute; the compute
  switch in its settings computes the outer folds whose file is missing and then the final model,
  which takes a few hours on four cores.

Reading saved results therefore means running the notebooks as they are. Starting an experiment
means changing a setting in notebook 2 after moving the saved runs aside, since the notebook
refuses to run in read mode with settings that differ from the saved file.

## Data and attribution

No data is stored in this repository. Patient-level values never appear in a tracked file or in
a notebook output; the notebooks show counts, summaries and plots.

The sources are the TCGA-BRCA expression and clinical files of the NCI Genomic Data Commons, the
TCGA Pan-Cancer Clinical Data Resource (Liu et al., 2018, Cell 173:400), and METABRIC from the
cBioPortal datahub (Curtis et al., 2012, Nature 486:346; Pereira et al., 2016, Nature
Communications 7:11479), used under the Open Database License. The results shown here are in
whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.

No software licence has been declared for this repository yet.

## Branches and weekly workflow

| Branch | Purpose |
|---|---|
| `main` | The reviewed version. Each state of `main` is one that the author has explicitly approved after a weekly supervisor meeting. |
| `develop` | Ongoing work. Everything is committed here first. |
| `depricated` | The earlier pipeline, frozen as a labelled historical snapshot (see its `DEPRECATED.md`). Not developed. |

Work happens on `develop`. After the author explicitly confirms approval of a particular weekly
version, that reviewed version is merged into `main` and pushed. Nothing is merged because a
week has passed or because checks have succeeded, and a merge into `main` records the author's
approval of that version, not the supervisors' acceptance of its conclusions.

    git switch develop                      # daily work
    git add notebooks results && git commit
    git push origin develop

    # after the explicit approval of a reviewed version:
    git switch main
    git merge --ff-only develop             # or the approved commit
    git push origin main
    git switch develop

Author: Mahdi Sarhangi.
