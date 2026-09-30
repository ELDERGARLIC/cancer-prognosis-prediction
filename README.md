# Breast cancer prognosis from gene expression

This repository holds the code of an MSc thesis on predicting overall survival in breast cancer
from tumour gene expression and clinical variables. The models are developed on TCGA-BRCA, and
METABRIC is kept as an independent cohort. The pipeline is written as Jupyter notebooks that are
read in order, one block at a time.

## Branches

- `rebuild` is the current work. It starts from an empty tree and shares no history with the
  other branches.
- `deprecated` is the earlier version of the pipeline, written as Python scripts. It is kept as
  a reference and is not developed further.
- `main`, `jbi-pivot`, `cbc-pivot` and `figures-fonttype42` are earlier states of that same code.

## Setup

The environment is pinned in `pyproject.toml` and `uv.lock`. It needs
[uv](https://docs.astral.sh/uv/) and Python 3.12.

    uv sync

## Running

The notebooks are in `notebooks/` and are numbered in the order they run. Each one executes from
a fresh kernel with one command and keeps its outputs.

    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb

To read or edit a notebook, start `uv run jupyter lab`. The aggregate tables a notebook produces
are written to `results/<notebook name>/`.

## Data

No data is stored in this repository. The notebooks download every source into `data/`, which
git ignores, and record the version and checksum of each file. Patient-level values never appear
in a tracked file or in a notebook output.
