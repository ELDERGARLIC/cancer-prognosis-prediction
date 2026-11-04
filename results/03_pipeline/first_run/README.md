# First run of notebook 3

The files of the first run of `notebooks/03_pipeline.ipynb`, kept unchanged: the fold files and gene lists of the five outer folds, the comparison rows of 31 October in `variants/`, the final model of that run in `final_model/`, its tables and its records. The first run was computed from commit dcf0a42 on 29 October 2026 and its variant rows from commit f9966dc on 31 October 2026.

In that run the selector's candidate penalties and the gene counts that decided which settings were eligible came from the path computed on the whole outer training part. The current notebook computes them on each inner fit part alone and reads the run in the folder above; nothing reads this folder any more.
