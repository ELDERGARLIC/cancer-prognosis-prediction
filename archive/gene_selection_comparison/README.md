# Archive: the gene selection comparison study

The two notebooks in this folder computed and showed the comparison of four gene selection methods
(most variable genes, one-gene Cox ranking, LASSO-Cox, elastic-net Cox) under nested cross-validation
on the earlier data version, a development set of 851 TCGA-BRCA patients with a locked test set. They
ran last at commit f8c4a00 of this repository (20 October 2026), when they were named
`notebooks/02_gene_selection_runs.ipynb` and `notebooks/03_gene_selection_results.ipynb`.

Their saved runs and tables now live in `results/appendix_gene_selection_comparison/` (`runs/` and
`tables/`), and `notebooks/appendix_gene_selection_comparison.ipynb` reads those tables. The paths
inside these two notebooks are not maintained, they are not executed again, and they are kept for
provenance: the run record at the end of each names the commit, the package versions, the seeds and
the input hashes of the runs. The roadmap document shared at the supervisors' meeting of 8 October
2026 names elastic-net Cox as the selector, confirmed on 22 October; the current pipeline is
`notebooks/03_pipeline.ipynb`.
