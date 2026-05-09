# Abstract

Graph neural networks are increasingly applied to cancer prognosis from
gene expression, but published comparisons commonly combine within-cohort
gene selection with within-cohort evaluation, which inflates apparent
gains over linear baselines. We present a leakage-corrected, pathway-interpretable GraphSAGE for TCGA-BRCA
survival using per-fold LASSO gene selection on training partitions only,
biological-prior gene topology from STRING PPI edges, and Cox partial-likelihood
loss with clinical late-fusion. On TCGA-BRCA (n = 1074), the model matches the
strongest Cox PH baseline internally and beats matched Cox PH on external
METABRIC validation (n = 1466, 824 events) with paired-bootstrap significance:
Δ = +0.053, 95% CI [+0.031, +0.076], P < 0.001 over 2000 resamples. Two
architectural elaborations — Reactome pathway pooling and BioBERT-derived
gene priors — significantly underperform the minimal architecture on external
validation despite competitive internal performance (paired Δ = −0.016 and
−0.039 respectively, both with CI strictly below zero), demonstrating that
complexity-driven gains require external paired-bootstrap testing to verify.
Methodological contributions: (i) a per-fold LASSO leakage audit revealing the
prior pipeline's 0.748 C-index as a full-cohort-selection artifact, and (ii)
a paired-bootstrap framework on identical external patients that
distinguishes harmful elaborations from genuine improvements. Limitation: the underlying
gene universe was full-cohort-selected; full correction would require per-fold
knowledge-graph rebuilds.
