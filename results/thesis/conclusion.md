# Conclusion

This thesis built and applied a framework that converts within-cohort
graph neural network comparisons into resolvable findings: per-fold
leakage correction during training, and paired-bootstrap on identical
external patients during evaluation. Applied to TCGA-BRCA prognosis,
the framework finds a leakage-corrected GraphSAGE beating matched Cox PH
on external METABRIC validation with paired-bootstrap significance
(Δ = +0.053, P < 0.001), with the gene-graph contribution above
non-linear clinical baselines a small but stable +0.013 c-index, and
two architectural elaborations — Reactome pathway pooling and
BioBERT-derived gene priors — significantly underperforming the minimal
architecture on the same external test. The framework is reusable beyond
TCGA-BRCA — the GenePT-versus-BioBERT comparison is its most informative
near-term application — and its broader contribution is to make external
paired-bootstrap testing on identical patients a discipline the field
can adopt without new infrastructure.
