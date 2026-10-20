# Historical branch: the earlier pipeline

This branch is a frozen snapshot of the earlier version of this project, a GraphSAGE pipeline
written as Python scripts and used for the thesis version submitted in 2026. It is kept for
reference only and is not developed further. The current work is the notebook rebuild on the
`main` branch; it shares no code or history with this branch.

Source: the tree of commit 6480428 ("Embed TrueType fonts in figure PDFs and record 2026-08-04
regenerated figures", branch `figures-fonttype42`), which is the last state of the earlier pipeline
and lies two commits past the previously published `main` (4dcb947). The commit history of that
branch is kept privately and is not part of this snapshot.

Omitted from the snapshot: eleven result files under `results/` that held one row per patient
(follow-up time, event indicator and predicted risk for each patient of a cohort or fold):
`stage_3_ref_mlp_clinical.json`, `stage_3a_sage_clinical.json`,
`stage_3b_sage_clinical_lasso_honest.json`, `stage_3c_sage_pathway_clinical.json`,
`stage_3d_knob_c_5fold_internal.json`, `stage_5_metabric_external.json`,
`stage_5b_knob_b_metabric.json`, `stage_5c_knob_c_biobert_metabric.json`,
`stage_10_baselines.json`, `stage_11_geneonly.json` and `stage_12_reviewer_experiments.json`.
Scripts that read those files will not run on this branch. The aggregate summaries, figure
statistics and figures remain.

The results reported in this branch's `README.md` and `results/` are those of the earlier
pipeline and are not the current results of the project. Their methodological problems are the
reason for the rebuild.
