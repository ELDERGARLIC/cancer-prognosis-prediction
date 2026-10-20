"""Verify manuscript-reported numbers against frozen result artifacts.

Phase 3 deliverable for the SADM -> Computational Biology and Chemistry (CBC)
pivot. Identical claim manifest to scripts/verify_jbi_numbers.py, since the
CBC package reframes the same frozen results rather than reporting new
numbers -- see research/CBC_NUMBER_AUDIT.md for the full claim table this
script backs. Reads only existing JSON/tex files under results/ and
journal-package tables; does not train or rerun anything. Prints
PASS / FAIL / NEEDS-REVIEW per claim.

Usage: python3 scripts/verify_cbc_numbers.py   (run from the code/ directory)
"""
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def close(a, b, tol=5e-4):
    return abs(a - b) <= tol


def check(label, reported, actual, tol=5e-4):
    ok = close(reported, actual, tol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: manuscript={reported!r} vs artifact={actual!r}")
    return ok


def main():
    results = []

    d = load("stage_0_baseline.json")
    results.append(check("TCGA-BRCA n", 1074, d["n_patients"], tol=0))
    results.append(check("TCGA-BRCA event rate", 0.140, d["event_rate"], tol=1e-3))

    d = load("stage_5_metabric_external.json")
    results.append(check("METABRIC n", 1466, d["metabric_n"], tol=0))
    results.append(check("METABRIC event rate", 0.561, d["metabric_event_rate"], tol=1e-3))
    results.append(check("769-gene universe", 769, d["feature_audit"]["n_total_kg_genes"], tol=0))
    results.append(check("TCGA-METABRIC overlap genes", 650, d["feature_audit"]["n_overlap_genes"], tol=0))
    results.append(check("M0 external C-index", 0.6443, d["full_tcga_run"]["metabric_gnn_cidx"]))
    results.append(check("PCA/ridge Cox external C-index", 0.5914, d["full_tcga_run"]["metabric_cox_cidx"]))
    results.append(check(
        "Paired delta M0 vs PCA/ridge Cox",
        0.053, d["full_tcga_run"]["metabric_paired_gnn_minus_cox"]["delta_point"],
    ))

    d = load("stage_0_lasso_audit.json")
    results.append(check("Leakage-corrected Cox PH mean", 0.6662, d["honest_baseline_mean"]))
    results.append(check("Leakage-corrected Cox PH std", 0.054, d["honest_baseline_std"], tol=2e-3))
    honest_baseline_mean = d["honest_baseline_mean"]

    # Leaky-Cox baseline: the correct source is stage_0_baseline.json's own
    # per-fold list, NOT stage_0_lasso_audit.json's leaky_baseline_mean field
    # (that field is a stale hardcoded constant left over from an earlier,
    # pre-ER/PR-recovery run of 00_baseline.py -- see research/JBI_NUMBER_AUDIT.md
    # Note 1 for the full trace). stage_0_baseline.json is the current,
    # post-recovery run and matches the manuscript exactly.
    d0 = load("stage_0_baseline.json")
    leaky_folds = [r["cindex_lifelines"] for r in d0["leaky_baseline_per_fold"]]
    leaky_mean = statistics.mean(leaky_folds)
    results.append(check("Leaky-Cox baseline mean", 0.7362, leaky_mean, tol=5e-4))
    results.append(check("Leakage cost (leaky - honest)", 0.070, leaky_mean - honest_baseline_mean, tol=5e-4))

    d = load("stage_3_ref_mlp_clinical.json")
    results.append(check("Clinical-only MLP internal mean", 0.7158, d["mlp_clinical_only_mean"]))

    d = load("stage_10_baselines.json")
    results.append(check("Elastic-net Cox external", 0.6392, d["external"]["coxnet"]["cindex"]))
    results.append(check("Random survival forest external", 0.5836, d["external"]["rsf"]["cindex"]))
    results.append(check("Gradient-boosted survival external", 0.5901, d["external"]["gbsa"]["cindex"]))

    d = load("stage_11_geneonly.json")
    results.append(check("Clinical-only Cox external", 0.6400, d["external"]["clinical_only_cidx"], tol=1e-3))
    results.append(check("Gene-only GNN external", 0.5058, d["external"]["gene_only_cidx"]))
    results.append(check("Gene-only GNN internal five-fold mean", 0.606, d["internal_geneonly_mean"], tol=1e-3))
    results.append(check(
        "Gene-only exported model's own TCGA val split (footnote)",
        0.539, d["external"]["gene_only_val_cidx_tcga"], tol=1e-3,
    ))
    results.append(check(
        "Gene-graph marginal over clinical",
        0.0043, d["gene_graph_marginal_over_clinical"]["delta_point"],
    ))
    results.append(check(
        "Clinical marginal over gene-graph",
        0.1384, d["clinical_marginal_over_genes"]["delta_point"],
    ))

    d = load("stage_12_reviewer_experiments.json")
    a = d["experiment_A_nonlinear_clinical"]
    results.append(check("Clinical-only MLP (matched non-linear) external", 0.6450, a["clin_mlp_external_cidx"]))
    results.append(check(
        "Gene-graph marginal over non-linear clinical",
        -0.0008, a["gene_marginal_over_nonlinear_clinical"]["delta_point"], tol=2e-3,
    ))
    b = d["experiment_B_fullbatch_riskset"]
    results.append(check("Mini-batch fold-0 val C-index", 0.7376, b["reproduced_batched_fold0_cidx"]))
    results.append(check("Full-batch (exact risk-set) fold-0 val C-index", 0.7368, b["fullbatch_fold0_cidx"]))

    d = load("stage_9_extended_metrics.json")
    results.append(check("Uno's C delta", 0.051, d["uno"]["delta_point"], tol=2e-3))
    results.append(check("Mean td-AUC delta", 0.050, d["tauc_mean"]["delta_point"], tol=2e-3))
    results.append(check("Integrated Brier delta", 0.010, d["ibs"]["delta_point"], tol=2e-3))

    d = load("stage_5b_knob_b_metabric.json")
    results.append(check("M_path external C-index", 0.6285, d["metabric_cidx"]))
    results.append(check("M_path external delta vs M0", -0.0158, d["paired_metabric_b_vs_a"]["delta_point"], tol=2e-3))

    d = load("stage_5c_knob_c_biobert_metabric.json")
    results.append(check("M_bio external C-index", 0.6054, d["metabric_cidx"]))
    results.append(check("M_bio external delta vs M0", -0.0389, d["paired_metabric_c_vs_a"]["delta_point"], tol=2e-3))

    d = load("stage_3d_knob_c_5fold_internal.json")
    results.append(check("M_bio internal five-fold mean", 0.7218, d["mean_val_cindex"], tol=1e-3))

    d = load("stage_3c_sage_pathway_clinical.json")
    m_path_folds = [r["best_val_cindex"] for r in d["fold_results"]]
    results.append(check("M_path internal five-fold mean", 0.7334, statistics.mean(m_path_folds), tol=1e-3))

    d = load("stage_3b_sage_clinical_lasso_honest.json")
    m0_folds = [r["best_val_cindex"] for r in d["fold_results"]]
    results.append(check("M0 internal five-fold mean", 0.7261, statistics.mean(m0_folds), tol=1e-3))

    fig1 = load("fig1_km_stats.json")
    results.append(check("TCGA log-rank chi-sq", 39.7, fig1["tcga"]["logrank_chi2"], tol=0.1))
    results.append(check("METABRIC log-rank chi-sq", 204.0, fig1["metabric"]["logrank_chi2"], tol=0.1))

    n_pass = sum(1 for r in results if r)
    print(f"\n{n_pass}/{len(results)} checks passed.")


if __name__ == "__main__":
    main()
