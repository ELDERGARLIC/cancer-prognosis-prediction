"""Stage 11 — Gene-only GNN ablation + clinical-only baseline (external decomposition).

Reviewer Q1: provide a gene-only ablation (no clinical covariates) for M0 to
directly quantify the gene-graph contribution, and a linear clinical-only Cox
baseline for symmetry.

We reuse Stage 5's loaders, per-fold LASSO, KG subsetting, data builder and
training loop *verbatim* (imported from 05_metabric_external.py) so the only
thing that changes is the model:

  * gene-only  : MinimalSAGE (2-layer SAGE + pool + MLP, NO clinical fusion)
  * gene+clin  : SAGEClinical = M0 (already in stage_5; external C = 0.6443)
  * clinical-only: Cox PH on the 3 shared clinical features only (no genes)

External protocol identical to Stage 5: full-TCGA seed-42 90/10 split -> METABRIC,
per-cohort z-score. We report the three external C-indices and the marginal
decomposition with patient-level paired bootstrap on identical METABRIC patients:

  gene-graph marginal over clinical = C(gene+clin) - C(clinical-only)
  clinical    marginal over genes   = C(gene+clin) - C(gene-only)

Outputs:
  results/stage_11_geneonly.json
  manuscript/table_decomposition.tex
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path("/home/eldergarlic/network_share/Projects/thesis-pipelines/thesis-research-v1")
CODE = ROOT / "code"
RESULTS = ROOT / "output/thesis-results-repo/results"
OUT_JSON = RESULTS / "stage_11_geneonly.json"
OUT_TEX = ROOT / "output/thesis-writing-repo/manuscript/table_decomposition.tex"
STAGE5 = RESULTS / "stage_5_metabric_external.json"

# import Stage 5 module by path (reuse its loaders/training verbatim)
sys.path.insert(0, str(CODE / "src"))
spec = importlib.util.spec_from_file_location("s5", CODE / "scripts/05_metabric_external.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)
from sage_models import MinimalSAGE  # noqa: E402

SEED = s5.SEED
N_FOLDS = s5.N_FOLDS
EPOCHS = s5.EPOCHS
BATCH_SIZE = s5.BATCH_SIZE
HIDDEN_DIM = s5.HIDDEN_DIM
DROPOUT = s5.DROPOUT
LR = s5.LR
WEIGHT_DECAY = s5.WEIGHT_DECAY
N_BOOT = 2000
ALPHA = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage11")


def cindex(T, E, risk):
    return float(concordance_index_censored(E.astype(bool), T, risk)[0])


def train_geneonly(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va, T_tr, T_va, E_tr, E_va,
                   fold_expr_cols, edge_index_local, fold_label=""):
    """Mirror s5.train_knob_a but with MinimalSAGE (gene-only, clinical ignored)."""
    from torch_geometric.loader import DataLoader
    train_list, scaler = s5.build_data_list(
        X_expr_tr, X_clin_tr, T_tr, E_tr, fold_expr_cols, edge_index_local, scaler=None)
    val_list, _ = s5.build_data_list(
        X_expr_va, X_clin_va, T_va, E_va, fold_expr_cols, edge_index_local, scaler=scaler)
    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cpu")
    model = MinimalSAGE(in_dim=1, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_c, best_log_h, best_state, best_ep = -1.0, None, None, -1
    for epoch in range(1, EPOCHS + 1):
        s5.run_one_epoch(model, train_loader, opt, device, train=True, clinical_dim=3)
        va = s5.run_one_epoch(model, val_loader, opt, device, train=False, clinical_dim=3)
        if va["cindex"] > best_c:
            best_c = va["cindex"]; best_log_h = va["log_h"].copy(); best_ep = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    log.info(f"    [{fold_label}] gene-only best val_cidx={best_c:.4f} (ep {best_ep})")
    return model, best_c, best_log_h


def infer_geneonly(model, X_expr, X_clin, T, E, fold_expr_cols, edge_index_local):
    from torch_geometric.loader import DataLoader
    data_list, _ = s5.build_data_list(X_expr, X_clin, T, E, fold_expr_cols,
                                      edge_index_local, scaler=None)
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    out = s5.run_one_epoch(model, loader, None, torch.device("cpu"),
                          train=False, clinical_dim=3)
    return out["log_h"], out["cindex"]


def clinical_only_cox(X_clin_tr, T_tr, E_tr, X_clin_ext, T_ext, E_ext):
    df = pd.DataFrame(X_clin_tr, columns=["age", "stage", "sex"])
    df["T"] = T_tr; df["E"] = E_tr.astype(int)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col="T", event_col="E")
    risk = cph.predict_partial_hazard(
        pd.DataFrame(X_clin_ext, columns=["age", "stage", "sex"])).values.ravel()
    return risk, cindex(T_ext, E_ext, risk)


def paired_delta(T, E, risk_a, risk_b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pa, pb = cindex(T, E, risk_a), cindex(T, E, risk_b)
    deltas = []
    n = len(T)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if E[idx].sum() < 1:
            continue
        try:
            deltas.append(cindex(T[idx], E[idx], risk_a[idx]) -
                          cindex(T[idx], E[idx], risk_b[idx]))
        except Exception:
            continue
    deltas = np.asarray(deltas)
    return {"c_a": pa, "c_b": pb, "delta_point": float(pa - pb),
            "ci_low": float(np.quantile(deltas, ALPHA / 2)),
            "ci_high": float(np.quantile(deltas, 1 - ALPHA / 2)),
            "p_a_le_b": float((deltas <= 0).mean()), "n_valid": int(len(deltas))}


def main():
    t0 = time.time()
    s5j = json.loads(STAGE5.read_text())
    gnn_both_risk = np.asarray(s5j["full_tcga_run"]["metabric_log_h_gnn_full"], float)
    gnn_both_cidx = s5j["full_tcga_run"]["metabric_gnn_cidx"]

    splits = json.loads(s5.TCGA_SPLITS.read_text())
    full_edge_index = torch.load(s5.KG_EDGES, weights_only=False)["gene_gene_edges"]
    tcga = s5.load_tcga()
    met = s5.load_metabric(tcga["gene_ids_769"], tcga["kg_gene_to_idx"])
    overlap_idx = met["overlap_expr_col_idx_in_tcga"]
    X_tcga_ov = tcga["X_expr"][:, overlap_idx]
    ov_col_to_kg = tcga["expr_col_to_kg_idx"][overlap_idx]
    assert len(met["T"]) == len(gnn_both_risk)

    # ----- internal 5-fold (gene-only) -----
    internal_geneonly = []
    for fold in range(N_FOLDS):
        s = splits[f"fold_{fold}"]
        tr, va = np.array(s["train_idx"]), np.array(s["val_idx"])
        nz, _ = s5.per_fold_lasso_within_universe(X_tcga_ov[tr], tcga["bins"][tr].astype(float),
                                                  fold_label=f"f{fold}")
        kg = s5.subset_kg_to_fold(nz, ov_col_to_kg, full_edge_index, n_kg_genes=769)
        _, best_c, _ = train_geneonly(
            X_tcga_ov[tr], X_tcga_ov[va], tcga["X_clin_3"][tr], tcga["X_clin_3"][va],
            tcga["T"][tr], tcga["T"][va], tcga["E"][tr], tcga["E"][va],
            kg["fold_expr_cols"], kg["edge_index_local"], fold_label=f"f{fold}")
        internal_geneonly.append(best_c)

    # ----- external: full-TCGA 90/10 -> METABRIC -----
    rng = np.random.default_rng(SEED)
    n = len(tcga["T"]); perm = rng.permutation(n); n_tr = int(0.9 * n)
    full_tr, full_va = perm[:n_tr], perm[n_tr:]
    nz_full, _ = s5.per_fold_lasso_within_universe(X_tcga_ov, tcga["bins"].astype(float),
                                                   fold_label="full")
    kg_full = s5.subset_kg_to_fold(nz_full, ov_col_to_kg, full_edge_index, n_kg_genes=769)

    model_go, go_va_c, _ = train_geneonly(
        X_tcga_ov[full_tr], X_tcga_ov[full_va], tcga["X_clin_3"][full_tr], tcga["X_clin_3"][full_va],
        tcga["T"][full_tr], tcga["T"][full_va], tcga["E"][full_tr], tcga["E"][full_va],
        kg_full["fold_expr_cols"], kg_full["edge_index_local"], fold_label="full")
    go_risk, go_cidx = infer_geneonly(model_go, met["X_expr"], met["X_clin_3"],
                                      met["T"], met["E"], kg_full["fold_expr_cols"],
                                      kg_full["edge_index_local"])

    # clinical-only Cox (same full_tr training rows)
    clin_risk, clin_cidx = clinical_only_cox(
        tcga["X_clin_3"][full_tr], tcga["T"][full_tr], tcga["E"][full_tr],
        met["X_clin_3"], met["T"], met["E"])

    T, E = met["T"], met["E"]
    gene_marginal = paired_delta(T, E, gnn_both_risk, clin_risk)   # both - clinical-only
    clin_marginal = paired_delta(T, E, gnn_both_risk, go_risk)     # both - gene-only

    log.info("=" * 60)
    log.info(f"  clinical-only  external C = {clin_cidx:.4f}")
    log.info(f"  gene-only      external C = {go_cidx:.4f}")
    log.info(f"  gene+clinical  external C = {gnn_both_cidx:.4f} (M0)")
    log.info(f"  gene-graph marginal over clinical = {gene_marginal['delta_point']:+.4f} "
             f"[{gene_marginal['ci_low']:+.4f},{gene_marginal['ci_high']:+.4f}]")
    log.info(f"  clinical marginal over genes      = {clin_marginal['delta_point']:+.4f} "
             f"[{clin_marginal['ci_low']:+.4f},{clin_marginal['ci_high']:+.4f}]")

    out = {
        "n_metabric": int(len(T)), "seed": SEED, "n_boot": N_BOOT,
        "internal_geneonly_mean": float(np.mean(internal_geneonly)),
        "internal_geneonly_std": float(np.std(internal_geneonly)),
        "internal_geneonly_folds": internal_geneonly,
        "external": {
            "clinical_only_cidx": clin_cidx,
            "gene_only_cidx": go_cidx,
            "gene_clin_cidx": gnn_both_cidx,
            "gene_only_val_cidx_tcga": go_va_c,
        },
        "gene_graph_marginal_over_clinical": gene_marginal,
        "clinical_marginal_over_genes": clin_marginal,
        "gene_only_risk": go_risk.tolist(),
        "clinical_only_risk": clin_risk.tolist(),
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {OUT_JSON}  ({out['elapsed_seconds']:.0f}s)")
    write_latex(out)
    log.info(f"wrote {OUT_TEX}")


def write_latex(o):
    e = o["external"]
    gm, cm = o["gene_graph_marginal_over_clinical"], o["clinical_marginal_over_genes"]
    lines = [
        r"% Auto-generated by scripts/11_geneonly_ablation.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{External (METABRIC, $n=" + str(o["n_metabric"]) + r"$) decomposition "
        r"of the base model into clinical and gene-graph contributions. Marginal "
        r"contributions and 95\% CIs are from patient-level paired bootstrap "
        r"($B=" + str(o["n_boot"]) + r"$) on identical patients; $p$ is the one-sided "
        r"bootstrap probability that the marginal is $\le 0$.}",
        r"\label{tab:decomposition}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Model & External C-index \\",
        r"\midrule",
        rf"Clinical-only (Cox, 3 features) & {e['clinical_only_cidx']:.4f} \\",
        rf"Gene-only GNN (no clinical) & {e['gene_only_cidx']:.4f} \\",
        rf"\textbf{{Gene$+$clinical GNN (M0)}} & \textbf{{{e['gene_clin_cidx']:.4f}}} \\",
        r"\midrule",
        r"\multicolumn{2}{l}{\emph{Marginal contributions (paired bootstrap)}}\\",
        rf"Gene-graph over clinical & {gm['delta_point']:+.4f} "
        rf"[{gm['ci_low']:+.3f}, {gm['ci_high']:+.3f}], $p={gm['p_a_le_b']:.3f}$ \\",
        rf"Clinical over gene-graph & {cm['delta_point']:+.4f} "
        rf"[{cm['ci_low']:+.3f}, {cm['ci_high']:+.3f}], $p={cm['p_a_le_b']:.3f}$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
