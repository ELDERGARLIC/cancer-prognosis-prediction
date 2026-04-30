"""Optional 4: Knob B (pathway pool + clinical) on METABRIC external validation.

Forecloses the reviewer question "does pathway pooling help externally?"

Re-uses Stage 5's full-TCGA training + METABRIC inference pipeline but swaps
SAGEClinical for SAGEPathwayClinical with per-fold pathway-membership matrix.

The Cox PH baseline predictions on METABRIC are loaded from
`results/stage_5_metabric_external.json` (full_tcga_run.metabric_risk_cox_full)
to keep the paired-bootstrap comparison apples-to-apples with knob A.

Output:
  - results/stage_5b_knob_b_metabric.json
  - prints headline numbers; appends a summary section to stage_5_summary.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse helpers from Stage 5 + Stage 3c
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s5 = _load("s5", ROOT / "scripts" / "05_metabric_external.py")
s3c = _load("s3c", ROOT / "scripts" / "03c_sage_pathway_clinical.py")
from cindex_bootstrap import bootstrap_cindex, paired_bootstrap_delta  # noqa: E402
from sage_models import (  # noqa: E402
    SAGEPathwayClinical, cox_partial_likelihood_loss, mean_pairwise_cosine,
)

RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
KG_EDGES_PATH = DATA / "kg_edges.pt"
KG_META_PATH = DATA / "kg_metadata.json"
S5_JSON = RESULTS / "stage_5_metabric_external.json"
OUT_JSON = RESULTS / "stage_5b_knob_b_metabric.json"
SUMMARY_MD = RESULTS / "stage_5_summary.md"

SEED = 42
HIDDEN_DIM = 128
DROPOUT = 0.4
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("06d_knob_b_metabric")


def build_dataset(X_expr, X_clin, T, E, fold_expr_cols, edge_index_local, scaler=None):
    """Same as Stage 5's build_data_list but inlined to avoid name drift."""
    return s5.build_data_list(X_expr, X_clin, T, E, fold_expr_cols, edge_index_local, scaler=scaler)


def train_knob_b(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va,
                 T_tr, T_va, E_tr, E_va,
                 fold_expr_cols, edge_index_local, membership,
                 clinical_dim, fold_label=""):
    train_list, scaler_used = build_dataset(
        X_expr_tr, X_clin_tr, T_tr, E_tr, fold_expr_cols, edge_index_local,
    )
    val_list, _ = build_dataset(
        X_expr_va, X_clin_va, T_va, E_va, fold_expr_cols, edge_index_local,
        scaler=scaler_used,
    )
    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)
    model = SAGEPathwayClinical(
        in_dim=1, hidden_dim=HIDDEN_DIM, clinical_dim=clinical_dim,
        dropout=DROPOUT, membership=membership,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_cidx = -1.0; best_log_h = None; best_ep = -1
    log.info(f"  ({fold_label}) train_knob_b: paths={membership.shape[0]} fold_genes={membership.shape[1]}")
    for ep in range(1, EPOCHS + 1):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            T = batch.y.view(-1); E = batch.event.view(-1)
            clinical = batch.clinical.view(-1, clinical_dim)
            if E.sum().item() < 1: continue
            log_h = model(batch.x, batch.edge_index, batch.batch, clinical=clinical)
            loss = cox_partial_likelihood_loss(log_h, T, E)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Val
        model.eval()
        all_log_h, all_T, all_E = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                T = batch.y.view(-1); E = batch.event.view(-1)
                clinical = batch.clinical.view(-1, clinical_dim)
                log_h = model(batch.x, batch.edge_index, batch.batch, clinical=clinical)
                all_log_h.append(log_h.detach())
                all_T.append(T.detach()); all_E.append(E.detach())
        log_h = torch.cat(all_log_h).cpu().numpy()
        T_np = torch.cat(all_T).cpu().numpy()
        E_np = torch.cat(all_E).cpu().numpy().astype(int)
        from lifelines.utils import concordance_index as _ci
        cidx = float(_ci(T_np, -log_h, E_np))
        if cidx > best_cidx:
            best_cidx = cidx; best_log_h = log_h.copy(); best_ep = ep
    log.info(f"  ({fold_label}) best val_cidx={best_cidx:.4f} (ep {best_ep})")
    return model, scaler_used, best_cidx, best_log_h, best_ep


def infer_knob_b(model, X_expr, X_clin, T, E, fold_expr_cols, edge_index_local,
                 clinical_dim, scaler):
    val_list, _ = build_dataset(
        X_expr, X_clin, T, E, fold_expr_cols, edge_index_local, scaler=scaler,
    )
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device(DEVICE)
    model.eval()
    all_log_h = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            clinical = batch.clinical.view(-1, clinical_dim)
            log_h = model(batch.x, batch.edge_index, batch.batch, clinical=clinical)
            all_log_h.append(log_h.detach())
    log_h = torch.cat(all_log_h).cpu().numpy()
    from lifelines.utils import concordance_index as _ci
    cidx = float(_ci(T, -log_h, E))
    return log_h, cidx


def main():
    t0 = time.time()
    # Load TCGA + METABRIC via Stage 5 helpers
    tcga = s5.load_tcga()
    metabric = s5.load_metabric(tcga["gene_ids_769"], tcga["kg_gene_to_idx"])

    # Cohort overlap universe (same as Stage 5)
    overlap_idx = metabric["overlap_expr_col_idx_in_tcga"]
    overlap_genes = metabric["overlap_genes"]
    log.info(f"Overlap universe (TCGA-769 ∩ METABRIC): {len(overlap_idx)} genes")
    X_tcga_overlap = tcga["X_expr"][:, overlap_idx]
    overlap_col_to_kg_idx = tcga["expr_col_to_kg_idx"][overlap_idx]

    full_edge_index = torch.load(KG_EDGES_PATH, weights_only=False)["gene_gene_edges"]
    kg_meta = json.loads(KG_META_PATH.read_text())
    reactome_edges = torch.load(KG_EDGES_PATH, weights_only=False)["gene_reactome_edges"]
    reactome_names_raw = kg_meta.get("reactome_names", None)

    # ---- Full-TCGA training (90/10 split for best-epoch selection) ----
    rng = np.random.default_rng(SEED)
    n_tcga = len(tcga["T"])
    perm = rng.permutation(n_tcga)
    n_tr = int(0.9 * n_tcga)
    full_tr = perm[:n_tr]; full_va = perm[n_tr:]
    log.info(f"Full-TCGA train: {n_tr}/{n_tcga} train, {n_tcga-n_tr} val")

    # 1. Per-fold (here: full) LASSO inside the leaky-769 ∩ METABRIC overlap universe
    y_all_tr = tcga["bins"][full_tr].astype(np.float64)
    nz_mask_full, _ = s5.per_fold_lasso_within_universe(
        X_tcga_overlap[full_tr], y_all_tr, fold_label="full-TCGA-knobB",
    )
    kg_info_full = s5.subset_kg_to_fold(
        nz_mask_full, overlap_col_to_kg_idx, full_edge_index, n_kg_genes=769,
    )
    log.info(f"  graph: {kg_info_full['n_nodes']} nodes, {kg_info_full['n_edges']} edges")

    # 2. Build pathway-membership matrix (R5 sentinel: drop pathways with < 3 fold genes)
    n_kg_genes = 769
    fold_kg_indices = overlap_col_to_kg_idx[kg_info_full["fold_expr_cols"]]
    fold_local_idx = np.arange(len(fold_kg_indices), dtype=np.int64)
    kg_to_local = -np.ones(n_kg_genes, dtype=np.int64)
    kg_to_local[fold_kg_indices] = fold_local_idx
    pw_info = s3c.build_pathway_membership(
        reactome_edges, kg_to_local, n_fold_genes=kg_info_full["n_nodes"],
        threshold=3, reactome_names=reactome_names_raw,
    )
    log.info(
        f"  R5 sentinel: kept {pw_info['n_kept']} / {pw_info['n_total_pathways_in_kg']} pathways "
        f"with >= 3 fold genes ({pw_info['n_pathways_with_any_fold_gene']} have any fold gene)"
    )
    if pw_info["n_kept"] < 5:
        log.warning(f"  R5 DEGENERATE: only {pw_info['n_kept']} pathways retained. Result may be unreliable.")

    # 3. Train knob B on full TCGA
    model_full, scaler_full, full_va_cidx, full_va_log_h, full_best_ep = train_knob_b(
        X_tcga_overlap[full_tr], X_tcga_overlap[full_va],
        tcga["X_clin_3"][full_tr], tcga["X_clin_3"][full_va],
        tcga["T"][full_tr], tcga["T"][full_va],
        tcga["E"][full_tr], tcga["E"][full_va],
        kg_info_full["fold_expr_cols"], kg_info_full["edge_index_local"],
        membership=pw_info["membership"],
        clinical_dim=3, fold_label="full-TCGA-knobB",
    )

    # 4. Infer on METABRIC
    met_log_h_b, met_cidx_b = infer_knob_b(
        model_full, metabric["X_expr"], metabric["X_clin_3"],
        metabric["T"], metabric["E"],
        kg_info_full["fold_expr_cols"], kg_info_full["edge_index_local"],
        clinical_dim=3, scaler=scaler_full,
    )
    met_boot_b = bootstrap_cindex(metabric["T"], metabric["E"], met_log_h_b,
                                  n_boot=2000, seed=SEED)
    log.info(
        f"Knob B  TCGA-10%-val={full_va_cidx:.4f}  METABRIC-external={met_cidx_b:.4f}  "
        f"95% CI=[{met_boot_b['ci_low']:.3f}, {met_boot_b['ci_high']:.3f}]"
    )

    # 5. Paired bootstrap vs Cox PH (predictions saved in Stage 5)
    s5_results = json.loads(S5_JSON.read_text())
    cox_met_risk = np.array(s5_results["full_tcga_run"]["metabric_risk_cox_full"])
    met_T = np.array(s5_results["full_tcga_run"]["metabric_T"])
    met_E = np.array(s5_results["full_tcga_run"]["metabric_E"])
    paired_b_vs_cox = paired_bootstrap_delta(
        met_T, met_E, risk_a=met_log_h_b, risk_b=cox_met_risk,
        n_boot=2000, seed=SEED,
    )
    knob_a_met_log_h = np.array(s5_results["full_tcga_run"]["metabric_log_h_gnn_full"])
    paired_b_vs_a = paired_bootstrap_delta(
        met_T, met_E, risk_a=met_log_h_b, risk_b=knob_a_met_log_h,
        n_boot=2000, seed=SEED,
    )
    log.info(
        f"PAIRED METABRIC Δ(KnobB - Cox)   = {paired_b_vs_cox['delta_point']:+.4f}  "
        f"95% CI=[{paired_b_vs_cox['delta_ci_low']:+.4f}, {paired_b_vs_cox['delta_ci_high']:+.4f}]  "
        f"P(B≤Cox)={paired_b_vs_cox['p_a_le_b']:.3f}"
    )
    log.info(
        f"PAIRED METABRIC Δ(KnobB - KnobA) = {paired_b_vs_a['delta_point']:+.4f}  "
        f"95% CI=[{paired_b_vs_a['delta_ci_low']:+.4f}, {paired_b_vs_a['delta_ci_high']:+.4f}]  "
        f"P(B≤A)={paired_b_vs_a['p_a_le_b']:.3f}"
    )

    total_dt = time.time() - t0
    payload = {
        "model": "SAGEPathwayClinical (knob B) — full-TCGA -> METABRIC",
        "tcga_n": len(tcga["T"]),
        "metabric_n": len(metabric["T"]),
        "n_pathways_kept": int(pw_info["n_kept"]),
        "n_pathways_with_any_fold_gene": int(pw_info["n_pathways_with_any_fold_gene"]),
        "n_total_pathways": int(pw_info["n_total_pathways_in_kg"]),
        "kept_path_names": pw_info["kept_path_names"],
        "kept_path_sizes": pw_info["kept_path_sizes"],
        "n_nodes": kg_info_full["n_nodes"],
        "n_edges": kg_info_full["n_edges"],
        "tcga_10pct_val_cidx": float(full_va_cidx),
        "tcga_10pct_val_best_epoch": int(full_best_ep),
        "metabric_cidx": float(met_cidx_b),
        "metabric_bootstrap": met_boot_b,
        "metabric_log_h_full": met_log_h_b.tolist(),
        "paired_metabric_b_vs_cox": paired_b_vs_cox,
        "paired_metabric_b_vs_a": paired_b_vs_a,
        "total_seconds": total_dt,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Saved {OUT_JSON}")
    log.info(f"Total: {total_dt:.0f}s")


if __name__ == "__main__":
    main()
