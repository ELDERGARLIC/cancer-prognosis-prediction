"""Optional 5: Knob C (BioBERT gene init) on METABRIC.

Substitutes the per-gene scalar-expression node feature for a 32-dimensional
BioBERT-projected gene prior multiplied element-wise by the patient's
expression scalar for that gene:

    x[gene] = expression_z[patient, gene] * biobert_pca32[gene]

The 768-dim BioBERT embeddings (cached in `data/embeddings/`) are
PCA-projected to 32-dim using full-cohort BioBERT (no patient labels
involved -- the projection is a fixed function of gene identity). This
matches the design doc's `gene_embed_dim=32` choice and Chen & Zou 2023's
recommendation to use a small learnable head on top of LLM-derived priors.

Pipeline:
  1. Load BioBERT 768-dim cache.
  2. Restrict to overlap-universe genes (650 = leaky-769 ∩ METABRIC).
  3. PCA(32) of overlap BioBERT (fit cohort-wide; no label peek).
  4. Per-fold LASSO inside leaky-769 ∩ METABRIC (Stage 5 knob A pipeline).
  5. Build per-patient node features = expression × biobert_pca32 (in_dim=32).
  6. SAGEClinical with in_dim=32; same hyperparameters as knob A.
  7. Train on full TCGA -> infer on METABRIC -> paired bootstrap.
  8. Compare against knob A (in_dim=1 scalar expression) and Cox PH using
     identical METABRIC patients (Stage 5 saved predictions).

Output:
  - results/stage_5c_knob_c_biobert_metabric.json
  - prints headline numbers; appends summary section to stage_5_summary.md
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s5 = _load("s5", ROOT / "scripts" / "05_metabric_external.py")
from cindex_bootstrap import bootstrap_cindex, paired_bootstrap_delta  # noqa: E402
from sage_models import cox_partial_likelihood_loss  # noqa: E402

DATA = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
RESULTS = ROOT / "results"

KG_EDGES_PATH = DATA / "kg_edges.pt"
S5_JSON = RESULTS / "stage_5_metabric_external.json"
BIOBERT_NPY = EMB / "gene_embeddings.npy"
BIOBERT_NAMES = EMB / "gene_names.json"
OUT_JSON = RESULTS / "stage_5c_knob_c_biobert_metabric.json"
SUMMARY_MD = RESULTS / "stage_5_summary.md"

SEED = 42
HIDDEN_DIM = 128
DROPOUT = 0.4
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"
GENE_EMBED_DIM = 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("06e_knob_c_biobert")


class SAGEClinicalEmb(nn.Module):
    """Same as SAGEClinical but takes in_dim=32 gene-embedding input."""
    def __init__(self, in_dim=32, hidden_dim=128, clinical_dim=3, dropout=0.4):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        fused = hidden_dim + clinical_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, clinical, return_emb=False):
        h = F.relu(self.sage1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.sage2(h, edge_index))
        emb = global_mean_pool(h, batch)
        fused = torch.cat([emb, clinical], dim=-1)
        log_h = self.mlp(fused).squeeze(-1)
        if return_emb:
            return log_h, emb
        return log_h


def load_biobert_pca32(overlap_genes):
    """Load BioBERT cache, restrict to overlap_genes, PCA-project to 32-dim.

    Returns:
        gene_to_pca32: dict from gene_symbol -> np.ndarray (32,) float32.
        pca: fitted PCA(32) object (fit on the OVERLAP subset).
    """
    emb_full = np.load(BIOBERT_NPY)  # (1500, 768)
    names_full = json.loads(BIOBERT_NAMES.read_text())  # list of 1500
    name_to_idx = {n: i for i, n in enumerate(names_full)}
    overlap_idx = []
    for g in overlap_genes:
        if g in name_to_idx:
            overlap_idx.append(name_to_idx[g])
        else:
            log.warning(f"  gene {g} not in BioBERT cache; using zeros")
            overlap_idx.append(-1)
    overlap_idx = np.array(overlap_idx)

    # Build the embedding matrix; missing genes (idx == -1) get zero rows.
    emb_overlap = np.zeros((len(overlap_genes), emb_full.shape[1]), dtype=np.float32)
    valid = overlap_idx >= 0
    emb_overlap[valid] = emb_full[overlap_idx[valid]]
    log.info(f"  BioBERT lookup: {valid.sum()}/{len(overlap_genes)} overlap genes have embeddings")

    pca = PCA(n_components=GENE_EMBED_DIM, random_state=SEED)
    emb_pca = pca.fit_transform(emb_overlap)
    log.info(
        f"  PCA(32) on BioBERT (n_genes={len(overlap_genes)}, in_dim=768): "
        f"explained variance = {pca.explained_variance_ratio_.sum():.3f}"
    )
    return emb_pca.astype(np.float32), pca


def build_dataset_emb(X_expr, X_clin, T, E, fold_expr_cols, edge_index_local,
                     biobert_pca32_overlap, scaler=None):
    """Same as Stage 5's build_data_list but with 32-d gene-embed × expression input."""
    # Per-patient expression for the fold gene set
    X_fold = X_expr[:, fold_expr_cols]  # (n_patients, n_fold_genes)
    if scaler is None:
        sc = StandardScaler()
        X_z = sc.fit_transform(X_fold).astype(np.float32)
    else:
        sc = scaler
        X_z = sc.transform(X_fold).astype(np.float32)

    # Gene priors restricted to fold genes
    biobert_fold = biobert_pca32_overlap[fold_expr_cols]  # (n_fold_genes, 32)

    data_list = []
    n = X_z.shape[0]
    for i in range(n):
        # Each gene's node feature = expression_z[patient, gene] * biobert_pca32[gene]
        # Shape: (n_fold_genes, 32)
        node_feat = X_z[i, :, None] * biobert_fold  # broadcast
        d = Data(
            x=torch.from_numpy(node_feat.astype(np.float32)),
            edge_index=edge_index_local,
            y=torch.tensor([T[i]], dtype=torch.float32),
            event=torch.tensor([E[i]], dtype=torch.float32),
            clinical=torch.from_numpy(X_clin[i].astype(np.float32)).unsqueeze(0),
        )
        data_list.append(d)
    return data_list, sc


def train_knob_c(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va,
                 T_tr, T_va, E_tr, E_va,
                 fold_expr_cols, edge_index_local, biobert_pca32_overlap,
                 clinical_dim=3, fold_label=""):
    train_list, scaler = build_dataset_emb(
        X_expr_tr, X_clin_tr, T_tr, E_tr, fold_expr_cols, edge_index_local,
        biobert_pca32_overlap,
    )
    val_list, _ = build_dataset_emb(
        X_expr_va, X_clin_va, T_va, E_va, fold_expr_cols, edge_index_local,
        biobert_pca32_overlap, scaler=scaler,
    )
    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(DEVICE)
    model = SAGEClinicalEmb(
        in_dim=GENE_EMBED_DIM, hidden_dim=HIDDEN_DIM, clinical_dim=clinical_dim,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log.info(f"  ({fold_label}) train_knob_c: in_dim=32 fold_genes={len(fold_expr_cols)}")
    best_cidx = -1.0; best_log_h = None; best_ep = -1
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            T = batch.y.view(-1); E = batch.event.view(-1)
            clinical = batch.clinical.view(-1, clinical_dim)
            if E.sum().item() < 1: continue
            log_h = model(batch.x, batch.edge_index, batch.batch, clinical=clinical)
            loss = cox_partial_likelihood_loss(log_h, T, E)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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
        log_h_arr = torch.cat(all_log_h).cpu().numpy()
        T_np = torch.cat(all_T).cpu().numpy()
        E_np = torch.cat(all_E).cpu().numpy().astype(int)
        from lifelines.utils import concordance_index as _ci
        cidx = float(_ci(T_np, -log_h_arr, E_np))
        if cidx > best_cidx:
            best_cidx = cidx; best_log_h = log_h_arr.copy(); best_ep = ep
    log.info(f"  ({fold_label}) best val_cidx={best_cidx:.4f} (ep {best_ep})")
    return model, scaler, best_cidx, best_log_h, best_ep


def infer_knob_c(model, X_expr, X_clin, T, E, fold_expr_cols, edge_index_local,
                 biobert_pca32_overlap, scaler, clinical_dim=3):
    val_list, _ = build_dataset_emb(
        X_expr, X_clin, T, E, fold_expr_cols, edge_index_local,
        biobert_pca32_overlap, scaler=scaler,
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
    log_h_arr = torch.cat(all_log_h).cpu().numpy()
    from lifelines.utils import concordance_index as _ci
    cidx = float(_ci(T, -log_h_arr, E))
    return log_h_arr, cidx


def main():
    t0 = time.time()
    tcga = s5.load_tcga()
    metabric = s5.load_metabric(tcga["gene_ids_769"], tcga["kg_gene_to_idx"])

    overlap_idx = metabric["overlap_expr_col_idx_in_tcga"]
    overlap_genes = metabric["overlap_genes"]
    log.info(f"Overlap universe: {len(overlap_genes)} genes")
    X_tcga_overlap = tcga["X_expr"][:, overlap_idx]
    overlap_col_to_kg_idx = tcga["expr_col_to_kg_idx"][overlap_idx]

    # Load + project BioBERT to 32-d on the overlap universe (cohort-wide; no labels)
    biobert_pca32_overlap, _ = load_biobert_pca32(overlap_genes)
    log.info(f"  biobert_pca32 shape: {biobert_pca32_overlap.shape}")

    full_edge_index = torch.load(KG_EDGES_PATH, weights_only=False)["gene_gene_edges"]

    # ---- Full-TCGA training (90/10 split for best-epoch selection, same as Stage 5) ----
    rng = np.random.default_rng(SEED)
    n_tcga = len(tcga["T"])
    perm = rng.permutation(n_tcga)
    n_tr = int(0.9 * n_tcga)
    full_tr = perm[:n_tr]; full_va = perm[n_tr:]
    log.info(f"Full-TCGA train: {n_tr}/{n_tcga} train, {n_tcga-n_tr} val")

    y_all_tr = tcga["bins"][full_tr].astype(np.float64)
    nz_mask_full, _ = s5.per_fold_lasso_within_universe(
        X_tcga_overlap[full_tr], y_all_tr, fold_label="full-TCGA-knobC",
    )
    kg_info_full = s5.subset_kg_to_fold(
        nz_mask_full, overlap_col_to_kg_idx, full_edge_index, n_kg_genes=769,
    )
    log.info(f"  graph: {kg_info_full['n_nodes']} nodes, {kg_info_full['n_edges']} edges")

    model_full, scaler_full, full_va_cidx, full_va_log_h, full_best_ep = train_knob_c(
        X_tcga_overlap[full_tr], X_tcga_overlap[full_va],
        tcga["X_clin_3"][full_tr], tcga["X_clin_3"][full_va],
        tcga["T"][full_tr], tcga["T"][full_va],
        tcga["E"][full_tr], tcga["E"][full_va],
        kg_info_full["fold_expr_cols"], kg_info_full["edge_index_local"],
        biobert_pca32_overlap, clinical_dim=3, fold_label="full-TCGA-knobC",
    )

    met_log_h_c, met_cidx_c = infer_knob_c(
        model_full, metabric["X_expr"], metabric["X_clin_3"],
        metabric["T"], metabric["E"],
        kg_info_full["fold_expr_cols"], kg_info_full["edge_index_local"],
        biobert_pca32_overlap, scaler_full, clinical_dim=3,
    )
    met_boot_c = bootstrap_cindex(metabric["T"], metabric["E"], met_log_h_c,
                                  n_boot=2000, seed=SEED)
    log.info(
        f"Knob C  TCGA-10%-val={full_va_cidx:.4f}  METABRIC-external={met_cidx_c:.4f}  "
        f"95% CI=[{met_boot_c['ci_low']:.3f}, {met_boot_c['ci_high']:.3f}]"
    )

    # Paired bootstrap vs Cox PH and knob A (Stage 5 saved predictions)
    s5_results = json.loads(S5_JSON.read_text())
    cox_met_risk = np.array(s5_results["full_tcga_run"]["metabric_risk_cox_full"])
    met_T = np.array(s5_results["full_tcga_run"]["metabric_T"])
    met_E = np.array(s5_results["full_tcga_run"]["metabric_E"])
    knob_a_met_log_h = np.array(s5_results["full_tcga_run"]["metabric_log_h_gnn_full"])

    paired_c_vs_cox = paired_bootstrap_delta(
        met_T, met_E, risk_a=met_log_h_c, risk_b=cox_met_risk,
        n_boot=2000, seed=SEED,
    )
    paired_c_vs_a = paired_bootstrap_delta(
        met_T, met_E, risk_a=met_log_h_c, risk_b=knob_a_met_log_h,
        n_boot=2000, seed=SEED,
    )
    log.info(
        f"PAIRED METABRIC Δ(KnobC - Cox)   = {paired_c_vs_cox['delta_point']:+.4f}  "
        f"95% CI=[{paired_c_vs_cox['delta_ci_low']:+.4f}, {paired_c_vs_cox['delta_ci_high']:+.4f}]  "
        f"P(C≤Cox)={paired_c_vs_cox['p_a_le_b']:.3f}"
    )
    log.info(
        f"PAIRED METABRIC Δ(KnobC - KnobA) = {paired_c_vs_a['delta_point']:+.4f}  "
        f"95% CI=[{paired_c_vs_a['delta_ci_low']:+.4f}, {paired_c_vs_a['delta_ci_high']:+.4f}]  "
        f"P(C≤A)={paired_c_vs_a['p_a_le_b']:.3f}"
    )

    total_dt = time.time() - t0
    payload = {
        "model": "SAGEClinicalEmb (knob C, in_dim=32 BioBERT-PCA × expression) — full-TCGA -> METABRIC",
        "tcga_n": len(tcga["T"]),
        "metabric_n": len(metabric["T"]),
        "n_overlap_genes": len(overlap_genes),
        "biobert_pca_dim": GENE_EMBED_DIM,
        "n_nodes": kg_info_full["n_nodes"],
        "n_edges": kg_info_full["n_edges"],
        "tcga_10pct_val_cidx": float(full_va_cidx),
        "tcga_10pct_val_best_epoch": int(full_best_ep),
        "metabric_cidx": float(met_cidx_c),
        "metabric_bootstrap": met_boot_c,
        "metabric_log_h_full": met_log_h_c.tolist(),
        "paired_metabric_c_vs_cox": paired_c_vs_cox,
        "paired_metabric_c_vs_a": paired_c_vs_a,
        "total_seconds": total_dt,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Saved {OUT_JSON}")
    log.info(f"Total: {total_dt:.0f}s")


if __name__ == "__main__":
    main()
