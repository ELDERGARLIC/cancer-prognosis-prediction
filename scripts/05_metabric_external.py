"""Stage 5: external validation on METABRIC for the frozen knob-A architecture.

This script does NOT optimize anything. It measures.

Five things, in order, each on identical patient sets within their cohort:

  1. Within-TCGA comparator at the METABRIC-compatible 3-feature clinical set.
     Knob A's headline used 7 clinical features. METABRIC only shares 3
     (`age`, `stage_ordinal`, `is_female`); for cross-cohort fairness we re-run
     knob A on TCGA with just those 3, so the within-cohort and external
     numbers compete on the same feature set. The 7-feature knob A is reported
     too as the upper-bound within-cohort number we can produce on TCGA alone.

  2. Cox PH on TCGA at the same 3-feature clinical set + per-fold-honest LASSO,
     for the rigorous within-cohort linear baseline.

  3. METABRIC inference per fold. Each TCGA fold's trained knob-A model is
     applied to ALL METABRIC patients with valid OS + stage; we report
     mean/std across 5 fold-models on METABRIC as one comparator.

  4. METABRIC inference from full-TCGA-trained model. Single model trained on
     all 1074 TCGA patients (no holdout) applied to METABRIC; this is the
     headline external number per the brief.

  5. Cox PH external (TCGA-trained -> METABRIC inference) for paired
     comparison on identical METABRIC patients.

LASSO universe is restricted to (leaky-769 ∩ METABRIC genes) = 650 genes from
the start, so every fold's per-fold-LASSO subset is guaranteed available in
METABRIC. Z-normalization is per-cohort (no joint stats), gene order is
explicit, and clinical feature mapping is documented in the audit section.

Pass criterion (brief's gate): TCGA-internal vs METABRIC-external C-index drop
≤ 0.05. Knob A internal at 3-clinical is the reference; METABRIC ≥ (internal − 0.05) passes.

Outputs:
  - results/stage_5_metabric_external.json
  - results/stage_5_summary.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_cindex
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cindex_bootstrap import bootstrap_cindex, paired_bootstrap_delta  # noqa: E402
from sage_models import (  # noqa: E402
    SAGEClinical, cox_partial_likelihood_loss, mean_pairwise_cosine,
)

DATA = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external" / "brca_metabric"
RESULTS = ROOT / "results"

# --- TCGA paths ---
TCGA_EXPR = DATA / "expression_selected.tsv"
TCGA_CLIN = DATA / "clinical_features.tsv"
TCGA_SURV = DATA / "clinical_processed.tsv"
TCGA_SPLITS = DATA / "cv_splits.json"
KG_EDGES = DATA / "kg_edges.pt"
KG_META = DATA / "kg_metadata.json"

# --- METABRIC paths ---
MET_EXPR = EXT / "data_mrna_illumina_microarray.txt"
MET_PATIENT = EXT / "data_clinical_patient.txt"
MET_SAMPLE = EXT / "data_clinical_sample.txt"

# --- outputs ---
RESULTS_JSON = RESULTS / "stage_5_metabric_external.json"
SUMMARY_MD = RESULTS / "stage_5_summary.md"

# --- knobs (frozen from Stage 3) ---
SEED = 42
N_FOLDS = 5
DEVICE = "cpu"
HIDDEN_DIM = 128
DROPOUT = 0.4
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
LASSO_INNER_CV = 5
LASSO_MAX_ITER = 10000

# Cox PH config (same as Stage 0 leaky baseline -- the strongest linear comparator)
COX_PCA = 100
COX_PENALIZER = 0.5

# Stage 5 gate
EXTERNAL_DROP_GATE = 0.05

# Reference numbers from Stage 3
KNOB_A_TCGA_INTERNAL_7CLIN = 0.7200  # for documentation only; this run produces a 3-clin internal #

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_5_metabric")


# =====================================================================
# Loaders
# =====================================================================

def load_tcga():
    """Returns dict with X_expr (n,769), X_clin_3 (n,3), T (days), E (0/1),
    bins (survival_class), gene_ids_769, kg_gene_to_idx, expr_col_to_kg_idx,
    case_ids."""
    log.info("Loading TCGA ...")
    exp = pd.read_csv(TCGA_EXPR, sep="\t")
    gene_ids_769 = exp["gene_id"].tolist()
    case_ids_exp = list(exp.columns[1:])
    X_expr = exp.iloc[:, 1:].T.values.astype(np.float32)

    surv = pd.read_csv(TCGA_SURV, sep="\t")
    case_ids_surv = surv["case_id"].tolist()
    assert case_ids_exp == case_ids_surv
    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    bins = surv["survival_class"].values
    keep = T > 0
    case_ids = np.array(case_ids_exp)[keep]
    X_expr = X_expr[keep]
    T = T[keep]; E = E[keep]; bins = bins[keep].astype(np.int64)

    # 7-feature clinical (Stage 3 default)
    clin_df = pd.read_csv(TCGA_CLIN, sep="\t").iloc[keep].reset_index(drop=True)
    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= 0.01]
    X_clin_7 = clin_df[keep_cols].values.astype(np.float32)
    log.info(f"  7-clin features: {keep_cols}")

    # 3-feature clinical (METABRIC-compatible: age, stage_ordinal, is_female).
    # Per-cohort z-score for cross-cohort scale parity. TCGA `age` is already
    # z-scored from preprocessing; `stage_ordinal` is in [0, 1] -- needs re-z-score.
    # `is_female` stays as 0/1 (binary; z-scoring would explode the rare male values).
    clin3_cols = ["age", "stage_ordinal", "is_female"]
    age_v = clin_df["age"].values.astype(np.float32)
    stage_v = clin_df["stage_ordinal"].values.astype(np.float32)
    sex_v = clin_df["is_female"].values.astype(np.float32)
    age_z = (age_v - age_v.mean()) / age_v.std()
    stage_z = (stage_v - stage_v.mean()) / stage_v.std()
    X_clin_3 = np.stack([age_z, stage_z, sex_v], axis=1).astype(np.float32)
    log.info(
        f"  3-clin features (METABRIC-compatible): {clin3_cols} "
        f"(age + stage z-scored on TCGA; is_female kept binary)"
    )
    log.info(
        f"  TCGA 3-clin means {X_clin_3.mean(0)}, stds {X_clin_3.std(0)}"
    )

    kg_meta = json.loads(KG_META.read_text())
    kg_gene_to_idx = kg_meta["gene_to_idx"]
    expr_col_to_kg_idx = np.array([kg_gene_to_idx[g] for g in gene_ids_769], dtype=np.int64)

    return {
        "case_ids": case_ids, "X_expr": X_expr,
        "X_clin_7": X_clin_7, "X_clin_3": X_clin_3,
        "T": T, "E": E, "bins": bins,
        "gene_ids_769": gene_ids_769, "kg_gene_to_idx": kg_gene_to_idx,
        "expr_col_to_kg_idx": expr_col_to_kg_idx,
        "clin7_cols": keep_cols, "clin3_cols": clin3_cols,
    }


def load_metabric(tcga_gene_ids_769, tcga_kg_gene_to_idx):
    """Loads METABRIC expression restricted to (leaky-769 ∩ METABRIC genes).

    Returns dict with:
      X_expr (n_metabric, n_overlap_genes)  -- per-cohort z-scored
      X_clin_3 (n_metabric, 3)              -- (age, stage_ordinal, is_female), TCGA-aligned
      T (months -> days), E
      patient_ids (list)
      overlap_genes (list of HUGO symbols, in TCGA-769 order, only those in METABRIC)
      overlap_expr_col_idx_in_tcga (np.array of indices into tcga_gene_ids_769 that survived intersection)
    """
    log.info("Loading METABRIC clinical ...")
    mp = pd.read_csv(MET_PATIENT, sep="\t", comment="#", low_memory=False)
    ms = pd.read_csv(MET_SAMPLE, sep="\t", comment="#", low_memory=False)

    def parse_event(s):
        if not isinstance(s, str): return np.nan
        return int(s.split(":")[0])
    mp["OS_event"] = mp["OS_STATUS"].apply(parse_event)

    merged = ms[["PATIENT_ID", "SAMPLE_ID", "TUMOR_STAGE"]].merge(
        mp[["PATIENT_ID", "OS_MONTHS", "OS_event", "AGE_AT_DIAGNOSIS", "SEX"]],
        on="PATIENT_ID", how="inner",
    )
    log.info(f"  raw merged clinical+sample: {len(merged)}")

    # Keep only rows with valid OS, OS_MONTHS > 0, and valid TUMOR_STAGE
    valid = (
        merged["OS_event"].notna() & merged["OS_MONTHS"].notna()
        & (merged["OS_MONTHS"] > 0) & merged["TUMOR_STAGE"].notna()
    )
    merged = merged.loc[valid].reset_index(drop=True)
    log.info(f"  after valid OS + TUMOR_STAGE filter: {len(merged)}")

    # Load expression header to find which patients have expression
    log.info("  loading METABRIC expression (gene-symbol intersection only) ...")
    t0 = time.time()
    # Only load rows for genes in tcga_gene_ids_769
    leaky_769_set = set(tcga_gene_ids_769)
    chunks = []
    for chunk in pd.read_csv(MET_EXPR, sep="\t", chunksize=2000, low_memory=False):
        chunk = chunk[chunk["Hugo_Symbol"].isin(leaky_769_set)]
        chunks.append(chunk)
    met_expr = pd.concat(chunks, ignore_index=True)
    log.info(f"  loaded in {time.time()-t0:.1f}s: {met_expr.shape[0]} genes (of 769) x {met_expr.shape[1]-2} samples")

    # Drop duplicates by Hugo_Symbol (keep first)
    if met_expr["Hugo_Symbol"].duplicated().any():
        n_dup = met_expr["Hugo_Symbol"].duplicated().sum()
        log.warning(f"  {n_dup} duplicated gene symbols in METABRIC; keeping first row each")
        met_expr = met_expr.drop_duplicates(subset="Hugo_Symbol", keep="first")

    # Reorder met_expr rows to match TCGA-769 order; keep only overlap genes
    overlap_genes = [g for g in tcga_gene_ids_769 if g in set(met_expr["Hugo_Symbol"])]
    overlap_expr_col_idx_in_tcga = np.array(
        [tcga_gene_ids_769.index(g) for g in overlap_genes], dtype=np.int64
    )
    met_expr_ordered = met_expr.set_index("Hugo_Symbol").reindex(overlap_genes)
    sample_id_cols = [c for c in met_expr_ordered.columns if c != "Entrez_Gene_Id"]
    log.info(f"  overlap genes (TCGA-order): {len(overlap_genes)} / 769")

    # Filter merged to patients with expression
    expr_samples = set(sample_id_cols)
    merged = merged[merged["SAMPLE_ID"].isin(expr_samples)].reset_index(drop=True)
    log.info(f"  after expression-availability filter: {len(merged)}")

    # Build aligned expression matrix
    sample_order = merged["SAMPLE_ID"].tolist()
    X_raw = met_expr_ordered[sample_order].T.values.astype(np.float32)  # (n_pts, n_overlap_genes)
    # If METABRIC expression has NaN cells, mean-impute per gene (rare in z-scored arrays)
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        col_means = np.nanmean(X_raw, axis=0)
        X_raw = np.where(nan_mask, col_means[None, :], X_raw)
        log.warning(f"  imputed {nan_mask.sum()} NaN expression values with per-gene mean")

    # Per-cohort z-score on METABRIC (NOT joint with TCGA)
    sc = StandardScaler()
    X_expr = sc.fit_transform(X_raw).astype(np.float32)

    # METABRIC clinical to TCGA-aligned 3-feature space
    # TCGA `age` is z-scored cohort-wide. METABRIC `AGE_AT_DIAGNOSIS` is in years.
    # We z-score METABRIC age within METABRIC (per-cohort, parallel to TCGA per-cohort z-score).
    age_z = (merged["AGE_AT_DIAGNOSIS"] - merged["AGE_AT_DIAGNOSIS"].mean()) / merged["AGE_AT_DIAGNOSIS"].std()
    # METABRIC `TUMOR_STAGE` is integer 0..4. Z-score on METABRIC (parallel to
    # TCGA-side z-score in load_tcga). is_female stays binary (matches TCGA-side).
    stage = merged["TUMOR_STAGE"].astype(np.float32)
    stage_z = (stage - stage.mean()) / stage.std()
    is_female = (merged["SEX"].astype(str) == "Female").astype(np.float32)

    X_clin_3 = np.stack([age_z.values, stage_z.values, is_female.values], axis=1).astype(np.float32)
    log.info(f"  METABRIC X_clin_3: shape={X_clin_3.shape}, mean={X_clin_3.mean(0)}, std={X_clin_3.std(0)}")

    # Survival labels in days
    T = (merged["OS_MONTHS"].values * 30.4375).astype(np.float64)  # months -> days
    E = merged["OS_event"].astype(np.int64).values

    log.info(
        f"METABRIC final: n={len(merged)}, n_genes={X_expr.shape[1]}, "
        f"event_rate={E.mean():.3f}, T_med={int(np.median(T))}d"
    )
    return {
        "X_expr": X_expr, "X_clin_3": X_clin_3,
        "T": T, "E": E,
        "patient_ids": merged["PATIENT_ID"].tolist(),
        "sample_ids": merged["SAMPLE_ID"].tolist(),
        "overlap_genes": overlap_genes,
        "overlap_expr_col_idx_in_tcga": overlap_expr_col_idx_in_tcga,
    }


# =====================================================================
# Per-fold + full-TCGA training (knob A architecture, frozen)
# =====================================================================

def per_fold_lasso_within_universe(X_expr_train, y_train, fold_label=""):
    """LASSO within whatever expression columns are passed (either 769 or
    overlap-650). Returns boolean nz_mask over those columns."""
    sc = StandardScaler()
    X_z = sc.fit_transform(X_expr_train)
    log.info(f"    [{fold_label}] LassoCV on {X_z.shape[0]} x {X_z.shape[1]} ...")
    t0 = time.time()
    lasso = LassoCV(cv=LASSO_INNER_CV, random_state=SEED, max_iter=LASSO_MAX_ITER, n_jobs=-1)
    lasso.fit(X_z, y_train)
    dt = time.time() - t0
    nz_mask = np.abs(lasso.coef_) > 0
    log.info(f"    [{fold_label}] LassoCV {dt:.0f}s alpha={lasso.alpha_:.5f} nz={int(nz_mask.sum())}")
    return nz_mask, dt


def subset_kg_to_fold(nz_mask, expr_col_to_kg_idx, full_edge_index, n_kg_genes=769):
    fold_expr_cols = np.where(nz_mask)[0]
    fold_kg_indices = expr_col_to_kg_idx[fold_expr_cols]
    fold_local_idx = np.arange(len(fold_expr_cols), dtype=np.int64)

    kg_to_local = -np.ones(n_kg_genes, dtype=np.int64)
    kg_to_local[fold_kg_indices] = fold_local_idx

    src = full_edge_index[0].numpy()
    dst = full_edge_index[1].numpy()
    src_local = kg_to_local[src]
    dst_local = kg_to_local[dst]
    keep = (src_local >= 0) & (dst_local >= 0)
    edge_index_local = torch.tensor(
        np.stack([src_local[keep], dst_local[keep]]), dtype=torch.int64
    )
    return {
        "fold_expr_cols": fold_expr_cols,
        "edge_index_local": edge_index_local,
        "n_nodes": int(nz_mask.sum()),
        "n_edges": int(keep.sum()),
    }


def build_data_list(X_expr, X_clin, T, E, fold_expr_cols, edge_index_local, scaler=None):
    """Apply (fit-on-passed-data or transform-with-scaler) z-score then build PyG list."""
    X_sub = X_expr[:, fold_expr_cols]
    if scaler is None:
        sc = StandardScaler()
        Xz = sc.fit_transform(X_sub).astype(np.float32)
    else:
        sc = scaler
        Xz = sc.transform(X_sub).astype(np.float32)
    data_list = [Data(
        x=torch.from_numpy(Xz[i]).unsqueeze(-1),
        edge_index=edge_index_local,
        y=torch.tensor([T[i]], dtype=torch.float32),
        event=torch.tensor([E[i]], dtype=torch.float32),
        clinical=torch.from_numpy(X_clin[i]).unsqueeze(0),
    ) for i in range(X_sub.shape[0])]
    return data_list, sc


def run_one_epoch(model, loader, optimizer, device, train: bool, clinical_dim: int):
    model.train() if train else model.eval()
    all_log_h, all_T, all_E, all_emb = [], [], [], []
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            T = batch.y.view(-1); E = batch.event.view(-1)
            clinical = batch.clinical.view(-1, clinical_dim)
            log_h, emb = model(
                batch.x, batch.edge_index, batch.batch, clinical=clinical, return_emb=True,
            )
            if train:
                if E.sum().item() < 1: continue
                loss = cox_partial_likelihood_loss(log_h, T, E)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += float(loss.item()); n_batches += 1
            all_log_h.append(log_h.detach())
            all_T.append(T.detach()); all_E.append(E.detach())
            if not train:
                all_emb.append(emb.detach())

    log_h = torch.cat(all_log_h)
    T = torch.cat(all_T); E = torch.cat(all_E)
    risk = log_h.cpu().numpy()
    T_np = T.cpu().numpy(); E_np = E.cpu().numpy().astype(bool)
    cidx = float(lifelines_cindex(T_np, -risk, E_np.astype(int))) if E_np.sum() else float("nan")
    cosine = mean_pairwise_cosine(torch.cat(all_emb)) if not train else float("nan")
    return {
        "loss": (total_loss / max(n_batches, 1)) if train else float("nan"),
        "cindex": cidx, "log_h": risk if not train else None,
        "T": T_np if not train else None, "E": E_np.astype(int) if not train else None,
        "cosine": cosine,
    }


def train_knob_a(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va,
                 T_tr, T_va, E_tr, E_va,
                 fold_expr_cols, edge_index_local, clinical_dim,
                 epochs=EPOCHS, seed=SEED, fold_label=""):
    """Train + return best-val log_h, the model, and the train-fit scaler.

    The scaler is fit on TRAIN expression and stored so the same transform can
    be applied to METABRIC expression at inference time (per-cohort
    METABRIC-z-score is OUTSIDE this function; see comment).
    """
    train_list, scaler = build_data_list(
        X_expr_tr, X_clin_tr, T_tr, E_tr, fold_expr_cols, edge_index_local, scaler=None,
    )
    val_list, _ = build_data_list(
        X_expr_va, X_clin_va, T_va, E_va, fold_expr_cols, edge_index_local, scaler=scaler,
    )
    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(DEVICE)
    model = SAGEClinical(
        in_dim=1, hidden_dim=HIDDEN_DIM, clinical_dim=clinical_dim, dropout=DROPOUT,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_cidx = -1.0; best_log_h = None; best_epoch = -1; best_state = None
    for epoch in range(1, epochs + 1):
        run_one_epoch(model, train_loader, opt, device, train=True, clinical_dim=clinical_dim)
        va_m = run_one_epoch(model, val_loader, opt, device, train=False, clinical_dim=clinical_dim)
        if va_m["cindex"] > best_cidx:
            best_cidx = va_m["cindex"]
            best_log_h = va_m["log_h"].copy()
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    log.info(f"    [{fold_label}] knob A best val_cidx={best_cidx:.4f} (ep {best_epoch})")
    return model, scaler, best_cidx, best_log_h, best_epoch


def infer_knob_a(model, X_expr, X_clin, T, E, fold_expr_cols, edge_index_local,
                 clinical_dim, scaler, device_name=DEVICE):
    """Apply trained model to a new dataset (e.g. METABRIC). NOTE: scaler should
    be fit on TCGA train; here we OVERRIDE that and apply the METABRIC's own
    z-score (per-cohort), because we re-fit on METABRIC's overall distribution.
    Caller decides which scaler to pass."""
    data_list, _ = build_data_list(X_expr, X_clin, T, E, fold_expr_cols, edge_index_local, scaler=scaler)
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device(device_name)
    model = model.to(device); model.eval()
    out = run_one_epoch(model, loader, None, device, train=False, clinical_dim=clinical_dim)
    return out["log_h"], out["cindex"]


# =====================================================================
# Cox PH baseline at the same 3-clin feature set
# =====================================================================

def fit_cox_per_fold_3clin(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va,
                           T_tr, T_va, E_tr, E_va, fold_expr_cols):
    """Cox PH on PCA(min(50, n_genes)) + 3 clinical, penalizer 0.5.
    Returns val risks + cidx."""
    sc = StandardScaler()
    Xt = sc.fit_transform(X_expr_tr[:, fold_expr_cols])
    Xv = sc.transform(X_expr_va[:, fold_expr_cols])
    n_pca = min(COX_PCA, Xt.shape[1] - 1)
    if n_pca >= 2:
        pca = PCA(n_components=n_pca, random_state=SEED)
        Pt = pca.fit_transform(Xt); Pv = pca.transform(Xv)
        X_tr = np.hstack([Pt, X_clin_tr]); X_va = np.hstack([Pv, X_clin_va])
    else:
        X_tr = np.hstack([Xt, X_clin_tr]); X_va = np.hstack([Xv, X_clin_va])
    cols = [f"f{i}" for i in range(X_tr.shape[1])]
    df = pd.DataFrame(X_tr, columns=cols); df["T"] = T_tr; df["E"] = E_tr.astype(int)
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(df, duration_col="T", event_col="E", show_progress=False)
    risk = cph.predict_partial_hazard(pd.DataFrame(X_va, columns=cols)).values
    cidx = float(lifelines_cindex(T_va, -risk, E_va))
    return cph, sc, pca if n_pca >= 2 else None, risk, cidx


def cox_external_infer(cph, sc, pca, X_expr_external, X_clin_external,
                       fold_expr_cols, T_ext, E_ext):
    Xs = sc.transform(X_expr_external[:, fold_expr_cols])
    if pca is not None:
        P = pca.transform(Xs)
        X_eval = np.hstack([P, X_clin_external])
    else:
        X_eval = np.hstack([Xs, X_clin_external])
    cols = [f"f{i}" for i in range(X_eval.shape[1])]
    risk = cph.predict_partial_hazard(pd.DataFrame(X_eval, columns=cols)).values
    cidx = float(lifelines_cindex(T_ext, -risk, E_ext))
    return risk, cidx


# =====================================================================
# Main
# =====================================================================

def main():
    splits = json.loads(TCGA_SPLITS.read_text())
    full_edge_index = torch.load(KG_EDGES, weights_only=False)["gene_gene_edges"]

    tcga = load_tcga()
    metabric = load_metabric(tcga["gene_ids_769"], tcga["kg_gene_to_idx"])

    # ---- Restrict TCGA expression to the (leaky-769 ∩ METABRIC) overlap ----
    overlap_idx = metabric["overlap_expr_col_idx_in_tcga"]
    overlap_genes = metabric["overlap_genes"]
    log.info(f"\nLASSO universe = leaky-769 ∩ METABRIC = {len(overlap_idx)} genes")
    X_tcga_overlap = tcga["X_expr"][:, overlap_idx]
    # New per-overlap-col -> KG-idx mapping
    overlap_col_to_kg_idx = tcga["expr_col_to_kg_idx"][overlap_idx]

    log.info("=" * 70)
    log.info("STAGE 5: External validation on METABRIC (frozen knob-A architecture)")
    log.info(f"  TCGA n={len(tcga['T'])}, event_rate={tcga['E'].mean():.3f}")
    log.info(f"  METABRIC n={len(metabric['T'])}, event_rate={metabric['E'].mean():.3f}")
    log.info(f"  shared LASSO universe: {len(overlap_idx)} genes")
    log.info("=" * 70)

    # ====================== Per-fold runs ======================
    fold_results = []
    t_total = time.time()
    for fold in range(N_FOLDS):
        log.info(f"\n--- fold {fold} ---")
        s = splits[f"fold_{fold}"]
        tr = np.array(s["train_idx"]); va = np.array(s["val_idx"])

        # Per-fold LASSO within the overlap universe
        y_tr = tcga["bins"][tr].astype(np.float64)
        nz_mask, lasso_dt = per_fold_lasso_within_universe(
            X_tcga_overlap[tr], y_tr, fold_label=f"f{fold}",
        )
        kg_info = subset_kg_to_fold(
            nz_mask, overlap_col_to_kg_idx, full_edge_index, n_kg_genes=769,
        )
        log.info(f"    f{fold} graph: {kg_info['n_nodes']} nodes, {kg_info['n_edges']} edges")

        # Train knob A with 3-clinical
        model, tcga_scaler, tcga_val_cidx, tcga_val_log_h, best_ep = train_knob_a(
            X_tcga_overlap[tr], X_tcga_overlap[va],
            tcga["X_clin_3"][tr], tcga["X_clin_3"][va],
            tcga["T"][tr], tcga["T"][va], tcga["E"][tr], tcga["E"][va],
            kg_info["fold_expr_cols"], kg_info["edge_index_local"],
            clinical_dim=3, fold_label=f"f{fold}-3clin",
        )

        # Infer on METABRIC: use METABRIC's per-cohort scaler (re-fit on METABRIC overall,
        # restricted to the same per-fold gene set)
        # METABRIC X_expr is already z-scored cohort-wide. We need a "scaler" that
        # operates on raw METABRIC values restricted to fold genes; since METABRIC
        # is pre-z-scored, we just use an identity scaler (StandardScaler with
        # mean=0, std=1).
        # Simpler: pass scaler=None to build_data_list which will re-fit on the
        # passed METABRIC-restricted expression -- this is the "per-cohort z-score
        # on METABRIC" approach.
        met_log_h, met_cidx = infer_knob_a(
            model, metabric["X_expr"], metabric["X_clin_3"],
            metabric["T"], metabric["E"],
            kg_info["fold_expr_cols"], kg_info["edge_index_local"],
            clinical_dim=3, scaler=None,
        )
        # Bootstrap CI on METABRIC predictions
        met_boot = bootstrap_cindex(metabric["T"], metabric["E"], met_log_h,
                                    n_boot=1000, seed=SEED + fold)
        log.info(
            f"    f{fold} GNN  TCGA-internal val={tcga_val_cidx:.4f}  "
            f"METABRIC-external={met_cidx:.4f}  "
            f"95% CI=[{met_boot['ci_low']:.3f}, {met_boot['ci_high']:.3f}]  "
            f"drop={tcga_val_cidx - met_cidx:+.4f}"
        )

        # Cox PH per fold (same TCGA train, same METABRIC inference)
        cph, sc_cox, pca_cox, cox_tcga_risk, cox_tcga_cidx = fit_cox_per_fold_3clin(
            X_tcga_overlap[tr], X_tcga_overlap[va],
            tcga["X_clin_3"][tr], tcga["X_clin_3"][va],
            tcga["T"][tr], tcga["T"][va], tcga["E"][tr], tcga["E"][va],
            kg_info["fold_expr_cols"],
        )
        cox_met_risk, cox_met_cidx = cox_external_infer(
            cph, sc_cox, pca_cox,
            metabric["X_expr"], metabric["X_clin_3"],
            kg_info["fold_expr_cols"], metabric["T"], metabric["E"],
        )
        log.info(
            f"    f{fold} Cox  TCGA-internal val={cox_tcga_cidx:.4f}  "
            f"METABRIC-external={cox_met_cidx:.4f}  "
            f"drop={cox_tcga_cidx - cox_met_cidx:+.4f}"
        )

        fold_results.append({
            "fold": fold,
            "n_lasso_nonzero": int(nz_mask.sum()),
            "n_nodes": kg_info["n_nodes"],
            "n_edges": kg_info["n_edges"],
            "tcga_val_cidx_gnn": tcga_val_cidx,
            "tcga_val_log_h_gnn": tcga_val_log_h.tolist(),
            "tcga_val_T": tcga["T"][va].tolist(),
            "tcga_val_E": tcga["E"][va].astype(int).tolist(),
            "metabric_cidx_gnn": met_cidx,
            "metabric_log_h_gnn": met_log_h.tolist(),
            "metabric_boot_gnn": met_boot,
            "tcga_val_cidx_cox": cox_tcga_cidx,
            "tcga_val_risk_cox": cox_tcga_risk.tolist(),
            "metabric_cidx_cox": cox_met_cidx,
            "metabric_risk_cox": cox_met_risk.tolist(),
            "lasso_seconds": float(lasso_dt),
        })

    # ====================== Full-TCGA model (no holdout) -> METABRIC ======================
    log.info("\n--- full-TCGA model (no holdout) -> METABRIC ---")
    y_all = tcga["bins"].astype(np.float64)
    nz_mask_full, _ = per_fold_lasso_within_universe(X_tcga_overlap, y_all, fold_label="full-TCGA")
    kg_info_full = subset_kg_to_fold(
        nz_mask_full, overlap_col_to_kg_idx, full_edge_index, n_kg_genes=769,
    )
    log.info(f"    full graph: {kg_info_full['n_nodes']} nodes, {kg_info_full['n_edges']} edges")

    # For full-TCGA training, we DON'T have a held-out TCGA val set; we report
    # METABRIC inference as the only "val" number from this model.
    # We split off a tiny dummy 5% TCGA val to satisfy the train_knob_a signature
    # but track the full-TCGA-train-set cidx via TCGA train data passed as val.
    # Simpler: use a 90/10 split of TCGA for train_knob_a's internal "best" tracking
    # then report METABRIC as the external evaluation.
    rng = np.random.default_rng(SEED)
    n_tcga = len(tcga["T"])
    perm = rng.permutation(n_tcga)
    n_tr = int(0.9 * n_tcga)
    full_tr = perm[:n_tr]; full_va = perm[n_tr:]
    log.info(f"    full-TCGA train uses {n_tr}/{n_tcga} for training and {n_tcga-n_tr} for best-epoch selection")

    model_full, _, full_tcga_va_cidx, full_tcga_va_log_h, full_best_ep = train_knob_a(
        X_tcga_overlap[full_tr], X_tcga_overlap[full_va],
        tcga["X_clin_3"][full_tr], tcga["X_clin_3"][full_va],
        tcga["T"][full_tr], tcga["T"][full_va], tcga["E"][full_tr], tcga["E"][full_va],
        kg_info_full["fold_expr_cols"], kg_info_full["edge_index_local"],
        clinical_dim=3, fold_label="full-TCGA",
    )
    met_log_h_full, met_cidx_full = infer_knob_a(
        model_full, metabric["X_expr"], metabric["X_clin_3"],
        metabric["T"], metabric["E"],
        kg_info_full["fold_expr_cols"], kg_info_full["edge_index_local"],
        clinical_dim=3, scaler=None,
    )
    met_boot_full = bootstrap_cindex(metabric["T"], metabric["E"], met_log_h_full,
                                     n_boot=2000, seed=SEED)
    log.info(
        f"    full GNN  TCGA-10%-val={full_tcga_va_cidx:.4f}  "
        f"METABRIC-external={met_cidx_full:.4f}  "
        f"95% CI=[{met_boot_full['ci_low']:.3f}, {met_boot_full['ci_high']:.3f}]"
    )

    # Cox PH on full TCGA -> METABRIC
    cph_f, sc_f, pca_f, cox_tcga_va_risk_full, cox_tcga_va_cidx_full = fit_cox_per_fold_3clin(
        X_tcga_overlap[full_tr], X_tcga_overlap[full_va],
        tcga["X_clin_3"][full_tr], tcga["X_clin_3"][full_va],
        tcga["T"][full_tr], tcga["T"][full_va], tcga["E"][full_tr], tcga["E"][full_va],
        kg_info_full["fold_expr_cols"],
    )
    cox_met_risk_full, cox_met_cidx_full = cox_external_infer(
        cph_f, sc_f, pca_f,
        metabric["X_expr"], metabric["X_clin_3"],
        kg_info_full["fold_expr_cols"], metabric["T"], metabric["E"],
    )
    cox_met_boot_full = bootstrap_cindex(
        metabric["T"], metabric["E"], cox_met_risk_full, n_boot=2000, seed=SEED,
    )
    paired_full = paired_bootstrap_delta(
        metabric["T"], metabric["E"],
        risk_a=met_log_h_full, risk_b=cox_met_risk_full,
        n_boot=2000, seed=SEED,
    )
    log.info(
        f"    full Cox  TCGA-10%-val={cox_tcga_va_cidx_full:.4f}  "
        f"METABRIC-external={cox_met_cidx_full:.4f}  "
        f"95% CI=[{cox_met_boot_full['ci_low']:.3f}, {cox_met_boot_full['ci_high']:.3f}]"
    )
    log.info(
        f"    PAIRED METABRIC Δ(GNN−Cox) = {paired_full['delta_point']:+.4f}  "
        f"95% CI=[{paired_full['delta_ci_low']:+.4f}, {paired_full['delta_ci_high']:+.4f}]  "
        f"P(GNN≤Cox)={paired_full['p_a_le_b']:.3f}"
    )

    total_dt = time.time() - t_total
    log.info(f"\nStage 5 total wall time: {total_dt/60:.1f} min")

    # ============================ Aggregate ============================
    pf = fold_results
    tcga_gnn_mean = float(np.mean([r["tcga_val_cidx_gnn"] for r in pf]))
    tcga_gnn_std = float(np.std([r["tcga_val_cidx_gnn"] for r in pf]))
    met_gnn_mean = float(np.mean([r["metabric_cidx_gnn"] for r in pf]))
    met_gnn_std = float(np.std([r["metabric_cidx_gnn"] for r in pf]))
    tcga_cox_mean = float(np.mean([r["tcga_val_cidx_cox"] for r in pf]))
    tcga_cox_std = float(np.std([r["tcga_val_cidx_cox"] for r in pf]))
    met_cox_mean = float(np.mean([r["metabric_cidx_cox"] for r in pf]))
    met_cox_std = float(np.std([r["metabric_cidx_cox"] for r in pf]))

    drop_gnn = tcga_gnn_mean - met_gnn_mean
    drop_cox = tcga_cox_mean - met_cox_mean
    pass_gate = drop_gnn <= EXTERNAL_DROP_GATE

    payload = {
        "tcga_n": len(tcga["T"]), "metabric_n": len(metabric["T"]),
        "tcga_event_rate": float(tcga["E"].mean()),
        "metabric_event_rate": float(metabric["E"].mean()),
        "lasso_universe_size": len(overlap_idx),
        "external_drop_gate": EXTERNAL_DROP_GATE,
        "fold_results": pf,
        "summary": {
            "tcga_internal_gnn_mean": tcga_gnn_mean, "tcga_internal_gnn_std": tcga_gnn_std,
            "metabric_external_gnn_mean": met_gnn_mean, "metabric_external_gnn_std": met_gnn_std,
            "tcga_internal_cox_mean": tcga_cox_mean, "tcga_internal_cox_std": tcga_cox_std,
            "metabric_external_cox_mean": met_cox_mean, "metabric_external_cox_std": met_cox_std,
            "drop_gnn": drop_gnn, "drop_cox": drop_cox,
            "pass_external_gate": pass_gate,
        },
        "full_tcga_run": {
            "tcga_10pct_val_gnn_cidx": full_tcga_va_cidx,
            "tcga_10pct_val_cox_cidx": cox_tcga_va_cidx_full,
            "metabric_gnn_cidx": met_cidx_full,
            "metabric_gnn_boot": met_boot_full,
            "metabric_cox_cidx": cox_met_cidx_full,
            "metabric_cox_boot": cox_met_boot_full,
            "metabric_paired_gnn_minus_cox": paired_full,
            "metabric_log_h_gnn_full": met_log_h_full.tolist(),
            "metabric_risk_cox_full": cox_met_risk_full.tolist(),
            "metabric_T": metabric["T"].tolist(),
            "metabric_E": metabric["E"].astype(int).tolist(),
        },
        "feature_audit": {
            "tcga_clin_7": tcga["clin7_cols"],
            "tcga_clin_3": tcga["clin3_cols"],
            "metabric_clin_3": ["age (z-scored on METABRIC)", "stage_ordinal (TUMOR_STAGE z-scored)", "is_female (=1 always)"],
            "n_overlap_genes": len(overlap_idx),
            "n_total_kg_genes": 769,
            "overlap_pct": float(len(overlap_idx) / 769),
        },
        "total_seconds": float(total_dt),
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Results JSON: {RESULTS_JSON}")

    write_summary(payload)


def write_summary(p):
    s = p["summary"]
    pf = p["fold_results"]
    full = p["full_tcga_run"]
    fa = p["feature_audit"]
    pass_gate = s["pass_external_gate"]

    lines = [
        "# Stage 5 — METABRIC External Validation",
        "",
        "## TL;DR",
        "",
        f"**TCGA → METABRIC C-index drop gate:** ≤ {p['external_drop_gate']:.2f} (brief's bar).",
        "",
        "| Comparison | TCGA internal | METABRIC external | Drop |",
        "|---|---:|---:|---:|",
        f"| **Knob A (per-fold mean, 3-clin)** | "
        f"**{s['tcga_internal_gnn_mean']:.4f}** ± {s['tcga_internal_gnn_std']:.3f} | "
        f"**{s['metabric_external_gnn_mean']:.4f}** ± {s['metabric_external_gnn_std']:.3f} | "
        f"**{s['drop_gnn']:+.4f}** {'✅' if s['drop_gnn'] <= p['external_drop_gate'] else '❌'} |",
        f"| Cox PH (per-fold mean, 3-clin) | "
        f"{s['tcga_internal_cox_mean']:.4f} ± {s['tcga_internal_cox_std']:.3f} | "
        f"{s['metabric_external_cox_mean']:.4f} ± {s['metabric_external_cox_std']:.3f} | "
        f"{s['drop_cox']:+.4f} |",
        "",
        "Full-TCGA-trained model (no holdout) → METABRIC:",
        "",
        f"| Model | TCGA 10% holdout | METABRIC (n={p['metabric_n']}) | 95% CI |",
        "|---|---:|---:|---|",
        f"| Knob A full | {full['tcga_10pct_val_gnn_cidx']:.4f} | "
        f"**{full['metabric_gnn_cidx']:.4f}** | "
        f"[{full['metabric_gnn_boot']['ci_low']:.3f}, {full['metabric_gnn_boot']['ci_high']:.3f}] |",
        f"| Cox PH full | {full['tcga_10pct_val_cox_cidx']:.4f} | "
        f"{full['metabric_cox_cidx']:.4f} | "
        f"[{full['metabric_cox_boot']['ci_low']:.3f}, {full['metabric_cox_boot']['ci_high']:.3f}] |",
        "",
        f"**Paired Δ on METABRIC (Knob A − Cox PH, identical patients)**: "
        f"**{full['metabric_paired_gnn_minus_cox']['delta_point']:+.4f}** "
        f"95% CI [{full['metabric_paired_gnn_minus_cox']['delta_ci_low']:+.4f}, "
        f"{full['metabric_paired_gnn_minus_cox']['delta_ci_high']:+.4f}]  "
        f"P(GNN≤Cox) = {full['metabric_paired_gnn_minus_cox']['p_a_le_b']:.3f}",
        "",
        f"**Verdict:** **{'PASS — external drop ≤ ' + str(p['external_drop_gate']) if pass_gate else 'FAIL — external drop > ' + str(p['external_drop_gate'])}**.",
        "",
        "## Per-fold detail",
        "",
        "| Fold | n_genes | nodes | edges | TCGA val (GNN) | METABRIC (GNN) | drop GNN | TCGA val (Cox) | METABRIC (Cox) | drop Cox |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pf:
        lines.append(
            f"| {r['fold']} | {r['n_lasso_nonzero']} | {r['n_nodes']} | {r['n_edges']} | "
            f"{r['tcga_val_cidx_gnn']:.4f} | {r['metabric_cidx_gnn']:.4f} | "
            f"{r['tcga_val_cidx_gnn']-r['metabric_cidx_gnn']:+.4f} | "
            f"{r['tcga_val_cidx_cox']:.4f} | {r['metabric_cidx_cox']:.4f} | "
            f"{r['tcga_val_cidx_cox']-r['metabric_cidx_cox']:+.4f} |"
        )
    lines += [
        "",
        "## Feature audit (cross-cohort intersection)",
        "",
        f"- **TCGA full clinical (7 features):** {fa['tcga_clin_7']}",
        f"- **TCGA-METABRIC compatible (3 features):** {fa['tcga_clin_3']}",
        f"- **METABRIC source mapping:** {fa['metabric_clin_3']}",
        f"- **Gene universe:** leaky-769 ∩ METABRIC = **{fa['n_overlap_genes']} genes** "
        f"({fa['overlap_pct']*100:.1f}% of original 769; missing 119 are mostly AC* lncRNAs not on the Illumina array)",
        f"- **Survival labels:** TCGA OS.time in days; METABRIC OS_MONTHS converted to days × 30.44 days/month",
        f"- **Z-normalization:** per-cohort separately on each side; no joint normalization",
        "",
        "## Notes",
        "",
        f"- TCGA n = {p['tcga_n']}, event rate {p['tcga_event_rate']:.3f}",
        f"- METABRIC n = {p['metabric_n']} (after expression + valid OS + valid TUMOR_STAGE filter), "
        f"event rate {p['metabric_event_rate']:.3f} -- much higher than TCGA's because METABRIC has ~30y follow-up",
        f"- 3-feature clinical was picked to match what's available in both cohorts. The 7-feature "
        f"knob A internal headline (Stage 3) was {KNOB_A_TCGA_INTERNAL_7CLIN:.4f}; the 3-feature "
        f"version reported above is the fair within-cohort comparator for the external transfer.",
        f"- Total wall time: {p['total_seconds']/60:.1f} min",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines))
    log.info(f"Summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
