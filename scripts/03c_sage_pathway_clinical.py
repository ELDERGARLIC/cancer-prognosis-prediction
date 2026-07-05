"""Stage 3 step 3 (knob B): pathway pooling + clinical, on top of knob A.

What knob B adds vs knob A:
  After the 2-layer GraphSAGE on each fold's per-fold-LASSO gene subgraph
  (knob A's setup), we replace the global mean pool with a Reactome-pathway
  pooling: for each retained pathway, mean over gene embeddings of fold genes
  in that pathway; single-head attention with uniform-init query produces the
  patient embedding; concat clinical and feed to the MLP head.

R5 sparsity sentinel:
  Reactome has 200 pathways. Knob A's per-fold gene set has 39-72 genes (within
  the leaky-769 universe). Random-model expectation: 0-4 pathways have >=5
  fold genes per fold; 3-16 pathways have >=3 fold genes. The design-doc R5
  threshold of 5 is too tight here, so we use threshold = 3 (option (a) from
  the design discussion). Per-fold retained-pathway counts are logged; folds
  with <5 retained pathways are flagged as degenerate but still trained for
  completeness.

Pass criterion: paired bootstrap delta vs knob A on identical val patients.
  - lower CI > 0 with point >= +0.02 = "thesis claim supported" (real lift)
  - CI crosses zero around 0 = "pathway pool doesn't help beyond gene GNN"
    (also reportable, also informative)
  - CI clearly < 0 = over-smoothing / sparsity issue
    (would point to R5 sentinel triggering meaningfully -- catch in advance)

Interpretability artifact: at end of training, save attention weights per val
patient. For each fold, log top-5 pathways for the 5 highest-risk and 5
lowest-risk val patients.

Outputs:
  - results/stage_3c_sage_pathway_clinical.json
  - results/stage_3_summary.md (knob B section prepended)
  - results/stage_3c_attention_per_fold.json (interpretability log)
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
from lifelines.utils import concordance_index as lifelines_cindex
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cindex_bootstrap import bootstrap_cindex, paired_bootstrap_delta  # noqa: E402
from sage_models import (  # noqa: E402
    SAGEPathwayClinical, cox_partial_likelihood_loss, mean_pairwise_cosine,
)

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

EXPRESSION_PATH = DATA / "expression_selected.tsv"
CLINICAL_PATH = DATA / "clinical_features.tsv"
SURVIVAL_PATH = DATA / "clinical_processed.tsv"
SPLITS_PATH = DATA / "cv_splits.json"
KG_EDGES_PATH = DATA / "kg_edges.pt"
KG_META_PATH = DATA / "kg_metadata.json"

KNOB_A_RESULTS = RESULTS / "stage_3b_sage_clinical_lasso_honest.json"
RESULTS_JSON = RESULTS / "stage_3c_sage_pathway_clinical.json"
ATTN_JSON = RESULTS / "stage_3c_attention_per_fold.json"
SUMMARY_MD = RESULTS / "stage_3_summary.md"

SEED = 42
N_FOLDS = 5
DEVICE = "cpu"
HIDDEN_DIM = 128
DROPOUT = 0.4
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
LOW_VAR = 0.01

LASSO_INNER_CV = 5
LASSO_MAX_ITER = 10000

R5_PATHWAY_GENE_THRESHOLD = 3   # lowered from design-doc 5 (see R5 sentinel comment)
R5_DEGENERATE_PATH_COUNT = 5    # fold flagged degenerate if fewer kept pathways

CINDEX_HONEST = 0.6605
COSINE_CATASTROPHIC = 0.99
DIFFERENTIATION_DELTA = 0.02

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_3c_sage_pathway")


def load_aligned():
    exp = pd.read_csv(EXPRESSION_PATH, sep="\t")
    gene_ids_769 = exp["gene_id"].tolist()
    case_ids_exp = list(exp.columns[1:])
    X_expr = exp.iloc[:, 1:].T.values.astype(np.float32)

    surv = pd.read_csv(SURVIVAL_PATH, sep="\t")
    case_ids_surv = surv["case_id"].tolist()
    assert case_ids_exp == case_ids_surv

    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    bins_raw = surv["survival_class"].values
    keep = T > 0

    case_ids = np.array(case_ids_exp)[keep]
    X_expr = X_expr[keep]
    T = T[keep]; E = E[keep]; bins = bins_raw[keep].astype(np.int64)

    clin_df = pd.read_csv(CLINICAL_PATH, sep="\t").iloc[keep].reset_index(drop=True)
    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= LOW_VAR]
    X_clin = clin_df[keep_cols].values.astype(np.float32)

    kg_meta = json.loads(KG_META_PATH.read_text())
    kg_gene_to_idx = kg_meta["gene_to_idx"]
    reactome_to_idx = kg_meta["reactome_names"]   # actually gene_to_idx is the only reliable key; will use names later
    reactome_names = kg_meta.get("reactome_names")  # could be dict {idx: name} or list
    expr_col_to_kg_idx = np.array([kg_gene_to_idx[g] for g in gene_ids_769], dtype=np.int64)

    return (case_ids, X_expr, X_clin, T, E, bins,
            gene_ids_769, kg_gene_to_idx, expr_col_to_kg_idx, reactome_names, keep_cols)


def per_fold_lasso_within_769(X_expr_train, y_train):
    sc = StandardScaler()
    X_z = sc.fit_transform(X_expr_train)
    log.info(f"    LassoCV on {X_z.shape[0]} samples x {X_z.shape[1]} (leaky-769) genes ...")
    t0 = time.time()
    lasso = LassoCV(cv=LASSO_INNER_CV, random_state=SEED, max_iter=LASSO_MAX_ITER, n_jobs=-1)
    lasso.fit(X_z, y_train)
    dt = time.time() - t0
    nz_mask = np.abs(lasso.coef_) > 0
    n_nz = int(nz_mask.sum())
    log.info(f"    LassoCV done in {dt:.0f}s, alpha={lasso.alpha_:.5f}, non-zero={n_nz}")
    return nz_mask, n_nz, dt


def subset_kg_to_fold(nz_mask, expr_col_to_kg_idx, full_edge_index):
    fold_expr_cols = np.where(nz_mask)[0]
    fold_kg_indices = expr_col_to_kg_idx[fold_expr_cols]
    fold_local_idx = np.arange(len(fold_expr_cols), dtype=np.int64)

    n_kg_genes = 769
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
        "fold_kg_indices": fold_kg_indices,
        "kg_to_local": kg_to_local,
        "edge_index_local": edge_index_local,
        "n_in_kg": int(nz_mask.sum()),
        "n_edges_after_mask": int(keep.sum()),
    }


def build_pathway_membership(reactome_edges, kg_to_local, n_fold_genes,
                             threshold=R5_PATHWAY_GENE_THRESHOLD,
                             reactome_names=None):
    """Build (n_paths_kept, n_fold_genes) row-normalized membership matrix.

    reactome_edges: (2, n_edges) tensor; row 0 = gene KG idx, row 1 = pathway idx.
    Each retained pathway has >= threshold fold genes; rows are normalized to
    mean (1/k where k = #fold genes in pathway).
    """
    src = reactome_edges[0].numpy()
    dst = reactome_edges[1].numpy()

    # Map gene KG idx -> fold-local idx (or -1 if outside fold)
    src_local = kg_to_local[src]
    keep = src_local >= 0
    src_local = src_local[keep]
    pw_idx = dst[keep]

    n_kg_pathways = int(dst.max() + 1) if len(dst) else 0

    # path_to_fold_genes[p] = list of fold-local gene indices in pathway p
    path_to_fold_genes = {}
    for g_local, p in zip(src_local, pw_idx):
        path_to_fold_genes.setdefault(int(p), []).append(int(g_local))

    kept_path_ids = sorted([p for p, gs in path_to_fold_genes.items() if len(gs) >= threshold])
    n_kept = len(kept_path_ids)

    membership = np.zeros((n_kept, n_fold_genes), dtype=np.float32)
    for row_i, p in enumerate(kept_path_ids):
        gs = path_to_fold_genes[p]
        for g in gs:
            membership[row_i, g] = 1.0
        membership[row_i] /= len(gs)

    # Map pathway id -> name (best effort; reactome_names may be list or dict)
    if reactome_names is None:
        names = [f"pathway_{p}" for p in kept_path_ids]
    elif isinstance(reactome_names, dict):
        names = [reactome_names.get(str(p)) or reactome_names.get(p)
                 or f"pathway_{p}" for p in kept_path_ids]
    elif isinstance(reactome_names, list):
        names = [reactome_names[p] if p < len(reactome_names) else f"pathway_{p}"
                 for p in kept_path_ids]
    else:
        names = [f"pathway_{p}" for p in kept_path_ids]

    # Stats: pathway sizes, all-pathways count
    all_path_sizes = sorted([len(gs) for gs in path_to_fold_genes.values()], reverse=True)

    return {
        "membership": torch.from_numpy(membership),
        "kept_path_ids": kept_path_ids,
        "kept_path_names": names,
        "n_kept": n_kept,
        "n_pathways_with_any_fold_gene": len(path_to_fold_genes),
        "n_total_pathways_in_kg": n_kg_pathways,
        "all_path_sizes": all_path_sizes,
        "kept_path_sizes": [len(path_to_fold_genes[p]) for p in kept_path_ids],
    }


def build_dataset(X_expr_train, X_expr_val, X_clin_train, X_clin_val,
                  T_tr, T_va, E_tr, E_va,
                  fold_expr_cols, edge_index_local):
    sc = StandardScaler()
    Xt = sc.fit_transform(X_expr_train[:, fold_expr_cols]).astype(np.float32)
    Xv = sc.transform(X_expr_val[:, fold_expr_cols]).astype(np.float32)

    train_list = [Data(
        x=torch.from_numpy(Xt[i]).unsqueeze(-1), edge_index=edge_index_local,
        y=torch.tensor([T_tr[i]], dtype=torch.float32),
        event=torch.tensor([E_tr[i]], dtype=torch.float32),
        clinical=torch.from_numpy(X_clin_train[i]).unsqueeze(0),
    ) for i in range(Xt.shape[0])]
    val_list = [Data(
        x=torch.from_numpy(Xv[i]).unsqueeze(-1), edge_index=edge_index_local,
        y=torch.tensor([T_va[i]], dtype=torch.float32),
        event=torch.tensor([E_va[i]], dtype=torch.float32),
        clinical=torch.from_numpy(X_clin_val[i]).unsqueeze(0),
    ) for i in range(Xv.shape[0])]
    return train_list, val_list


def run_one_epoch(model, loader, optimizer, device, train: bool, clinical_dim: int):
    model.train() if train else model.eval()
    all_log_h, all_T, all_E, all_emb, all_attn = [], [], [], [], []
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            T = batch.y.view(-1); E = batch.event.view(-1)
            clinical = batch.clinical.view(-1, clinical_dim)
            log_h, emb, attn = model(
                batch.x, batch.edge_index, batch.batch, clinical=clinical, return_attn=True,
            )
            if train:
                if E.sum().item() < 1: continue
                loss = cox_partial_likelihood_loss(log_h, T, E)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += float(loss.item()); n_batches += 1
            all_log_h.append(log_h.detach()); all_T.append(T.detach()); all_E.append(E.detach())
            if not train:
                all_emb.append(emb.detach()); all_attn.append(attn.detach())

    log_h = torch.cat(all_log_h)
    T = torch.cat(all_T); E = torch.cat(all_E)
    risk = log_h.cpu().numpy()
    T_np = T.cpu().numpy(); E_np = E.cpu().numpy().astype(bool)
    cidx_l = float(lifelines_cindex(T_np, -risk, E_np.astype(int))) if E_np.sum() else float("nan")
    cosine = mean_pairwise_cosine(torch.cat(all_emb)) if not train else float("nan")
    attn = torch.cat(all_attn).cpu().numpy() if not train else None
    return {
        "loss": (total_loss / max(n_batches, 1)) if train else float("nan"),
        "cindex_lifelines": cidx_l, "log_h": risk if not train else None,
        "T": T_np if not train else None, "E": E_np.astype(int) if not train else None,
        "mean_pairwise_cosine": cosine,
        "attn": attn,
    }


def train_fold(fold, splits, X_expr, X_clin, T, E, bins, expr_col_to_kg_idx,
               full_edge_index, reactome_edges, reactome_names, clinical_dim):
    log.info(f"--- fold {fold} ---")
    s = splits[f"fold_{fold}"]
    tr = np.array(s["train_idx"]); va = np.array(s["val_idx"])

    # Step 1: per-fold LASSO (same as knob A)
    y_train = bins[tr].astype(np.float64)
    nz_mask, n_nz, lasso_dt = per_fold_lasso_within_769(X_expr[tr], y_train)

    # Step 2: subset KG (same as knob A)
    kg_info = subset_kg_to_fold(nz_mask, expr_col_to_kg_idx, full_edge_index)
    log.info(f"    fold {fold} gene-graph: {kg_info['n_in_kg']} nodes, "
             f"{kg_info['n_edges_after_mask']} edges")
    if kg_info["n_in_kg"] < 10 or kg_info["n_edges_after_mask"] < 20:
        log.error(f"    fold {fold}: gene-graph too sparse -- skipping")
        return None

    # Step 3 (R5 sentinel): build pathway membership
    pw_info = build_pathway_membership(
        reactome_edges, kg_info["kg_to_local"], kg_info["n_in_kg"],
        threshold=R5_PATHWAY_GENE_THRESHOLD, reactome_names=reactome_names,
    )
    n_kept = pw_info["n_kept"]
    n_have_any = pw_info["n_pathways_with_any_fold_gene"]
    log.info(
        f"    fold {fold} R5 sentinel: kept {n_kept}/{pw_info['n_total_pathways_in_kg']} pathways "
        f"with >={R5_PATHWAY_GENE_THRESHOLD} fold genes "
        f"({n_have_any} pathways have any fold gene)"
    )
    if n_kept < R5_DEGENERATE_PATH_COUNT:
        log.warning(
            f"    fold {fold} R5 DEGENERATE: only {n_kept} pathways retained "
            f"(< {R5_DEGENERATE_PATH_COUNT}). Knob B on this fold is structurally "
            f"close to a {n_kept}-attention-slot model. Training anyway for completeness."
        )
    if n_kept < 1:
        log.error(f"    fold {fold}: zero pathways retained -- cannot build pathway pool, skipping")
        return None

    # Step 4: build dataset, model, optimizer
    train_list, val_list = build_dataset(
        X_expr[tr], X_expr[va], X_clin[tr], X_clin[va],
        T[tr], T[va], E[tr], E[va],
        kg_info["fold_expr_cols"], kg_info["edge_index_local"],
    )
    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
    device = torch.device(DEVICE)
    model = SAGEPathwayClinical(
        in_dim=1, hidden_dim=HIDDEN_DIM, clinical_dim=clinical_dim, dropout=DROPOUT,
        membership=pw_info["membership"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    init_m = run_one_epoch(model, val_loader, optimizer, device, train=False,
                           clinical_dim=clinical_dim)
    cosine_init = init_m["mean_pairwise_cosine"]
    log.info(f"    ep  0 (untrained): cosine={cosine_init:.4f}")

    history = []; best_cidx = -1.0; best_log_h = None; best_attn = None; best_epoch = -1
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_m = run_one_epoch(model, train_loader, optimizer, device, train=True,
                             clinical_dim=clinical_dim)
        va_m = run_one_epoch(model, val_loader, optimizer, device, train=False,
                             clinical_dim=clinical_dim)
        cosine = va_m["mean_pairwise_cosine"]
        catastrophic = cosine > COSINE_CATASTROPHIC
        log.info(
            f"    ep {epoch:>2d}: train_loss={tr_m['loss']:.4f} "
            f"train_cidx={tr_m['cindex_lifelines']:.4f} "
            f"val_cidx={va_m['cindex_lifelines']:.4f} cosine={cosine:.4f}"
            + ("  [R1 CATASTROPHIC]" if catastrophic else "")
        )
        history.append({
            "epoch": epoch, "train_loss": tr_m["loss"],
            "train_cindex": tr_m["cindex_lifelines"],
            "val_cindex_lifelines": va_m["cindex_lifelines"],
            "val_cosine": cosine,
            "val_cosine_catastrophic": bool(catastrophic),
            "val_cosine_delta_from_init": float(cosine - cosine_init),
        })
        if va_m["cindex_lifelines"] > best_cidx:
            best_cidx = va_m["cindex_lifelines"]
            best_log_h = va_m["log_h"].copy()
            best_attn = va_m["attn"].copy()  # (n_val, n_paths_kept)
            best_epoch = epoch

    fold_dt = time.time() - t0

    # Per-fold bootstrap CI on best val predictions
    T_va = T[va].astype(np.float64); E_va = E[va].astype(np.int64)
    boot = bootstrap_cindex(T_va, E_va, best_log_h, n_boot=1000, seed=SEED + fold)
    log.info(
        f"    fold {fold} GNN (knob B) best val_cidx={best_cidx:.4f} "
        f"95% CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] "
        f"(n_kept_pathways={n_kept})"
    )

    # Interpretability: top-5 pathways for top-5 highest-risk and top-5 lowest-risk val patients
    risk_order = np.argsort(best_log_h)[::-1]   # high to low risk
    top5_high = risk_order[:5].tolist()
    top5_low = risk_order[-5:][::-1].tolist()
    interp = {"high_risk": [], "low_risk": []}
    for label, idxs in (("high_risk", top5_high), ("low_risk", top5_low)):
        for i in idxs:
            attn_vec = best_attn[i]  # (n_paths_kept,)
            top_paths = np.argsort(attn_vec)[::-1][:5]
            interp[label].append({
                "val_patient_local_idx": int(i),
                "log_hazard": float(best_log_h[i]),
                "T_days": float(T_va[i]),
                "event": int(E_va[i]),
                "top5_pathways": [
                    {"name": pw_info["kept_path_names"][p],
                     "weight": float(attn_vec[p]),
                     "n_genes_in_pathway_in_fold": int(pw_info["kept_path_sizes"][p])}
                    for p in top_paths
                ],
            })

    final_cosine = float(history[-1]["val_cosine"])
    differentiated = (cosine_init - final_cosine) > DIFFERENTIATION_DELTA
    return {
        "fold": int(fold),
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "events_train": int(E[tr].sum()), "events_val": int(E[va].sum()),
        "lasso_seconds": float(lasso_dt),
        "n_lasso_nonzero": int(n_nz),
        "n_nodes": kg_info["n_in_kg"],
        "n_edges": kg_info["n_edges_after_mask"],
        "n_pathways_kept": int(n_kept),
        "n_pathways_with_any_fold_gene": int(n_have_any),
        "n_total_pathways": int(pw_info["n_total_pathways_in_kg"]),
        "kept_path_sizes": pw_info["kept_path_sizes"],
        "degenerate": bool(n_kept < R5_DEGENERATE_PATH_COUNT),
        "cosine_init": float(cosine_init),
        "history": history,
        "best_val_cindex": float(best_cidx),
        "best_val_cindex_epoch": int(best_epoch),
        "final_val_cindex": float(history[-1]["val_cindex_lifelines"]),
        "final_val_cosine": final_cosine,
        "cosine_delta": float(final_cosine - cosine_init),
        "differentiated": bool(differentiated),
        "any_cosine_catastrophic": any(h["val_cosine_catastrophic"] for h in history),
        "fold_seconds": float(fold_dt),
        "best_val_log_h": best_log_h.tolist(),
        "val_T": T_va.tolist(), "val_E": E_va.tolist(),
        "bootstrap": boot,
        "interpretability": interp,
    }


def write_summary(payload):
    fr = payload["fold_results"]
    best = np.array([r["best_val_cindex"] for r in fr])
    cosines_init = np.array([r["cosine_init"] for r in fr])
    cosines_final = np.array([r["final_val_cosine"] for r in fr])
    cosines_delta = np.array([r["cosine_delta"] for r in fr])
    any_catastrophic = any(r["any_cosine_catastrophic"] for r in fr)
    all_differentiated = all(r["differentiated"] for r in fr)
    n_paths = [r["n_pathways_kept"] for r in fr]
    n_degenerate = sum(1 for r in fr if r["degenerate"])

    knob_a = json.loads(KNOB_A_RESULTS.read_text())
    knob_a_fr = {r["fold"]: r for r in knob_a["fold_results"]}

    T_pool = np.concatenate([np.array(r["val_T"]) for r in fr])
    E_pool = np.concatenate([np.array(r["val_E"]) for r in fr])
    knob_b_pool = np.concatenate([np.array(r["best_val_log_h"]) for r in fr])
    knob_a_pool = np.concatenate([np.array(knob_a_fr[r["fold"]]["best_val_log_h"]) for r in fr])
    pooled_paired_vs_a = paired_bootstrap_delta(
        T_pool, E_pool, risk_a=knob_b_pool, risk_b=knob_a_pool, n_boot=2000, seed=SEED,
    )
    knob_a_mean = np.mean([knob_a_fr[r["fold"]]["best_val_cindex"] for r in fr])

    pass_parity = best.mean() >= CINDEX_HONEST
    pass_std = best.std() <= 0.05
    paired_clearly_above = pooled_paired_vs_a["delta_ci_low"] > 0
    paired_clearly_below = pooled_paired_vs_a["delta_ci_high"] < 0
    paired_crosses_zero = (not paired_clearly_above) and (not paired_clearly_below)

    if paired_clearly_above and pooled_paired_vs_a["delta_point"] >= 0.02:
        verdict = "PASS_THESIS_SUPPORTED"
    elif paired_clearly_above:
        verdict = "PASS_SMALL_LIFT"
    elif paired_crosses_zero:
        verdict = "TIE_NO_HARM"
    else:  # paired_clearly_below
        verdict = "REGRESSION"

    section = [
        "# Stage 3 — Knob B: Reactome Pathway Pooling (on top of Knob A)",
        "",
        "## What knob B adds vs knob A",
        "",
        "After the 2-layer GraphSAGE on each fold's per-fold-LASSO gene subgraph, knob B "
        "replaces global mean pool with a Reactome-pathway pool: for each retained pathway, "
        "mean over fold-gene embeddings in that pathway; single-head attention with "
        "uniform-init query (zeros) weighted-sums pathway representations into the patient "
        "embedding; concat clinical, MLP head, Cox loss.",
        "",
        f"**R5 sparsity sentinel:** knob A's per-fold gene set has 39-72 of 769 leaky-LASSO "
        f"genes. Reactome has 200 pathways. The design-doc threshold of >=5 fold genes per "
        f"pathway leaves 0-4 retained pathways per fold (degenerate). We use threshold = "
        f"**{R5_PATHWAY_GENE_THRESHOLD}** instead, documented as design-doc option (a). "
        f"Folds with fewer than {R5_DEGENERATE_PATH_COUNT} retained pathways are flagged "
        f"as 'degenerate' but trained for completeness.",
        "",
        "## TL;DR (knob B vs knob A)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cox PH HONEST baseline | 0.6605 ± 0.014 |",
        f"| Knob A (Stage 3b) | {knob_a_mean:.4f} |",
        f"| **Knob B (this run, knob A + pathway pool)** | **{best.mean():.4f}** ± {best.std():.4f} |",
        f"| Δ(knob B − knob A) point (fold-mean) | {best.mean() - knob_a_mean:+.4f} |",
        f"| **Paired Δ(knob B − knob A), pooled** | "
        f"**{pooled_paired_vs_a['delta_point']:+.4f}** "
        f"95% CI [{pooled_paired_vs_a['delta_ci_low']:+.4f}, {pooled_paired_vs_a['delta_ci_high']:+.4f}] "
        f"P(B≤A)={pooled_paired_vs_a['p_a_le_b']:.3f} |",
        f"| Δ vs Cox HONEST | {best.mean() - CINDEX_HONEST:+.4f} |",
        f"| R5 retained pathways per fold | {n_paths} (mean {np.mean(n_paths):.1f}) |",
        f"| R5 degenerate folds (n_paths < {R5_DEGENERATE_PATH_COUNT}) | {n_degenerate} of {len(fr)} |",
        f"| Mean cosine: init → final | {cosines_init.mean():.4f} → {cosines_final.mean():.4f} (Δ {cosines_delta.mean():+.4f}) |",
        f"| R1 catastrophic ever (>0.99) | {'YES' if any_catastrophic else 'NO'} |",
        f"| R1 all folds differentiated | {'YES' if all_differentiated else 'NO'} |",
        f"| Honest-parity gate (mean ≥ 0.6605) | **{'PASS' if pass_parity else 'FAIL'}** |",
        f"| Variance-floor gate (std ≤ 0.05) | **{'PASS' if pass_std else 'FAIL'}** |",
        f"| Paired vs knob A | "
        + ("**clearly above (real lift)**" if paired_clearly_above
           else "**clearly below (regression)**" if paired_clearly_below
           else "**crosses zero (tie / no harm)**") + " |",
        f"| **Verdict** | **{verdict}** |",
        f"| Total wall time | {payload['total_seconds']/60:.1f} min |",
        "",
        "## Per-fold details",
        "",
        "| Fold | LASSO genes | nodes | edges | retained pathways | degenerate | best val cidx | 95% CI | Δ vs knob A |",
        "|---:|---:|---:|---:|---:|:-:|---:|---|---:|",
    ]
    for r in fr:
        a_v = knob_a_fr[r["fold"]]["best_val_cindex"]
        section.append(
            f"| {r['fold']} | {r['n_lasso_nonzero']} | {r['n_nodes']} | {r['n_edges']} | "
            f"{r['n_pathways_kept']} | {'YES' if r['degenerate'] else 'no'} | "
            f"{r['best_val_cindex']:.4f} | "
            f"[{r['bootstrap']['ci_low']:.3f}, {r['bootstrap']['ci_high']:.3f}] | "
            f"{r['best_val_cindex'] - a_v:+.4f} |"
        )
    section += ["", "## Per-epoch curves (mean across folds, knob B)", "",
                "| Epoch | train_loss | train_cidx | val_cidx | val_cosine |",
                "|---:|---:|---:|---:|---:|"]
    n_e = max(len(r["history"]) for r in fr)
    for ep in range(n_e):
        cols = []
        for key in ("train_loss", "train_cindex", "val_cindex_lifelines", "val_cosine"):
            vals = []
            for r in fr:
                if ep < len(r["history"]):
                    v = r["history"][ep][key]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(float(v))
            cols.append(np.mean(vals) if vals else float("nan"))
        section.append(
            f"| {ep+1} | {cols[0]:.4f} | {cols[1]:.4f} | {cols[2]:.4f} | {cols[3]:.4f} |"
        )

    section += ["", "## Verdict (knob B)", ""]
    if verdict == "PASS_THESIS_SUPPORTED":
        section.append(
            f"**PASS_THESIS_SUPPORTED** — pathway pooling delivers a real lift over knob A: "
            f"point Δ {pooled_paired_vs_a['delta_point']:+.4f}, paired CI clearly above zero "
            f"({pooled_paired_vs_a['delta_ci_low']:+.4f} to {pooled_paired_vs_a['delta_ci_high']:+.4f}), "
            f"P(B≤A)={pooled_paired_vs_a['p_a_le_b']:.3f}. "
            f"The design's central architectural claim — *biological pathway structure in pooling beats global mean pool* — is supported on this cohort."
        )
    elif verdict == "PASS_SMALL_LIFT":
        section.append(
            f"**PASS_SMALL_LIFT** — paired CI vs knob A clearly above zero "
            f"({pooled_paired_vs_a['delta_ci_low']:+.4f} to {pooled_paired_vs_a['delta_ci_high']:+.4f}) "
            f"but point lift {pooled_paired_vs_a['delta_point']:+.4f} is below the +0.02 threshold "
            f"for a strong claim. Pathway pooling helps consistently but modestly."
        )
    elif verdict == "TIE_NO_HARM":
        section.append(
            f"**TIE_NO_HARM** — paired CI vs knob A crosses zero "
            f"({pooled_paired_vs_a['delta_ci_low']:+.4f} to {pooled_paired_vs_a['delta_ci_high']:+.4f}). "
            f"Pathway pooling does not measurably help beyond gene-level GNN at this gene-set "
            f"sparsity (knob A's per-fold LASSO + pathway pool is degenerate on at least some "
            f"folds — see R5 sentinel). This is reportable: the biology-named pooling does not "
            f"add accuracy at this scale, though it does deliver an interpretability artifact "
            f"(pathway attention weights) that flat GNN doesn't."
        )
    else:
        section.append(
            f"**REGRESSION** — paired CI vs knob A clearly below zero "
            f"({pooled_paired_vs_a['delta_ci_low']:+.4f} to {pooled_paired_vs_a['delta_ci_high']:+.4f}). "
            f"Pathway pooling on top of knob A's sparse gene set hurts more than it helps. "
            f"Likely cause: R5 sparsity (only {np.mean(n_paths):.1f} pathways retained mean) "
            f"makes the attention-over-pathways layer effectively a low-rank projection that "
            f"throws away information."
        )

    section += [
        "",
        "## Interpretability artifact",
        "",
        "Per-fold top-5 pathway attention weights for the 5 highest-risk and 5 lowest-risk "
        "val patients are saved to `results/stage_3c_attention_per_fold.json`. This is the "
        "Stage 5 interpretability figure starting point.",
        "",
    ]

    section_text = "\n".join(section)
    existing = SUMMARY_MD.read_text() if SUMMARY_MD.exists() else ""
    SUMMARY_MD.write_text(section_text + "\n---\n\n" + existing)
    log.info(f"Summary: {SUMMARY_MD}")

    # Save interpretability artifact separately
    interp_payload = {
        "fold_results": [
            {"fold": r["fold"], "interpretability": r["interpretability"]}
            for r in fr
        ]
    }
    ATTN_JSON.write_text(json.dumps(interp_payload, indent=2))
    log.info(f"Interpretability: {ATTN_JSON}")


def main():
    (case_ids, X_expr, X_clin, T, E, bins,
     gene_ids_769, kg_gene_to_idx, expr_col_to_kg_idx, reactome_names, clin_cols
     ) = load_aligned()
    splits = json.loads(SPLITS_PATH.read_text())
    ed = torch.load(KG_EDGES_PATH, weights_only=False)
    full_edge_index = ed["gene_gene_edges"]
    reactome_edges = ed["gene_reactome_edges"]

    log.info("=" * 70)
    log.info("STAGE 3 step 3 (knob B): per-fold LASSO + per-fold KG masking + Reactome pathway pool")
    log.info(
        f"  cohort n={len(case_ids)}, leaky-769={X_expr.shape[1]}, "
        f"reactome edges={reactome_edges.shape[1]}, n_pathways_total={int(reactome_edges[1].max()+1)}, "
        f"R5_threshold={R5_PATHWAY_GENE_THRESHOLD}"
    )
    log.info("=" * 70)

    fold_results = []
    t_total = time.time()
    for fold in range(N_FOLDS):
        r = train_fold(
            fold, splits, X_expr, X_clin, T, E, bins,
            expr_col_to_kg_idx, full_edge_index, reactome_edges, reactome_names,
            len(clin_cols),
        )
        if r is not None:
            fold_results.append(r)

    total_dt = time.time() - t_total
    payload = {
        "n_folds": N_FOLDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT, "device": DEVICE, "seed": SEED,
        "r5_threshold": R5_PATHWAY_GENE_THRESHOLD,
        "r5_degenerate_count": R5_DEGENERATE_PATH_COUNT,
        "clinical_cols": clin_cols,
        "fold_results": fold_results,
        "total_seconds": float(total_dt),
        "cox_ph_honest_baseline": CINDEX_HONEST,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Results JSON: {RESULTS_JSON}")

    write_summary(payload)


if __name__ == "__main__":
    main()
