"""Stage 0 LASSO leakage audit: refit gene selection per fold's training partition.

The existing 769-gene set in `data/processed/expression_selected.tsv` was selected
by `LassoCV(cv=5).fit(X, y)` on the FULL cohort labels (see
`src/preprocessing.py:191`). Every gene in that set was chosen with knowledge of
every patient's survival_class — including patients in val folds. This is
gene-selection leakage.

This script produces the HONEST baseline by:
  1. Loading raw HTSeq counts (data/raw/tcga_brca_htseq_counts.tsv).
  2. For each of the 5 stratified CV folds (data/processed/cv_splits.json):
       a. Filter low-count genes on TRAIN only (sum >= min_total_counts).
       b. log2(x + 1) transform.
       c. Z-score on TRAIN only; apply to VAL.
       d. LassoCV on TRAIN only (target = survival_class, treated as continuous).
       e. Take the genes with non-zero coef as that fold's gene set.
       f. PCA(100) on those genes (train-fit, val-transform) + 7 clinical.
       g. Cox PH penalizer=0.5; score val with lifelines + sksurv C-index.
  3. Report per-fold honest C-index and the leakage delta vs the leaky baseline.

Outputs:
  - results/stage_0_lasso_audit.json   : per-fold honest C-indices + gene counts
  - appends a section to results/stage_0_summary.md
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_cindex
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

RAW_COUNTS_PATH = RAW / "tcga_brca_htseq_counts.tsv"
CLINICAL_PATH = DATA / "clinical_features.tsv"
SURVIVAL_PATH = DATA / "clinical_processed.tsv"
SPLITS_PATH = DATA / "cv_splits.json"

AUDIT_JSON = RESULTS / "stage_0_lasso_audit.json"
SUMMARY_MD = RESULTS / "stage_0_summary.md"

SEED = 42
N_FOLDS = 5
N_PCA = 100
PENALIZER = 0.5
LOW_VAR = 0.01

# Preprocessing knobs (mirroring src/preprocessing.py defaults)
MIN_TOTAL_COUNTS = 1000
LASSO_MAX_ITER = 10000
LASSO_INNER_CV = 5

# Honest baseline reads as the "leakage delta" vs this number from 00_baseline.py
LEAKY_BASELINE_CIDX = 0.7324
LEAKY_BASELINE_STD = 0.0141

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_0_lasso_audit")


def load_aligned_raw():
    """Load raw counts + clinical + survival; align by case_id; drop T <= 0."""
    log.info(f"Loading raw HTSeq counts from {RAW_COUNTS_PATH} ({RAW_COUNTS_PATH.stat().st_size/1e6:.0f} MB) ...")
    t0 = time.time()
    raw = pd.read_csv(RAW_COUNTS_PATH, sep="\t")
    log.info(f"  loaded in {time.time()-t0:.1f}s: {raw.shape[0]} genes x {raw.shape[1]-1} patients")

    gene_ids = raw["gene_id"].values
    case_ids_raw = list(raw.columns[1:])
    X_raw = raw.iloc[:, 1:].T.values.astype(np.float32)  # (n_patients, n_genes)

    surv = pd.read_csv(SURVIVAL_PATH, sep="\t")
    case_ids_surv = surv["case_id"].tolist()
    assert case_ids_raw == case_ids_surv, "raw counts cols vs clinical_processed rows mismatch"

    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    keep = T > 0

    case_ids = np.array(case_ids_raw)[keep]
    X_raw = X_raw[keep]
    T = T[keep]
    E = E[keep]
    bins_raw = surv["survival_class"].values
    bins = bins_raw[keep].astype(np.int64)

    clin_df = pd.read_csv(CLINICAL_PATH, sep="\t").iloc[keep].reset_index(drop=True)
    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= LOW_VAR]
    X_clin = clin_df[keep_cols].values.astype(np.float32)

    log.info(
        f"Aligned cohort: n={len(case_ids)}, n_genes_raw={X_raw.shape[1]}, "
        f"n_clinical={X_clin.shape[1]}, event_rate={E.mean():.3f}"
    )
    return case_ids, X_raw, X_clin, T, E, bins, gene_ids, keep_cols


def fold_lasso_select(X_raw_train, y_train, gene_ids, min_total_counts=1000):
    """Per-fold preprocessing + LassoCV. Returns (selected_gene_idx, train_z, scaler, retained_idx)."""
    # 1. low-count filter on TRAIN ONLY
    counts_train = X_raw_train.sum(axis=0)
    keep_genes = counts_train >= min_total_counts
    n_kept = int(keep_genes.sum())

    # 2. log2(x + 1) transform
    X_log = np.log2(X_raw_train[:, keep_genes] + 1.0)

    # 3. z-score on train only
    sc = StandardScaler()
    X_z = sc.fit_transform(X_log)

    # 4. LassoCV regression on z-scored expression -> survival_class (as continuous)
    log.info(f"    LassoCV on {X_z.shape[0]} samples x {n_kept} genes (cv={LASSO_INNER_CV}) ...")
    t0 = time.time()
    lasso = LassoCV(
        cv=LASSO_INNER_CV,
        random_state=SEED,
        max_iter=LASSO_MAX_ITER,
        n_jobs=-1,
    )
    lasso.fit(X_z, y_train)
    dt = time.time() - t0
    log.info(f"    LassoCV done in {dt:.1f}s, alpha = {lasso.alpha_:.6f}")

    # 5. genes with non-zero coefs
    nz = np.abs(lasso.coef_) > 0
    n_nz = int(nz.sum())
    log.info(f"    LASSO non-zero genes: {n_nz} / {n_kept}")

    # selected_idx is the index in the original raw matrix space (post low-count keep)
    return keep_genes, sc, nz, n_nz, dt


def fold_cox(
    X_raw_train, X_raw_val,
    X_clin_train, X_clin_val,
    T_train, E_train, T_val, E_val,
    keep_genes_idx, scaler, lasso_nz_idx,
    n_pca=100, penalizer=0.5,
):
    """Apply train-fit preprocessing to both train and val, run Cox PH, score val."""
    # apply low-count filter (from train) to both
    X_tr_log = np.log2(X_raw_train[:, keep_genes_idx] + 1.0)
    X_va_log = np.log2(X_raw_val[:, keep_genes_idx] + 1.0)

    # apply train-fit z-score
    X_tr_z = scaler.transform(X_tr_log)
    X_va_z = scaler.transform(X_va_log)

    # subset to LASSO-selected
    X_tr_z = X_tr_z[:, lasso_nz_idx]
    X_va_z = X_va_z[:, lasso_nz_idx]

    # PCA on train, transform val
    n_comp = min(n_pca, X_tr_z.shape[1], X_tr_z.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    Pt = pca.fit_transform(X_tr_z)
    Pv = pca.transform(X_va_z)

    X_tr = np.hstack([Pt, X_clin_train])
    X_va = np.hstack([Pv, X_clin_val])

    cols = [f"pc{i}" for i in range(Pt.shape[1])] + [f"c{i}" for i in range(X_clin_train.shape[1])]
    df_tr = pd.DataFrame(X_tr, columns=cols)
    df_tr["T"] = T_train
    df_tr["E"] = E_train.astype(int)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_tr, duration_col="T", event_col="E", show_progress=False)

    df_va = pd.DataFrame(X_va, columns=cols)
    risk = cph.predict_partial_hazard(df_va).values
    cidx_lifelines = float(lifelines_cindex(T_val, -risk, E_val))
    cidx_sksurv = float(concordance_index_censored(E_val.astype(bool), T_val, risk)[0])
    return cidx_lifelines, cidx_sksurv, n_comp, float(pca.explained_variance_ratio_.sum())


def main():
    splits = json.loads(SPLITS_PATH.read_text())
    case_ids, X_raw, X_clin, T, E, bins, gene_ids, clin_cols = load_aligned_raw()

    log.info("=" * 70)
    log.info(f"LASSO LEAKAGE AUDIT: {N_FOLDS}-fold per-fold-train LASSO refit")
    log.info("=" * 70)

    rows = []
    total_t0 = time.time()
    for fold in range(N_FOLDS):
        log.info(f"--- fold {fold} ---")
        s = splits[f"fold_{fold}"]
        tr = np.array(s["train_idx"])
        va = np.array(s["val_idx"])

        # LASSO target: survival_class (continuous regression on ordinal label,
        # matching src/preprocessing.py behavior)
        y_train = bins[tr].astype(np.float64)

        keep_genes_idx, scaler, lasso_nz_idx, n_nz, lasso_dt = fold_lasso_select(
            X_raw[tr], y_train, gene_ids, min_total_counts=MIN_TOTAL_COUNTS
        )

        cidx_l, cidx_s, n_pca_used, pca_var = fold_cox(
            X_raw[tr], X_raw[va],
            X_clin[tr], X_clin[va],
            T[tr], E[tr], T[va], E[va],
            keep_genes_idx, scaler, lasso_nz_idx,
            n_pca=N_PCA, penalizer=PENALIZER,
        )

        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(va),
                "events_val": int(E[va].sum()),
                "n_genes_after_count_filter": int(keep_genes_idx.sum()),
                "n_genes_lasso_nonzero": n_nz,
                "n_pca_used": n_pca_used,
                "pca_variance_explained": pca_var,
                "lasso_seconds": lasso_dt,
                "cindex_lifelines": cidx_l,
                "cindex_sksurv": cidx_s,
            }
        )
        log.info(
            f"    fold {fold} HONEST: c-idx (lifelines) = {cidx_l:.4f}, "
            f"(sksurv) = {cidx_s:.4f}, PCA_var = {pca_var:.3f}"
        )

    total_dt = time.time() - total_t0
    cidx = np.array([r["cindex_lifelines"] for r in rows])
    log.info(f"")
    log.info(f"HONEST baseline (lifelines): {cidx.mean():.4f} ± {cidx.std():.4f}")
    log.info(f"LEAKY baseline (from 00_baseline.py): {LEAKY_BASELINE_CIDX:.4f} ± {LEAKY_BASELINE_STD:.4f}")
    log.info(f"Leakage delta (LEAKY - HONEST): {LEAKY_BASELINE_CIDX - cidx.mean():+.4f}")
    log.info(f"Total audit time: {total_dt:.1f}s")

    payload = {
        "n_patients": int(len(case_ids)),
        "event_rate": float(E.mean()),
        "n_folds": N_FOLDS,
        "n_pca_target": N_PCA,
        "penalizer": PENALIZER,
        "min_total_counts": MIN_TOTAL_COUNTS,
        "lasso_inner_cv": LASSO_INNER_CV,
        "seed": SEED,
        "honest_baseline_per_fold": rows,
        "honest_baseline_mean": float(cidx.mean()),
        "honest_baseline_std": float(cidx.std()),
        "leaky_baseline_mean": LEAKY_BASELINE_CIDX,
        "leaky_baseline_std": LEAKY_BASELINE_STD,
        "leakage_delta": LEAKY_BASELINE_CIDX - float(cidx.mean()),
        "total_seconds": total_dt,
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Audit JSON: {AUDIT_JSON}")

    # Append section to summary
    lines = [
        "",
        "## LASSO Leakage Audit (HONEST baseline)",
        "",
        "Refit LASSO inside each fold's training partition on the raw 60k-gene matrix, ",
        f"then Cox PH PCA({N_PCA}) + clinical with penalizer={PENALIZER} (same model config).",
        "",
        "| Fold | n_train | n_val | n_genes after count filter | LASSO non-zero | PCA used | C-idx (lifelines) | C-idx (sksurv) | LASSO time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['n_train']} | {r['n_val']} | {r['n_genes_after_count_filter']} | "
            f"{r['n_genes_lasso_nonzero']} | {r['n_pca_used']} | "
            f"{r['cindex_lifelines']:.4f} | {r['cindex_sksurv']:.4f} | "
            f"{r['lasso_seconds']:.1f} |"
        )
    lines += [
        f"| **mean** |  |  |  |  |  | **{cidx.mean():.4f}** ± {cidx.std():.4f} |  |  |",
        "",
        f"- **Honest baseline (per-fold LASSO refit):** `{cidx.mean():.4f} ± {cidx.std():.4f}`",
        f"- **Leaky baseline (existing 769-gene set):** `{LEAKY_BASELINE_CIDX:.4f} ± {LEAKY_BASELINE_STD:.4f}`",
        f"- **Leakage delta (leaky − honest):** `{LEAKY_BASELINE_CIDX - cidx.mean():+.4f}`",
        "",
        "**This `{:.4f}` is the new north-star number for the rest of the thesis.**".format(cidx.mean()),
        "Any GNN result reported below this is a regression vs Cox PH; any result above it is signal.",
        "",
    ]
    existing = SUMMARY_MD.read_text() if SUMMARY_MD.exists() else ""
    SUMMARY_MD.write_text(existing + "\n".join(lines))
    log.info(f"Summary appended to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
