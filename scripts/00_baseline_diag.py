"""Stage 0 diagnostic: find what Cox PH config gets to ~0.748.

The leaky baseline run (00_baseline.py) gave 0.6952 with PCA(50) + clinical + penalizer=0.01.
The prior code (src/evaluate.py:436) reported 0.748 with a different feature recipe:
  - features = patient_embeddings.mean(axis=2)  # mean over BioBERT 768-dim
  - X_train[:, :50]                             # FIRST 50 columns, not PCA
  - penalizer = 0.1                             # 10x stronger ridge

This diagnostic sweeps several Cox PH configurations against the same splits to find
which one reproduces ~0.748 (and to characterize the search space for the honest baseline).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_cindex
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

EXPRESSION_PATH = DATA / "expression_selected.tsv"
CLINICAL_PATH = DATA / "clinical_features.tsv"
SURVIVAL_PATH = DATA / "clinical_processed.tsv"
SPLITS_PATH = DATA / "cv_splits.json"

SEED = 42
N_FOLDS = 5
LOW_VAR = 0.01

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_0_diag")


def load_aligned():
    exp = pd.read_csv(EXPRESSION_PATH, sep="\t")
    case_ids_exp = list(exp.columns[1:])
    X_expr = exp.iloc[:, 1:].T.values.astype(np.float32)

    clin_df = pd.read_csv(CLINICAL_PATH, sep="\t")
    surv = pd.read_csv(SURVIVAL_PATH, sep="\t")

    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    keep = T > 0

    case_ids = np.array(case_ids_exp)[keep]
    X_expr = X_expr[keep]
    clin_df = clin_df.iloc[keep].reset_index(drop=True)
    T = T[keep]
    E = E[keep]

    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= LOW_VAR]
    X_clin = clin_df[keep_cols].values.astype(np.float32)
    return case_ids, X_expr, X_clin, T, E


def fit_cox(X_train, X_val, T_train, E_train, T_val, E_val, penalizer=0.01):
    cols = [f"f{i}" for i in range(X_train.shape[1])]
    df_tr = pd.DataFrame(X_train, columns=cols)
    df_tr["T"] = T_train
    df_tr["E"] = E_train.astype(int)
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df_tr, duration_col="T", event_col="E", show_progress=False)
    except Exception as e:
        return float("nan")
    df_va = pd.DataFrame(X_val, columns=cols)
    risk = cph.predict_partial_hazard(df_va).values
    return float(lifelines_cindex(T_val, -risk, E_val))


def cv_run(name, build_features, penalizer=0.01):
    """build_features: callable(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va) -> (X_tr, X_va)"""
    splits = json.loads(SPLITS_PATH.read_text())
    case_ids, X_expr, X_clin, T, E = load_aligned()
    cidx = []
    for fold in range(N_FOLDS):
        s = splits[f"fold_{fold}"]
        tr = np.array(s["train_idx"])
        va = np.array(s["val_idx"])
        X_tr, X_va = build_features(X_expr[tr], X_expr[va], X_clin[tr], X_clin[va])
        c = fit_cox(X_tr, X_va, T[tr], E[tr], T[va], E[va], penalizer=penalizer)
        cidx.append(c)
    cidx = np.array(cidx)
    log.info(
        f"  {name:50s} mean={cidx.mean():.4f} ± {cidx.std():.4f} "
        f"per-fold={[f'{c:.3f}' for c in cidx]}"
    )
    return cidx


def main():
    log.info("=" * 100)
    log.info("STAGE 0 DIAGNOSTIC: Cox PH config sweep -- find what hits ~0.748")
    log.info("=" * 100)
    log.info("All runs use the same splits (data/processed/cv_splits.json) and the same T/E.")
    log.info("All expression is z-scored per fold (train-fit, val-transform) before any reduction.")
    log.info("")

    def fnone(Xe_tr, Xe_va, Xc_tr, Xc_va):
        # clinical only
        return Xc_tr, Xc_va

    def fpca(n_components, with_clinical=True):
        def _f(Xe_tr, Xe_va, Xc_tr, Xc_va):
            sc = StandardScaler()
            zt = sc.fit_transform(Xe_tr)
            zv = sc.transform(Xe_va)
            pca = PCA(n_components=n_components, random_state=SEED)
            pt = pca.fit_transform(zt)
            pv = pca.transform(zv)
            if with_clinical:
                return np.hstack([pt, Xc_tr]), np.hstack([pv, Xc_va])
            return pt, pv
        return _f

    def ffirst_n_genes(n, with_clinical=True):
        def _f(Xe_tr, Xe_va, Xc_tr, Xc_va):
            sc = StandardScaler()
            zt = sc.fit_transform(Xe_tr[:, :n])
            zv = sc.transform(Xe_va[:, :n])
            if with_clinical:
                return np.hstack([zt, Xc_tr]), np.hstack([zv, Xc_va])
            return zt, zv
        return _f

    def ftopvar_genes(n, with_clinical=True):
        def _f(Xe_tr, Xe_va, Xc_tr, Xc_va):
            v = Xe_tr.var(axis=0)
            top = np.argsort(v)[::-1][:n]
            sc = StandardScaler()
            zt = sc.fit_transform(Xe_tr[:, top])
            zv = sc.transform(Xe_va[:, top])
            if with_clinical:
                return np.hstack([zt, Xc_tr]), np.hstack([zv, Xc_va])
            return zt, zv
        return _f

    log.info("--- Clinical-only baselines ---")
    cv_run("clinical_only (penalizer=0.01)", fnone, penalizer=0.01)
    cv_run("clinical_only (penalizer=0.1)", fnone, penalizer=0.1)

    log.info("--- PCA + clinical (varying n_components, varying penalizer) ---")
    for npc in (20, 50, 100, 200):
        cv_run(f"pca({npc}) + clinical, p=0.01", fpca(npc, True), penalizer=0.01)
    for pen in (0.001, 0.01, 0.1, 0.5):
        cv_run(f"pca(50) + clinical, p={pen}", fpca(50, True), penalizer=pen)

    log.info("--- PCA expression-only (no clinical) ---")
    for npc in (20, 50):
        cv_run(f"pca({npc}) only, p=0.1", fpca(npc, False), penalizer=0.1)

    log.info("--- First-N-genes (mimicking the prior baseline's first-50-cols recipe) ---")
    for n in (20, 50, 100):
        cv_run(f"first_{n}_genes + clinical, p=0.1", ffirst_n_genes(n, True), penalizer=0.1)
        cv_run(f"first_{n}_genes only, p=0.1", ffirst_n_genes(n, False), penalizer=0.1)

    log.info("--- Top-variance N genes + clinical ---")
    for n in (20, 50, 100):
        cv_run(f"top_var_{n}_genes + clinical, p=0.1", ftopvar_genes(n, True), penalizer=0.1)


if __name__ == "__main__":
    main()
