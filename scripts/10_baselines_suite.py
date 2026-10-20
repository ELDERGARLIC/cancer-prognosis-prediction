"""Stage 10 — Stronger survival baselines on the matched external protocol.

Addresses reviewer requests for a broader, fairer baseline family than the
PCA(100)+ridge Cox used in Stage 5:

  * Elastic-net Cox (Coxnet) directly on the per-fold selected genes -- no PCA,
    survival-aware penalty -- the linear comparator the reviewer asked for.
  * Random Survival Forest (RSF).
  * Gradient Boosting Survival Analysis (GBSA).

Everything is held identical to Stage 5 so the numbers are directly comparable
to the base GNN (M0):

  - gene universe = leaky-769 ∩ METABRIC overlap (same loader)
  - per-fold gene selection = LassoCV on the discretized survival class (same)
  - 3 shared clinical features (age, stage, sex), per-cohort z-score (same)
  - internal: 5-fold on cv_splits.json (same folds)
  - external: full-TCGA model on the identical seed-42 90/10 split -> METABRIC
  - external paired bootstrap on identical METABRIC patients vs the SAVED GNN
    predictions (metabric_log_h_gnn_full from stage_5_metabric_external.json)

Outputs:
  results/stage_10_baselines.json
  manuscript/table_baselines.tex
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = Path("/home/eldergarlic/network_share/Projects/thesis-pipelines/thesis-research-v1")
DATA = ROOT / "code/data/processed"
EXT = ROOT / "code/data/external/brca_metabric"
RESULTS = ROOT / "output/thesis-results-repo/results"
OUT_JSON = RESULTS / "stage_10_baselines.json"
OUT_TEX = ROOT / "output/thesis-writing-repo/manuscript/table_baselines.tex"
STAGE5 = RESULTS / "stage_5_metabric_external.json"

TCGA_EXPR = DATA / "expression_selected.tsv"
TCGA_CLIN = DATA / "clinical_features.tsv"
TCGA_SURV = DATA / "clinical_processed.tsv"
TCGA_SPLITS = DATA / "cv_splits.json"
MET_EXPR = EXT / "data_mrna_illumina_microarray.txt"
MET_PATIENT = EXT / "data_clinical_patient.txt"
MET_SAMPLE = EXT / "data_clinical_sample.txt"

SEED = 42
N_FOLDS = 5
LASSO_INNER_CV = 5
LASSO_MAX_ITER = 10000
N_BOOT = 2000
ALPHA = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage10")


# ---------------------------------------------------------------- loaders (mirror Stage 5)
def load_tcga():
    exp = pd.read_csv(TCGA_EXPR, sep="\t")
    gene_ids = exp["gene_id"].tolist()
    case_ids_exp = list(exp.columns[1:])
    X_expr = exp.iloc[:, 1:].T.values.astype(np.float32)
    surv = pd.read_csv(TCGA_SURV, sep="\t")
    assert case_ids_exp == surv["case_id"].tolist()
    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    bins = surv["survival_class"].values
    keep = T > 0
    X_expr, T, E, bins = X_expr[keep], T[keep], E[keep], bins[keep].astype(np.int64)
    clin_df = pd.read_csv(TCGA_CLIN, sep="\t").iloc[keep].reset_index(drop=True)
    age = clin_df["age"].values.astype(np.float32)
    stage = clin_df["stage_ordinal"].values.astype(np.float32)
    sex = clin_df["is_female"].values.astype(np.float32)
    X_clin_3 = np.stack([(age - age.mean()) / age.std(),
                         (stage - stage.mean()) / stage.std(), sex], axis=1).astype(np.float32)
    return {"X_expr": X_expr, "X_clin_3": X_clin_3, "T": T, "E": E, "bins": bins,
            "gene_ids": gene_ids}


def load_metabric(tcga_gene_ids):
    mp = pd.read_csv(MET_PATIENT, sep="\t", comment="#", low_memory=False)
    ms = pd.read_csv(MET_SAMPLE, sep="\t", comment="#", low_memory=False)
    mp["OS_event"] = mp["OS_STATUS"].apply(
        lambda s: int(s.split(":")[0]) if isinstance(s, str) else np.nan)
    merged = ms[["PATIENT_ID", "SAMPLE_ID", "TUMOR_STAGE"]].merge(
        mp[["PATIENT_ID", "OS_MONTHS", "OS_event", "AGE_AT_DIAGNOSIS", "SEX"]],
        on="PATIENT_ID", how="inner")
    valid = (merged["OS_event"].notna() & merged["OS_MONTHS"].notna()
             & (merged["OS_MONTHS"] > 0) & merged["TUMOR_STAGE"].notna())
    merged = merged.loc[valid].reset_index(drop=True)
    leaky = set(tcga_gene_ids)
    chunks = []
    for chunk in pd.read_csv(MET_EXPR, sep="\t", chunksize=2000, low_memory=False):
        chunks.append(chunk[chunk["Hugo_Symbol"].isin(leaky)])
    met_expr = pd.concat(chunks, ignore_index=True).drop_duplicates(
        subset="Hugo_Symbol", keep="first")
    overlap_genes = [g for g in tcga_gene_ids if g in set(met_expr["Hugo_Symbol"])]
    overlap_idx = np.array([tcga_gene_ids.index(g) for g in overlap_genes], dtype=np.int64)
    met_ordered = met_expr.set_index("Hugo_Symbol").reindex(overlap_genes)
    expr_cols = [c for c in met_ordered.columns if c != "Entrez_Gene_Id"]
    merged = merged[merged["SAMPLE_ID"].isin(set(expr_cols))].reset_index(drop=True)
    sample_order = merged["SAMPLE_ID"].tolist()
    X_raw = met_ordered[sample_order].T.values.astype(np.float32)
    nan = np.isnan(X_raw)
    if nan.any():
        X_raw = np.where(nan, np.nanmean(X_raw, axis=0)[None, :], X_raw)
    X_expr = StandardScaler().fit_transform(X_raw).astype(np.float32)
    age = merged["AGE_AT_DIAGNOSIS"]
    stage = merged["TUMOR_STAGE"].astype(np.float32)
    age_z = ((age - age.mean()) / age.std()).values
    stage_z = ((stage - stage.mean()) / stage.std()).values
    sex = (merged["SEX"].astype(str) == "Female").astype(np.float32).values
    X_clin_3 = np.stack([age_z, stage_z, sex], axis=1).astype(np.float32)
    T = (merged["OS_MONTHS"].values * 30.4375).astype(np.float64)
    E = merged["OS_event"].astype(np.int64).values
    return {"X_expr": X_expr, "X_clin_3": X_clin_3, "T": T, "E": E,
            "overlap_idx": overlap_idx, "overlap_genes": overlap_genes}


# ---------------------------------------------------------------- helpers
def lasso_select(X_train, y_bins, label=""):
    Xz = StandardScaler().fit_transform(X_train)
    lasso = LassoCV(cv=LASSO_INNER_CV, random_state=SEED, max_iter=LASSO_MAX_ITER, n_jobs=-1)
    lasso.fit(Xz, y_bins.astype(np.float64))
    mask = np.abs(lasso.coef_) > 0
    log.info(f"  [{label}] LASSO nz={int(mask.sum())}/{X_train.shape[1]}")
    return mask


def make_xy(X_expr, cols, X_clin, T, E, scaler=None):
    Xg = X_expr[:, cols]
    if scaler is None:
        scaler = StandardScaler().fit(Xg)
    Xz = scaler.transform(Xg)
    X = np.hstack([Xz, X_clin]).astype(np.float64)
    y = Surv.from_arrays(event=E.astype(bool), time=T)
    return X, y, scaler


def cindex(y, risk):
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def fit_coxnet(X, y):
    """Elastic-net Cox with alpha chosen by 3-fold internal CV on the alpha path."""
    try:
        base = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.05, max_iter=200000)
        base.fit(X, y)
        alphas = base.alphas_
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = np.zeros(len(alphas))
        for tr, va in kf.split(X):
            m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=alphas, max_iter=200000)
            m.fit(X[tr], y[tr])
            for j, a in enumerate(alphas):
                try:
                    r = m.predict(X[va], alpha=a)
                    scores[j] += cindex(y[va], r)
                except Exception:
                    scores[j] += 0.5
        best = alphas[int(np.argmax(scores))]
        model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[best], fit_baseline_model=False,
                                       max_iter=200000)
        model.fit(X, y)
        return model
    except Exception as e:  # convergence fallback: ridge Cox
        log.warning(f"  coxnet failed ({e}); falling back to ridge CoxPH")
        m = CoxPHSurvivalAnalysis(alpha=1.0)
        m.fit(X, y)
        return m


def fit_models(X, y):
    models = {}
    models["coxnet"] = fit_coxnet(X, y)
    rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt",
                               random_state=SEED, n_jobs=-1)
    rsf.fit(X, y); models["rsf"] = rsf
    gbsa = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05, max_depth=2,
                                            subsample=0.8, random_state=SEED)
    gbsa.fit(X, y); models["gbsa"] = gbsa
    return models


def predict_risk(model, X):
    r = model.predict(X)
    return np.asarray(r, dtype=np.float64).ravel()


def paired_delta(T, E, risk_a, risk_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap Δ = C(a) - C(b) on identical patients (Stage-5 protocol)."""
    rng = np.random.default_rng(seed)
    y = Surv.from_arrays(event=E.astype(bool), time=T)
    pa, pb = cindex(y, risk_a), cindex(y, risk_b)
    deltas = []
    n = len(T)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if E[idx].sum() < 1:
            continue
        try:
            yy = Surv.from_arrays(event=E[idx].astype(bool), time=T[idx])
            deltas.append(cindex(yy, risk_a[idx]) - cindex(yy, risk_b[idx]))
        except Exception:
            continue
    deltas = np.asarray(deltas)
    return {"c_a": pa, "c_b": pb, "delta_point": float(pa - pb),
            "delta_ci_low": float(np.quantile(deltas, ALPHA / 2)),
            "delta_ci_high": float(np.quantile(deltas, 1 - ALPHA / 2)),
            "p_a_le_b": float((deltas <= 0).mean()),
            "p_a_ge_b": float((deltas >= 0).mean()), "n_valid": int(len(deltas))}


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    splits = json.loads(TCGA_SPLITS.read_text())
    s5 = json.loads(STAGE5.read_text())
    gnn_ext = np.asarray(s5["full_tcga_run"]["metabric_log_h_gnn_full"], float)
    cox_ext_pcaridge = np.asarray(s5["full_tcga_run"]["metabric_risk_cox_full"], float)
    gnn_ext_cidx = s5["full_tcga_run"]["metabric_gnn_cidx"]

    log.info("loading TCGA + METABRIC ...")
    tcga = load_tcga()
    met = load_metabric(tcga["gene_ids"])
    ov = met["overlap_idx"]
    X_tcga = tcga["X_expr"][:, ov]            # (n_tcga, 650)
    log.info(f"TCGA n={len(tcga['T'])}  METABRIC n={len(met['T'])}  overlap genes={len(ov)}")
    assert len(met["T"]) == len(gnn_ext), (len(met["T"]), len(gnn_ext))

    model_names = ["coxnet", "rsf", "gbsa"]

    # -------- internal 5-fold --------
    internal = {m: [] for m in model_names}
    for fold in range(N_FOLDS):
        s = splits[f"fold_{fold}"]
        tr, va = np.array(s["train_idx"]), np.array(s["val_idx"])
        mask = lasso_select(X_tcga[tr], tcga["bins"][tr], label=f"fold{fold}")
        cols = np.where(mask)[0]
        Xtr, ytr, sc = make_xy(X_tcga, cols, tcga["X_clin_3"], tcga["T"], tcga["E"], None)
        Xtr_f = Xtr[tr]; ytr_f = ytr[tr]
        Xva = Xtr[va]; yva = ytr[va]
        models = fit_models(Xtr_f, ytr_f)
        for m in model_names:
            c = cindex(yva, predict_risk(models[m], Xva))
            internal[m].append(c)
        log.info(f"  fold{fold} internal: " +
                 " ".join(f"{m}={internal[m][-1]:.4f}" for m in model_names))

    # -------- external: full-TCGA seed-42 90/10 -> METABRIC --------
    rng = np.random.default_rng(SEED)
    n = len(tcga["T"]); perm = rng.permutation(n); n_tr = int(0.9 * n)
    full_tr = perm[:n_tr]
    mask_full = lasso_select(X_tcga, tcga["bins"], label="full-TCGA")
    cols_full = np.where(mask_full)[0]

    # TCGA train features (scaler fit on TCGA train genes)
    Xtr_g = X_tcga[full_tr][:, cols_full]
    sc_tcga = StandardScaler().fit(Xtr_g)
    Xtr = np.hstack([sc_tcga.transform(Xtr_g), tcga["X_clin_3"][full_tr]]).astype(np.float64)
    ytr = Surv.from_arrays(event=tcga["E"][full_tr].astype(bool), time=tcga["T"][full_tr])

    # METABRIC features (per-cohort z-score, mirrors Stage 5 scaler=None)
    Xmet_g = met["X_expr"][:, cols_full]
    sc_met = StandardScaler().fit(Xmet_g)
    Xmet = np.hstack([sc_met.transform(Xmet_g), met["X_clin_3"]]).astype(np.float64)
    ymet = Surv.from_arrays(event=met["E"].astype(bool), time=met["T"])

    models_full = fit_models(Xtr, ytr)
    external = {}
    for m in model_names:
        risk = predict_risk(models_full[m], Xmet)
        c = cindex(ymet, risk)
        vs_gnn = paired_delta(met["T"], met["E"], risk, gnn_ext)  # Δ = baseline - GNN
        external[m] = {"cindex": c, "risk": risk.tolist(),
                       "delta_vs_gnn": vs_gnn["delta_point"],
                       "delta_vs_gnn_ci": [vs_gnn["delta_ci_low"], vs_gnn["delta_ci_high"]],
                       "p_baseline_le_gnn": vs_gnn["p_a_le_b"],
                       "p_baseline_ge_gnn": vs_gnn["p_a_ge_b"]}
        log.info(f"  EXTERNAL {m}: C={c:.4f}  Δ(vs GNN)={vs_gnn['delta_point']:+.4f} "
                 f"[{vs_gnn['delta_ci_low']:+.4f},{vs_gnn['delta_ci_high']:+.4f}]")

    # sanity: reproduce the saved PCA+ridge Cox external number too (context row)
    cox_pcaridge_cidx = cindex(ymet, cox_ext_pcaridge)

    out = {
        "n_tcga": int(n), "n_metabric": int(len(met["T"])), "n_overlap_genes": int(len(ov)),
        "n_boot": N_BOOT, "seed": SEED,
        "gnn_external_cidx": gnn_ext_cidx,
        "cox_pcaridge_external_cidx": cox_pcaridge_cidx,
        "internal_mean": {m: float(np.mean(internal[m])) for m in model_names},
        "internal_std": {m: float(np.std(internal[m])) for m in model_names},
        "internal_folds": internal,
        "external": external,
        "lasso_full_nz": int(mask_full.sum()),
        "elapsed_seconds": time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {OUT_JSON}  ({out['elapsed_seconds']:.0f}s)")
    write_latex(out)
    log.info(f"wrote {OUT_TEX}")


def write_latex(o):
    pretty = {"coxnet": "Elastic-net Cox (genes, no PCA)",
              "rsf": "Random Survival Forest",
              "gbsa": "Gradient Boosting Survival"}
    lines = [
        r"% Auto-generated by scripts/10_baselines_suite.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stronger survival baselines on the matched external protocol "
        rf"(overlap-{o['n_overlap_genes']} genes, identical folds, 3 shared clinical "
        r"features, per-cohort normalisation). Internal = 5-fold TCGA "
        r"mean$\pm$SD; External = full-TCGA model on METABRIC "
        rf"($n={o['n_metabric']}$). $\Delta$(vs GNN) and 95\% CI from patient-level "
        rf"paired bootstrap ($B={o['n_boot']}$) against the base GNN's saved "
        r"predictions; $p$ is the one-sided bootstrap probability that the baseline "
        r"is \emph{not} worse than the GNN.}",
        r"\label{tab:baselines_strong}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Internal (5-fold) & External & $\Delta$ vs GNN (95\% CI) \\",
        r"\midrule",
        rf"\textbf{{Base GNN (M0)}} & -- & \textbf{{{o['gnn_external_cidx']:.4f}}} & -- \\",
        rf"Cox PH, PCA(100)+ridge & -- & {o['cox_pcaridge_external_cidx']:.4f} & "
        rf"(Stage~5 baseline) \\",
        r"\midrule",
    ]
    for m in ["coxnet", "rsf", "gbsa"]:
        e = o["external"][m]
        ci = f"[{e['delta_vs_gnn_ci'][0]:+.3f}, {e['delta_vs_gnn_ci'][1]:+.3f}]"
        p_str = f", $p={e['p_baseline_ge_gnn']:.3f}$" if "p_baseline_ge_gnn" in e else ""
        lines.append(
            f"{pretty[m]} & {o['internal_mean'][m]:.4f}$\\pm${o['internal_std'][m]:.3f} "
            f"& {e['cindex']:.4f} & {e['delta_vs_gnn']:+.3f} {ci}{p_str} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\footnotesize",
        r"$\Delta$ vs GNN is (baseline $-$ GNN); negative favours the GNN. All "
        r"baselines select genes with the same per-fold LASSO as the GNN, then fit "
        r"on the selected genes directly (no PCA for Coxnet).",
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
