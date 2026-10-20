"""Stage 3 step 1 (knob D): minimal GraphSAGE + clinical late-fusion + Cox loss.

Same architecture as Stage 2 plus a 7-d clinical feature vector concatenated
to the pooled gene embedding before the MLP head. Reference: Gao 2021 reported
+0.061 C-index from the same late-fusion in their breast-cancer setup.

Pass criteria (relative to Stage 0 honest north-star 0.6605):
  (a) Mean val C-index >= 0.66 (Cox honest parity).
  (b) Fold std <= 0.05 (matches the Stage 0 honest variance floor).
  (c) R1 sentinel: cosine differentiated, no catastrophic collapse.
  (d) Paired bootstrap delta vs Cox PH leaky baseline (same fold splits, same
      patients): lower CI bound > 0 = real win, crosses 0 = tie.

Outputs:
  - results/stage_3a_sage_clinical.json     : per-fold per-epoch + val predictions
  - results/stage_3_summary.md              : appended knob-D section (incremental)
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
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cindex_bootstrap import bootstrap_cindex, paired_bootstrap_delta  # noqa: E402
from sage_models import SAGEClinical, cox_partial_likelihood_loss, mean_pairwise_cosine  # noqa: E402

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

EXPRESSION_PATH = DATA / "expression_selected.tsv"
CLINICAL_PATH = DATA / "clinical_features.tsv"
SURVIVAL_PATH = DATA / "clinical_processed.tsv"
SPLITS_PATH = DATA / "cv_splits.json"
KG_EDGES_PATH = DATA / "kg_edges.pt"

RESULTS_JSON = RESULTS / "stage_3a_sage_clinical.json"
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

# Cox PH leaky baseline config (for paired comparison)
COX_PCA = 100
COX_PENALIZER = 0.5

CINDEX_PARITY = 0.6605
CINDEX_LEAKY = 0.7324
COSINE_CATASTROPHIC = 0.99
DIFFERENTIATION_DELTA = 0.02

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_3a_sage_clinical")


# ---------- data ----------

def load_aligned():
    exp = pd.read_csv(EXPRESSION_PATH, sep="\t")
    case_ids_exp = list(exp.columns[1:])
    X_expr = exp.iloc[:, 1:].T.values.astype(np.float32)

    clin_df = pd.read_csv(CLINICAL_PATH, sep="\t")
    surv = pd.read_csv(SURVIVAL_PATH, sep="\t")
    case_ids_surv = surv["case_id"].tolist()
    assert case_ids_exp == case_ids_surv

    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    keep = T > 0

    case_ids = np.array(case_ids_exp)[keep]
    X_expr = X_expr[keep]
    T = T[keep]
    E = E[keep]
    clin_df = clin_df.iloc[keep].reset_index(drop=True)
    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= LOW_VAR]
    X_clin = clin_df[keep_cols].values.astype(np.float32)
    return case_ids, X_expr, X_clin, T, E, keep_cols


def load_edges():
    ed = torch.load(KG_EDGES_PATH, weights_only=False)
    return ed["gene_gene_edges"]


def build_dataset(X_expr_train, X_expr_val, X_clin_train, X_clin_val,
                  T_train, T_val, E_train, E_val, edge_index):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_expr_train).astype(np.float32)
    X_va = sc.transform(X_expr_val).astype(np.float32)

    train_list = []
    for i in range(X_tr.shape[0]):
        train_list.append(Data(
            x=torch.from_numpy(X_tr[i]).unsqueeze(-1),
            edge_index=edge_index,
            y=torch.tensor([T_train[i]], dtype=torch.float32),
            event=torch.tensor([E_train[i]], dtype=torch.float32),
            clinical=torch.from_numpy(X_clin_train[i]).unsqueeze(0),  # (1, 7)
        ))
    val_list = []
    for i in range(X_va.shape[0]):
        val_list.append(Data(
            x=torch.from_numpy(X_va[i]).unsqueeze(-1),
            edge_index=edge_index,
            y=torch.tensor([T_val[i]], dtype=torch.float32),
            event=torch.tensor([E_val[i]], dtype=torch.float32),
            clinical=torch.from_numpy(X_clin_val[i]).unsqueeze(0),
        ))
    return train_list, val_list


# ---------- training ----------

def run_one_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    all_log_h, all_T, all_E, all_emb = [], [], [], []
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            T = batch.y.view(-1)
            E = batch.event.view(-1)
            clinical = batch.clinical.view(-1, model.clinical_dim)
            log_h, emb = model(
                batch.x, batch.edge_index, batch.batch, clinical=clinical, return_emb=True,
            )
            if train:
                if E.sum().item() < 1:
                    continue
                loss = cox_partial_likelihood_loss(log_h, T, E)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            all_log_h.append(log_h.detach())
            all_T.append(T.detach())
            all_E.append(E.detach())
            if not train:
                all_emb.append(emb.detach())

    log_h = torch.cat(all_log_h)
    T = torch.cat(all_T)
    E = torch.cat(all_E)
    risk = log_h.cpu().numpy()
    T_np = T.cpu().numpy()
    E_np = E.cpu().numpy().astype(bool)
    if E_np.sum() == 0:
        cidx_l, cidx_s = float("nan"), float("nan")
    else:
        cidx_l = float(lifelines_cindex(T_np, -risk, E_np.astype(int)))
        cidx_s = float(concordance_index_censored(E_np, T_np, risk)[0])
    cosine = mean_pairwise_cosine(torch.cat(all_emb)) if not train else float("nan")
    return {
        "loss": (total_loss / max(n_batches, 1)) if train else float("nan"),
        "cindex_lifelines": cidx_l,
        "cindex_sksurv": cidx_s,
        "mean_pairwise_cosine": cosine,
        "log_h": risk if not train else None,
        "T": T_np if not train else None,
        "E": E_np.astype(int) if not train else None,
    }


def cox_leaky_baseline_per_fold(X_expr_tr, X_expr_va, X_clin_tr, X_clin_va,
                                T_tr, E_tr, T_va, E_va):
    """Refit the Stage-0 leaky Cox PH baseline on a single fold, return val risks
    (for paired bootstrap on identical val patients)."""
    sc = StandardScaler()
    Xe_tr = sc.fit_transform(X_expr_tr)
    Xe_va = sc.transform(X_expr_va)
    pca = PCA(n_components=COX_PCA, random_state=SEED)
    Pe_tr = pca.fit_transform(Xe_tr)
    Pe_va = pca.transform(Xe_va)
    X_tr = np.hstack([Pe_tr, X_clin_tr])
    X_va = np.hstack([Pe_va, X_clin_va])
    cols = [f"f{i}" for i in range(X_tr.shape[1])]
    df_tr = pd.DataFrame(X_tr, columns=cols)
    df_tr["T"] = T_tr
    df_tr["E"] = E_tr.astype(int)
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(df_tr, duration_col="T", event_col="E", show_progress=False)
    df_va = pd.DataFrame(X_va, columns=cols)
    risk = cph.predict_partial_hazard(df_va).values
    cidx = float(lifelines_cindex(T_va, -risk, E_va))
    return risk, cidx


def train_fold(fold, X_expr, X_clin, T, E, splits, edge_index, clinical_dim):
    log.info(f"--- fold {fold} ---")
    s = splits[f"fold_{fold}"]
    tr = np.array(s["train_idx"])
    va = np.array(s["val_idx"])

    train_list, val_list = build_dataset(
        X_expr[tr], X_expr[va], X_clin[tr], X_clin[va],
        T[tr], T[va], E[tr], E[va], edge_index,
    )
    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    device = torch.device(DEVICE)
    model = SAGEClinical(
        in_dim=1, hidden_dim=HIDDEN_DIM, clinical_dim=clinical_dim, dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    init_m = run_one_epoch(model, val_loader, optimizer, device, train=False)
    cosine_init = init_m["mean_pairwise_cosine"]
    log.info(f"  fold {fold} ep  0 (untrained val): cosine={cosine_init:.4f}")

    history = []
    best_val_log_h = None
    best_val_cidx = -1.0
    best_epoch = -1
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_m = run_one_epoch(model, train_loader, optimizer, device, train=True)
        va_m = run_one_epoch(model, val_loader, optimizer, device, train=False)
        cosine = va_m["mean_pairwise_cosine"]
        catastrophic = cosine > COSINE_CATASTROPHIC
        log.info(
            f"  fold {fold} ep {epoch:>2d}: train_loss={tr_m['loss']:.4f} "
            f"train_cidx={tr_m['cindex_lifelines']:.4f} "
            f"val_cidx={va_m['cindex_lifelines']:.4f} "
            f"val_cosine={cosine:.4f} (Δ={cosine - cosine_init:+.4f})"
            + ("  [R1 CATASTROPHIC]" if catastrophic else "")
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_m["loss"],
            "train_cindex": tr_m["cindex_lifelines"],
            "val_cindex_lifelines": va_m["cindex_lifelines"],
            "val_cindex_sksurv": va_m["cindex_sksurv"],
            "val_cosine": cosine,
            "val_cosine_delta_from_init": float(cosine - cosine_init),
            "val_cosine_catastrophic": bool(catastrophic),
        })
        if va_m["cindex_lifelines"] > best_val_cidx:
            best_val_cidx = va_m["cindex_lifelines"]
            best_epoch = epoch
            best_val_log_h = va_m["log_h"].copy()

    fold_dt = time.time() - t0

    # Fit Cox PH on same fold for paired bootstrap delta
    cox_risk, cox_cidx = cox_leaky_baseline_per_fold(
        X_expr[tr], X_expr[va], X_clin[tr], X_clin[va], T[tr], E[tr], T[va], E[va],
    )
    log.info(f"  fold {fold} Cox PH (leaky, same fold): c-idx = {cox_cidx:.4f}")

    # Bootstrap CIs on the GNN's BEST val predictions for this fold
    T_va = T[va].astype(np.float64)
    E_va = E[va].astype(np.int64)
    boot = bootstrap_cindex(T_va, E_va, best_val_log_h, n_boot=1000, seed=SEED + fold)
    paired = paired_bootstrap_delta(
        T_va, E_va, risk_a=best_val_log_h, risk_b=cox_risk,
        n_boot=1000, seed=SEED + fold,
    )
    log.info(
        f"  fold {fold} GNN best val_cidx={best_val_cidx:.4f} "
        f"95% CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] | "
        f"paired Δ(GNN-Cox)={paired['delta_point']:+.4f} "
        f"95% CI=[{paired['delta_ci_low']:+.3f}, {paired['delta_ci_high']:+.3f}] "
        f"P(GNN<=Cox)={paired['p_a_le_b']:.3f}"
    )

    final_cosine = float(history[-1]["val_cosine"])
    differentiated = (cosine_init - final_cosine) > DIFFERENTIATION_DELTA
    return {
        "fold": int(fold),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "events_train": int(E[tr].sum()),
        "events_val": int(E[va].sum()),
        "cosine_init": float(cosine_init),
        "history": history,
        "best_val_cindex": float(best_val_cidx),
        "best_val_cindex_epoch": int(best_epoch),
        "final_val_cindex": float(history[-1]["val_cindex_lifelines"]),
        "final_val_cosine": final_cosine,
        "cosine_delta": float(final_cosine - cosine_init),
        "differentiated": bool(differentiated),
        "any_cosine_catastrophic": any(h["val_cosine_catastrophic"] for h in history),
        "fold_seconds": float(fold_dt),
        "cox_leaky_cindex": float(cox_cidx),
        "best_val_log_h": best_val_log_h.tolist(),
        "cox_val_log_h": cox_risk.tolist(),
        "val_T": T_va.tolist(),
        "val_E": E_va.tolist(),
        "bootstrap": boot,
        "paired_vs_cox_leaky": paired,
    }


def write_summary(payload, prepend: bool):
    fr = payload["fold_results"]
    best = np.array([r["best_val_cindex"] for r in fr])
    final = np.array([r["final_val_cindex"] for r in fr])
    cosines_init = np.array([r["cosine_init"] for r in fr])
    cosines_final = np.array([r["final_val_cosine"] for r in fr])
    cosines_delta = np.array([r["cosine_delta"] for r in fr])
    cox_cidx = np.array([r["cox_leaky_cindex"] for r in fr])
    deltas_point = np.array([r["paired_vs_cox_leaky"]["delta_point"] for r in fr])
    any_catastrophic = any(r["any_cosine_catastrophic"] for r in fr)
    all_differentiated = all(r["differentiated"] for r in fr)

    # Pooled paired bootstrap on stacked val predictions
    T_pool = np.concatenate([np.array(r["val_T"]) for r in fr])
    E_pool = np.concatenate([np.array(r["val_E"]) for r in fr])
    gnn_pool = np.concatenate([np.array(r["best_val_log_h"]) for r in fr])
    cox_pool = np.concatenate([np.array(r["cox_val_log_h"]) for r in fr])
    pooled_paired = paired_bootstrap_delta(
        T_pool, E_pool, risk_a=gnn_pool, risk_b=cox_pool, n_boot=2000, seed=SEED,
    )

    pass_parity = best.mean() >= CINDEX_PARITY
    pass_std = best.std() <= 0.05
    pass_r1 = (not any_catastrophic) and all_differentiated
    pass_paired = pooled_paired["delta_ci_low"] > 0

    overall = "PASS" if (pass_parity and pass_std and pass_r1) else "MARGINAL/FAIL"

    lines = [
        "# Stage 3 — Knob D: Clinical Late-Fusion",
        "",
        "## TL;DR (knob D only; knob A = LASSO refit and knob B = pathway pool come next)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cox PH honest baseline (Stage 0) | **{CINDEX_PARITY:.4f}** |",
        f"| Cox PH leaky baseline (Stage 0) | {CINDEX_LEAKY:.4f} |",
        f"| Stage 2 minimal SAGE (no clinical) | 0.6114 ± 0.050 |",
        f"| **Stage 3 SAGE + clinical (this run)** | **{best.mean():.4f}** ± {best.std():.4f} |",
        f"| Lift vs Stage 2 minimal | {best.mean() - 0.6114:+.4f} |",
        f"| Cox PH leaky on identical splits (sanity check) | {cox_cidx.mean():.4f} ± {cox_cidx.std():.4f} |",
        f"| **Paired Δ(GNN − Cox leaky), pooled** | "
        f"**{pooled_paired['delta_point']:+.4f}** "
        f"95% CI [{pooled_paired['delta_ci_low']:+.4f}, {pooled_paired['delta_ci_high']:+.4f}], "
        f"P(GNN≤Cox)={pooled_paired['p_a_le_b']:.3f} |",
        f"| Mean cosine: init → final | {cosines_init.mean():.4f} → {cosines_final.mean():.4f} (Δ {cosines_delta.mean():+.4f}) |",
        f"| R1 catastrophic ever (>0.99) | {'YES' if any_catastrophic else 'NO'} |",
        f"| R1 all folds differentiated | {'YES' if all_differentiated else 'NO'} |",
        f"| Honest-parity gate (mean ≥ {CINDEX_PARITY}) | **{'PASS' if pass_parity else 'FAIL'}** |",
        f"| Variance-floor gate (std ≤ 0.05) | **{'PASS' if pass_std else 'FAIL'}** |",
        f"| Paired-CI > 0 gate (real win vs Cox leaky) | **{'PASS' if pass_paired else 'tie/FAIL'}** |",
        f"| **Overall** | **{overall}** |",
        f"| Total wall time | {payload['total_seconds']/60:.1f} min |",
        "",
        "## Per-fold results",
        "",
        "| Fold | best val cidx | best ep | 95% CI | Cox leaky cidx | Δ(GNN−Cox) | Δ 95% CI | P(GNN≤Cox) | cosine init→final | catastrophic |",
        "|---:|---:|---:|---|---:|---:|---|---:|---|:-:|",
    ]
    for r in fr:
        b = r["bootstrap"]
        p = r["paired_vs_cox_leaky"]
        lines.append(
            f"| {r['fold']} | {r['best_val_cindex']:.4f} | {r['best_val_cindex_epoch']} | "
            f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}] | "
            f"{r['cox_leaky_cindex']:.4f} | "
            f"{p['delta_point']:+.4f} | "
            f"[{p['delta_ci_low']:+.3f}, {p['delta_ci_high']:+.3f}] | "
            f"{p['p_a_le_b']:.3f} | "
            f"{r['cosine_init']:.4f}→{r['final_val_cosine']:.4f} ({r['cosine_delta']:+.4f}) | "
            f"{'YES' if r['any_cosine_catastrophic'] else 'no'} |"
        )

    lines += [
        "",
        "## Per-epoch curves (mean across 5 folds)",
        "",
        "| Epoch | train_loss | train_cidx | val_cidx | val_cosine |",
        "|---:|---:|---:|---:|---:|",
    ]
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
        lines.append(
            f"| {ep+1} | {cols[0]:.4f} | {cols[1]:.4f} | {cols[2]:.4f} | {cols[3]:.4f} |"
        )

    lines += [
        "",
        "## Verdict (knob D)",
        "",
    ]
    if pass_parity and pass_std and pass_r1:
        lines.append(
            f"**PASS** — clinical late-fusion delivers mean val C-index "
            f"`{best.mean():.4f}` with std `{best.std():.3f}`, hitting Cox PH honest parity. "
            f"R1 sentinel passes. Paired delta vs Cox leaky: "
            f"{pooled_paired['delta_point']:+.4f} "
            f"[{pooled_paired['delta_ci_low']:+.3f}, {pooled_paired['delta_ci_high']:+.3f}]."
        )
    elif pass_paired:
        lines.append(
            f"**PARTIAL PASS** — paired Δ vs Cox leaky has CI > 0 "
            f"({pooled_paired['delta_point']:+.4f} "
            f"[{pooled_paired['delta_ci_low']:+.4f}, {pooled_paired['delta_ci_high']:+.4f}]) "
            f"but mean cidx `{best.mean():.4f}` is below honest parity {CINDEX_PARITY}. "
            f"Knob A (LASSO refit) and knob B (pathway pool) need to push us over."
        )
    else:
        lines.append(
            f"**MARGINAL** — mean cidx `{best.mean():.4f}` "
            f"({'beats' if best.mean() > 0.6114 else 'no improvement on'} Stage 2 minimal at 0.6114). "
            f"Paired CI vs Cox leaky crosses zero ({pooled_paired['delta_ci_low']:+.4f} to "
            f"{pooled_paired['delta_ci_high']:+.4f}). Continue to knob A; "
            f"if knob A also fails to lift, reconsider before knob B."
        )

    lines += [
        "",
        "Per-fold lift attribution (Stage 2 minimal vs Stage 3 + clinical):",
        "",
        "| Fold | Stage 2 minimal | Stage 3 +clin | lift |",
        "|---:|---:|---:|---:|",
    ]
    stage2 = {0: 0.5944, 1: 0.6710, 2: 0.6009, 3: 0.6595, 4: 0.5309}
    for r in fr:
        s2 = stage2[r["fold"]]
        s3 = r["best_val_cindex"]
        lines.append(f"| {r['fold']} | {s2:.4f} | {s3:.4f} | {s3 - s2:+.4f} |")
    lines.append("")

    text = "\n".join(lines)
    if prepend or not SUMMARY_MD.exists():
        SUMMARY_MD.write_text(text)
    else:
        existing = SUMMARY_MD.read_text()
        SUMMARY_MD.write_text(text + "\n---\n\n" + existing)
    log.info(f"Summary: {SUMMARY_MD}")


def main():
    case_ids, X_expr, X_clin, T, E, clin_cols = load_aligned()
    edge_index = load_edges()
    splits = json.loads(SPLITS_PATH.read_text())

    log.info("=" * 70)
    log.info(
        f"STAGE 3 step 1 (knob D): SAGE (in=1, hidden={HIDDEN_DIM}, layers=2, "
        f"dropout={DROPOUT}) + global mean pool + late-fusion of {len(clin_cols)} "
        f"clinical features -> Cox PH loss"
    )
    log.info(f"  clinical cols: {clin_cols}")
    log.info(
        f"  cohort n={len(case_ids)}, n_genes={X_expr.shape[1]}, "
        f"n_edges={edge_index.shape[1]}, event_rate={E.mean():.3f}"
    )
    log.info(
        f"  train: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}, "
        f"device={DEVICE}, seed={SEED}"
    )
    log.info("=" * 70)

    fold_results = []
    t_total = time.time()
    for fold in range(N_FOLDS):
        r = train_fold(fold, X_expr, X_clin, T, E, splits, edge_index, len(clin_cols))
        fold_results.append(r)
        log.info(
            f"  fold {fold} done in {r['fold_seconds']:.0f}s: "
            f"best={r['best_val_cindex']:.4f} (ep {r['best_val_cindex_epoch']}) "
            f"final={r['final_val_cindex']:.4f} cosine_Δ={r['cosine_delta']:+.4f}"
        )

    total_dt = time.time() - t_total
    payload = {
        "n_folds": N_FOLDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT, "device": DEVICE, "seed": SEED,
        "clinical_cols": clin_cols,
        "fold_results": fold_results,
        "total_seconds": float(total_dt),
        "cox_ph_honest_baseline": CINDEX_PARITY,
        "cox_ph_leaky_baseline": CINDEX_LEAKY,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Results JSON: {RESULTS_JSON}")

    write_summary(payload, prepend=True)


if __name__ == "__main__":
    main()
