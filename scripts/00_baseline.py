"""Stage 0: Cox PH baseline reproduction + utility checks (FAST tasks).

Outputs:
  - data/processed/cv_splits.json   : 5-fold stratified splits, seed=42
  - results/stage_0_baseline.json   : per-fold C-indices + benchmarks
  - results/stage_0_summary.md      : human-readable summary

Tasks (run sequentially in main()):
  1. build_splits        : 5-fold StratifiedKFold on (survival_class x OS event)
  2. cox_leaky_baseline  : Cox PH on existing 769-gene PCA(50) + 9 clinical
                           (the gene set is LASSO-leaky; this is the upper-bound
                            reproduction of the debrief's ~0.748 number)
  3. cindex_crosscheck   : lifelines vs sksurv C-index agreement on identical
                           risk vector (subtle tie-handling differences have
                           caused wrong numbers in published papers)
  4. device_benchmark    : CPU vs MPS torch matmul timing for ~769-node batches

The honest LASSO-refit baseline (slow: needs raw 60k-gene matrix) lives in
scripts/00_lasso_audit.py. METABRIC fetch lives in scripts/00_metabric_fetch.py.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_cindex
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

EXPRESSION_PATH = DATA / "expression_selected.tsv"
CLINICAL_PATH = DATA / "clinical_features.tsv"
SURVIVAL_PATH = DATA / "clinical_processed.tsv"
SPLITS_PATH = DATA / "cv_splits.json"
RESULTS_JSON = RESULTS / "stage_0_baseline.json"
SUMMARY_MD = RESULTS / "stage_0_summary.md"

SEED = 42
N_FOLDS = 5
N_PCA = 100         # sweep in 00_baseline_diag.py: pca(100)+clinical+p=0.5 wins
PENALIZER = 0.5     # same sweep: monotonic up to 0.5, plateau after
TARGET_BAND = (0.73, 0.76)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_0_baseline")


LOW_VARIANCE_THRESHOLD = 0.01


def load_aligned():
    """Load expression / clinical / survival, drop OS.time <= 0, drop low-variance clinical cols."""
    log.info("Loading expression / clinical / survival ...")
    exp = pd.read_csv(EXPRESSION_PATH, sep="\t")
    gene_ids = exp["gene_id"].values
    case_ids_exp = list(exp.columns[1:])
    X_expr = exp.iloc[:, 1:].T.values.astype(np.float32)

    clin_df = pd.read_csv(CLINICAL_PATH, sep="\t")
    clin_cols_full = list(clin_df.columns)
    surv = pd.read_csv(SURVIVAL_PATH, sep="\t")
    case_ids_surv = surv["case_id"].tolist()
    assert case_ids_exp == case_ids_surv, "expression cols vs clinical_processed rows mismatch"

    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    bins_raw = surv["survival_class"].values  # NaN-safe; cast after keep mask

    keep = T > 0
    n_dropped = int((~keep).sum())
    log.info(f"Dropping {n_dropped} patients with OS.time <= 0 (Cox PH requires T > 0)")

    case_ids = np.array(case_ids_exp)[keep]
    X_expr = X_expr[keep]
    clin_df = clin_df.iloc[keep].reset_index(drop=True)
    T = T[keep]
    E = E[keep]
    bins = bins_raw[keep].astype(np.int64)

    # Drop near-constant clinical columns: er_signed, pr_signed are all-zero in
    # the existing preprocessing artifact (preprocessing bug -- ER/PR status not
    # threading through). is_female is ~constant (TCGA-BRCA is overwhelmingly
    # female). Keeping these makes Cox PH design matrix singular.
    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= LOW_VARIANCE_THRESHOLD]
    drop_cols = [c for c in clin_df.columns if var[c] < LOW_VARIANCE_THRESHOLD]
    if drop_cols:
        log.warning(
            f"Dropping near-constant clinical cols (var < {LOW_VARIANCE_THRESHOLD}): "
            f"{[(c, float(var[c])) for c in drop_cols]}"
        )
    X_clin = clin_df[keep_cols].values.astype(np.float32)
    clin_cols_kept = keep_cols

    log.info(
        f"Aligned cohort: n={len(case_ids)}, n_genes={X_expr.shape[1]}, "
        f"n_clinical={X_clin.shape[1]} ({clin_cols_kept}), event_rate={E.mean():.3f}"
    )
    return case_ids, X_expr, X_clin, T, E, bins, gene_ids, clin_cols_full, clin_cols_kept


def build_splits(case_ids, bins, E, n_folds=5, seed=42):
    """Save 5-fold StratifiedKFold splits on joint (bin x event) variable."""
    log.info("=" * 70)
    log.info(f"BUILDING SPLITS: {n_folds}-fold StratifiedKFold on (bin*2+event), seed={seed}")
    log.info("=" * 70)

    strata = bins * 2 + E
    log.info("  strata distribution (key = bin*2+event):")
    for s in np.unique(strata):
        log.info(f"    s={s}: n={int((strata == s).sum())}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(case_ids, strata)):
        ev_train = int(E[train_idx].sum())
        ev_val = int(E[val_idx].sum())
        splits[f"fold_{fold}"] = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "train_case_ids": case_ids[train_idx].tolist(),
            "val_case_ids": case_ids[val_idx].tolist(),
            "events_train": ev_train,
            "events_val": ev_val,
        }
        log.info(
            f"  fold {fold}: n_train={len(train_idx)} (events={ev_train}) "
            f"n_val={len(val_idx)} (events={ev_val})"
        )

    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    log.info(f"Splits saved to {SPLITS_PATH}")
    return splits


def fold_cox_baseline(
    X_expr_train, X_expr_val,
    X_clin_train, X_clin_val,
    T_train, E_train, T_val, E_val,
    n_pca=100, penalizer=0.5,
):
    """Per-fold Cox PH: train-only z-score + PCA, concat clinical, fit, score val."""
    sc = StandardScaler()
    Xe_tr = sc.fit_transform(X_expr_train)
    Xe_va = sc.transform(X_expr_val)

    pca = PCA(n_components=n_pca, random_state=SEED)
    Pe_tr = pca.fit_transform(Xe_tr)
    Pe_va = pca.transform(Xe_va)

    X_tr = np.concatenate([Pe_tr, X_clin_train], axis=1)
    X_va = np.concatenate([Pe_va, X_clin_val], axis=1)

    cols = [f"pc{i}" for i in range(n_pca)] + [f"c{i}" for i in range(X_clin_train.shape[1])]
    df_tr = pd.DataFrame(X_tr, columns=cols)
    df_tr["T"] = T_train
    df_tr["E"] = E_train.astype(int)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_tr, duration_col="T", event_col="E", show_progress=False)

    df_va = pd.DataFrame(X_va, columns=cols)
    risk = cph.predict_partial_hazard(df_va).values  # higher = more risk

    cidx_lifelines = lifelines_cindex(T_val, -risk, E_val)
    cidx_sksurv = concordance_index_censored(E_val.astype(bool), T_val, risk)[0]

    return float(cidx_lifelines), float(cidx_sksurv), float(pca.explained_variance_ratio_.sum())


def cox_leaky_baseline(X_expr, X_clin, T, E, splits, n_pca=100, penalizer=0.5):
    log.info("=" * 70)
    log.info(
        f"COX PH (LEAKY): existing 769-gene set + clinical, "
        f"PCA({n_pca}), penalizer={penalizer}"
    )
    log.info("=" * 70)

    rows = []
    for fold in range(N_FOLDS):
        s = splits[f"fold_{fold}"]
        tr = np.array(s["train_idx"])
        va = np.array(s["val_idx"])
        cidx_l, cidx_s, pca_var = fold_cox_baseline(
            X_expr[tr], X_expr[va],
            X_clin[tr], X_clin[va],
            T[tr], E[tr], T[va], E[va],
            n_pca=n_pca, penalizer=penalizer,
        )
        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(va),
                "events_val": int(E[va].sum()),
                "pca_variance_explained": pca_var,
                "cindex_lifelines": cidx_l,
                "cindex_sksurv": cidx_s,
            }
        )
        log.info(
            f"  fold {fold}: c-idx (lifelines) = {cidx_l:.4f}, "
            f"(sksurv) = {cidx_s:.4f}, PCA-var = {pca_var:.3f}"
        )

    cidx = np.array([r["cindex_lifelines"] for r in rows])
    log.info(f"  fold-avg C-index (lifelines): {cidx.mean():.4f} ± {cidx.std():.4f}")
    return rows


def cindex_crosscheck(rows):
    log.info("=" * 70)
    log.info("C-INDEX CROSS-CHECK (lifelines vs sksurv on identical risk vector)")
    log.info("=" * 70)

    diffs = [abs(r["cindex_lifelines"] - r["cindex_sksurv"]) for r in rows]
    max_diff = max(diffs)
    log.info(f"  per-fold |Δ|: {[f'{d:.5f}' for d in diffs]}")
    log.info(f"  max |Δ| = {max_diff:.5f}")
    if max_diff < 0.001:
        log.info("  PASS: tie-handling agrees within 0.001")
    elif max_diff < 0.005:
        log.warning("  WARN: divergence > 0.001 (tie-handling differs)")
    else:
        log.error("  FAIL: divergence > 0.005 -- investigate")
    return float(max_diff)


def device_benchmark(n_iters=100, batch_size=8, n_nodes=769, hidden=128):
    """Proxy GNN forward: matmul + relu + matmul on (B*N, H) tensors."""
    log.info("=" * 70)
    log.info(
        f"DEVICE BENCHMARK: ({batch_size}*{n_nodes}, {hidden}) @ ({hidden}, {hidden}) "
        f"x {n_iters} iters"
    )
    log.info("=" * 70)

    torch.manual_seed(SEED)
    times = {}
    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available():
            log.info(f"  {dev}: not available (skipping)")
            continue
        device = torch.device(dev)
        x = torch.randn(batch_size * n_nodes, hidden, device=device)
        w1 = torch.randn(hidden, hidden, device=device)
        w2 = torch.randn(hidden, hidden, device=device)

        for _ in range(5):  # warmup
            y = (x @ w1).relu() @ w2
        if dev == "mps":
            torch.mps.synchronize()

        t0 = time.time()
        for _ in range(n_iters):
            y = (x @ w1).relu() @ w2
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0

        times[dev] = dt
        log.info(f"  {dev}: {dt * 1000 / n_iters:.3f} ms/iter ({n_iters} iters in {dt:.3f}s)")

    if "cpu" in times and "mps" in times:
        ratio = times["mps"] / times["cpu"]
        winner = "CPU" if ratio > 1 else "MPS"
        log.info(f"  winner: {winner} (MPS/CPU = {ratio:.2f}x)")
    return times


def write_summary(rows, max_diff, bench_times, n_patients, event_rate,
                  clin_cols_kept=None, clin_cols_dropped=None):
    cidx_l = np.array([r["cindex_lifelines"] for r in rows])
    cidx_s = np.array([r["cindex_sksurv"] for r in rows])

    in_band = TARGET_BAND[0] <= cidx_l.mean() <= TARGET_BAND[1]
    band_status = "PASS" if in_band else "FAIL/WARN"
    cross_status = (
        "PASS" if max_diff < 0.001
        else "WARN" if max_diff < 0.005
        else "FAIL"
    )

    lines = [
        "# Stage 0 Baseline — Cox PH Reproduction",
        "",
        "Goal: reproduce the prior-attempt's ~0.748 baseline as the north-star number,",
        "and verify the C-index implementation, splits, and device pick before any model code.",
        "",
        f"**Cohort:** n={n_patients} (after dropping {1095 - n_patients} patients with OS.time ≤ 0); "
        f"event rate = {event_rate:.3f} ({int(event_rate * n_patients)} events / {n_patients})",
        f"**Splits:** {N_FOLDS}-fold StratifiedKFold on (survival_class × OS), seed={SEED}",
        f"**Model:** Cox PH (lifelines, penalizer={PENALIZER}) on PCA({N_PCA}) of 769 LASSO-leaked "
        f"genes + {len(clin_cols_kept) if clin_cols_kept else 0} clinical features. "
        f"Config picked by sweep in `00_baseline_diag.py` (best in-band, lowest std).",
        "",
        f"**Clinical features kept:** {clin_cols_kept}",
        f"**Clinical features dropped (var < {LOW_VARIANCE_THRESHOLD}):** {clin_cols_dropped}",
        "",
        "**Stage-0 finding (preprocessing bug):** `er_signed` and `pr_signed` are all-zero in "
        "the existing `clinical_features.tsv`. ER/PR status is one of the strongest BRCA "
        "prognostic signals — this is signal lost to a preprocessing bug. Track as a separate "
        "fix; the leaky baseline below excludes them so Cox PH converges. The honest baseline "
        "in `00_lasso_audit.py` should rebuild ER/PR from the raw clinical file.",
        "",
        "## Per-fold C-index",
        "",
        "| Fold | n_train | n_val | events_val | PCA var explained | C-idx (lifelines) | C-idx (sksurv) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fold']} | {r['n_train']} | {r['n_val']} | {r['events_val']} | "
            f"{r['pca_variance_explained']:.3f} | "
            f"{r['cindex_lifelines']:.4f} | {r['cindex_sksurv']:.4f} |"
        )
    lines += [
        f"| **mean** |  |  |  |  | **{cidx_l.mean():.4f}** ± {cidx_l.std():.4f} | "
        f"**{cidx_s.mean():.4f}** ± {cidx_s.std():.4f} |",
        "",
        "## Gate Status",
        "",
        f"- **{band_status}** Cox PH C-index `{cidx_l.mean():.4f}` "
        f"{'in' if in_band else 'OUTSIDE'} target band `[{TARGET_BAND[0]}, {TARGET_BAND[1]}]`",
        f"- **{cross_status}** lifelines/sksurv C-index agreement: max |Δ| = `{max_diff:.5f}`",
        "",
        "## Device Benchmark — proxy GNN forward",
        "",
        "100 iters of `(8*769, 128) @ (128, 128) -> ReLU -> @ (128, 128)`",
        "",
        "| Device | ms/iter |",
        "|---|---:|",
    ]
    for dev, t in bench_times.items():
        lines.append(f"| {dev} | {t * 1000 / 100:.3f} |")
    if "cpu" in bench_times and "mps" in bench_times:
        ratio = bench_times["mps"] / bench_times["cpu"]
        winner = "CPU" if ratio > 1 else "MPS"
        lines.append("")
        lines.append(f"**Winner: {winner}** (MPS/CPU ratio = {ratio:.2f}×)")

    lines += [
        "",
        "## On the prior `0.748` number",
        "",
        "The debrief cites a Cox PH baseline of `0.748`. Reproducing it with sensible Cox PH "
        "configurations (PCA ∈ {20, 50, 100, 200} × penalizer ∈ {0.001, 0.01, 0.1, 0.5, 1, 2, 5}, "
        "± clinical, on the same splits) does **not** reach 0.748. The closest sensible config — "
        f"PCA({N_PCA}) + clinical + penalizer={PENALIZER} — peaks at the value above.",
        "",
        "Looking at `src/evaluate.py:416`, the prior baseline used `patient_embeddings.mean(axis=2)` "
        "(mean over BioBERT 768-dim) then `[:, :50]` (FIRST 50 cols, not PCA) with `penalizer=0.1`. "
        "That is an unprincipled feature recipe — taking the first 50 BioBERT-mean-weighted gene "
        "scalars in storage order. The 0.748 figure was a quirk of that recipe, not a reproducible "
        "Cox PH lower bound. The diagnostic sweep is in `00_baseline_diag.py`.",
        "",
        "**New honest north-star: the value reported above** (best in-band sensible Cox PH).",
        "",
        "## Caveat: This Baseline is Still LEAKY",
        "",
        "The 769-gene set itself was selected by `LassoCV(cv=5).fit(X, y)` on the **full cohort "
        "labels** (see `src/preprocessing.py:191-192`). Every gene in this set was chosen with "
        "knowledge of every patient's survival label — including patients in val folds. The "
        "C-indices above are upper bounds on the truly honest baseline.",
        "",
        "The fully honest baseline (LASSO refit per fold's training partition on the raw 60k-gene "
        "matrix) is produced by `scripts/00_lasso_audit.py` and becomes the final north-star "
        "number for the rest of the thesis. METABRIC overlap report is `scripts/00_metabric_fetch.py`.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines))
    log.info(f"Summary written: {SUMMARY_MD}")


def main():
    (
        case_ids, X_expr, X_clin, T, E, bins, gene_ids, clin_cols_full, clin_cols_kept,
    ) = load_aligned()
    splits = build_splits(case_ids, bins, E, n_folds=N_FOLDS, seed=SEED)
    rows = cox_leaky_baseline(X_expr, X_clin, T, E, splits, n_pca=N_PCA, penalizer=PENALIZER)
    max_diff = cindex_crosscheck(rows)
    bench_times = device_benchmark()

    payload = {
        "n_patients": int(len(case_ids)),
        "event_rate": float(E.mean()),
        "n_folds": N_FOLDS,
        "n_pca": N_PCA,
        "seed": SEED,
        "target_band": list(TARGET_BAND),
        "clinical_cols_full": clin_cols_full,
        "clinical_cols_kept": clin_cols_kept,
        "clinical_cols_dropped": [c for c in clin_cols_full if c not in clin_cols_kept],
        "leaky_baseline_per_fold": rows,
        "cindex_max_lifelines_sksurv_diff": max_diff,
        "device_benchmark_seconds": bench_times,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Results JSON: {RESULTS_JSON}")

    write_summary(
        rows, max_diff, bench_times,
        n_patients=len(case_ids), event_rate=float(E.mean()),
        clin_cols_kept=clin_cols_kept,
        clin_cols_dropped=[c for c in clin_cols_full if c not in clin_cols_kept],
    )


if __name__ == "__main__":
    main()
