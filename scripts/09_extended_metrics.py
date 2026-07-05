"""Stage 9 — Extended external-validation metrics (no retraining).

Addresses reviewer requests for time-dependent discrimination and calibration
metrics beyond Harrell's C-index, computed from the *saved* per-patient external
predictions in ``stage_5_metabric_external.json`` (full_tcga_run block):

    metabric_log_h_gnn_full : GNN linear predictor eta (higher = worse)
    metabric_risk_cox_full  : Cox relative risk exp(eta) (higher = worse)
    metabric_T, metabric_E  : METABRIC follow-up time (days) and event indicator

Metrics reported for both the base GNN (M0) and the matched leakage-corrected
Cox baseline, with patient-level paired bootstrap on identical METABRIC patients
(same protocol as src/cindex_bootstrap.py: seed=42, n_boot=1000, one-sided
p_a_le_b = fraction of bootstrap deltas <= 0):

  1. Harrell's C            (reproduces the headline number; sanity check)
  2. Uno's C (IPCW)         censoring-distribution-adjusted concordance at tau
  3. Antolini's td-C        time-dependent concordance using S(t_i | x)
  4. Time-dependent AUC     cumulative/dynamic AUC at clinical horizons + mean
  5. Integrated Brier Score calibration; lower is better

Notes / honest caveats (surfaced in the manuscript):
  * Only risk scores were exported, not full survival curves. For Antolini's C
    and the IBS we reconstruct S(t | x) with a Breslow baseline cumulative
    hazard estimated on the *external* cohort given the fixed linear predictors.
    Discrimination metrics (Harrell, Uno, tAUC) do not use the baseline; IBS
    therefore reflects the externally *recalibrated* model and is reported as a
    complement to discrimination, not a deployment calibration claim.
  * The IPCW censoring distribution for Uno's C, tAUC and IBS is estimated on
    METABRIC (passed as both survival_train and survival_test) because METABRIC
    is the only cohort with exported (T, E); standard when only the evaluation
    cohort is available.
  * Because both models are proportional-hazards by construction, the ordering
    of S(t | x) is time-invariant, so Antolini's td-C coincides with Harrell's C
    up to tie handling -- reported to confirm the ranking is not a PH artifact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

DAYS_PER_YEAR = 365.25
SEED = 42
N_BOOT = 1000
ALPHA = 0.05

REPO = Path("/home/eldergarlic/network_share/Projects/thesis-pipelines/thesis-research-v1")
RESULTS = REPO / "output/thesis-results-repo/results"
IN_JSON = RESULTS / "stage_5_metabric_external.json"
OUT_JSON = RESULTS / "stage_9_extended_metrics.json"
OUT_TEX = REPO / "output/thesis-writing-repo/manuscript/table_extended_metrics.tex"


# ----------------------------------------------------------------------------- helpers
def surv(T, E):
    return Surv.from_arrays(event=E.astype(bool), time=T)


def breslow_survival(eta, T, E, times):
    """S(t | x) at `times` via Breslow baseline cumulative hazard.

    H0(t) = sum_{event times t_k <= t} d_k / sum_{j in risk set at t_k} exp(eta_j)
    S(t | x_i) = exp(-H0(t) * exp(eta_i)).  Returns (n_samples, n_times).
    """
    order = np.argsort(T, kind="mergesort")
    Ts, Es, etas = T[order], E[order], eta[order]
    exp_eta = np.exp(etas)
    # risk set sum at each sorted position = sum of exp_eta for time >= t (suffix sum)
    risk_suffix = np.cumsum(exp_eta[::-1])[::-1]
    # baseline hazard increments at distinct event times
    h0_times, h0_incr = [], []
    i, n = 0, len(Ts)
    while i < n:
        j = i
        while j < n and Ts[j] == Ts[i]:
            j += 1
        d = int(Es[i:j].sum())
        if d > 0:
            denom = risk_suffix[i]
            if denom > 0:
                h0_times.append(Ts[i])
                h0_incr.append(d / denom)
        i = j
    h0_times = np.asarray(h0_times)
    H0_cum = np.cumsum(np.asarray(h0_incr))
    # H0 at each requested time = cumulative increment up to that time (step function)
    idx = np.searchsorted(h0_times, times, side="right") - 1
    H0_at = np.where(idx >= 0, H0_cum[np.clip(idx, 0, len(H0_cum) - 1)], 0.0)
    # S(t|x) = exp(-H0(t) * exp(eta))
    return np.exp(-np.outer(np.exp(eta), H0_at))  # (n_samples, n_times)


def antolini_c(eta, T, E, times):
    """Antolini's time-dependent concordance using S(t_i | x).

    Concordant pair (i events at t_i, j with T_j > t_i): S(t_i|x_i) < S(t_i|x_j).
    Vectorised over event subjects; O(n_events * n).
    """
    Smat = breslow_survival(eta, T, E, times)  # at grid; need S at each event time
    # interpolate S at each event time t_i for all subjects: use nearest grid <= t_i
    ev = np.where(E == 1)[0]
    num = den = 0.0
    for i in ev:
        ti = T[i]
        comparable = T > ti  # j outlives i (still at risk after t_i)
        if not comparable.any():
            continue
        gi = max(np.searchsorted(times, ti, side="right") - 1, 0)
        S_at_ti = breslow_survival(eta, T, E, np.array([ti]))[:, 0]
        si = S_at_ti[i]
        sj = S_at_ti[comparable]
        den += comparable.sum()
        num += (si < sj).sum() + 0.5 * (si == sj).sum()
    return float(num / den) if den > 0 else float("nan")


def harrell(T, E, score):
    return float(concordance_index_censored(E.astype(bool), T, score)[0])


def uno(T, E, score, tau):
    y = surv(T, E)
    return float(concordance_index_ipcw(y, y, score, tau=tau)[0])


def tauc_mean(T, E, score, times):
    y = surv(T, E)
    _, mean_auc = cumulative_dynamic_auc(y, y, score, times)
    return float(mean_auc)


def ibs(eta, T, E, times):
    y = surv(T, E)
    Smat = breslow_survival(eta, T, E, times)
    return float(integrated_brier_score(y, y, Smat, times))


def paired_bootstrap(fn_a, fn_b, T, E, n_boot=N_BOOT, seed=SEED, lower_better=False):
    """Generic paired bootstrap on identical resampled patients.

    fn_a / fn_b are callables taking a boolean/int index array -> scalar metric.
    Returns dict with points, delta and one-sided p_a_le_b (fraction delta<=0).
    For lower-is-better metrics (IBS) the 'improvement' delta is b - a.
    """
    rng = np.random.default_rng(seed)
    n = len(T)
    pa, pb = fn_a(np.arange(n)), fn_b(np.arange(n))
    delta_point = (pb - pa) if lower_better else (pa - pb)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if E[idx].sum() < 5:
            continue
        try:
            a, b = fn_a(idx), fn_b(idx)
            if np.isnan(a) or np.isnan(b):
                continue
            deltas.append((b - a) if lower_better else (a - b))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
    deltas = np.asarray(deltas)
    return {
        "metric_gnn": pa,
        "metric_cox": pb,
        "delta_point": float(delta_point),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "delta_ci_low": float(np.quantile(deltas, ALPHA / 2)),
        "delta_ci_high": float(np.quantile(deltas, 1 - ALPHA / 2)),
        "p_delta_le_0": float((deltas <= 0).mean()),  # one-sided: GNN not better
        "n_boot": n_boot,
        "n_valid": int(len(deltas)),
    }


# ----------------------------------------------------------------------------- main
def main():
    d = json.loads(IN_JSON.read_text())
    ft = d["full_tcga_run"]
    T = np.asarray(ft["metabric_T"], float)
    E = np.asarray(ft["metabric_E"], int)
    eta_gnn = np.asarray(ft["metabric_log_h_gnn_full"], float)
    risk_cox = np.asarray(ft["metabric_risk_cox_full"], float)
    eta_cox = np.log(risk_cox)  # linear predictor for Breslow baseline

    # discrimination scores (higher = worse); monotonic transform irrelevant for ranking
    score_gnn, score_cox = eta_gnn, risk_cox

    # time grids (within follow-up; days)
    horizons_yr = [3, 5, 8, 10, 12, 15]
    horizons = np.array([h * DAYS_PER_YEAR for h in horizons_yr])
    horizons = horizons[horizons < T.max() * 0.98]
    ibs_grid = np.linspace(1 * DAYS_PER_YEAR, 15 * DAYS_PER_YEAR, 30)
    ibs_grid = ibs_grid[ibs_grid < T.max() * 0.98]
    tau = 10 * DAYS_PER_YEAR  # Uno's C truncation horizon

    print(f"n={len(T)} events={int(E.sum())} ({E.mean():.1%}) "
          f"max_fu={T.max()/DAYS_PER_YEAR:.1f}y")

    # sanity: reproduce headline Harrell C
    cg, cc = harrell(T, E, score_gnn), harrell(T, E, score_cox)
    assert abs(cg - ft["metabric_gnn_cidx"]) < 1e-6, (cg, ft["metabric_gnn_cidx"])
    assert abs(cc - ft["metabric_cox_cidx"]) < 1e-6, (cc, ft["metabric_cox_cidx"])
    print(f"[ok] reproduced Harrell C  GNN={cg:.4f} Cox={cc:.4f}")

    t0 = time.time()
    out = {
        "n": int(len(T)), "n_events": int(E.sum()), "event_rate": float(E.mean()),
        "max_followup_years": float(T.max() / DAYS_PER_YEAR),
        "seed": SEED, "n_boot": N_BOOT,
        "horizons_years": [float(h) for h in horizons / DAYS_PER_YEAR],
        "uno_tau_years": float(tau / DAYS_PER_YEAR),
        "ibs_grid_years": [float(round(x / DAYS_PER_YEAR, 2)) for x in ibs_grid],
    }

    # 1. Harrell C (paired bootstrap). The Harrell delta and its interval are the
    # paper's headline external comparison; to avoid reporting the same quantity
    # two ways, adopt the authoritative interval from the Stage-5 headline run
    # (n_boot=2000) rather than an independent re-estimate.
    out["harrell"] = paired_bootstrap(
        lambda ix: harrell(T[ix], E[ix], score_gnn[ix]),
        lambda ix: harrell(T[ix], E[ix], score_cox[ix]), T, E)
    _hd = ft["metabric_paired_gnn_minus_cox"]
    out["harrell"]["delta_ci_low"] = float(_hd["delta_ci_low"])
    out["harrell"]["delta_ci_high"] = float(_hd["delta_ci_high"])
    out["harrell"]["p_delta_le_0"] = float(_hd["p_a_le_b"])
    out["harrell"]["ci_source"] = "stage_5_headline_n_boot_2000"
    print(f"[1] Harrell  GNN={out['harrell']['metric_gnn']:.4f} "
          f"Cox={out['harrell']['metric_cox']:.4f} "
          f"d={out['harrell']['delta_point']:+.4f} "
          f"[{out['harrell']['delta_ci_low']:+.4f},{out['harrell']['delta_ci_high']:+.4f}]")

    # 2. Uno's C (IPCW) at tau
    out["uno"] = paired_bootstrap(
        lambda ix: uno(T[ix], E[ix], score_gnn[ix], tau),
        lambda ix: uno(T[ix], E[ix], score_cox[ix], tau), T, E)
    print(f"[2] Uno(IPCW) GNN={out['uno']['metric_gnn']:.4f} "
          f"Cox={out['uno']['metric_cox']:.4f} "
          f"d={out['uno']['delta_point']:+.4f} "
          f"[{out['uno']['delta_ci_low']:+.4f},{out['uno']['delta_ci_high']:+.4f}]")

    # 3. Time-dependent AUC (mean over horizons)
    out["tauc_mean"] = paired_bootstrap(
        lambda ix: tauc_mean(T[ix], E[ix], score_gnn[ix], horizons),
        lambda ix: tauc_mean(T[ix], E[ix], score_cox[ix], horizons), T, E)
    print(f"[3] mean tAUC GNN={out['tauc_mean']['metric_gnn']:.4f} "
          f"Cox={out['tauc_mean']['metric_cox']:.4f} "
          f"d={out['tauc_mean']['delta_point']:+.4f} "
          f"[{out['tauc_mean']['delta_ci_low']:+.4f},{out['tauc_mean']['delta_ci_high']:+.4f}]")

    # per-horizon tAUC point estimates (no bootstrap) for a curve/table
    y = surv(T, E)
    auc_gnn, _ = cumulative_dynamic_auc(y, y, score_gnn, horizons)
    auc_cox, _ = cumulative_dynamic_auc(y, y, score_cox, horizons)
    out["tauc_curve"] = {
        "years": out["horizons_years"],
        "gnn": [float(x) for x in auc_gnn],
        "cox": [float(x) for x in auc_cox],
    }

    # 4. Integrated Brier Score (lower better; improvement delta = cox - gnn)
    out["ibs"] = paired_bootstrap(
        lambda ix: ibs(eta_gnn[ix], T[ix], E[ix], ibs_grid),
        lambda ix: ibs(eta_cox[ix], T[ix], E[ix], ibs_grid), T, E, lower_better=True)
    print(f"[4] IBS      GNN={out['ibs']['metric_gnn']:.4f} "
          f"Cox={out['ibs']['metric_cox']:.4f} "
          f"improvement(cox-gnn)={out['ibs']['delta_point']:+.4f} "
          f"[{out['ibs']['delta_ci_low']:+.4f},{out['ibs']['delta_ci_high']:+.4f}]")

    # 5. Antolini's td-C (point estimates only; PH coincidence note)
    ant_gnn = antolini_c(eta_gnn, T, E, ibs_grid)
    ant_cox = antolini_c(eta_cox, T, E, ibs_grid)
    out["antolini"] = {"metric_gnn": ant_gnn, "metric_cox": ant_cox,
                       "delta_point": float(ant_gnn - ant_cox)}
    print(f"[5] Antolini GNN={ant_gnn:.4f} Cox={ant_cox:.4f} d={ant_gnn-ant_cox:+.4f}")

    out["elapsed_seconds"] = time.time() - t0
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}  ({out['elapsed_seconds']:.0f}s)")

    write_latex(out)
    print(f"wrote {OUT_TEX}")


def write_latex(out):
    def row(name, blk, fmt="{:.3f}", pkey="p_delta_le_0", lower=False):
        g, c = fmt.format(blk["metric_gnn"]), fmt.format(blk["metric_cox"])
        d = blk["delta_point"]
        lo, hi = blk["delta_ci_low"], blk["delta_ci_high"]
        ci = "--" if (np.isnan(lo) or np.isnan(hi)) else f"[{lo:+.3f}, {hi:+.3f}]"
        p = blk.get(pkey)
        psig = "" if p is None else (r"\textbf{%.3f}" % p if p < 0.05 else f"{p:.3f}")
        return f"{name} & {g} & {c} & {d:+.3f} {ci} & {psig} \\\\"

    lines = [
        r"% Auto-generated by scripts/09_extended_metrics.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Extended external-validation metrics on identical METABRIC "
        rf"patients ($n={out['n']}$, {out['n_events']} events). "
        r"Base GNN (M0) vs.\ matched leakage-corrected Cox. "
        r"$\Delta$ and 95\% CI from patient-level paired bootstrap "
        rf"($B={out['n_boot']}$, seed~{out['seed']}; Harrell's interval is the "
        r"headline $B=2000$ estimate of Section~\ref{sec:res-external}); "
        r"$p$ is the one-sided bootstrap probability that the GNN is \emph{not} "
        r"better (fraction of resamples with $\Delta\le 0$).}",
        r"\label{tab:extended_metrics}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & GNN (M0) & Cox & $\Delta$ (GNN$-$Cox) & $p$ \\",
        r"\midrule",
        row("Harrell's C", out["harrell"]),
        row("Uno's C (IPCW, 10\\,y)", out["uno"]),
        row("Antolini's td-C$^{\\dagger}$",
            {**out["antolini"], "delta_ci_low": float("nan"),
             "delta_ci_high": float("nan"), "p_delta_le_0": None}),
        row("Mean td-AUC (3--15\\,y)", out["tauc_mean"]),
        r"\midrule",
        row("Integrated Brier$^{\\ddagger}$", out["ibs"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}\footnotesize",
        r"$^{\dagger}$Point estimates; under the proportional-hazards "
        r"construction Antolini's td-C coincides with Harrell's C. "
        r"$^{\ddagger}$Lower is better; $\Delta$ and $p$ are for the improvement "
        r"(Cox$-$GNN), so $p<0.05$ favours the GNN.",
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
