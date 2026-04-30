"""Figure 2: Calibration — decile-bin predicted risk vs Kaplan-Meier observed survival.

For each decile of predicted log-hazard, compute KM survival at 3y (36 mo) and
5y (60 mo). A well-calibrated model produces monotonically decreasing survival
across deciles (decile 1 = lowest predicted risk = highest observed survival).
Steeper descent = better discrimination.

Inputs: same as fig1.
Output:
  - results/figures/fig2_calibration.png / .pdf
  - results/fig2_calibration_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

KNOB_A_PATH = RESULTS / "stage_3b_sage_clinical_lasso_honest.json"
STAGE5_PATH = RESULTS / "stage_5_metabric_external.json"
OUT_PNG = FIGS / "fig2_calibration.png"
OUT_PDF = FIGS / "fig2_calibration.pdf"
OUT_STATS = RESULTS / "fig2_calibration_stats.json"

DAYS_PER_MONTH = 30.4375
N_BINS = 10  # deciles
TIME_POINTS_MONTHS = [36, 60]  # 3-year, 5-year
COLORS_TIMES = {"36": "#2c7bb6", "60": "#d7191c"}
LABELS_TIMES = {"36": "3-year survival", "60": "5-year survival"}


def pool_tcga_oof():
    d = json.loads(KNOB_A_PATH.read_text())
    log_h, T, E = [], [], []
    for fr in d["fold_results"]:
        log_h.append(np.array(fr["best_val_log_h"]))
        T.append(np.array(fr["val_T"]))
        E.append(np.array(fr["val_E"]))
    return np.concatenate(log_h), np.concatenate(T), np.concatenate(E)


def load_metabric():
    d = json.loads(STAGE5_PATH.read_text())
    ft = d["full_tcga_run"]
    return (
        np.array(ft["metabric_log_h_gnn_full"]),
        np.array(ft["metabric_T"]),
        np.array(ft["metabric_E"]),
    )


def km_at_time(T_months, E, t):
    """KM survival probability at time t (with 95% CI).

    Returns (survival, lower, upper) at the largest time index <= t.
    Uses lifelines' default Greenwood CI.
    """
    if len(T_months) == 0:
        return np.nan, np.nan, np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(T_months, E)
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    # Find largest timeline value <= t
    times = sf.index.values
    mask = times <= t
    if not mask.any():
        # No event/censor reached t -- use the first timeline value as 1.0
        return 1.0, 1.0, 1.0
    idx = times[mask].max()
    s = float(sf.loc[idx].values[0])
    lo_col = ci.columns[0]; hi_col = ci.columns[1]
    lo = float(ci.loc[idx, lo_col])
    hi = float(ci.loc[idx, hi_col])
    return s, lo, hi


def calibration_panel(ax, log_h, T_months, E, title):
    # Decile-bin by predicted risk (lower decile = lower predicted risk)
    qs = np.quantile(log_h, np.linspace(0, 1, N_BINS + 1))
    # Right-edges; use np.digitize to assign bins 0..N_BINS-1
    bin_edges = qs[1:-1]
    bins = np.digitize(log_h, bin_edges)
    # Decile labels 1..10 (1 = lowest risk)
    deciles = bins + 1

    panel_stats = {"deciles": []}
    for tp_str, color in COLORS_TIMES.items():
        tp = float(tp_str)
        x_dec, y_surv, y_lo, y_hi, n_per, ev_per = [], [], [], [], [], []
        for d in range(1, N_BINS + 1):
            mask = (deciles == d)
            n = int(mask.sum())
            ev = int(E[mask].sum())
            s, lo, hi = km_at_time(T_months[mask], E[mask], tp)
            x_dec.append(d)
            y_surv.append(s)
            y_lo.append(lo)
            y_hi.append(hi)
            n_per.append(n)
            ev_per.append(ev)
            panel_stats["deciles"].append({
                "decile": d, "tp_months": tp,
                "n": n, "events": ev, "survival": s, "ci_low": lo, "ci_high": hi,
            })
        x = np.array(x_dec)
        y = np.array(y_surv)
        lo_arr = np.array(y_lo)
        hi_arr = np.array(y_hi)
        # CI band
        ax.fill_between(x, lo_arr, hi_arr, color=color, alpha=0.15)
        ax.plot(x, y, "o-", color=color, lw=2.0, markersize=7,
                label=LABELS_TIMES[tp_str])

    # Spearman rank correlation between decile and survival at each time
    from scipy.stats import spearmanr
    for tp_str in COLORS_TIMES:
        tp = float(tp_str)
        d_pts = [p for p in panel_stats["deciles"] if p["tp_months"] == tp]
        rho, p = spearmanr([p["decile"] for p in d_pts], [p["survival"] for p in d_pts])
        panel_stats[f"spearman_rho_{tp_str}m"] = float(rho)
        panel_stats[f"spearman_p_{tp_str}m"] = float(p)

    rho_36 = panel_stats["spearman_rho_36m"]
    rho_60 = panel_stats["spearman_rho_60m"]
    ax.text(0.97, 0.97, f"Spearman ρ (3y) = {rho_36:.3f}\nSpearman ρ (5y) = {rho_60:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.6", boxstyle="round,pad=0.3"))

    ax.set_xlim(0.5, N_BINS + 0.5)
    ax.set_xticks(range(1, N_BINS + 1))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Predicted risk decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Observed Kaplan-Meier survival")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return panel_stats


def main():
    tcga_log_h, tcga_T_days, tcga_E = pool_tcga_oof()
    met_log_h, met_T_days, met_E = load_metabric()
    tcga_T = tcga_T_days / DAYS_PER_MONTH
    met_T = met_T_days / DAYS_PER_MONTH
    print(f"TCGA OOF: n={len(tcga_T)}, events={int(tcga_E.sum())}")
    print(f"METABRIC: n={len(met_T)}, events={int(met_E.sum())}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    stats = {}
    stats["tcga"] = calibration_panel(
        axes[0], tcga_log_h, tcga_T, tcga_E,
        title=f"TCGA-BRCA — pooled out-of-fold (n={len(tcga_T)})",
    )
    stats["metabric"] = calibration_panel(
        axes[1], met_log_h, met_T, met_E,
        title=f"METABRIC — external (n={len(met_T)})",
    )

    fig.suptitle(
        "Calibration: predicted risk decile vs observed Kaplan-Meier survival",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"Saved {OUT_STATS}")
    print()
    print("Headline calibration metrics (Spearman ρ between decile and observed survival):")
    print(f"  TCGA  3y rho = {stats['tcga']['spearman_rho_36m']:+.3f}  (p = {stats['tcga']['spearman_p_36m']:.2e})")
    print(f"  TCGA  5y rho = {stats['tcga']['spearman_rho_60m']:+.3f}  (p = {stats['tcga']['spearman_p_60m']:.2e})")
    print(f"  METABRIC 3y rho = {stats['metabric']['spearman_rho_36m']:+.3f}  (p = {stats['metabric']['spearman_p_36m']:.2e})")
    print(f"  METABRIC 5y rho = {stats['metabric']['spearman_rho_60m']:+.3f}  (p = {stats['metabric']['spearman_p_60m']:.2e})")


if __name__ == "__main__":
    main()
