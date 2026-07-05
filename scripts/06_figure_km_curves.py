"""Figure 1: Risk-stratified Kaplan-Meier curves on TCGA (pooled OOF) and METABRIC.

Standardised (2026-07-05) to source the pooled out-of-fold TCGA predictions
from the stage-5 M0 harness (fold_results[*].tcga_val_log_h_gnn), the same
source used by regenerate_figures.py and by fig2's 06b_figure_calibration.py
(standardised 2026-07-03). Previously this read
stage_3b_sage_clinical_lasso_honest.json (best_val_log_h), a different,
superseded M0 run -- that gave TCGA log-rank chi^2 = 46.6, which did not match
the paper's chi^2 = 39.7. Both figures now use one consistent OOF prediction
vector.

Inputs:
  - results/stage_5_metabric_external.json  (TCGA per-fold val predictions,
    fold_results[*].tcga_val_log_h_gnn/tcga_val_T/tcga_val_E; and the
    full-TCGA-trained METABRIC predictions, full_tcga_run.*)

Output:
  - results/figures/fig1_km_curves.png  (and .pdf)
  - results/fig1_km_stats.json          (log-rank p, quartile cidx, n per quartile)

Style requirements (from brief):
  - Quartile split (4 risk groups, not tertile, not median)
  - Months on x-axis (convert days -> months: /30.44)
  - METABRIC truncated at 240 months (long follow-up)
  - 95% CI bands per curve
  - Multivariate log-rank p-value inside each panel
  - Sample size per quartile in the legend
  - Same axes/style for TCGA and METABRIC
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(exist_ok=True, parents=True)

STAGE5_PATH = RESULTS / "stage_5_metabric_external.json"

OUT_PNG = FIGS / "fig1_km_curves.png"
OUT_PDF = FIGS / "fig1_km_curves.pdf"
OUT_STATS = RESULTS / "fig1_km_stats.json"

DAYS_PER_MONTH = 30.4375  # 365.25/12

# Plot settings
TCGA_X_MAX_MONTHS = 240   # consistent with METABRIC for paired figure
METABRIC_X_MAX_MONTHS = 240
QUARTILES = 4
COLORS = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]  # blue->red, low->high risk
LABELS = ["Q1 (lowest risk)", "Q2", "Q3", "Q4 (highest risk)"]


def pool_tcga_oof():
    """Concat per-fold val predictions into one array per patient (1074 total)."""
    d = json.loads(STAGE5_PATH.read_text())
    log_h, T, E, fold_id = [], [], [], []
    for fr in d["fold_results"]:
        log_h.append(np.array(fr["tcga_val_log_h_gnn"]))
        T.append(np.array(fr["tcga_val_T"]))
        E.append(np.array(fr["tcga_val_E"]))
        fold_id.append(np.full(len(fr["tcga_val_T"]), fr["fold"], dtype=int))
    log_h = np.concatenate(log_h)
    T = np.concatenate(T)
    E = np.concatenate(E)
    fold_id = np.concatenate(fold_id)
    print(f"TCGA pooled OOF: n={len(log_h)}, events={int(E.sum())}, "
          f"folds covered: {sorted(set(fold_id.tolist()))}")
    return log_h, T, E


def load_metabric():
    d = json.loads(STAGE5_PATH.read_text())
    ft = d["full_tcga_run"]
    log_h = np.array(ft["metabric_log_h_gnn_full"])
    T = np.array(ft["metabric_T"])
    E = np.array(ft["metabric_E"])
    print(f"METABRIC: n={len(log_h)}, events={int(E.sum())}")
    return log_h, T, E


def quartile_groups(risk):
    """Quartile labels 0..3 (0 = lowest risk, 3 = highest)."""
    q = np.quantile(risk, [0.25, 0.5, 0.75])
    g = np.digitize(risk, q)
    return g.astype(int)


def plot_km_panel(ax, T_months, E, groups, title, x_max):
    """Plot 4 KM curves with 95% CI bands; multivariate log-rank in panel."""
    kms = []
    n_per = []
    events_per = []
    for q in range(QUARTILES):
        mask = (groups == q)
        n_per.append(int(mask.sum()))
        events_per.append(int(E[mask].sum()))
        kmf = KaplanMeierFitter()
        kmf.fit(T_months[mask], E[mask], label=f"{LABELS[q]} (n={int(mask.sum())}, ev={int(E[mask].sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS[q], linewidth=2.0, ci_alpha=0.15)
        kms.append(kmf)

    # Multivariate log-rank across all 4 groups
    mlr = multivariate_logrank_test(T_months, groups, E)
    p = mlr.p_value
    if p < 1e-4:
        p_str = "p < 1e-4"
    elif p < 1e-3:
        p_str = f"p = {p:.1e}"
    else:
        p_str = f"p = {p:.4f}"

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Months from diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.97, 0.05, f"Log-rank (4 groups): {p_str}\nχ² = {mlr.test_statistic:.2f}, df = {QUARTILES-1}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.6", boxstyle="round,pad=0.3"))
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    return {
        "logrank_p": float(p),
        "logrank_chi2": float(mlr.test_statistic),
        "df": QUARTILES - 1,
        "n_per_quartile": n_per,
        "events_per_quartile": events_per,
    }


def main():
    # Load
    tcga_log_h, tcga_T_days, tcga_E = pool_tcga_oof()
    met_log_h, met_T_days, met_E = load_metabric()

    # Convert to months
    tcga_T = tcga_T_days / DAYS_PER_MONTH
    met_T = met_T_days / DAYS_PER_MONTH
    print(f"TCGA T_months: max={tcga_T.max():.1f}, median={np.median(tcga_T):.1f}")
    print(f"METABRIC T_months: max={met_T.max():.1f}, median={np.median(met_T):.1f}")

    # Quartile split per cohort (NOT joint -- cohort-specific)
    tcga_q = quartile_groups(tcga_log_h)
    met_q = quartile_groups(met_log_h)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    stats = {}

    stats["tcga"] = plot_km_panel(
        axes[0], tcga_T, tcga_E, tcga_q,
        title=f"TCGA-BRCA — pooled out-of-fold (n={len(tcga_T)}, events={int(tcga_E.sum())})",
        x_max=TCGA_X_MAX_MONTHS,
    )
    stats["metabric"] = plot_km_panel(
        axes[1], met_T, met_E, met_q,
        title=f"METABRIC — external validation (n={len(met_T)}, events={int(met_E.sum())})",
        x_max=METABRIC_X_MAX_MONTHS,
    )

    # Suptitle
    fig.suptitle(
        "Risk-stratified survival by knob A predicted log-hazard quartile",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()

    # Save
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    # Stats JSON
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"Saved {OUT_STATS}")
    print()
    print("Stats:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
