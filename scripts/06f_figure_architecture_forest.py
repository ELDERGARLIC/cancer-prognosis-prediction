"""Figure 4: Architecture-ablation forest plot.

Three panels showing knob A / B / C / Cox PH on:
  Panel 1 — TCGA 10%-val C-index (internal anchor; no CI; small markers)
  Panel 2 — METABRIC C-index (external; bootstrap 95% CI as horizontal bars)
  Panel 3 — Paired bootstrap Δ vs Knob A on identical METABRIC patients
            (95% CI as horizontal bars; vertical dashed line at zero)

The point of the figure is the third panel: knob B and knob C are
significantly *below zero* on identical METABRIC patients vs knob A. This
is the architectural-minimalism story in one image.

Inputs:
  - results/stage_5_metabric_external.json   (knob A full-TCGA → METABRIC)
  - results/stage_5b_knob_b_metabric.json    (knob B full-TCGA → METABRIC)
  - results/stage_5c_knob_c_biobert_metabric.json  (knob C full-TCGA → METABRIC)

Outputs:
  - results/figures/fig4_architecture_forest.png / .pdf
  - results/fig4_architecture_forest_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

S5A = RESULTS / "stage_5_metabric_external.json"
S5B = RESULTS / "stage_5b_knob_b_metabric.json"
S5C = RESULTS / "stage_5c_knob_c_biobert_metabric.json"
OUT_PNG = FIGS / "fig4_architecture_forest.png"
OUT_PDF = FIGS / "fig4_architecture_forest.pdf"
OUT_STATS = RESULTS / "fig4_architecture_forest_stats.json"


def main():
    s5a = json.loads(S5A.read_text())
    s5b = json.loads(S5B.read_text())
    s5c = json.loads(S5C.read_text())

    full_a = s5a["full_tcga_run"]

    rows = [
        # (label, color, tcga_val, met_cidx, met_ci_low, met_ci_high,
        #  delta_vs_a, delta_ci_low, delta_ci_high, p_le_a)
        ("Knob A (scalar expression GNN)", "#1b9e77",
         full_a["tcga_10pct_val_gnn_cidx"],
         full_a["metabric_gnn_cidx"],
         full_a["metabric_gnn_boot"]["ci_low"],
         full_a["metabric_gnn_boot"]["ci_high"],
         0.0, 0.0, 0.0, None),
        ("Knob B (pathway pool)", "#7570b3",
         s5b["tcga_10pct_val_cidx"],
         s5b["metabric_cidx"],
         s5b["metabric_bootstrap"]["ci_low"],
         s5b["metabric_bootstrap"]["ci_high"],
         s5b["paired_metabric_b_vs_a"]["delta_point"],
         s5b["paired_metabric_b_vs_a"]["delta_ci_low"],
         s5b["paired_metabric_b_vs_a"]["delta_ci_high"],
         s5b["paired_metabric_b_vs_a"]["p_a_le_b"]),
        ("Knob C (BioBERT-PCA gene init)", "#d95f02",
         s5c["tcga_10pct_val_cidx"],
         s5c["metabric_cidx"],
         s5c["metabric_bootstrap"]["ci_low"],
         s5c["metabric_bootstrap"]["ci_high"],
         s5c["paired_metabric_c_vs_a"]["delta_point"],
         s5c["paired_metabric_c_vs_a"]["delta_ci_low"],
         s5c["paired_metabric_c_vs_a"]["delta_ci_high"],
         s5c["paired_metabric_c_vs_a"]["p_a_le_b"]),
        ("Cox PH (linear; matched 3-clin)", "#666666",
         full_a["tcga_10pct_val_cox_cidx"],
         full_a["metabric_cox_cidx"],
         full_a["metabric_cox_boot"]["ci_low"],
         full_a["metabric_cox_boot"]["ci_high"],
         -full_a["metabric_paired_gnn_minus_cox"]["delta_point"],   # negate for "vs A"
         -full_a["metabric_paired_gnn_minus_cox"]["delta_ci_high"],
         -full_a["metabric_paired_gnn_minus_cox"]["delta_ci_low"],
         1.0 - full_a["metabric_paired_gnn_minus_cox"]["p_a_le_b"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                              gridspec_kw={"width_ratios": [1.4, 1.8, 1.8]})
    n = len(rows)
    y_pos = np.arange(n)[::-1]  # top row = knob A

    # Panel 1: TCGA internal
    ax = axes[0]
    for i, (label, color, tv, _, _, _, _, _, _, _) in enumerate(rows):
        yp = y_pos[i]
        ax.plot([tv], [yp], "o", color=color, markersize=10)
        ax.text(tv + 0.005, yp, f"{tv:.4f}", va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0.65, 0.80)
    ax.set_xlabel("C-index")
    ax.set_title("TCGA internal\n(10%-val held-out for best-epoch)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Panel 2: METABRIC C-index with 95% CI
    ax = axes[1]
    for i, (_, color, _, mc, ci_lo, ci_hi, _, _, _, _) in enumerate(rows):
        yp = y_pos[i]
        ax.errorbar(
            x=[mc], y=[yp], xerr=[[mc - ci_lo], [ci_hi - mc]],
            fmt="o", color=color, markersize=10, capsize=4, lw=2,
        )
        ax.text(ci_hi + 0.003, yp, f"{mc:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]",
                va="center", fontsize=8.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(0.55, 0.74)
    ax.set_xlabel("C-index (with 95% bootstrap CI)")
    ax.set_title("METABRIC external\n(n=1466, 824 events)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Panel 3: Paired Δ vs Knob A (METABRIC, identical patients)
    ax = axes[2]
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    for i, (label, color, _, _, _, _, dp, dlo, dhi, p_le_a) in enumerate(rows):
        yp = y_pos[i]
        if i == 0:  # Knob A reference; just dot at zero
            ax.plot([0], [yp], "o", color=color, markersize=10)
            ax.text(0.005, yp, "reference", va="center", fontsize=9)
        else:
            ax.errorbar(
                x=[dp], y=[yp], xerr=[[dp - dlo], [dhi - dp]],
                fmt="o", color=color, markersize=10, capsize=4, lw=2,
            )
            sig = ""
            if p_le_a is not None:
                if p_le_a >= 0.999:
                    sig = "  ★★★ (P=1.000)"
                elif p_le_a >= 0.95:
                    sig = f"  ★★ (P={p_le_a:.3f})"
                elif p_le_a >= 0.90:
                    sig = f"  ★ (P={p_le_a:.3f})"
                else:
                    sig = f"  (P={p_le_a:.3f})"
            ax.text(dhi + 0.003, yp,
                    f"Δ={dp:+.4f} [{dlo:+.3f}, {dhi:+.3f}]{sig}",
                    va="center", fontsize=8.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(-0.10, 0.04)
    ax.set_xlabel("Δ vs Knob A (paired bootstrap, identical patients)")
    ax.set_title("Paired Δ vs Knob A on METABRIC\n(★★★ = P(model ≤ A) ≥ 0.999, CI strictly below zero)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Architecture-ablation forest: simpler beats more elaborate at external validation",
        fontsize=13, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    stats = {
        "rows": [
            {
                "label": r[0],
                "tcga_internal_cidx": r[2],
                "metabric_cidx": r[3],
                "metabric_ci_low": r[4],
                "metabric_ci_high": r[5],
                "delta_vs_knob_a": r[6],
                "delta_ci_low": r[7],
                "delta_ci_high": r[8],
                "p_le_a": r[9],
            }
            for r in rows
        ],
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
