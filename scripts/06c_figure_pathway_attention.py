"""Figure 3: Pathway-attention heatmap from knob B.

Layout:
  - 10 rows: top-10 Reactome pathways by total attention weight across all folds
    and patients (high-risk and low-risk pooled).
  - 10 columns: 5 high-risk patient slots + 5 low-risk patient slots. Slot k =
    "k-th highest/lowest risk val patient" within each fold; values shown are
    fold-averaged across the 5 folds.
  - Color = mean attention weight at that (pathway, slot) cell.

Reads: results/stage_3c_attention_per_fold.json (knob B saved attention).
Writes:
  - results/figures/fig3_pathway_attention.png / .pdf
  - results/fig3_pathway_attention_stats.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

INPUT = RESULTS / "stage_3c_attention_per_fold.json"
OUT_PNG = FIGS / "fig3_pathway_attention.png"
OUT_PDF = FIGS / "fig3_pathway_attention.pdf"
OUT_STATS = RESULTS / "fig3_pathway_attention_stats.json"

N_PATHWAYS_TO_SHOW = 10
N_HIGH_RISK_SLOTS = 5
N_LOW_RISK_SLOTS = 5


def shorten_pathway_label(name: str, max_len: int = 38) -> str:
    """Make Reactome names readable in a figure axis."""
    s = name.replace("REACTOME_", "").replace("_", " ")
    # Common compressions
    repl = [
        ("SIGNALING BY ", ""),
        ("FAMILY ", ""),
        ("INTERLEUKIN ", "IL-"),
        ("RECEPTOR ", "RX "),
        ("INTRACELLULAR ", "INTRACEL "),
        ("REGULATION OF ", "REG "),
        ("ESTROGEN DEPENDENT GENE EXPRESSION", "ESTROGEN-DEP GENE EXPR"),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    # Capitalize first letter of each word, keep all-caps short ones
    s = " ".join(w if (len(w) <= 4 and w.isupper()) else w.capitalize() for w in s.split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def main():
    data = json.loads(INPUT.read_text())
    folds = data["fold_results"]
    n_folds = len(folds)
    print(f"Folds: {n_folds}")

    # Step 1: aggregate (pathway, patient_class, fold, slot) -> attention
    # "slot" = rank within {high_risk, low_risk} list (0..4)
    # Patients within high_risk are sorted descending by log_hazard.
    # We collect (pathway_name, attention_weight) for each (fold, class, slot)
    pathway_total = {}
    cells = {}  # (class, slot) -> list of {pathway: weight} per fold
    for fr in folds:
        fold = fr["fold"]
        interp = fr["interpretability"]
        for cls_key, cls_label in (("high_risk", "high"), ("low_risk", "low")):
            for slot_idx, p in enumerate(interp[cls_key]):
                key = (cls_label, slot_idx)
                cells.setdefault(key, []).append({
                    "fold": fold,
                    "pathways": {tp["name"]: float(tp["weight"]) for tp in p["top5_pathways"]},
                })
                for tp in p["top5_pathways"]:
                    pathway_total[tp["name"]] = pathway_total.get(tp["name"], 0.0) + tp["weight"]

    # Step 2: top-N pathways by total attention
    top_pathways = sorted(pathway_total.items(), key=lambda kv: -kv[1])[:N_PATHWAYS_TO_SHOW]
    pathway_names = [p[0] for p in top_pathways]
    print("Top pathways by total attention:")
    for n, w in top_pathways:
        print(f"  {w:.3f}  {n}")

    # Step 3: build matrix [pathway, slot] of mean attention weight (fold-averaged)
    # Slots ordered: high_risk slot 0..4 then low_risk slot 0..4
    slot_keys = [("high", i) for i in range(N_HIGH_RISK_SLOTS)] \
              + [("low", i) for i in range(N_LOW_RISK_SLOTS)]
    matrix = np.zeros((N_PATHWAYS_TO_SHOW, len(slot_keys)))
    for j, key in enumerate(slot_keys):
        fold_entries = cells.get(key, [])
        for i, pname in enumerate(pathway_names):
            weights = []
            for fe in fold_entries:
                # If this slot's patient in this fold attended to this pathway among top-5, use its weight; else 0.
                w = fe["pathways"].get(pname, 0.0)
                weights.append(w)
            matrix[i, j] = np.mean(weights) if weights else 0.0

    # Step 4: plot
    fig, ax = plt.subplots(figsize=(11, 5.6))

    # Use a sequential cmap. Mean attention will typically be 0.0 to ~0.13.
    cmap = "YlOrRd"
    vmin = 0.0
    vmax = max(matrix.max(), 0.05)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Y-tick labels: shortened pathway names
    ax.set_yticks(range(N_PATHWAYS_TO_SHOW))
    ax.set_yticklabels([shorten_pathway_label(p) for p in pathway_names], fontsize=9)

    # X-tick labels: slot indices grouped by class
    x_labels = [f"H{i+1}" for i in range(N_HIGH_RISK_SLOTS)] \
             + [f"L{i+1}" for i in range(N_LOW_RISK_SLOTS)]
    ax.set_xticks(range(len(slot_keys)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Patient slot (H = highest-risk, L = lowest-risk; 1 = most extreme; fold-averaged)")
    ax.set_ylabel("Reactome pathway (top 10 by total attention)")

    # Vertical separator between high-risk and low-risk
    ax.axvline(N_HIGH_RISK_SLOTS - 0.5, color="black", linewidth=1.5)

    # Annotate "HIGH RISK" / "LOW RISK" group labels
    ax.text(
        (N_HIGH_RISK_SLOTS - 1) / 2, -0.7, "HIGH-RISK PATIENTS",
        ha="center", fontsize=10, fontweight="bold", transform=ax.transData,
    )
    ax.text(
        N_HIGH_RISK_SLOTS + (N_LOW_RISK_SLOTS - 1) / 2, -0.7, "LOW-RISK PATIENTS",
        ha="center", fontsize=10, fontweight="bold", transform=ax.transData,
    )

    # Annotate cells with values
    for i in range(N_PATHWAYS_TO_SHOW):
        for j in range(len(slot_keys)):
            v = matrix[i, j]
            if v >= 0.005:  # only annotate non-zero
                txt_color = "white" if v > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)

    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cb.set_label("Mean attention weight (fold-averaged)", rotation=270, labelpad=14)

    ax.set_title(
        "Knob B Reactome-pathway attention — top 10 pathways × 5 highest-risk + 5 lowest-risk val patients\n"
        "(fold-averaged across 5 TCGA-BRCA CV folds; uniform prior 1/n attention before training)",
        fontsize=10, fontweight="bold", pad=18,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    # Stats JSON
    stats = {
        "n_pathways_shown": N_PATHWAYS_TO_SHOW,
        "top_pathways_by_total_attention": [
            {"name": n, "short_label": shorten_pathway_label(n), "total_weight": w}
            for n, w in top_pathways
        ],
        "matrix_shape": list(matrix.shape),
        "slot_keys": [{"class": c, "slot": s} for (c, s) in slot_keys],
        "matrix": matrix.tolist(),
        "n_folds": n_folds,
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
