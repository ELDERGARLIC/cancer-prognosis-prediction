"""Stage 14 -- TCGA-internal paired bootstrap Delta for Mpath (Knob B) vs M0
(Knob A), backing Table 2.

Table 2's Mpath row reports a paired-bootstrap Delta vs M0 on identical TCGA
patients (+0.005 [-0.015, +0.025], P=0.32, Section 3.7) that was previously
only recorded in a markdown log (results/stage_3_summary.md), not any
checked-in JSON -- the same gap class Stage 13 closed for Table 2's other
hand-maintained numbers.

This script recomputes it from the already-saved, frozen per-fold TCGA
validation predictions (no model refit), using the identical function,
pooling order, seed and n_boot that 03c_sage_pathway_clinical.py's own
main() already used to produce that number:

    M0 (Knob A)    <- stage_3b_sage_clinical_lasso_honest.json
                      (fold_results[*].best_val_log_h/val_T/val_E)
    Mpath (Knob B) <- stage_3c_sage_pathway_clinical.json
                      (fold_results[*].best_val_log_h/val_T/val_E)

Bootstrap: paired_bootstrap_delta (src/cindex_bootstrap.py), B=2000, seed=42
-- identical to the pooled_paired_vs_a computation in
03c_sage_pathway_clinical.py's main().

No manuscript numbers change. This purely adds the missing generating code so
the repo backs up Table 2's Mpath row end to end.

Outputs:
  results/stage_14_table2_mpath_tcga_delta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cindex_bootstrap import paired_bootstrap_delta  # noqa: E402

RESULTS = ROOT / "results"
KNOB_A_JSON = RESULTS / "stage_3b_sage_clinical_lasso_honest.json"
KNOB_B_JSON = RESULTS / "stage_3c_sage_pathway_clinical.json"
OUT_JSON = RESULTS / "stage_14_table2_mpath_tcga_delta.json"

SEED = 42
N_BOOT = 2000

MANUSCRIPT_REPORTED = {"delta_point": 0.005, "ci_low": -0.015, "ci_high": 0.025, "p_a_le_b": 0.32}


def main():
    knob_a = json.loads(KNOB_A_JSON.read_text())
    knob_b = json.loads(KNOB_B_JSON.read_text())
    knob_a_fr = {r["fold"]: r for r in knob_a["fold_results"]}
    fr = knob_b["fold_results"]

    T_pool = np.concatenate([np.array(r["val_T"]) for r in fr])
    E_pool = np.concatenate([np.array(r["val_E"]) for r in fr])
    knob_b_pool = np.concatenate([np.array(r["best_val_log_h"]) for r in fr])
    knob_a_pool = np.concatenate([np.array(knob_a_fr[r["fold"]]["best_val_log_h"]) for r in fr])

    result = paired_bootstrap_delta(
        T_pool, E_pool, risk_a=knob_b_pool, risk_b=knob_a_pool, n_boot=N_BOOT, seed=SEED,
    )

    match = (
        abs(result["delta_point"] - MANUSCRIPT_REPORTED["delta_point"]) < 1e-3
        and abs(result["delta_ci_low"] - MANUSCRIPT_REPORTED["ci_low"]) < 1e-3
        and abs(result["delta_ci_high"] - MANUSCRIPT_REPORTED["ci_high"]) < 1e-3
        and abs(result["p_a_le_b"] - MANUSCRIPT_REPORTED["p_a_le_b"]) < 5e-3
    )

    out = dict(result)
    out["description"] = "Paired Delta(Mpath - M0), pooled TCGA out-of-fold, identical patients"
    out["seed"] = SEED
    out["manuscript_reported"] = MANUSCRIPT_REPORTED
    out["matches_manuscript"] = match

    print(f"delta_point={result['delta_point']:+.4f}  "
          f"CI=[{result['delta_ci_low']:+.4f}, {result['delta_ci_high']:+.4f}]  "
          f"p_a_le_b={result['p_a_le_b']:.3f}")
    print("manuscript: +0.005 [-0.015, +0.025], P=0.32")
    print(f"matches_manuscript: {match}")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
