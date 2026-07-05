"""FIX 2: Knob C (BioBERT-PCA gene init) 5-fold cross-validation on TCGA-BRCA.

The thesis Results §4 currently reports Knob C as a single 10%-val
C-index (0.7414), while Knobs A and B both have 5-fold mean ± std. This
script runs Knob C through the same 5-fold stratified splits as Knobs A
and B and reports per-fold + mean ± std.

Architecture: SAGEClinicalEmb with in_dim=32 (BioBERT-PCA gene priors
multiplied element-wise by expression scalar), per-fold LASSO selection
within the leaky-769 universe (matching Knob A), clinical late-fusion,
Cox partial-likelihood loss. Reuses helpers from scripts/05 and the
SAGEClinicalEmb model defined in scripts/06e.

Output:
  - results/stage_3d_knob_c_5fold_internal.json
  - prints headline numbers; standalone — does NOT regenerate METABRIC
    external. METABRIC external for Knob C remains at scripts/06e (full-TCGA
    trained, single inference); the 5-fold internal numbers here replace
    only the within-TCGA point estimate in Table 2 / Results §4.
"""
from __future__ import annotations
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse Stage 5 helpers (load_tcga, per_fold_lasso_within_universe,
# subset_kg_to_fold) and the BioBERT loader + train fn from 06e.
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s5 = _load("s5", ROOT / "scripts" / "05_metabric_external.py")
s6e = _load("s6e", ROOT / "scripts" / "06e_knob_c_biobert_metabric.py")

from cindex_bootstrap import bootstrap_cindex  # noqa: E402

DATA = ROOT / "data" / "processed"
KG_EDGES_PATH = DATA / "kg_edges.pt"
SPLITS_PATH = DATA / "cv_splits.json"
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "stage_3d_knob_c_5fold_internal.json"

SEED = 42
N_FOLDS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("06g_knob_c_5fold")


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("KNOB C — 5-fold internal CV (BioBERT-PCA gene init × expression)")
    log.info("=" * 70)

    # Load TCGA + BioBERT priors (overlap universe = full leaky-769 here, since
    # we don't intersect with METABRIC for the internal-only run).
    tcga = s5.load_tcga()

    # For consistency with Knob A's internal run, use the FULL leaky-769 universe
    # (not the METABRIC-overlap 650). This is the same comparator basis as
    # Knob A's internal numbers in scripts/03b.
    leaky_genes = tcga["gene_ids_769"]  # 769-element list
    log.info(f"  Using full leaky-769 universe (matches Knob A internal): {len(leaky_genes)} genes")

    # PCA(32) of BioBERT on the FULL 769-gene universe (matching the spirit of
    # 06e's PCA, but on 769 instead of 650). No patient labels involved.
    biobert_pca32, _ = s6e.load_biobert_pca32(leaky_genes)
    log.info(f"  biobert_pca32 shape: {biobert_pca32.shape}")

    full_edge_index = torch.load(KG_EDGES_PATH, weights_only=False)["gene_gene_edges"]

    # 5 stratified folds (the canonical CV splits)
    splits = json.loads(SPLITS_PATH.read_text())

    # Use FULL TCGA expression (no METABRIC-overlap restriction)
    overlap_col_to_kg_idx = tcga["expr_col_to_kg_idx"]
    X_tcga = tcga["X_expr"]
    clinical_dim = tcga["X_clin_7"].shape[1]
    log.info(f"  clinical_dim = {clinical_dim} (was 7 pre-ER/PR fix; up to 10 if recovery applied)")

    fold_results = []
    for fold in range(N_FOLDS):
        log.info(f"--- fold {fold} ---")
        s = splits[f"fold_{fold}"]
        tr = np.array(s["train_idx"]); va = np.array(s["val_idx"])

        # Per-fold LASSO within 769 (same as Knob A)
        y_train = tcga["bins"][tr].astype(np.float64)
        nz_mask, _ = s5.per_fold_lasso_within_universe(
            X_tcga[tr], y_train, fold_label=f"f{fold}",
        )
        kg_info = s5.subset_kg_to_fold(
            nz_mask, overlap_col_to_kg_idx, full_edge_index, n_kg_genes=769,
        )
        log.info(f"    fold {fold} graph: {kg_info['n_nodes']} nodes, {kg_info['n_edges']} edges")

        # Train Knob C on this fold
        model, scaler, va_cidx, va_log_h, best_ep = s6e.train_knob_c(
            X_tcga[tr], X_tcga[va],
            tcga["X_clin_7"][tr], tcga["X_clin_7"][va],
            tcga["T"][tr], tcga["T"][va],
            tcga["E"][tr], tcga["E"][va],
            kg_info["fold_expr_cols"], kg_info["edge_index_local"],
            biobert_pca32, clinical_dim=clinical_dim,
            fold_label=f"f{fold}",
        )

        # Bootstrap CI on best val predictions
        boot = bootstrap_cindex(
            tcga["T"][va].astype(np.float64),
            tcga["E"][va].astype(np.int64),
            va_log_h, n_boot=1000, seed=SEED + fold,
        )
        log.info(f"    fold {fold} Knob C best val_cidx = {va_cidx:.4f} "
                 f"(ep {best_ep})  95% CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")

        fold_results.append({
            "fold": fold,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "events_val": int(tcga["E"][va].sum()),
            "n_nodes": kg_info["n_nodes"],
            "n_edges": kg_info["n_edges"],
            "best_val_cindex": float(va_cidx),
            "best_val_cindex_epoch": int(best_ep),
            "best_val_log_h": va_log_h.tolist(),
            "val_T": tcga["T"][va].tolist(),
            "val_E": tcga["E"][va].astype(int).tolist(),
            "bootstrap": boot,
        })

    cidx = np.array([r["best_val_cindex"] for r in fold_results])
    log.info("")
    log.info(f"KNOB C 5-FOLD INTERNAL: mean = {cidx.mean():.4f} ± {cidx.std():.4f}")
    log.info(f"  per-fold: {[f'{c:.4f}' for c in cidx]}")
    log.info(f"  total time: {time.time()-t0:.0f}s")

    payload = {
        "model": "SAGEClinicalEmb (Knob C, in_dim=32 BioBERT-PCA × expression) — 5-fold internal CV",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "biobert_pca_dim": 32,
        "gene_universe": "leaky-769 (matches Knob A internal)",
        "fold_results": fold_results,
        "mean_val_cindex": float(cidx.mean()),
        "std_val_cindex": float(cidx.std()),
        "per_fold_cindex": cidx.tolist(),
        "total_seconds": float(time.time() - t0),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
