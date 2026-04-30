"""Stage 3 reference: clinical-features-only MLP + Cox loss.

Anchor for the lift-attribution analysis:
  - Cox PH clinical-only (linear): 0.7000 (Stage 0 diag sweep)
  - MLP clinical-only (non-linear, this run): ?
  - GNN + clinical (knob D, prior run): 0.7256

The gene-graph contribution above the *non-linear* clinical baseline (this MLP)
is the honest gene-graph lift to compare against. If MLP-clinical-only also
hits ~0.70, the GNN's gene-graph contribution above non-linear clinical is
~0.025 rather than ~0.04.

Same hyperparameters as knob D except no GNN: 2-layer MLP head straight from
the 7-d clinical vector to scalar log-hazard, trained with Cox partial likelihood.
Same seeds, same splits, same epochs, same patient-level bootstrap CI machinery.
"""
from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cindex_bootstrap import bootstrap_cindex  # noqa: E402
from sage_models import cox_partial_likelihood_loss  # noqa: E402

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

CLINICAL_PATH = DATA / "clinical_features.tsv"
SURVIVAL_PATH = DATA / "clinical_processed.tsv"
SPLITS_PATH = DATA / "cv_splits.json"

RESULTS_JSON = RESULTS / "stage_3_ref_mlp_clinical.json"

SEED = 42
N_FOLDS = 5
HIDDEN_DIM = 128
DROPOUT = 0.4
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
LOW_VAR = 0.01
DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_3_ref_mlp_clin")


class ClinicalOnlyMLP(nn.Module):
    """clinical -> Linear -> ReLU -> Dropout -> Linear -> scalar log-hazard."""
    def __init__(self, clinical_dim=7, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, clinical):
        return self.mlp(clinical).squeeze(-1)


def load_aligned():
    clin_df = pd.read_csv(CLINICAL_PATH, sep="\t")
    surv = pd.read_csv(SURVIVAL_PATH, sep="\t")
    T = surv["OS.time"].values.astype(np.float64)
    E = surv["OS"].values.astype(np.int64)
    keep = T > 0
    T = T[keep]; E = E[keep]
    clin_df = clin_df.iloc[keep].reset_index(drop=True)
    var = clin_df.var(axis=0)
    keep_cols = [c for c in clin_df.columns if var[c] >= LOW_VAR]
    X_clin = clin_df[keep_cols].values.astype(np.float32)
    return X_clin, T, E, keep_cols


def train_fold(fold, X_clin, T, E, splits):
    s = splits[f"fold_{fold}"]
    tr = np.array(s["train_idx"]); va = np.array(s["val_idx"])
    Xc_tr = torch.from_numpy(X_clin[tr])
    Xc_va = torch.from_numpy(X_clin[va])
    T_tr = torch.from_numpy(T[tr].astype(np.float32))
    E_tr = torch.from_numpy(E[tr].astype(np.float32))
    T_va = torch.from_numpy(T[va].astype(np.float32))
    E_va = torch.from_numpy(E[va].astype(np.float32))

    torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
    model = ClinicalOnlyMLP(clinical_dim=X_clin.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_cidx = -1.0
    best_log_h = None
    best_epoch = -1

    n_train = Xc_tr.shape[0]
    perm = np.arange(n_train)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(perm)
        total_loss = 0.0; n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb = Xc_tr[idx]; Tb = T_tr[idx]; Eb = E_tr[idx]
            if Eb.sum() < 1: continue
            log_h = model(Xb)
            loss = cox_partial_likelihood_loss(log_h, Tb, Eb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()); n_batches += 1

        model.eval()
        with torch.no_grad():
            log_h_va = model(Xc_va).numpy()
        from lifelines.utils import concordance_index as _ci
        val_cidx = float(_ci(T[va], -log_h_va, E[va]))
        history.append({"epoch": epoch, "train_loss": total_loss / max(n_batches, 1), "val_cidx": val_cidx})
        if val_cidx > best_cidx:
            best_cidx = val_cidx; best_log_h = log_h_va.copy(); best_epoch = epoch

    boot = bootstrap_cindex(T[va].astype(np.float64), E[va].astype(np.int64), best_log_h,
                            n_boot=1000, seed=SEED + fold)
    log.info(
        f"  fold {fold}: best val_cidx={best_cidx:.4f} (ep {best_epoch})  "
        f"95% CI=[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]"
    )
    return {
        "fold": fold, "n_train": int(len(tr)), "n_val": int(len(va)),
        "events_val": int(E[va].sum()),
        "best_val_cindex": float(best_cidx), "best_val_cindex_epoch": int(best_epoch),
        "best_val_log_h": best_log_h.tolist(),
        "val_T": T[va].tolist(), "val_E": E[va].astype(int).tolist(),
        "bootstrap": boot,
        "history": history,
    }


def main():
    X_clin, T, E, clin_cols = load_aligned()
    splits = json.loads(SPLITS_PATH.read_text())

    log.info("=" * 70)
    log.info(f"Stage 3 REFERENCE: clinical-only MLP (clinical_dim={X_clin.shape[1]}, "
             f"hidden={HIDDEN_DIM}) + Cox loss; cols: {clin_cols}")
    log.info(f"  cohort n={len(T)}, event_rate={E.mean():.3f}")
    log.info("=" * 70)

    fold_results = []
    t0 = time.time()
    for fold in range(N_FOLDS):
        fold_results.append(train_fold(fold, X_clin, T, E, splits))
    total_dt = time.time() - t0

    bests = np.array([r["best_val_cindex"] for r in fold_results])
    log.info(f"")
    log.info(f"MLP clinical-only mean = {bests.mean():.4f} ± {bests.std():.4f}  "
             f"(reference: Cox PH clinical-only ~0.700, GNN+clinical knob D = 0.7256)")
    log.info(f"Total: {total_dt:.0f}s")

    payload = {
        "n_folds": N_FOLDS, "epochs": EPOCHS, "hidden_dim": HIDDEN_DIM,
        "clinical_cols": clin_cols, "fold_results": fold_results,
        "mlp_clinical_only_mean": float(bests.mean()),
        "mlp_clinical_only_std": float(bests.std()),
        "total_seconds": float(total_dt),
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Results JSON: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
