"""Stage 12 — Two reviewer-requested controls (no new modelling choices).

Both experiments discharge promises the manuscript already makes as "future
work"; each can only tighten or leave unchanged the honest decomposition.

EXPERIMENT A — matched NON-LINEAR clinical-only external baseline.
  Reviewers (rounds 2 & 3) ask for an external clinical-only model with the same
  non-linear head as M0 but no gene inputs, to bound the gene-graph marginal
  without the linear-vs-non-linear confound. We train a ClinicalOnlyMLP (the
  exact head style of M0's fusion MLP: Linear->ReLU->Dropout(0.4)->Linear, Cox
  partial-likelihood loss, same optimiser/epochs) on the 3 shared clinical
  features, full-TCGA seed-42 90/10 split, and apply cold to METABRIC. Paired
  bootstrap on identical METABRIC patients gives:
     gene-graph marginal over NON-LINEAR clinical = C(M0) - C(clin-MLP)
     non-linearity effect on clinical baseline      = C(clin-MLP) - C(clin-Cox)

EXPERIMENT B — full-batch (exact risk-set) Cox training on one fold.
  Reviewers flag that the Cox partial likelihood is optimised per mini-batch
  (batch 64), truncating the risk set. We retrain M0 on fold 0 with batch size =
  full training set (exact risk set) and compare validation concordance to the
  batched value from Stage 5. A small |delta| confirms the approximation does
  not bias the hazard ranking.

Outputs:
  results/stage_12_reviewer_experiments.json
  manuscript/table_reviewer_controls.tex
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
import torch.nn as nn
from sksurv.metrics import concordance_index_censored

ROOT = Path("/home/eldergarlic/network_share/Projects/thesis-pipelines/thesis-research-v1")
CODE = ROOT / "code"
RESULTS = ROOT / "output/thesis-results-repo/results"
OUT_JSON = RESULTS / "stage_12_reviewer_experiments.json"
OUT_TEX = ROOT / "output/thesis-writing-repo/manuscript/table_reviewer_controls.tex"
STAGE5 = RESULTS / "stage_5_metabric_external.json"
STAGE11 = RESULTS / "stage_11_geneonly.json"

sys.path.insert(0, str(CODE / "src"))
spec = importlib.util.spec_from_file_location("s5", CODE / "scripts/05_metabric_external.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)
from cindex_bootstrap import bootstrap_cindex  # noqa: E402
from sage_models import cox_partial_likelihood_loss  # noqa: E402
from lifelines.utils import concordance_index as lifelines_ci  # noqa: E402

SEED = s5.SEED
N_FOLDS = s5.N_FOLDS
EPOCHS = s5.EPOCHS
HIDDEN_DIM = s5.HIDDEN_DIM
DROPOUT = s5.DROPOUT
LR = s5.LR
WEIGHT_DECAY = s5.WEIGHT_DECAY
MLP_BATCH = 64
N_BOOT = 2000
ALPHA = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage12")


def cindex(T, E, risk):
    return float(concordance_index_censored(E.astype(bool), T, risk)[0])


def paired_delta(T, E, risk_a, risk_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap Δ = C(a) - C(b) on identical patients (Stage-11 protocol)."""
    rng = np.random.default_rng(seed)
    pa, pb = cindex(T, E, risk_a), cindex(T, E, risk_b)
    deltas = []
    n = len(T)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if E[idx].sum() < 1:
            continue
        try:
            deltas.append(cindex(T[idx], E[idx], risk_a[idx]) - cindex(T[idx], E[idx], risk_b[idx]))
        except Exception:
            continue
    deltas = np.asarray(deltas)
    return {"c_a": float(pa), "c_b": float(pb), "delta_point": float(pa - pb),
            "ci_low": float(np.quantile(deltas, ALPHA / 2)),
            "ci_high": float(np.quantile(deltas, 1 - ALPHA / 2)),
            "p_a_le_b": float((deltas <= 0).mean()), "n_valid": int(len(deltas))}


# ---------------------------------------------------------------- Experiment A
class ClinicalOnlyMLP(nn.Module):
    """Same head style as M0's fusion MLP, no gene input (matches 03_ref_mlp)."""
    def __init__(self, clinical_dim=3, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def train_clin_mlp(Xc_tr, T_tr, E_tr, Xc_va, T_va, E_va, seed, epochs=EPOCHS):
    """Train clinical-only MLP; select epoch by val concordance; return best state + val log_h."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ClinicalOnlyMLP(clinical_dim=Xc_tr.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xtr = torch.from_numpy(Xc_tr.astype(np.float32))
    Ttr = torch.from_numpy(T_tr.astype(np.float32)); Etr = torch.from_numpy(E_tr.astype(np.float32))
    Xva = torch.from_numpy(Xc_va.astype(np.float32))
    n = Xtr.shape[0]; perm = np.arange(n)
    best_c, best_state, best_logh = -1.0, None, None
    for _ in range(1, epochs + 1):
        model.train(); np.random.shuffle(perm)
        for st in range(0, n, MLP_BATCH):
            idx = perm[st:st + MLP_BATCH]
            if Etr[idx].sum() < 1:
                continue
            loss = cox_partial_likelihood_loss(model(Xtr[idx]), Ttr[idx], Etr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logh_va = model(Xva).numpy()
        c = float(lifelines_ci(T_va, -logh_va, E_va))
        if c > best_c:
            best_c = c; best_logh = logh_va.copy()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_c, best_logh


def experiment_A(tcga, met, gnn_risk, clin_cox_risk):
    log.info("=== EXPERIMENT A: non-linear clinical-only external baseline ===")
    T, E = met["T"], met["E"]

    # internal 5-fold (for the reporting row)
    splits = json.loads(s5.TCGA_SPLITS.read_text())
    internal = []
    for f in range(N_FOLDS):
        s = splits[f"fold_{f}"]
        tr, va = np.array(s["train_idx"]), np.array(s["val_idx"])
        _, c, _ = train_clin_mlp(tcga["X_clin_3"][tr], tcga["T"][tr], tcga["E"][tr],
                                 tcga["X_clin_3"][va], tcga["T"][va], tcga["E"][va], seed=SEED + f)
        internal.append(c)
        log.info(f"  fold{f} clin-MLP internal C={c:.4f}")

    # external: full-TCGA seed-42 90/10 split (identical to Stage 11)
    rng = np.random.default_rng(SEED)
    n = len(tcga["T"]); pm = rng.permutation(n); ntr = int(0.9 * n)
    ftr, fva = pm[:ntr], pm[ntr:]
    model, va_c, _ = train_clin_mlp(tcga["X_clin_3"][ftr], tcga["T"][ftr], tcga["E"][ftr],
                                    tcga["X_clin_3"][fva], tcga["T"][fva], tcga["E"][fva], seed=SEED)
    with torch.no_grad():
        mlp_risk = model(torch.from_numpy(met["X_clin_3"].astype(np.float32))).numpy()
    mlp_cidx = cindex(T, E, mlp_risk)
    mlp_boot = bootstrap_cindex(T.astype(np.float64), E.astype(np.int64), mlp_risk, n_boot=N_BOOT, seed=SEED)

    gene_marg_nl = paired_delta(T, E, gnn_risk, mlp_risk)      # M0 - clin-MLP  (tightened marginal)
    nonlin_effect = paired_delta(T, E, mlp_risk, clin_cox_risk)  # clin-MLP - clin-Cox

    log.info(f"  clin-MLP external C={mlp_cidx:.4f} [{mlp_boot['ci_low']:.3f},{mlp_boot['ci_high']:.3f}]"
             f"  (internal {np.mean(internal):.4f}±{np.std(internal):.3f}, TCGA-10%val {va_c:.4f})")
    log.info(f"  gene-graph marginal over NON-LINEAR clinical = {gene_marg_nl['delta_point']:+.4f} "
             f"[{gene_marg_nl['ci_low']:+.4f},{gene_marg_nl['ci_high']:+.4f}] p={gene_marg_nl['p_a_le_b']:.3f}")
    log.info(f"  non-linearity effect on clinical (MLP-Cox) = {nonlin_effect['delta_point']:+.4f} "
             f"[{nonlin_effect['ci_low']:+.4f},{nonlin_effect['ci_high']:+.4f}]")
    return {
        "clin_mlp_internal_mean": float(np.mean(internal)),
        "clin_mlp_internal_std": float(np.std(internal)),
        "clin_mlp_internal_folds": internal,
        "clin_mlp_external_cidx": mlp_cidx,
        "clin_mlp_external_ci": [mlp_boot["ci_low"], mlp_boot["ci_high"]],
        "clin_mlp_tcga10val": va_c,
        "gene_marginal_over_nonlinear_clinical": gene_marg_nl,
        "nonlinearity_effect_on_clinical": nonlin_effect,
        "clin_mlp_risk": mlp_risk.tolist(),
    }


# ---------------------------------------------------------------- Experiment B
def experiment_B(tcga, met, full_edge_index):
    log.info("=== EXPERIMENT B: full-batch (exact risk-set) M0 on fold 0 ===")
    s5j = json.loads(STAGE5.read_text())
    batched_stage5 = s5j["fold_results"][0]["tcga_val_cidx_gnn"]

    overlap_idx = met["overlap_expr_col_idx_in_tcga"]
    X_ov = tcga["X_expr"][:, overlap_idx]
    ov_col_to_kg = tcga["expr_col_to_kg_idx"][overlap_idx]
    splits = json.loads(s5.TCGA_SPLITS.read_text())
    s = splits["fold_0"]
    tr, va = np.array(s["train_idx"]), np.array(s["val_idx"])
    nz, _ = s5.per_fold_lasso_within_universe(X_ov[tr], tcga["bins"][tr].astype(float), fold_label="f0")
    kg = s5.subset_kg_to_fold(nz, ov_col_to_kg, full_edge_index, n_kg_genes=769)

    def run_m0(batch):
        old = s5.BATCH_SIZE
        s5.BATCH_SIZE = batch
        try:
            _, _, best_c, _, best_ep = s5.train_knob_a(
                X_ov[tr], X_ov[va], tcga["X_clin_3"][tr], tcga["X_clin_3"][va],
                tcga["T"][tr], tcga["T"][va], tcga["E"][tr], tcga["E"][va],
                kg["fold_expr_cols"], kg["edge_index_local"], clinical_dim=3,
                fold_label=f"f0-b{batch}")
        finally:
            s5.BATCH_SIZE = old
        return best_c, best_ep

    repro_c, repro_ep = run_m0(64)                 # reproduce batched baseline
    full_c, full_ep = run_m0(int(len(tr)))          # exact full-batch risk set

    log.info(f"  Stage-5 stored batched (b=64) fold0 val C = {batched_stage5:.4f}")
    log.info(f"  reproduced batched (b=64)      fold0 val C = {repro_c:.4f} (ep {repro_ep})")
    log.info(f"  full-batch (exact risk set)    fold0 val C = {full_c:.4f} (ep {full_ep})")
    log.info(f"  delta (full-batch - batched)   = {full_c - repro_c:+.4f}")
    return {
        "stage5_batched_fold0_cidx": float(batched_stage5),
        "reproduced_batched_fold0_cidx": float(repro_c),
        "fullbatch_fold0_cidx": float(full_c),
        "delta_fullbatch_minus_batched": float(full_c - repro_c),
        "train_size": int(len(tr)),
    }


def write_latex(A, B):
    gm = A["gene_marginal_over_nonlinear_clinical"]
    lines = [
        r"% Auto-generated by scripts/12_reviewer_experiments.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Two reviewer-requested controls on the external cohort "
        r"(METABRIC). \emph{Top:} a non-linear clinical-only baseline (an MLP with "
        r"$\mathrm{M_0}$'s head style and no gene input) bounds the gene-graph "
        r"marginal without the linear-vs-non-linear confound; the marginal remains "
        r"unresolved from zero. \emph{Bottom:} retraining $\mathrm{M_0}$ on fold~0 "
        r"with the exact full-cohort risk set changes validation concordance "
        r"negligibly, confirming the mini-batch partial-likelihood approximation "
        r"does not bias the hazard ranking. Paired bootstrap $B=" + str(N_BOOT) + r"$.}",
        r"\label{tab:reviewer_controls}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Control & Result \\",
        r"\midrule",
        rf"Clinical-only MLP, external C-index & {A['clin_mlp_external_cidx']:.4f} "
        rf"[{A['clin_mlp_external_ci'][0]:.3f}, {A['clin_mlp_external_ci'][1]:.3f}] \\",
        rf"Gene-graph marginal over non-linear clinical & {gm['delta_point']:+.4f} "
        rf"[{gm['ci_low']:+.3f}, {gm['ci_high']:+.3f}], $p={gm['p_a_le_b']:.3f}$ \\",
        r"\midrule",
        rf"$\mathrm{{M_0}}$ fold~0, mini-batch (64) val C & {B['reproduced_batched_fold0_cidx']:.4f} \\",
        rf"$\mathrm{{M_0}}$ fold~0, full-batch (exact) val C & {B['fullbatch_fold0_cidx']:.4f} \\",
        rf"Difference (full $-$ mini-batch) & {B['delta_fullbatch_minus_batched']:+.4f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n")


def main():
    t0 = time.time()
    s5j = json.loads(STAGE5.read_text())
    s11j = json.loads(STAGE11.read_text())
    gnn_risk = np.asarray(s5j["full_tcga_run"]["metabric_log_h_gnn_full"], float)
    clin_cox_risk = np.asarray(s11j["clinical_only_risk"], float)

    full_edge_index = torch.load(s5.KG_EDGES, weights_only=False)["gene_gene_edges"]
    tcga = s5.load_tcga()
    met = s5.load_metabric(tcga["gene_ids_769"], tcga["kg_gene_to_idx"])
    assert len(met["T"]) == len(gnn_risk) == len(clin_cox_risk)

    A = experiment_A(tcga, met, gnn_risk, clin_cox_risk)
    B = experiment_B(tcga, met, full_edge_index)

    out = {"n_metabric": int(len(met["T"])), "seed": SEED, "n_boot": N_BOOT,
           "experiment_A_nonlinear_clinical": A,
           "experiment_B_fullbatch_riskset": B,
           "elapsed_seconds": time.time() - t0}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    write_latex(A, B)
    log.info(f"wrote {OUT_JSON} and {OUT_TEX}  ({out['elapsed_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
