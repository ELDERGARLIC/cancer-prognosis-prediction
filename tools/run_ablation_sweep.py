"""Phase 1.5 loss-ablation sweep driver.

Runs 4 configurations, 2-fold CV, max 10 epochs each, with dead-early kill
at epoch 5 when train_ce > 1.35 AND val_acc < 0.25.

  A  ce                        -- classifier alone
  B  ce + aux_stage            -- classifier + AuxNet tumor-stage head
  C  deephit                   -- survival-bin head alone
  D  ce + deephit @ 0.1        -- low-DH-weight coexistence

Before the sweep, benchmarks 1 epoch * 1 fold on CPU using spec A (the
fastest config -- single head, no grad-norm probe overhead to speak of).
Compares against the already-measured MPS 26 min/epoch and picks the
faster device for the full sweep.

Outputs:
  results/ablation_{RUN_NAME}.log              -- per-run log file
  results/ablation_{RUN_NAME}_results.json     -- full metrics + history
  results/ablation_summary.md                  -- aggregate table + decision

Usage:
  poetry run python -m tools.run_ablation_sweep [--device cpu|mps]
                                                [--skip-benchmark]
                                                [--only A,B,C,D]
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
import yaml

from src.dataset import build_dataset
from src.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sweep")


# ----- Ablation specs ----------------------------------------------------

SPECS = [
    {
        "run_name": "A_ce_only",
        "active_losses": ["ce"],
        "desc": "classifier head alone",
        "overrides": {},
    },
    {
        "run_name": "B_ce_aux",
        "active_losses": ["ce", "aux_stage"],
        "desc": "classifier + AuxNet tumor-stage",
        "overrides": {},
    },
    {
        "run_name": "C_deephit_only",
        "active_losses": ["deephit"],
        "desc": "DeepHit survival-bin head alone",
        "overrides": {},
    },
    {
        "run_name": "D_ce_dh_low",
        "active_losses": ["ce", "deephit"],
        "desc": "CE + DeepHit at weight 0.1",
        "overrides": {"training.deephit_aux_weight": 0.1},
    },
]


# ----- Config helpers ----------------------------------------------------

def _apply_override(cfg: dict, path: str, val) -> None:
    parts = path.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def make_config(
    base: dict,
    spec: dict,
    device: str,
    max_epochs: int = 10,
    cv_folds: int = 2,
    skip_rf: bool = True,
    dead_early_kill: bool = True,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["training"]["cv_folds"] = cv_folds
    cfg["training"]["device"] = device
    # Keep epochs reasonable to allow normal early stopping too
    cfg["training"]["epochs"] = max(cfg["training"].get("epochs", 200), max_epochs)
    cfg.setdefault("ablation", {})
    cfg["ablation"]["enabled"] = True
    cfg["ablation"]["run_name"] = spec["run_name"]
    cfg["ablation"]["active_losses"] = list(spec["active_losses"])
    cfg["ablation"]["max_epochs"] = max_epochs
    cfg["ablation"]["skip_rf"] = skip_rf
    cfg["ablation"]["dead_early_kill"] = dead_early_kill
    for p, v in spec.get("overrides", {}).items():
        _apply_override(cfg, p, v)
    return cfg


# ----- Summary extraction -----------------------------------------------

def _history_at(history: dict, key: str, epoch_index: int, default=None):
    """history[key][epoch_index] with defaults for short/missing series."""
    vals = history.get(key, [])
    if not vals:
        return default
    if epoch_index < len(vals):
        return vals[epoch_index]
    return vals[-1]


def summarize_run(results: dict) -> dict:
    """Pull the key numbers from a per-run results dict, fold-averaged."""
    spec_name = results.get("run_name", "?")
    active = results.get("active_losses", [])
    folds = results.get("fold_results", [])
    if not folds:
        return {"run": spec_name, "active": active, "ok": False}

    # Averaged across folds: epoch-10 (or last available) metrics.
    val_acc = []
    c_idx_ord = []
    train_ce = []
    grad_ce = []
    grad_dh = []
    grad_cox = []
    grad_ord = []
    grad_aux = []
    max_epochs_run = 0
    any_dead = False

    for fr in folds:
        hist = fr["history"]
        n = len(hist.get("train_ce_loss", []))
        max_epochs_run = max(max_epochs_run, n)
        if fr.get("dead_early"):
            any_dead = True
        last = n - 1 if n > 0 else 0
        val_acc.append(_history_at(hist, "val_accuracy", last, 0.0))
        c_idx_ord.append(_history_at(hist, "val_c_index_ord", last, 0.0))
        train_ce.append(_history_at(hist, "train_ce_loss", last, float("nan")))
        grad_ce.append(_history_at(hist, "grad_ce", last, 0.0))
        grad_dh.append(_history_at(hist, "grad_deephit", last, 0.0))
        grad_cox.append(_history_at(hist, "grad_cox", last, 0.0))
        grad_ord.append(_history_at(hist, "grad_ordinal", last, 0.0))
        grad_aux.append(_history_at(hist, "grad_aux_stage", last, 0.0))

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "run": spec_name,
        "active": active,
        "epochs_run": max_epochs_run,
        "dead_early": any_dead,
        "val_acc_last": _mean(val_acc),
        "c_idx_ord_last": _mean(c_idx_ord),
        "train_ce_last": _mean(train_ce),
        "gnn_grad_ce": _mean(grad_ce),
        "gnn_grad_dh": _mean(grad_dh),
        "gnn_grad_cox": _mean(grad_cox),
        "gnn_grad_ord": _mean(grad_ord),
        "gnn_grad_aux": _mean(grad_aux),
        "ok": True,
    }


def decision_from_summaries(rows: list) -> str:
    """Apply the user's branching logic based on A/C outcomes."""
    by = {r["run"]: r for r in rows if r.get("ok")}
    A = by.get("A_ce_only")
    C = by.get("C_deephit_only")
    D = by.get("D_ce_dh_low")

    if A is None:
        return (
            "**Branch unavailable**: run A did not complete. Re-run the sweep "
            "before making an architecture call."
        )

    A_learns = (A["val_acc_last"] >= 0.35) and (A["train_ce_last"] < 1.0)
    A_dead = (A["train_ce_last"] > 1.35) or (A["val_acc_last"] < 0.25)

    if A_dead:
        return (
            "**Branch 2 — Run A failed the same way.** The classifier cannot "
            "learn even with every other loss off. The bug is upstream of the "
            "multi-task balance (likely dataset / model / pooling). "
            "**Next step:** build a minimal repro (GNN + CE head, same "
            "preprocessed data, no decoupling, no aux). If that also fails, "
            "the bug is in `src/dataset.py` or `src/model.py`, not `src/train.py`."
        )

    if not A_learns:
        return (
            "**Branch inconclusive:** run A did not clearly pass "
            f"(val_acc={A['val_acc_last']:.3f}, train_ce={A['train_ce_last']:.3f}). "
            "Before committing to an architecture change, extend run A to 20 "
            "epochs to see if it's slow-learning rather than stuck."
        )

    # A learns. Check C.
    if C is not None:
        C_survival_ok = C["c_idx_ord_last"] >= 0.60
    else:
        C_survival_ok = None

    if C_survival_ok is False:
        return (
            f"**Branch 3 — DeepHit head is broken, not the classifier.** Run A "
            f"passes (val_acc={A['val_acc_last']:.3f}) but Run C's c-index stays "
            f"at {C['c_idx_ord_last']:.3f} <= 0.60. Likely a bug in the "
            "censoring mask or the discrete-time bin target. "
            "**Next step:** drop DeepHit; use only CE + ordinal (these worked "
            "in earlier runs). Re-verify with 2-fold CV."
        )

    # Both A and C look OK individually. Check D.
    if D is not None:
        # D should be within 0.05 of A on val_acc and within 0.05 of C on c-idx
        d_acc_gap = abs(D["val_acc_last"] - A["val_acc_last"])
        d_cidx_gap = abs(D["c_idx_ord_last"] - (C["c_idx_ord_last"] if C else 0.0))
        if d_acc_gap > 0.05 or d_cidx_gap > 0.05:
            return (
                f"**Branch 4 — pure gradient-scale incompatibility.** Run A "
                f"(val_acc={A['val_acc_last']:.3f}) and run C "
                f"(c-idx={C['c_idx_ord_last']:.3f}) each succeed alone, but "
                f"run D's blend (val_acc={D['val_acc_last']:.3f}, "
                f"c-idx={D['c_idx_ord_last']:.3f}) drops. Fix: separate GNN "
                "encoders per head, OR GradNorm / PCGrad on the shared encoder."
            )

    return (
        "**Branch 1 — multi-task gradient imbalance confirmed.** Run A shows "
        f"the classifier can learn (val_acc={A['val_acc_last']:.3f}, "
        f"train_ce={A['train_ce_last']:.3f}). "
        "**Next step:** pick one of (i) two separate GNN encoders per head, "
        "(ii) `deephit_aux_weight = 0.05` with gradient clipping. "
        "Use run D's numbers to decide which is more promising."
    )


def write_summary(rows: list, base: dict, device: str, out_path: str) -> None:
    lines = []
    lines.append("# Phase 1.5 — Loss-Ablation Sweep Summary")
    lines.append("")
    lines.append(f"Device: **{device}** · "
                 f"n_classes={len(base['data']['survival_labels'])} · "
                 f"cv_folds=2 · max_epochs=10 per run · "
                 f"dead_early_kill at epoch 5 (train_ce>1.35 AND val_acc<0.25)")
    lines.append("")
    lines.append("## Results (fold-averaged, last-epoch snapshot)")
    lines.append("")
    lines.append("| Run | Active | Epochs | Val Acc | C-idx (ord) | Train CE | "
                 "GNN grad CE | GNN grad DH | GNN grad Cox | GNN grad Ord | "
                 "GNN grad Aux | Dead? |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|")
    for r in rows:
        if not r.get("ok"):
            lines.append(
                f"| {r.get('run', '?')} | {','.join(r.get('active', []))} | - | "
                "- | - | - | - | - | - | - | - | ERR |"
            )
            continue
        lines.append(
            f"| {r['run']} | {','.join(r['active'])} | {r['epochs_run']} | "
            f"{r['val_acc_last']:.4f} | {r['c_idx_ord_last']:.4f} | "
            f"{r['train_ce_last']:.4f} | "
            f"{r['gnn_grad_ce']:.2e} | {r['gnn_grad_dh']:.2e} | "
            f"{r['gnn_grad_cox']:.2e} | {r['gnn_grad_ord']:.2e} | "
            f"{r['gnn_grad_aux']:.2e} | "
            f"{'YES' if r['dead_early'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"Classifier at uniform softmax: train_ce should be "
                 f"`ln({len(base['data']['survival_labels'])}) ≈ "
                 f"{float(__import__('math').log(len(base['data']['survival_labels']))):.3f}`.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(decision_from_summaries(rows))
    lines.append("")
    Path(out_path).write_text("\n".join(lines))
    logger.info(f"[SUMMARY] wrote {out_path}")


# ----- Benchmark ---------------------------------------------------------

def benchmark(base: dict, device: str, dataset_cache: dict) -> float:
    """Run spec A, 1 fold, 1 epoch; return elapsed seconds for that fold.

    Relies on run_training honoring `training.cv_folds` even when the passed
    dataset_cache was built with a higher fold count.
    """
    spec = copy.deepcopy(SPECS[0])
    spec["run_name"] = f"bench_{device}"
    cfg = make_config(
        base, spec, device=device, max_epochs=1, cv_folds=1, skip_rf=True,
        dead_early_kill=False,
    )
    t0 = time.time()
    run_training(cfg, dataset_cache=dataset_cache)
    return time.time() - t0


# ----- Main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"],
                    default="auto",
                    help="Override device selection (skip benchmark).")
    ap.add_argument("--skip-benchmark", action="store_true",
                    help="Skip CPU benchmark; use --device or auto.")
    ap.add_argument("--only", default="A,B,C,D",
                    help="Comma-separated subset of A,B,C,D to run.")
    args = ap.parse_args()

    base = yaml.safe_load(Path("configs/config.yaml").read_text())

    # Build dataset ONCE. Subsequent runs pass it through via dataset_cache.
    logger.info("[SETUP] building dataset (shared across all runs)...")
    t0 = time.time()
    dataset_cache = build_dataset(base)
    logger.info(f"[SETUP] dataset built in {time.time() - t0:.1f}s")

    # Device pick
    if args.skip_benchmark or args.device != "auto":
        device = args.device if args.device != "auto" else (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"[PERF] using device={device} (benchmark skipped)")
    else:
        logger.info("[PERF] benchmarking CPU (MPS was previously measured at "
                    "~26 min/epoch; benchmark only the unknown side)...")
        cpu_t = benchmark(base, "cpu", dataset_cache)
        logger.info(f"[PERF] CPU epoch 1 time: {cpu_t / 60:.1f} min")
        logger.info(f"[PERF] MPS epoch 1 time (prior): 26.0 min")
        if cpu_t / 60 < 15.0:  # within a reasonable margin of the 10-min cutoff
            device = "cpu"
            logger.info(f"[PERF] Using: CPU for sweep (faster by "
                        f"{26.0 / (cpu_t / 60):.2f}x)")
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"[PERF] Using: {device} for sweep "
                        "(CPU too slow vs MPS baseline)")

    # Filter specs by --only
    wanted = set(args.only.split(","))
    specs_to_run = [
        s for s in SPECS if s["run_name"].split("_")[0] in wanted
    ]
    logger.info(f"[SWEEP] running {len(specs_to_run)} specs: "
                f"{[s['run_name'] for s in specs_to_run]}")

    results = []
    for spec in specs_to_run:
        logger.info("=" * 70)
        logger.info(f"[SWEEP] starting {spec['run_name']} "
                    f"({spec['desc']}): active={spec['active_losses']}")
        logger.info("=" * 70)
        cfg = make_config(base, spec, device=device)
        t0 = time.time()
        try:
            run_training(cfg, dataset_cache=dataset_cache)
            elapsed = time.time() - t0
            logger.info(f"[SWEEP] {spec['run_name']} done in {elapsed / 60:.1f} min")
            # Reload JSON the trainer wrote, extract summary
            results_path = (
                Path(base["paths"]["results"])
                / f"ablation_{spec['run_name']}_results.json"
            )
            with open(results_path) as f:
                run_results = json.load(f)
            results.append(summarize_run(run_results))
        except Exception as e:
            logger.exception(f"[SWEEP] {spec['run_name']} FAILED: {e}")
            results.append({
                "run": spec["run_name"],
                "active": spec["active_losses"],
                "ok": False,
                "error": str(e),
            })

    summary_path = Path(base["paths"]["results"]) / "ablation_summary.md"
    write_summary(results, base, device, str(summary_path))


if __name__ == "__main__":
    main()
