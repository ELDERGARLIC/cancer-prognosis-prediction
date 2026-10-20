"""One-shot diagnostic for the head-decoupled HybridModel.

Before the refactor this script diagnosed why blending DeepHit + CE on a
shared `main_out` collapsed val_acc to 0.2421. The refactor gave DeepHit
its own head (`self.survival_bin_head`) and left the CE classifier
(`self.fc`) untouched. The script now verifies the three invariants the
refactor needs to hold *before* we commit to another 2-fold CV run:

  INV-A  cls_logits gradient through self.fc is healthy under pure CE.
  INV-B  cls_logits.argmax at init is diverse across >=2 classes.
         If all rows predict the same class at random init, the model
         is born in the collapsed state and SMOTE + class weights won't
         pull it out.
  INV-C  DeepHit loss operates on bin_logits, NOT cls_logits.
         Verified two ways:
           (1) zero DeepHit and backward -> cls_logits still has grad
               (because pure CE supervises it).
           (2) zero CE + ord + cox + aux and backward with DeepHit only
               -> grad lands on self.survival_bin_head.* and self.gnn.*,
               and NOT on self.fc.*.

Also carried over from the pre-refactor version (kept because each one
has caught a real bug in this project):
  CHECK 1  runtime config values (deephit_aux_weight, use_deephit, trim).
  CHECK 5  graph tensor shapes (did the trim propagate to n_genes=769?).

No optimizer.step() is ever applied -- this is a read-only observation
of what the first training step *would* do. Runs on CPU to avoid MPS
startup cost; total runtime ~1-2 min dominated by the dataset build.
"""
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, ".")

from src.dataset import build_dataset, get_dataloaders
from src.model import build_model
from src.train import deephit_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("diag")


def _banner(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def _grad_norms_by_prefix(model: nn.Module, prefixes: tuple[str, ...]) -> dict:
    """Return {prefix: total_grad_norm} summed over all params starting with prefix."""
    out = {p: 0.0 for p in prefixes}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.norm().item())
        for pref in prefixes:
            if name.startswith(pref):
                out[pref] += g
                break
    return out


def main() -> None:
    # ---------- CHECK 1: config values at runtime ----------
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    train_cfg = config["training"]
    _banner("CHECK 1: runtime config values")
    logger.info(f"  training.deephit_aux_weight  = {train_cfg.get('deephit_aux_weight', 'MISSING')}")
    logger.info(f"  training.use_deephit         = {train_cfg.get('use_deephit', 'MISSING')}")
    logger.info(f"  training.cv_folds            = {train_cfg.get('cv_folds', 'MISSING')}")
    logger.info(f"  kg.max_isolated_pct          = {config.get('kg', {}).get('max_isolated_pct', 'MISSING')}")
    logger.info(f"  kg.strict_asserts            = {config.get('kg', {}).get('strict_asserts', 'MISSING')}")
    stale = train_cfg.get("deephit_loss_weight")
    if stale is not None:
        logger.warning(
            f"  ! legacy key `deephit_loss_weight={stale}` still in config -- "
            "delete it (replaced by `deephit_aux_weight`)"
        )
    logger.info(f"  full train_cfg keys: {sorted(train_cfg.keys())}")

    # ---------- Build dataset + one fold's train loader ----------
    device = torch.device("cpu")
    data = build_dataset(config)
    dataset = data["dataset"]
    splits = data["splits"]
    clinical_dim = data["clinical_dim"]
    n_genes = data["n_genes"]
    train_idx, val_idx = splits[0]
    train_loader, _val_loader, smote_applied = get_dataloaders(
        dataset, train_idx, val_idx,
        batch_size=train_cfg["batch_size"],
        smote=True,
        smote_strategy=config["data"]["smote_strategy"],
        seed=train_cfg["seed"],
    )
    logger.info(f"  smote_applied = {smote_applied}")
    logger.info(f"  n_genes = {n_genes}  (trim-propagated if ~769; stale if 1500)")

    # ---------- Build model via the project's own helper ----------
    model = build_model(
        config, clinical_dim=clinical_dim, device=str(device), num_genes=n_genes,
    )
    model.train()

    # Sanity: the decoupled head must exist.
    assert hasattr(model, "survival_bin_head"), (
        "HybridModel is missing `survival_bin_head`. The decoupling refactor "
        "did not land in src/model.py."
    )
    assert hasattr(model, "fc"), "HybridModel is missing `fc` (classifier head)."

    # ---------- Pull ONE training batch ----------
    batch = next(iter(train_loader)).to(device)
    _banner("CHECK 5: graph tensor shapes at runtime")
    logger.info(f"  batch.x.shape         = {tuple(batch.x.shape)}")
    logger.info(f"  batch.edge_index.shape= {tuple(batch.edge_index.shape)}")
    logger.info(f"  batch.num_graphs      = {batch.num_graphs}")
    logger.info(f"  batch.y.shape         = {tuple(batch.y.shape)}")
    if hasattr(batch, "clinical"):
        logger.info(f"  batch.clinical.shape  = {tuple(batch.clinical.shape)}")
    if hasattr(batch, "os_event") and batch.os_event is not None:
        ev = batch.os_event
        logger.info(
            f"  batch.os_event        : shape={tuple(ev.shape)}  "
            f"n_nan={torch.isnan(ev).sum().item()}  n_valid={(~torch.isnan(ev)).sum().item()}"
        )
    logger.info(
        f"  nodes_per_patient     = batch.x.shape[0] / batch.num_graphs "
        f"= {batch.x.shape[0]} / {batch.num_graphs} = {batch.x.shape[0] // batch.num_graphs}"
    )

    # ---------- Forward (shared; no grad needed for reporting) ----------
    gene_idx = getattr(batch, "gene_idx", None)
    with torch.no_grad():
        cls_logits, _aux, _emb, _cox, _ord, bin_logits = model(
            batch.x, batch.edge_index, batch.batch, batch.clinical,
            gene_idx=gene_idx,
        )
    assert cls_logits.shape == bin_logits.shape, (
        f"cls_logits {tuple(cls_logits.shape)} and bin_logits "
        f"{tuple(bin_logits.shape)} must have the same shape -- they are both "
        "[B, K] over the same K survival bins."
    )

    # ---------- INV-B: cls_logits.argmax diversity at init ----------
    _banner("INV-B: cls_logits.argmax diversity at init")
    n_classes = cls_logits.shape[1]
    cls_mean = cls_logits.mean(dim=0).cpu().numpy()
    cls_std = cls_logits.std(dim=0).cpu().numpy()
    bin_mean = bin_logits.mean(dim=0).cpu().numpy()
    logger.info(f"  cls_logits mean per class = {cls_mean.tolist()}")
    logger.info(f"  cls_logits std  per class = {cls_std.tolist()}")
    logger.info(f"  bin_logits mean per class = {bin_mean.tolist()}")
    cls_pred = cls_logits.argmax(dim=1)
    bin_pred = bin_logits.argmax(dim=1)
    cls_bincount = torch.bincount(cls_pred, minlength=n_classes).tolist()
    bin_bincount = torch.bincount(bin_pred, minlength=n_classes).tolist()
    y_bincount = torch.bincount(batch.y, minlength=n_classes).tolist()
    logger.info(f"  cls_logits pred bincount = {cls_bincount}")
    logger.info(f"  bin_logits pred bincount = {bin_bincount}")
    logger.info(f"  batch.y        bincount = {y_bincount}")
    n_unique_cls = int((torch.tensor(cls_bincount) > 0).sum().item())
    inv_b_pass = n_unique_cls >= 2
    logger.info(
        f"  INV-B: cls_logits covers {n_unique_cls}/{n_classes} classes at init "
        f"-> {'PASS' if inv_b_pass else 'FAIL (collapsed at birth)'}"
    )

    # ---------- CHECK 2 (info): per-term loss magnitudes ----------
    _banner("CHECK 2: per-term loss magnitudes (informational)")
    class_weights = torch.ones(n_classes, dtype=torch.float, device=device)
    main_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=train_cfg.get("label_smoothing", 0.1),
    )
    ce_val = main_criterion(cls_logits, batch.y).item()
    dh_val = float("nan")
    n_valid_dh = 0
    if (
        train_cfg.get("use_deephit", True)
        and hasattr(batch, "os_event")
        and batch.os_event is not None
    ):
        event = batch.os_event.float()
        censored = 1.0 - event
        valid_dh = ~torch.isnan(event)
        n_valid_dh = int(valid_dh.sum().item())
        if valid_dh.any():
            dh_val = deephit_loss(
                bin_logits[valid_dh], batch.y[valid_dh], censored[valid_dh]
            ).item()
    w_aux = float(train_cfg.get("deephit_aux_weight", 0.3))
    logger.info(f"  CE (cls_logits, y)                 = {ce_val:.4f}")
    logger.info(f"  DeepHit (bin_logits, y, censored)  = {dh_val:.4f}  "
                f"(on {n_valid_dh}/{batch.num_graphs} non-NaN samples)")
    logger.info(f"  deephit_aux_weight                  = {w_aux:.2f}")
    logger.info(f"  post-weight contribution: CE={ce_val:.3f}  DH={w_aux*dh_val:.3f}")

    # ---------- INV-A: cls_logits gradient under pure CE ----------
    _banner("INV-A: cls_logits gradient under PURE CE backward")
    model.zero_grad()
    cls_logits2, _, _, _, _, _bin2 = model(
        batch.x, batch.edge_index, batch.batch, batch.clinical,
        gene_idx=gene_idx,
    )
    ce_only = main_criterion(cls_logits2, batch.y)
    ce_only.backward()
    grads_a = _grad_norms_by_prefix(
        model,
        ("fc.", "survival_bin_head.", "gnn.", "cox_head.", "ordinal_head.", "aux_head."),
    )
    logger.info(f"  backward target: CE(cls_logits, y) = {ce_only.item():.4f}")
    for pref, g in grads_a.items():
        logger.info(f"    |grad| {pref:24s} = {g:.4e}")
    inv_a_pass = grads_a["fc."] > 1e-6 and grads_a["gnn."] > 1e-6
    # Under pure CE, the bin head is disconnected from the loss -> expect zero.
    bin_isolated = grads_a["survival_bin_head."] < 1e-10
    logger.info(
        f"  INV-A: fc grad {grads_a['fc.']:.2e} > 0, gnn grad {grads_a['gnn.']:.2e} > 0 "
        f"-> {'PASS' if inv_a_pass else 'FAIL'}"
    )
    logger.info(
        f"  (side check) bin_head grad under CE-only = {grads_a['survival_bin_head.']:.2e} "
        f"(expect ~0 since CE doesn't touch it) -> "
        f"{'OK' if bin_isolated else 'LEAK'}"
    )

    # ---------- INV-C: DeepHit lands on bin head, not classifier ----------
    _banner("INV-C: DeepHit-only backward lands on bin head, NOT self.fc")
    model.zero_grad()
    cls_logits3, _, _, _, _, bin_logits3 = model(
        batch.x, batch.edge_index, batch.batch, batch.clinical,
        gene_idx=gene_idx,
    )
    if (
        hasattr(batch, "os_event")
        and batch.os_event is not None
        and (~torch.isnan(batch.os_event.float())).any()
    ):
        event = batch.os_event.float()
        censored = 1.0 - event
        valid_dh = ~torch.isnan(event)
        dh_only = deephit_loss(
            bin_logits3[valid_dh], batch.y[valid_dh], censored[valid_dh]
        )
    else:
        dh_only = bin_logits3.sum() * 0.0  # keep the graph alive; nothing to backward
        logger.warning("  No valid DeepHit rows in batch -- INV-C is inconclusive.")
    dh_only.backward()
    grads_c = _grad_norms_by_prefix(
        model,
        ("fc.", "survival_bin_head.", "gnn.", "cox_head.", "ordinal_head.", "aux_head."),
    )
    logger.info(f"  backward target: DeepHit(bin_logits, y, censored) = {dh_only.item():.4f}")
    for pref, g in grads_c.items():
        logger.info(f"    |grad| {pref:24s} = {g:.4e}")
    bin_has_grad = grads_c["survival_bin_head."] > 1e-6
    fc_is_clean = grads_c["fc."] < 1e-10
    inv_c_pass = bin_has_grad and fc_is_clean
    logger.info(
        f"  INV-C: bin_head grad {grads_c['survival_bin_head.']:.2e} > 0 AND "
        f"fc grad {grads_c['fc.']:.2e} ~ 0 -> "
        f"{'PASS' if inv_c_pass else 'FAIL (DeepHit is leaking into cls_logits)'}"
    )

    # ---------- Verdict ----------
    _banner("VERDICT")
    logger.info(f"  INV-A (CE drives fc + gnn grads):           {'PASS' if inv_a_pass else 'FAIL'}")
    logger.info(f"  INV-B (cls_logits argmax diverse at init):  {'PASS' if inv_b_pass else 'FAIL'}")
    logger.info(f"  INV-C (DeepHit -> bin_head only, not fc):   {'PASS' if inv_c_pass else 'FAIL'}")
    all_pass = inv_a_pass and inv_b_pass and inv_c_pass
    if all_pass:
        logger.info("  ALL invariants hold. Safe to launch 2-fold CV.")
    else:
        logger.error("  One or more invariants FAILED. Do not launch CV; fix first.")
    logger.info("  No optimizer.step() applied; exiting clean.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
