"""
Stage 4: Training Loop with Cross-Validation

Implements:
    1. GAT training with multi-task loss (main + auxiliary)
    2. 5-fold stratified cross-validation
    3. Hybrid GAT -> Calibrated Random Forest pipeline
    4. Early stopping and learning rate scheduling

References:
    - Multi-task loss: Rahaman et al., 2023 (main + 0.3 * aux_loss)
    - CV strategy: 5-fold stratified (Alharbi et al., 2025)
    - Optimizer: Adam, lr=0.001, weight_decay=1e-4 (Alharbi et al., 2025)
    - Scheduler: ReduceLROnPlateau (Alharbi et al., 2025)
    - Hybrid GNN+RF: Palmal et al., 2024
"""

import os
import copy
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index

from src.model import build_model, HybridModel
from src.dataset import (
    BreastCancerGraphDataset,
    get_dataloaders,
    build_dataset,
    compute_class_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def deephit_loss(
    logits: torch.Tensor,
    bin_idx: torch.Tensor,
    censored: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DeepHit-style discrete-time survival loss (Lee et al., 2018).

    Mass vs. ranking trade-off: this implements just the mass term (L_1 in
    the paper). The ranking term is handled separately by the Cox head.

    Semantics per-patient:
      - Uncensored (event observed at bin k): maximize P(event=k).
      - Censored (last-seen at bin k): maximize P(survive past k)
        = sum_{j>k} P(event=j), i.e. the patient survived all bins up to k.

    Args:
        logits: [B, K] unnormalized class scores (main classification head).
        bin_idx: [B] int, which survival bin the patient is in.
        censored: [B] float 1.0 if censored, 0.0 if event observed.

    Returns:
        Scalar loss averaged over the batch. Returns 0 when the batch has
        no valid targets (bin_idx < 0).
    """
    valid = bin_idx >= 0
    if valid.sum() < 1:
        return torch.zeros((), device=logits.device)

    logits = logits[valid]
    bin_idx = bin_idx[valid].long()
    censored = censored[valid].float()

    probs = torch.softmax(logits, dim=-1).clamp(min=eps, max=1.0 - eps)  # [B, K]

    # P(event at bin_idx) -- uncensored signal
    event_p = probs.gather(1, bin_idx.unsqueeze(1)).squeeze(1)  # [B]
    event_ll = -torch.log(event_p + eps)

    # P(survive > bin_idx) = 1 - cumsum_up_to_bin_idx
    cum = probs.cumsum(dim=-1)  # [B, K]
    surv_p = 1.0 - cum.gather(1, bin_idx.unsqueeze(1)).squeeze(1)
    surv_p = surv_p.clamp(min=eps)
    surv_ll = -torch.log(surv_p)

    # Combine: censored -> surv_ll, uncensored -> event_ll
    loss = (1.0 - censored) * event_ll + censored * surv_ll
    return loss.mean()


def cox_ph_loss(log_hazards: torch.Tensor, times: torch.Tensor, events: torch.Tensor,
                eps: float = 1e-7) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow approximation).

    Reference: Katzman et al., 2018 -- DeepSurv.

    Args:
        log_hazards: [N] predicted scalar log-hazard per patient.
        times: [N] observed event/censoring times (>0).
        events: [N] binary indicator -- 1 if event observed, 0 if censored.

    Returns:
        Scalar loss averaged over observed events. Returns 0 when there are
        no events in the batch (gradient still valid because the subtraction
        short-circuits).
    """
    # Drop invalid rows (e.g. SMOTE-synthetic samples with NaN times)
    valid = (~torch.isnan(times)) & (~torch.isnan(events.float())) & (times > 0)
    if valid.sum() < 2:
        return torch.zeros((), device=log_hazards.device)

    h = log_hazards[valid]
    t = times[valid]
    e = events[valid].float()

    if e.sum() < 1:
        return torch.zeros((), device=log_hazards.device)

    # Sort by time descending so prefix cumsum gives the risk set
    _, sort_idx = torch.sort(t, descending=True)
    h = h[sort_idx]
    e = e[sort_idx]

    # Numerically stable cumulative logsumexp
    hmax = h.max().detach()
    log_cum = torch.log(torch.cumsum(torch.exp(h - hmax), dim=0) + eps) + hmax

    pll = (h - log_cum) * e
    n_events = e.sum().clamp(min=1.0)
    return -(pll.sum() / n_events)


def train_one_epoch(
    model: HybridModel,
    loader,
    optimizer,
    device,
    aux_loss_weight: float = 0.3,
    cox_loss_weight: float = 0.5,
    ordinal_loss_weight: float = 0.2,
    deephit_loss_weight: float = 1.0,
    label_smoothing: float = 0.1,
    class_weights: torch.Tensor = None,
    use_deephit: bool = True,
) -> dict:
    """Train for one epoch with multi-task survival objective.

    When use_deephit=True (recommended): main classification uses DeepHit
    discrete-time survival loss which properly handles censored patients:
        L = deephit_w * DeepHit(logits, bin, censored)
          + aux_w     * CE(aux_stage, ignore_index=-1)
          + cox_w     * CoxPartialLikelihood(log_hazard, T, E)
          + ord_w     * SmoothL1(ordinal_score, y_float)

    When use_deephit=False (legacy): main uses class-weighted smoothed CE.

    The Cox and ordinal heads directly optimize survival ordering, which
    was the biggest gap in the first run (GAT C-index=0.38 vs Cox=0.748).
    DeepHit adds censoring-aware mass learning on top.
    """
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    total_cox_loss = 0.0
    total_ord_loss = 0.0
    correct = 0
    total = 0

    main_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )
    aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    ordinal_criterion = nn.SmoothL1Loss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        gene_idx = getattr(batch, "gene_idx", None)
        main_out, aux_out, _, cox_out, ordinal_out = model(
            batch.x, batch.edge_index, batch.batch, batch.clinical,
            gene_idx=gene_idx,
        )

        # --- Main classification loss (DeepHit if requested and metadata present) ---
        if (
            use_deephit
            and hasattr(batch, "os_event")
            and batch.os_event is not None
        ):
            # censored = 1 - event; skip synthetic SMOTE rows (NaN event)
            event = batch.os_event.float()
            censored = 1.0 - event
            # Mask NaN (synthetic samples): train only on real patients for DeepHit.
            valid_dh = ~torch.isnan(event)
            if valid_dh.any():
                main_loss = deephit_loss(
                    main_out[valid_dh],
                    batch.y[valid_dh],
                    censored[valid_dh],
                )
                # Blend with class-weighted smoothed CE on the full batch so
                # synthetic SMOTE samples still contribute gradient (otherwise
                # they're dead weight in the batch). Scale deephit by its weight.
                main_loss = deephit_loss_weight * main_loss + (1.0 - deephit_loss_weight) * main_criterion(main_out, batch.y)
            else:
                main_loss = main_criterion(main_out, batch.y)
        else:
            main_loss = main_criterion(main_out, batch.y)

        # --- Auxiliary tumor-stage loss (skip unresolved rows) ---
        aux_loss = torch.zeros((), device=device)
        if aux_loss_weight > 0 and hasattr(batch, "tumor_stage") and batch.tumor_stage is not None:
            valid_aux = batch.tumor_stage >= 0
            if valid_aux.any():
                aux_loss = aux_criterion(aux_out[valid_aux], batch.tumor_stage[valid_aux])

        # --- Cox partial likelihood on OS.time / OS.event ---
        cox_loss = torch.zeros((), device=device)
        if cox_loss_weight > 0 and hasattr(batch, "os_time") and batch.os_time is not None:
            cox_loss = cox_ph_loss(cox_out, batch.os_time.float(), batch.os_event.float())

        # --- Ordinal regression: predict class index as a continuous score ---
        ordinal_loss = torch.zeros((), device=device)
        if ordinal_loss_weight > 0:
            ordinal_loss = ordinal_criterion(ordinal_out, batch.y.float())

        loss = (
            main_loss
            + aux_loss_weight * aux_loss
            + cox_loss_weight * cox_loss
            + ordinal_loss_weight * ordinal_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = batch.num_graphs
        total_loss += loss.item() * bs
        total_main_loss += main_loss.item() * bs
        total_aux_loss += float(aux_loss.item()) * bs
        total_cox_loss += float(cox_loss.item()) * bs
        total_ord_loss += float(ordinal_loss.item()) * bs

        pred = main_out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += bs

    n = total if total > 0 else 1
    return {
        "loss": total_loss / n,
        "main_loss": total_main_loss / n,
        "aux_loss": total_aux_loss / n,
        "cox_loss": total_cox_loss / n,
        "ordinal_loss": total_ord_loss / n,
        "accuracy": correct / n,
    }


@torch.no_grad()
def evaluate(model: HybridModel, loader, device) -> dict:
    """Evaluate model on validation data.

    Computes: loss, accuracy, AUC-ROC, C-index (from both classifier-expected
    class and from the Cox hazard head -- we report whichever is higher).
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_cox = []
    all_ordinal = []
    all_os_time = []
    all_os_event = []
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)

        gene_idx = getattr(batch, "gene_idx", None)
        main_out, _, _, cox_out, ordinal_out = model(
            batch.x, batch.edge_index, batch.batch, batch.clinical,
            gene_idx=gene_idx,
        )
        loss = criterion(main_out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs

        probs = F.softmax(main_out, dim=1)
        pred = main_out.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_cox.extend(cox_out.cpu().numpy())
        all_ordinal.extend(ordinal_out.cpu().numpy())

        if hasattr(batch, "os_time") and batch.os_time is not None:
            all_os_time.extend(batch.os_time.cpu().numpy())
            all_os_event.extend(batch.os_event.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_cox = np.array(all_cox)
    all_ordinal = np.array(all_ordinal)

    n = total if total > 0 else 1
    metrics = {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
    }

    # AUC-ROC (multi-class, one-vs-rest)
    try:
        if len(np.unique(all_labels)) > 1:
            metrics["auc_roc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        else:
            metrics["auc_roc"] = 0.0
    except ValueError:
        metrics["auc_roc"] = 0.0

    # C-index (concordance index) — THE standard for survival (Zohari et al., 2025)
    # lifelines convention: higher predicted_scores = longer survival time.
    if all_os_time and all_os_event:
        os_time = np.array(all_os_time)
        os_event = np.array(all_os_event)
        valid = ~np.isnan(os_time) & ~np.isnan(os_event) & (os_time > 0)

        if valid.sum() > 10:
            # 1) Classifier expected-class (higher class idx = longer survival)
            predicted_survival = np.sum(all_probs * np.arange(all_probs.shape[1]), axis=1)
            # 2) Cox head log-hazard (higher hazard = shorter survival, so negate)
            cox_score = -all_cox
            # 3) Ordinal head score (trained on float label -> already higher = longer)
            ord_score = all_ordinal

            def _safe_cidx(score):
                try:
                    return concordance_index(os_time[valid], score[valid], os_event[valid])
                except Exception:
                    return 0.5

            c_cls = _safe_cidx(predicted_survival)
            c_cox = _safe_cidx(cox_score)
            c_ord = _safe_cidx(ord_score)

            metrics["c_index_cls"] = c_cls
            metrics["c_index_cox"] = c_cox
            metrics["c_index_ord"] = c_ord
            # Report best head as the headline c_index so early stopping /
            # model selection tracks the most useful survival signal.
            metrics["c_index"] = max(c_cls, c_cox, c_ord)
        else:
            metrics["c_index"] = 0.5

    return metrics


@torch.no_grad()
def extract_embeddings(model: HybridModel, loader, device) -> tuple:
    """Extract GNN embeddings for all patients in the loader.

    Used for Hybrid GAT -> RF pipeline (Palmal et al., 2024).

    Returns:
        (embeddings, labels, clinical_features)
    """
    model.eval()
    all_emb = []
    all_labels = []
    all_clinical = []

    for batch in loader:
        batch = batch.to(device)
        gene_idx = getattr(batch, "gene_idx", None)
        emb = model.extract_embeddings(
            batch.x, batch.edge_index, batch.batch, batch.clinical, gene_idx=gene_idx,
        )
        all_emb.append(emb.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_clinical.append(batch.clinical.cpu().numpy())

    return (
        np.concatenate(all_emb),
        np.concatenate(all_labels),
        np.concatenate(all_clinical),
    )


def train_calibrated_rf(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    config: dict,
) -> tuple:
    """Train Calibrated Random Forest on GNN embeddings.

    Reference: Palmal et al., 2024 — hybrid GCN+RF approach.

    Returns:
        (calibrated_rf_model, val_metrics)
    """
    rf_cfg = config["hybrid"]

    rf = RandomForestClassifier(
        n_estimators=rf_cfg["rf_n_estimators"],
        max_depth=rf_cfg.get("rf_max_depth"),
        min_samples_split=rf_cfg.get("rf_min_samples_split", 5),
        random_state=config["training"]["seed"],
        n_jobs=-1,
    )

    if rf_cfg.get("rf_calibrated", True):
        model = CalibratedClassifierCV(rf, cv=3, method="isotonic")
    else:
        model = rf

    model.fit(train_emb, train_labels)

    # Evaluate
    val_pred = model.predict(val_emb)
    val_probs = model.predict_proba(val_emb)

    metrics = {"accuracy": accuracy_score(val_labels, val_pred)}

    try:
        if len(np.unique(val_labels)) > 1:
            metrics["auc_roc"] = roc_auc_score(val_labels, val_probs, multi_class="ovr", average="macro")
    except ValueError:
        pass

    logger.info(f"RF metrics: {metrics}")
    return model, metrics


def train_fold(
    fold: int,
    model: HybridModel,
    dataset: BreastCancerGraphDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Train one CV fold end-to-end (GAT + RF).

    Returns:
        Dictionary with best model, metrics, and RF model.
    """
    train_cfg = config["training"]

    # Create data loaders (SMOTE on training set only -- skipped if dims too large)
    train_loader, val_loader = get_dataloaders(
        dataset, train_idx, val_idx,
        batch_size=train_cfg["batch_size"],
        smote=True,
        smote_strategy=config["data"]["smote_strategy"],
        seed=train_cfg["seed"],
    )

    # Compute class weights for balanced loss (always used, critical when SMOTE is skipped)
    train_labels = dataset.survival_labels[train_idx]
    cw = compute_class_weights(train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float, device=device)
    logger.info(f"  Class weights: {dict(enumerate(cw.tolist()))}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    # Early-stop metric: 'auc_roc' (maximize) is more stable than val_loss,
    # especially with the multi-task loss whose magnitude varies epoch to epoch.
    es_metric = train_cfg.get("early_stop_metric", "auc_roc")
    es_mode = "max" if es_metric in ("auc_roc", "accuracy", "c_index") else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=es_mode, factor=0.5, patience=10
    )

    best_score = -float("inf") if es_mode == "max" else float("inf")
    best_model_state = None
    patience_counter = 0
    history = defaultdict(list)

    logger.info(
        f"--- Fold {fold + 1}/{config['training']['cv_folds']} "
        f"(early-stop on val_{es_metric} {es_mode}) ---"
    )

    for epoch in range(train_cfg["epochs"]):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            aux_loss_weight=train_cfg.get("aux_loss_weight", 0.3),
            cox_loss_weight=train_cfg.get("cox_loss_weight", 0.5),
            ordinal_loss_weight=train_cfg.get("ordinal_loss_weight", 0.2),
            deephit_loss_weight=train_cfg.get("deephit_loss_weight", 1.0),
            label_smoothing=train_cfg.get("label_smoothing", 0.1),
            class_weights=class_weights,
            use_deephit=train_cfg.get("use_deephit", True),
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Learning rate scheduling follows the early-stop metric
        scheduler.step(val_metrics.get(es_metric, val_metrics["loss"]))

        # Track history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        # Early stopping on chosen metric
        score = val_metrics.get(es_metric, val_metrics["loss"])
        improved = (score > best_score) if es_mode == "max" else (score < best_score)
        if improved:
            best_score = score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            c_idx = val_metrics.get("c_index", "N/A")
            c_str = f"{c_idx:.4f}" if isinstance(c_idx, float) else c_idx
            logger.info(
                f"  Epoch {epoch + 1:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(main={train_metrics['main_loss']:.3f} "
                f"cox={train_metrics['cox_loss']:.3f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val AUC: {val_metrics.get('auc_roc', 0):.4f} | "
                f"Val C-idx: {c_str}"
            )

        if patience_counter >= train_cfg["patience"]:
            logger.info(f"  Early stopping at epoch {epoch + 1} (best {es_metric}={best_score:.4f})")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final validation metrics
    final_metrics = evaluate(model, val_loader, device)
    logger.info(f"  Best val metrics: {final_metrics}")

    # Hybrid: Extract embeddings and train Calibrated RF
    logger.info("  Training Calibrated Random Forest on GNN embeddings...")
    train_emb, train_labels, _ = extract_embeddings(model, train_loader, device)
    val_emb, val_labels, _ = extract_embeddings(model, val_loader, device)

    rf_model, rf_metrics = train_calibrated_rf(train_emb, train_labels, val_emb, val_labels, config)

    return {
        "model_state": best_model_state,
        "gat_metrics": final_metrics,
        "rf_model": rf_model,
        "rf_metrics": rf_metrics,
        "history": dict(history),
        "val_emb": val_emb,
        "val_labels": val_labels,
    }


def run_training(config: dict) -> dict:
    """Run the complete training pipeline with cross-validation.

    Returns:
        Dictionary with all fold results and aggregated metrics.
    """
    seed = config["training"]["seed"]
    set_seed(seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Build dataset
    logger.info("=" * 60)
    logger.info("Building dataset...")
    logger.info("=" * 60)
    data = build_dataset(config)
    dataset = data["dataset"]
    splits = data["splits"]
    clinical_dim = data["clinical_dim"]
    num_genes = data["n_genes"]

    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    all_fold_results = []
    all_gat_metrics = defaultdict(list)
    all_rf_metrics = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Build fresh model for each fold
        model = build_model(
            config, clinical_dim=clinical_dim, device=str(device),
            num_genes=num_genes,
        )

        fold_result = train_fold(fold, model, dataset, train_idx, val_idx, config, device)
        all_fold_results.append(fold_result)

        for k, v in fold_result["gat_metrics"].items():
            all_gat_metrics[k].append(v)
        for k, v in fold_result["rf_metrics"].items():
            all_rf_metrics[k].append(v)

        # Save fold model
        torch.save(fold_result["model_state"], os.path.join(results_dir, f"model_fold{fold}.pt"))

    # Aggregate metrics
    logger.info("=" * 60)
    logger.info("Cross-Validation Results (GAT)")
    logger.info("=" * 60)
    gat_summary = {}
    for k, values in all_gat_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        gat_summary[k] = {"mean": mean, "std": std}
        logger.info(f"  {k}: {mean:.4f} +/- {std:.4f}")

    logger.info("=" * 60)
    logger.info("Cross-Validation Results (Calibrated RF)")
    logger.info("=" * 60)
    rf_summary = {}
    for k, values in all_rf_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        rf_summary[k] = {"mean": mean, "std": std}
        logger.info(f"  {k}: {mean:.4f} +/- {std:.4f}")

    # Save results
    results = {
        "gat_summary": gat_summary,
        "rf_summary": rf_summary,
        "fold_results": [
            {
                "gat_metrics": fr["gat_metrics"],
                "rf_metrics": fr["rf_metrics"],
                "history": fr["history"],
            }
            for fr in all_fold_results
        ],
        "config": config,
    }

    with open(os.path.join(results_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return {
        "fold_results": all_fold_results,
        "gat_summary": gat_summary,
        "rf_summary": rf_summary,
        "dataset": data,
    }


if __name__ == "__main__":
    config = load_config()
    results = run_training(config)
