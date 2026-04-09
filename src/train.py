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


def train_one_epoch(
    model: HybridModel,
    loader,
    optimizer,
    device,
    aux_loss_weight: float = 0.3,
    class_weights: torch.Tensor = None,
) -> dict:
    """Train for one epoch.

    Loss = CrossEntropy(main) + aux_weight * CrossEntropy(aux_stage)
    """
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    correct = 0
    total = 0

    main_criterion = nn.CrossEntropyLoss(weight=class_weights)
    aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        main_out, aux_out, _ = model(
            batch.x, batch.edge_index, batch.batch, batch.clinical
        )

        # Main loss: survival class prediction
        main_loss = main_criterion(main_out, batch.y)

        # Auxiliary loss: tumor stage prediction
        aux_loss = torch.tensor(0.0, device=device)
        if hasattr(batch, "tumor_stage") and batch.tumor_stage is not None:
            valid_aux = batch.tumor_stage >= 0
            if valid_aux.any():
                aux_loss = aux_criterion(aux_out[valid_aux], batch.tumor_stage[valid_aux])

        # Combined loss
        loss = main_loss + aux_loss_weight * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_main_loss += main_loss.item() * batch.num_graphs
        total_aux_loss += aux_loss.item() * batch.num_graphs

        pred = main_out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    n = total if total > 0 else 1
    return {
        "loss": total_loss / n,
        "main_loss": total_main_loss / n,
        "aux_loss": total_aux_loss / n,
        "accuracy": correct / n,
    }


@torch.no_grad()
def evaluate(model: HybridModel, loader, device) -> dict:
    """Evaluate model on validation data.

    Computes: loss, accuracy, AUC-ROC, C-index.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_os_time = []
    all_os_event = []
    total = 0

    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)

        main_out, _, _ = model(batch.x, batch.edge_index, batch.batch, batch.clinical)
        loss = criterion(main_out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        total += batch.num_graphs

        probs = F.softmax(main_out, dim=1)
        pred = main_out.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        if hasattr(batch, "os_time") and batch.os_time is not None:
            all_os_time.extend(batch.os_time.cpu().numpy())
            all_os_event.extend(batch.os_event.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

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
    if all_os_time and all_os_event:
        os_time = np.array(all_os_time)
        os_event = np.array(all_os_event)

        # Use predicted risk score (higher class = higher risk inverted)
        # Risk = negative expected survival bin
        risk_scores = np.sum(all_probs * np.arange(all_probs.shape[1]), axis=1)
        risk_scores = -risk_scores  # Higher bin = better survival, so negate for risk

        valid = ~np.isnan(os_time) & ~np.isnan(os_event) & (os_time > 0)
        if valid.sum() > 10:
            try:
                metrics["c_index"] = concordance_index(
                    os_time[valid], risk_scores[valid], os_event[valid]
                )
            except Exception:
                metrics["c_index"] = 0.5
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
        emb = model.extract_embeddings(batch.x, batch.edge_index, batch.batch, batch.clinical)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = defaultdict(list)

    logger.info(f"--- Fold {fold + 1}/{config['training']['cv_folds']} ---")

    for epoch in range(train_cfg["epochs"]):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            aux_loss_weight=train_cfg["aux_loss_weight"],
            class_weights=class_weights,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Track history
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            c_idx = val_metrics.get("c_index", "N/A")
            c_str = f"{c_idx:.4f}" if isinstance(c_idx, float) else c_idx
            logger.info(
                f"  Epoch {epoch + 1:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val AUC: {val_metrics.get('auc_roc', 0):.4f} | "
                f"Val C-idx: {c_str}"
            )

        if patience_counter >= train_cfg["patience"]:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
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

    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    all_fold_results = []
    all_gat_metrics = defaultdict(list)
    all_rf_metrics = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Build fresh model for each fold
        model = build_model(config, clinical_dim=clinical_dim, device=str(device))

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
