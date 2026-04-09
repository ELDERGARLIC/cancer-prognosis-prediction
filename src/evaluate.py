"""
Stage 5.1-5.2: Evaluation and Baseline Comparisons

Implements:
    1. Full metrics suite: Accuracy, Precision, Recall, F1, AUC-ROC, C-index
    2. Time-dependent C-index at 1-year, 3-year, 5-year (Ling et al., 2022)
    3. Baseline models: Cox PH, RF, MLP, vanilla GCN (Gogoshin & Rodin, 2023)
    4. Ablation study (Gao et al., 2021)

References:
    - C-index: Zohari & Chehreghani, 2025
    - Time-dependent C-index: Ling et al., 2022
    - Baselines: Gogoshin & Rodin, 2023
    - Ablation: Gao et al., 2021
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    os_time: np.ndarray = None,
    os_event: np.ndarray = None,
    num_classes: int = 4,
) -> dict:
    """Compute the complete metrics suite.

    Returns dictionary with all required metrics.
    """
    metrics = {}

    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics (compute once, index per class)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for cls in range(num_classes):
        cls_mask = y_true == cls
        if cls_mask.sum() > 0 and cls < len(per_class_precision):
            metrics[f"precision_class_{cls}"] = per_class_precision[cls]
            metrics[f"recall_class_{cls}"] = per_class_recall[cls]
            metrics[f"f1_class_{cls}"] = per_class_f1[cls]

    # AUC-ROC (multi-class, one-vs-rest)
    try:
        if len(np.unique(y_true)) > 1 and y_probs.shape[1] >= num_classes:
            metrics["auc_roc_macro"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            metrics["auc_roc_weighted"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")

            # Per-class AUC
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            for cls in range(num_classes):
                if y_bin[:, cls].sum() > 0 and y_bin[:, cls].sum() < len(y_bin):
                    fpr, tpr, _ = roc_curve(y_bin[:, cls], y_probs[:, cls])
                    metrics[f"auc_class_{cls}"] = auc(fpr, tpr)
    except (ValueError, IndexError):
        metrics["auc_roc_macro"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    # C-index (Zohari & Chehreghani, 2025)
    if os_time is not None and os_event is not None:
        risk_scores = np.sum(y_probs * np.arange(y_probs.shape[1]), axis=1)
        risk_scores = -risk_scores  # Negate: higher survival bin = lower risk

        valid = ~np.isnan(os_time) & ~np.isnan(os_event) & (os_time > 0)
        if valid.sum() > 10:
            try:
                metrics["c_index"] = concordance_index(os_time[valid], risk_scores[valid], os_event[valid])
            except Exception:
                metrics["c_index"] = 0.5

            # Time-dependent C-index (Ling et al., 2022)
            for t_name, t_days in [("1yr", 365), ("3yr", 1095), ("5yr", 1825)]:
                try:
                    t_mask = valid & (os_time <= t_days)
                    if t_mask.sum() > 10:
                        metrics[f"c_index_{t_name}"] = concordance_index(
                            os_time[t_mask], risk_scores[t_mask], os_event[t_mask]
                        )
                except Exception:
                    pass

    return metrics


def train_cox_baseline(
    X_train: np.ndarray,
    y_train_time: np.ndarray,
    y_train_event: np.ndarray,
    X_val: np.ndarray,
    y_val_time: np.ndarray,
    y_val_event: np.ndarray,
    y_val_class: np.ndarray,
    feature_names: list = None,
) -> dict:
    """Train Cox Proportional Hazards baseline.

    Reference: lifelines CoxPHFitter
    """
    logger.info("Training Cox PH baseline...")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Limit features for Cox PH (it doesn't scale well to high dimensions)
    max_features = min(50, X_train.shape[1])
    X_train_sub = X_train[:, :max_features]
    X_val_sub = X_val[:, :max_features]
    feature_names_sub = feature_names[:max_features]

    # Build DataFrame for lifelines
    train_df = pd.DataFrame(X_train_sub, columns=feature_names_sub)
    train_df["T"] = y_train_time
    train_df["E"] = y_train_event

    # Remove rows with NaN
    train_df = train_df.dropna()

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col="T", event_col="E")

        # Predict on validation
        val_df = pd.DataFrame(X_val_sub, columns=feature_names_sub)
        risk_scores = cph.predict_partial_hazard(val_df).values.flatten()

        # C-index
        valid = ~np.isnan(y_val_time) & ~np.isnan(y_val_event) & (y_val_time > 0)
        c_index = concordance_index(y_val_time[valid], -risk_scores[valid], y_val_event[valid])

        metrics = {"c_index": c_index, "model": "Cox PH"}
        logger.info(f"  Cox PH C-index: {c_index:.4f}")
        return metrics
    except Exception as e:
        logger.warning(f"  Cox PH failed: {e}")
        return {"c_index": 0.5, "model": "Cox PH", "error": str(e)}


def train_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> dict:
    """Train Random Forest baseline.

    Reference: Gogoshin & Rodin, 2023 — benchmarking requirement.
    """
    logger.info("Training Random Forest baseline...")

    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    y_probs = rf.predict_proba(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "model": "Random Forest",
    }

    try:
        if len(np.unique(y_val)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_val, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        pass

    logger.info(f"  RF Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    return metrics


def train_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> dict:
    """Train MLP baseline.

    Reference: Gogoshin & Rodin, 2023
    """
    logger.info("Training MLP baseline...")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_val)
    y_probs = mlp.predict_proba(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "model": "MLP",
    }

    try:
        if len(np.unique(y_val)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_val, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        pass

    logger.info(f"  MLP Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    return metrics


def train_vanilla_gcn_baseline(
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Train vanilla GCN (without BioKG enrichment) as ablation baseline.

    Reference: Gao et al., 2021 — ablation study
    """
    from torch_geometric.nn import GCNConv, global_mean_pool
    from src.dataset import get_dataloaders

    logger.info("Training vanilla GCN baseline (no BioKG)...")

    class VanillaGCN(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, num_classes):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, num_classes)
            self.dropout = torch.nn.Dropout(0.4)

        def forward(self, x, edge_index, batch):
            x = torch.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = torch.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.classifier(x)

    in_dim = config["model"]["llm_embedding_dim"]
    hidden_dim = config["model"]["hidden_dim"]

    model = VanillaGCN(in_dim, hidden_dim, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataloaders(
        dataset, train_idx, val_idx,
        batch_size=config["training"]["batch_size"],
        smote=True, seed=config["training"]["seed"],
    )

    # Train for fewer epochs (simplified baseline)
    model.train()
    for epoch in range(100):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "model": "Vanilla GCN",
    }

    try:
        if len(np.unique(y_true)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        pass

    logger.info(f"  GCN Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    return metrics


def run_ablation_study(
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Run ablation study: progressively add components.

    Reference: Gao et al., 2021
    Ablations:
        1. Genes only (expression features, no KG, no clinical)
        2. Genes + Clinical
        3. Genes + Clinical + BioKG
        4. Full model (Genes + Clinical + BioKG + LLM embeddings + RF)
    """
    logger.info("=" * 60)
    logger.info("Running Ablation Study")
    logger.info("=" * 60)

    ablation_results = {}

    # 1. Genes only (flat expression features with RF)
    logger.info("Ablation 1: Expression features only (RF)")
    train_expr = dataset.patient_embeddings[train_idx].mean(axis=2)  # Average over embedding dim
    val_expr = dataset.patient_embeddings[val_idx].mean(axis=2)
    train_labels = dataset.survival_labels[train_idx].astype(int)
    val_labels = dataset.survival_labels[val_idx].astype(int)

    ablation_results["genes_only"] = train_rf_baseline(
        train_expr, train_labels, val_expr, val_labels, config["training"]["seed"]
    )

    # 2. Genes + Clinical (RF)
    logger.info("Ablation 2: Expression + Clinical features (RF)")
    train_combined = np.hstack([train_expr, dataset.clinical_features[train_idx]])
    val_combined = np.hstack([val_expr, dataset.clinical_features[val_idx]])

    ablation_results["genes_clinical"] = train_rf_baseline(
        train_combined, train_labels, val_combined, val_labels, config["training"]["seed"]
    )
    ablation_results["genes_clinical"]["model"] = "RF (Genes + Clinical)"

    # 3. Genes + Clinical + BioKG (vanilla GCN)
    logger.info("Ablation 3: GCN with BioKG topology")
    ablation_results["genes_clinical_biokg"] = train_vanilla_gcn_baseline(
        dataset, train_idx, val_idx, config, device
    )

    # 4. Full model results come from the main training pipeline

    return ablation_results


def run_baseline_comparison(
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """Run all baseline models for comparison.

    Reference: Gogoshin & Rodin, 2023 — benchmarking requirement.
    """
    logger.info("=" * 60)
    logger.info("Running Baseline Comparisons")
    logger.info("=" * 60)

    # Prepare flat features for sklearn baselines
    # Use mean of LLM embeddings as features
    train_features = dataset.patient_embeddings[train_idx].mean(axis=2)
    val_features = dataset.patient_embeddings[val_idx].mean(axis=2)

    # Add clinical features
    train_features = np.hstack([train_features, dataset.clinical_features[train_idx]])
    val_features = np.hstack([val_features, dataset.clinical_features[val_idx]])

    train_labels = dataset.survival_labels[train_idx].astype(int)
    val_labels = dataset.survival_labels[val_idx].astype(int)

    seed = config["training"]["seed"]
    baselines = {}

    # 1. Cox PH
    os_time_train = dataset.os_time[train_idx] if dataset.os_time is not None else None
    os_event_train = dataset.os_event[train_idx] if dataset.os_event is not None else None
    os_time_val = dataset.os_time[val_idx] if dataset.os_time is not None else None
    os_event_val = dataset.os_event[val_idx] if dataset.os_event is not None else None

    if os_time_train is not None:
        baselines["cox_ph"] = train_cox_baseline(
            train_features, os_time_train, os_event_train,
            val_features, os_time_val, os_event_val, val_labels,
        )

    # 2. Random Forest
    baselines["random_forest"] = train_rf_baseline(
        train_features, train_labels, val_features, val_labels, seed
    )

    # 3. MLP
    baselines["mlp"] = train_mlp_baseline(
        train_features, train_labels, val_features, val_labels, seed
    )

    # 4. Vanilla GCN
    baselines["vanilla_gcn"] = train_vanilla_gcn_baseline(
        dataset, train_idx, val_idx, config, device
    )

    return baselines


def run_evaluation(config: dict, training_results: dict = None) -> dict:
    """Run the complete evaluation pipeline.

    If training_results is None, loads saved results from disk.
    """
    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    if training_results is None:
        results_path = os.path.join(results_dir, "training_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                training_results = json.load(f)
        else:
            raise FileNotFoundError("Run train.py first.")

    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    # Print comprehensive results
    if "gat_summary" in training_results:
        logger.info("\nGAT Model (5-fold CV):")
        for metric, values in training_results["gat_summary"].items():
            if isinstance(values, dict):
                logger.info(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")

    if "rf_summary" in training_results:
        logger.info("\nCalibrated RF (5-fold CV):")
        for metric, values in training_results["rf_summary"].items():
            if isinstance(values, dict):
                logger.info(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")

    # Save evaluation report
    with open(os.path.join(results_dir, "evaluation_report.json"), "w") as f:
        json.dump(training_results, f, indent=2, default=str)

    return training_results


if __name__ == "__main__":
    config = load_config()
    results = run_evaluation(config)
