"""
Stage 5.5: Visualizations

Generates all required figures:
    1. Kaplan-Meier survival curves (predicted High vs Low risk)
    2. t-SNE / UMAP of patient embeddings colored by survival class
    3. ROC curves (per class + macro)
    4. Confusion matrix
    5. Training loss / validation AUC curves
    6. BioKG subgraph heatmap (handled in explain.py)
    7. Bar chart: model comparison

References:
    - KM curves: lifelines
    - t-SNE/UMAP: sklearn / umap-learn
    - Visualization standards: Choudhry et al., 2025
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from lifelines import KaplanMeierFitter

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Color scheme
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
CLASS_NAMES = ["<1yr", "1-3yr", "3-5yr", ">5yr"]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def plot_kaplan_meier(
    os_time: np.ndarray,
    os_event: np.ndarray,
    risk_groups: np.ndarray,
    output_path: str = "results/kaplan_meier.png",
):
    """Plot Kaplan-Meier survival curves for predicted risk groups.

    Splits patients into High-Risk vs Low-Risk based on model predictions.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    valid = ~np.isnan(os_time) & ~np.isnan(os_event) & (os_time > 0)
    os_time = os_time[valid]
    os_event = os_event[valid]
    risk_groups = risk_groups[valid]

    unique_groups = np.unique(risk_groups)
    colors = COLORS[: len(unique_groups)]

    for group, color in zip(sorted(unique_groups), colors):
        mask = risk_groups == group
        if mask.sum() < 2:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            os_time[mask] / 365.25,  # Convert to years
            event_observed=os_event[mask],
            label=CLASS_NAMES[int(group)] if int(group) < len(CLASS_NAMES) else f"Group {group}",
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Kaplan-Meier Survival Curves by Predicted Risk Group", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Kaplan-Meier plot to {output_path}")


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str = "results/tsne_embeddings.png",
    perplexity: int = 30,
    seed: int = 42,
):
    """Plot t-SNE visualization of patient embeddings colored by survival class."""
    logger.info("Computing t-SNE projection...")

    # Handle NaN in embeddings
    valid = ~np.isnan(labels) & ~np.any(np.isnan(embeddings), axis=1)
    embeddings = embeddings[valid]
    labels = labels[valid].astype(int)

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1), random_state=seed)
    proj = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=COLORS[cls % len(COLORS)], label=label,
            alpha=0.6, s=40, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE of Patient Embeddings (colored by survival class)", fontsize=14)
    ax.legend(fontsize=11, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved t-SNE plot to {output_path}")


def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str = "results/umap_embeddings.png",
    seed: int = 42,
):
    """Plot UMAP visualization of patient embeddings (fallback if umap available)."""
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP plot.")
        return

    logger.info("Computing UMAP projection...")

    valid = ~np.isnan(labels) & ~np.any(np.isnan(embeddings), axis=1)
    embeddings = embeddings[valid]
    labels = labels[valid].astype(int)

    reducer = umap.UMAP(n_components=2, random_state=seed)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=COLORS[cls % len(COLORS)], label=label,
            alpha=0.6, s=40, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("UMAP of Patient Embeddings (colored by survival class)", fontsize=14)
    ax.legend(fontsize=11, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved UMAP plot to {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = 4,
    output_path: str = "results/roc_curves.png",
):
    """Plot ROC curves per class and macro-average."""
    fig, ax = plt.subplots(figsize=(10, 8))

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # Per-class ROC
    fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
    for cls in range(num_classes):
        if y_bin[:, cls].sum() > 0 and y_bin[:, cls].sum() < len(y_bin):
            fpr_dict[cls], tpr_dict[cls], _ = roc_curve(y_bin[:, cls], y_probs[:, cls])
            roc_auc_dict[cls] = auc(fpr_dict[cls], tpr_dict[cls])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
            ax.plot(
                fpr_dict[cls], tpr_dict[cls],
                color=COLORS[cls], linewidth=2,
                label=f"{label} (AUC = {roc_auc_dict[cls]:.3f})"
            )

    # Macro-average ROC
    if fpr_dict:
        all_fpr = np.unique(np.concatenate([fpr_dict[cls] for cls in fpr_dict]))
        mean_tpr = np.zeros_like(all_fpr)
        for cls in fpr_dict:
            mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
        mean_tpr /= len(fpr_dict)
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr, mean_tpr,
            color="navy", linewidth=3, linestyle="--",
            label=f"Macro Average (AUC = {macro_auc:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Multi-class Survival Prediction", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "results/confusion_matrix.png",
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES[:cm.shape[0]],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Survival Class Prediction", fontsize=14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_training_curves(
    history: dict,
    output_path: str = "results/training_curves.png",
):
    """Plot training loss and validation AUC curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    if "train_loss" in history and "val_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title("Training & Validation Loss", fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    if "train_accuracy" in history and "val_accuracy" in history:
        axes[1].plot(epochs, history["train_accuracy"], "b-", label="Train Acc", linewidth=2)
        axes[1].plot(epochs, history["val_accuracy"], "r-", label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Training & Validation Accuracy", fontsize=13)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

    # AUC / C-index curves
    if "val_auc_roc" in history:
        axes[2].plot(epochs, history["val_auc_roc"], "g-", label="Val AUC-ROC", linewidth=2)
        if "val_c_index" in history:
            axes[2].plot(epochs, history["val_c_index"], "m-", label="Val C-index", linewidth=2)
        axes[2].set_xlabel("Epoch", fontsize=12)
        axes[2].set_ylabel("Score", fontsize=12)
        axes[2].set_title("Validation AUC-ROC & C-index", fontsize=13)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def plot_model_comparison(
    model_metrics: dict,
    metric_name: str = "accuracy",
    output_path: str = "results/model_comparison.png",
):
    """Bar chart comparing model performance.

    Args:
        model_metrics: Dict mapping model_name -> dict with metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Metrics to compare
    compare_metrics = ["accuracy", "f1_macro", "auc_roc"]
    titles = ["Accuracy", "F1 Score (Macro)", "AUC-ROC"]

    for ax, metric, title in zip(axes, compare_metrics, titles):
        models = []
        values = []
        stds = []

        for model_name, metrics in model_metrics.items():
            if metric in metrics:
                val = metrics[metric]
                if isinstance(val, dict):
                    values.append(val.get("mean", 0))
                    stds.append(val.get("std", 0))
                else:
                    values.append(val)
                    stds.append(0)
                models.append(model_name)

        if not models:
            continue

        x = np.arange(len(models))
        bars = ax.bar(x, values, yerr=stds, capsize=5, color=COLORS[:len(models)], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison to {output_path}")


def plot_ablation_study(
    ablation_results: dict,
    output_path: str = "results/ablation_study.png",
):
    """Bar chart showing progressive improvement in ablation study.

    Reference: Gao et al., 2021
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = list(ablation_results.keys())
    stage_labels = {
        "genes_only": "Genes Only",
        "genes_clinical": "Genes + Clinical",
        "genes_clinical_biokg": "Genes + Clinical\n+ BioKG",
        "full_model": "Full Model\n(+ LLM + RF)",
    }

    labels = [stage_labels.get(s, s) for s in stages]
    accuracies = [ablation_results[s].get("accuracy", 0) for s in stages]

    x = np.arange(len(stages))
    colors_ablation = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]

    bars = ax.bar(x, accuracies, color=colors_ablation[:len(stages)], edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Ablation Study — Progressive Component Addition", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ablation study to {output_path}")


def generate_all_visualizations(config: dict, training_results: dict) -> list:
    """Generate all required visualizations.

    Returns list of generated file paths.
    """
    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)
    generated_files = []

    fold_results = training_results.get("fold_results", [])
    if not fold_results:
        logger.warning("No fold results available for visualization")
        return generated_files

    # Use best fold (fold 0) for visualizations
    best_fold = fold_results[0]

    # 1. Training curves
    if "history" in best_fold:
        path = os.path.join(results_dir, "training_curves.png")
        plot_training_curves(best_fold["history"], path)
        generated_files.append(path)

    # 2. Kaplan-Meier (from validation data)
    if "val_emb" in best_fold and "val_labels" in best_fold:
        val_labels = best_fold["val_labels"]

        # Get OS time and event from dataset
        dataset_info = training_results.get("dataset", {})
        if "dataset" in dataset_info:
            ds = dataset_info["dataset"]
            splits = dataset_info.get("splits", training_results.get("splits", []))
            if splits:
                _, val_idx = splits[0]
                if ds.os_time is not None:
                    path = os.path.join(results_dir, "kaplan_meier.png")
                    plot_kaplan_meier(ds.os_time[val_idx], ds.os_event[val_idx], val_labels, path)
                    generated_files.append(path)

    # 3. t-SNE of patient embeddings
    if "val_emb" in best_fold and "val_labels" in best_fold:
        path = os.path.join(results_dir, "tsne_embeddings.png")
        plot_tsne_embeddings(best_fold["val_emb"], best_fold["val_labels"], path)
        generated_files.append(path)

        # Also try UMAP
        path = os.path.join(results_dir, "umap_embeddings.png")
        plot_umap_embeddings(best_fold["val_emb"], best_fold["val_labels"], path)
        generated_files.append(path)

    # 4. Model comparison bar chart
    comparison = {}
    if "gat_summary" in training_results:
        comparison["GAT+Clinical"] = training_results["gat_summary"]
    if "rf_summary" in training_results:
        comparison["GAT+RF (Hybrid)"] = training_results["rf_summary"]

    if comparison:
        path = os.path.join(results_dir, "model_comparison.png")
        plot_model_comparison(comparison, output_path=path)
        generated_files.append(path)

    logger.info(f"Generated {len(generated_files)} visualization files")
    return generated_files


if __name__ == "__main__":
    config = load_config()
    results_dir = config["paths"]["results"]

    # Load training results if available
    results_path = os.path.join(results_dir, "training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            training_results = json.load(f)
        generate_all_visualizations(config, training_results)
    else:
        logger.info("Run train.py first to generate results for visualization")
