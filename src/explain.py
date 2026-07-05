"""
Stage 5.4: Explainability

Implements:
    1. GNNExplainer — identify top genes driving predictions
    2. SHAP values on Random Forest component
    3. Risk heatmap — BioKG subgraph with gene importance

References:
    - GNNExplainer: Ying et al., 2019 (via PyG)
    - SHAP: Lundberg & Lee, 2017
    - Explainability: Choudhry et al., 2025
    - SHAP on RF: Vaida et al., 2025
"""

import os
import json
import logging
import numpy as np
import torch
import yaml
import shap
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_gnn_explainer(
    model,
    data,
    gene_list: list,
    device: torch.device,
    top_k: int = 10,
    num_samples: int = 20,
) -> dict:
    """Run GNNExplainer to identify important genes.

    Reference: PyG torch_geometric.explain.GNNExplainer

    Args:
        model: Trained HybridModel.
        data: Single PyG Data object (a patient graph).
        gene_list: List of gene names.
        device: Compute device.
        top_k: Number of top genes to return.
        num_samples: Number of patients to explain.

    Returns:
        Dictionary with gene importance scores and top genes.
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    logger.info("Running GNNExplainer...")

    # Wrap the GNN part of the model for explanation
    class GNNWrapper(torch.nn.Module):
        def __init__(self, hybrid_model):
            super().__init__()
            self.gnn = hybrid_model.gnn
            self.classifier = torch.nn.Linear(hybrid_model.gnn.output_dim, 4)
            # Copy weights from the first linear layer of the main head
            with torch.no_grad():
                fc_weight = hybrid_model.fc[0].weight.data
                # Project from (gnn_out + clinical) to gnn_out
                self.classifier.weight.data = fc_weight[:4, :hybrid_model.gnn.output_dim]
                self.classifier.bias.data = hybrid_model.fc[-1].bias.data

        def forward(self, x, edge_index, batch=None):
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            emb = self.gnn(x, edge_index, batch)
            return self.classifier(emb)

    gnn_wrapper = GNNWrapper(model).to(device)
    gnn_wrapper.eval()

    explainer = Explainer(
        model=gnn_wrapper,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    # Aggregate node importance across samples
    gene_importance = np.zeros(len(gene_list))
    n_explained = 0

    for i in range(min(num_samples, len(data) if hasattr(data, '__len__') else 1)):
        try:
            if hasattr(data, '__getitem__'):
                sample = data[i].to(device)
            else:
                sample = data.to(device)

            explanation = explainer(sample.x, sample.edge_index)

            if explanation.node_mask is not None:
                node_scores = explanation.node_mask.mean(dim=1).cpu().numpy()
                # Map back to genes (handle batched nodes)
                n_nodes = min(len(node_scores), len(gene_list))
                gene_importance[:n_nodes] += node_scores[:n_nodes]
                n_explained += 1

        except Exception as e:
            logger.warning(f"GNNExplainer failed for sample {i}: {e}")
            continue

    if n_explained > 0:
        gene_importance /= n_explained

    # Get top-k genes
    top_indices = np.argsort(gene_importance)[::-1][:top_k]
    top_genes = [(gene_list[idx], float(gene_importance[idx])) for idx in top_indices]

    # Known breast cancer genes for validation
    known_bc_genes = {"BRCA1", "BRCA2", "TP53", "ERBB2", "ESR1", "PGR", "PTEN",
                      "PIK3CA", "AKT1", "CDH1", "GATA3", "CCND1", "MYC", "RB1"}
    top_gene_names = {g[0].upper() for g in top_genes}
    overlap = top_gene_names & known_bc_genes

    logger.info(f"Top-{top_k} genes identified by GNNExplainer:")
    for gene, score in top_genes:
        marker = " ***" if gene.upper() in known_bc_genes else ""
        logger.info(f"  {gene}: {score:.6f}{marker}")

    if overlap:
        logger.info(f"Overlap with known BC genes: {overlap}")

    results = {
        "gene_importance": {gene_list[i]: float(gene_importance[i]) for i in range(len(gene_list))},
        "top_genes": top_genes,
        "known_bc_overlap": list(overlap),
        "n_samples_explained": n_explained,
    }

    return results


def run_shap_analysis(
    rf_model,
    embeddings: np.ndarray,
    feature_names: list = None,
    top_k: int = 20,
    max_samples: int = 100,
) -> dict:
    """Run SHAP analysis on the Random Forest component.

    Reference: Vaida et al., 2025

    Args:
        rf_model: Trained (calibrated) Random Forest model.
        embeddings: [num_patients, num_features] input features.
        feature_names: Feature names for interpretability.
        top_k: Number of top features to highlight.
        max_samples: Maximum samples for SHAP computation.

    Returns:
        Dictionary with SHAP values and top features.
    """
    logger.info("Running SHAP analysis on Random Forest...")

    # Subsample for speed
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        X_sample = embeddings[indices]
    else:
        X_sample = embeddings

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(embeddings.shape[1])]

    # Get the underlying estimator for SHAP
    if hasattr(rf_model, "estimator"):
        # CalibratedClassifierCV wrapper
        base_model = rf_model.calibrated_classifiers_[0].estimator
    elif hasattr(rf_model, "estimators_"):
        base_model = rf_model
    else:
        base_model = rf_model

    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        logger.warning(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
        predict_fn = rf_model.predict_proba if hasattr(rf_model, 'predict_proba') else rf_model.predict
        background = shap.kmeans(X_sample, 10)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample[:50])

    # Compute mean absolute SHAP values across all classes
    if isinstance(shap_values, list):
        # Multi-class: list of arrays, one per class
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.mean(axis=1)

    # Top features
    top_indices = np.argsort(mean_shap)[::-1][:top_k]
    top_features = [(feature_names[i], float(mean_shap[i])) for i in top_indices]

    logger.info(f"Top-{top_k} SHAP features:")
    for name, value in top_features:
        logger.info(f"  {name}: {value:.6f}")

    results = {
        "shap_values": shap_values if isinstance(shap_values, np.ndarray) else [sv.tolist() for sv in shap_values],
        "mean_shap": {feature_names[i]: float(mean_shap[i]) for i in range(len(feature_names))},
        "top_features": top_features,
        "X_sample": X_sample,
        "feature_names": feature_names,
    }

    return results


def create_risk_heatmap(
    gene_importance: dict,
    kg_edges: torch.Tensor,
    gene_list: list,
    top_k: int = 30,
    output_path: str = "results/risk_heatmap.png",
) -> str:
    """Create BioKG subgraph heatmap with gene importance scores.

    Reference: Choudhry et al., 2025

    Visualizes the subgraph of top important genes colored by their
    importance scores using NetworkX + matplotlib.
    """
    import networkx as nx

    logger.info("Creating risk heatmap...")

    # Get top genes by importance
    sorted_genes = sorted(gene_importance.items(), key=lambda x: x[1], reverse=True)
    top_genes = dict(sorted_genes[:top_k])
    top_gene_set = set(top_genes.keys())

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Build subgraph of top genes
    G = nx.Graph()
    for gene, score in top_genes.items():
        G.add_node(gene, importance=score)

    # Add edges between top genes
    if kg_edges.size(1) > 0:
        for i in range(kg_edges.size(1)):
            src_idx = kg_edges[0, i].item()
            dst_idx = kg_edges[1, i].item()
            if src_idx < len(gene_list) and dst_idx < len(gene_list):
                src_gene = gene_list[src_idx]
                dst_gene = gene_list[dst_idx]
                if src_gene in top_gene_set and dst_gene in top_gene_set:
                    G.add_edge(src_gene, dst_gene)

    if len(G.nodes) == 0:
        logger.warning("No nodes in subgraph. Skipping heatmap.")
        return output_path

    # Layout
    pos = nx.spring_layout(G, k=2, seed=42)

    # Color nodes by importance
    node_colors = [top_genes.get(node, 0) for node in G.nodes()]
    max_importance = max(node_colors) if max(node_colors) > 0 else 1
    node_colors = [c / max_importance for c in node_colors]

    # Known BC genes for markers
    known_bc = {"BRCA1", "BRCA2", "TP53", "ERBB2", "ESR1", "PGR", "PTEN", "PIK3CA"}

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", ax=ax)

    # Draw nodes
    node_collection = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        node_size=[800 + 1500 * top_genes.get(n, 0) / max_importance for n in G.nodes()],
        ax=ax,
    )

    # Labels
    labels = {}
    for node in G.nodes():
        marker = "*" if node.upper() in known_bc else ""
        labels[node] = f"{node}{marker}"

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    plt.colorbar(node_collection, ax=ax, label="Gene Importance Score")
    ax.set_title("BioKG Risk Heatmap — Top Gene Subgraph\n(* = known breast cancer gene)", fontsize=14)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved risk heatmap to {output_path}")
    return output_path


def run_explainability(config: dict, training_results: dict) -> dict:
    """Run the complete explainability pipeline.

    Args:
        config: Configuration dictionary.
        training_results: Results from training pipeline.

    Returns:
        Dictionary with all explainability results.
    """
    results_dir = config["paths"]["results"]
    processed_dir = config["paths"]["processed_data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load gene list
    with open(os.path.join(processed_dir, "selected_genes.txt")) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    all_results = {}

    # 1. GNNExplainer on the best fold
    if training_results and "fold_results" in training_results:
        best_fold = training_results["fold_results"][0]

        # Load model
        from src.model import build_model
        model = build_model(config, clinical_dim=training_results["dataset"]["clinical_dim"], device=str(device))
        model.load_state_dict(best_fold["model_state"])
        model.eval()

        # Get validation data
        dataset = training_results["dataset"]["dataset"]
        gnn_results = run_gnn_explainer(model, dataset, gene_list, device)
        all_results["gnn_explainer"] = gnn_results

        # 2. SHAP on RF
        if "rf_model" in best_fold:
            rf_model = best_fold["rf_model"]
            val_emb = best_fold["val_emb"]

            # Create feature names
            gnn_features = [f"gnn_{i}" for i in range(model.gnn.output_dim)]
            clinical_features = [f"clinical_{i}" for i in range(model.clinical_dim)]
            feature_names = gnn_features + clinical_features

            shap_results = run_shap_analysis(rf_model, val_emb, feature_names)
            all_results["shap"] = {
                "top_features": shap_results["top_features"],
                "mean_shap": shap_results["mean_shap"],
            }

        # 3. Risk heatmap
        kg_edges = torch.load(os.path.join(processed_dir, "kg_edges.pt"), weights_only=True)
        if "gnn_explainer" in all_results:
            create_risk_heatmap(
                all_results["gnn_explainer"]["gene_importance"],
                kg_edges["gene_gene_edges"],
                gene_list,
                output_path=os.path.join(results_dir, "risk_heatmap.png"),
            )

    # Save results (without non-serializable objects)
    serializable_results = {}
    for key, value in all_results.items():
        if key == "shap":
            serializable_results[key] = {
                "top_features": value["top_features"],
            }
        elif key == "gnn_explainer":
            serializable_results[key] = {
                "top_genes": value["top_genes"],
                "known_bc_overlap": value["known_bc_overlap"],
            }

    with open(os.path.join(results_dir, "explainability_results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info("Explainability analysis complete")
    return all_results


if __name__ == "__main__":
    config = load_config()
    # This requires training_results; typically called from the main pipeline
    logger.info("Run this module via the main pipeline or after training.")
