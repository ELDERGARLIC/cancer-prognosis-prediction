"""
Stage 3: PyTorch Geometric Dataset and DataLoader

Builds per-patient graph data objects using BioKG topology
and patient-specific LLM-weighted node features.

Uses Option B (recommended): shared topology, per-patient node features
with DataLoader batching.

References:
    - PyG HeteroData: Fey & Lenssen, 2019
    - Per-patient graphs: Vavekanand, 2026
    - Batch approach: standard PyG mini-batching
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class BreastCancerGraphDataset(Dataset):
    """PyTorch Geometric dataset for breast cancer patient graphs.

    Each patient gets a graph with:
        - Gene nodes with patient-specific weighted LLM embeddings
        - Shared BioKG topology (gene-gene, gene-pathway, gene-disease edges)
        - Clinical features as a separate tensor
        - Survival class label
    """

    def __init__(
        self,
        patient_embeddings: np.ndarray,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        survival_labels: np.ndarray,
        clinical_features: np.ndarray,
        os_time: np.ndarray = None,
        os_event: np.ndarray = None,
        tumor_stages: np.ndarray = None,
        patient_ids: list = None,
    ):
        """
        Args:
            patient_embeddings: [num_patients, num_genes, embedding_dim]
            edge_index: [2, num_edges] shared topology
            edge_weights: [num_edges] edge weights
            survival_labels: [num_patients] discrete survival class (0-3)
            clinical_features: [num_patients, num_clinical_features]
            os_time: [num_patients] continuous overall survival time
            os_event: [num_patients] event indicator (1=death, 0=censored)
            tumor_stages: [num_patients] tumor stage labels for auxiliary task
            patient_ids: list of patient identifiers
        """
        super().__init__()
        self.patient_embeddings = patient_embeddings
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.survival_labels = survival_labels
        self.clinical_features = clinical_features
        self.os_time = os_time
        self.os_event = os_event
        self.tumor_stages = tumor_stages
        self.patient_ids = patient_ids or [f"patient_{i}" for i in range(len(survival_labels))]

    def __len__(self):
        return len(self.survival_labels)

    def __getitem__(self, idx):
        # Node features: patient-specific weighted embeddings
        x = torch.tensor(self.patient_embeddings[idx], dtype=torch.float)

        # Survival label
        y = torch.tensor(self.survival_labels[idx], dtype=torch.long)

        # Clinical features -- stored as [1, n_features] so PyG batches to [batch, n_features]
        clinical = torch.tensor(self.clinical_features[idx], dtype=torch.float).unsqueeze(0)

        # Build PyG Data object with shared topology
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights,
            y=y,
            clinical=clinical,
        )

        # Optional fields for C-index evaluation
        if self.os_time is not None:
            data.os_time = torch.tensor(self.os_time[idx], dtype=torch.float)
        if self.os_event is not None:
            data.os_event = torch.tensor(self.os_event[idx], dtype=torch.long)

        # Auxiliary task: tumor stage
        if self.tumor_stages is not None:
            data.tumor_stage = torch.tensor(self.tumor_stages[idx], dtype=torch.long)

        return data


MAX_SMOTE_FEATURES = 50_000


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights for balanced loss."""
    from collections import Counter
    valid = labels[~np.isnan(labels)].astype(int)
    counts = Counter(valid)
    n_samples = len(valid)
    n_classes = len(counts)
    weights = np.zeros(max(counts.keys()) + 1, dtype=np.float32)
    for cls, count in counts.items():
        weights[cls] = n_samples / (n_classes * count)
    return weights


def apply_smote(
    embeddings: np.ndarray,
    labels: np.ndarray,
    clinical_features: np.ndarray,
    strategy: str = "auto",
    seed: int = 42,
) -> tuple:
    """Apply SMOTE oversampling to balance classes.

    Reference: Vaida et al., 2025; Palmal et al., 2024
    IMPORTANT: Only apply on training data, never on validation/test.

    If the flattened feature dimension exceeds MAX_SMOTE_FEATURES, SMOTE is
    skipped to avoid OOM. The caller should use class-weighted loss instead.

    Args:
        embeddings: [num_patients, num_genes, embedding_dim]
        labels: [num_patients]
        clinical_features: [num_patients, num_features]
        strategy: SMOTE sampling strategy

    Returns:
        (resampled_embeddings, resampled_labels, resampled_clinical)
    """
    n_patients, n_genes, emb_dim = embeddings.shape
    total_features = n_genes * emb_dim + clinical_features.shape[1]

    valid_mask = ~np.isnan(labels)
    labels_valid = labels[valid_mask].astype(int)

    if total_features > MAX_SMOTE_FEATURES:
        logger.warning(
            f"Feature dim too large for SMOTE ({total_features:,} > {MAX_SMOTE_FEATURES:,}). "
            "Skipping SMOTE; use class-weighted loss instead."
        )
        return embeddings[valid_mask], labels_valid, clinical_features[valid_mask]

    flat_emb = embeddings.reshape(n_patients, -1)
    combined = np.hstack([flat_emb, clinical_features])

    combined = combined[valid_mask]

    from collections import Counter
    class_counts = Counter(labels_valid)
    min_count = min(class_counts.values())
    n_neighbors = min(5, min_count - 1) if min_count > 1 else 1

    if min_count < 2:
        logger.warning("Some classes have < 2 samples. Skipping SMOTE.")
        return embeddings[valid_mask], labels_valid, clinical_features[valid_mask]

    smote = SMOTE(sampling_strategy=strategy, random_state=seed, k_neighbors=n_neighbors)
    combined_resampled, labels_resampled = smote.fit_resample(combined, labels_valid)

    emb_resampled = combined_resampled[:, : n_genes * emb_dim].reshape(-1, n_genes, emb_dim)
    clinical_resampled = combined_resampled[:, n_genes * emb_dim :]

    logger.info(f"SMOTE: {len(labels_valid)} -> {len(labels_resampled)} samples")
    logger.info(f"  Before: {dict(Counter(labels_valid))}")
    logger.info(f"  After:  {dict(Counter(labels_resampled))}")

    return emb_resampled, labels_resampled, clinical_resampled


def create_cv_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list:
    """Create stratified k-fold cross-validation splits.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    valid_mask = ~np.isnan(labels)
    valid_indices = np.where(valid_mask)[0]
    valid_labels = labels[valid_mask].astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(valid_indices, valid_labels):
        splits.append((valid_indices[train_idx], valid_indices[val_idx]))

    logger.info(f"Created {n_folds}-fold CV splits")
    for i, (train, val) in enumerate(splits):
        logger.info(f"  Fold {i}: train={len(train)}, val={len(val)}")

    return splits


def extract_tumor_stages(clinical_df: pd.DataFrame) -> np.ndarray:
    """Extract numerical tumor stage labels for auxiliary task.

    Reference: Rahaman et al., 2023 — AuxNet predicts tumor stage.
    """
    stage_col = None
    for col in ["tumor_stage", "pathologic_stage", "ajcc_pathologic_stage"]:
        if col in clinical_df.columns:
            stage_col = col
            break

    if stage_col is None:
        return None

    stages = clinical_df[stage_col].astype(str).str.lower()

    stage_map = {}
    stage_map_val = {"i": 0, "ii": 1, "iii": 2, "iv": 3}
    for idx, s in stages.items():
        assigned = -1
        for key, val in stage_map_val.items():
            if key in s and f"i{key}" not in s:
                assigned = max(assigned, val)
        # More specific matching
        if "stage iv" in s or s.endswith("iv"):
            assigned = 3
        elif "stage iii" in s:
            assigned = 2
        elif "stage ii" in s:
            assigned = 1
        elif "stage i" in s:
            assigned = 0
        stage_map[idx] = assigned if assigned >= 0 else -1

    result = np.array([stage_map.get(i, -1) for i in clinical_df.index])
    # Replace -1 with most common stage
    valid = result[result >= 0]
    if len(valid) > 0:
        most_common = int(np.bincount(valid).argmax())
        result[result < 0] = most_common

    logger.info(f"Tumor stage distribution: {dict(zip(*np.unique(result, return_counts=True)))}")
    return result


def build_dataset(config: dict) -> dict:
    """Build the complete PyG dataset from processed data.

    Returns:
        Dictionary with dataset, splits, and metadata.
    """
    processed_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]
    seed = config["training"]["seed"]

    # Load processed expression data
    expr_df = pd.read_csv(os.path.join(processed_dir, "expression_selected.tsv"), sep="\t", index_col=0)

    # Load clinical data
    clinical_df = pd.read_csv(os.path.join(processed_dir, "clinical_processed.tsv"), sep="\t")
    clinical_features = pd.read_csv(
        os.path.join(processed_dir, "clinical_features.tsv"), sep="\t"
    ).values

    # Load gene embeddings
    gene_embeddings = np.load(os.path.join(emb_dir, "gene_embeddings.npy"))

    # Load knowledge graph edges
    kg_edges = torch.load(os.path.join(processed_dir, "kg_edges.pt"), weights_only=True)
    gene_gene_edges = kg_edges["gene_gene_edges"]
    gene_gene_weights = kg_edges["gene_gene_weights"]

    # Load gene list
    with open(os.path.join(processed_dir, "selected_genes.txt")) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    # Create patient-weighted embeddings
    from src.llm_embeddings import create_patient_weighted_embeddings

    expression_matrix = expr_df.values  # [n_genes, n_patients]
    patient_embeddings = create_patient_weighted_embeddings(
        gene_embeddings, expression_matrix, gene_list
    )

    # Get survival labels and OS time/event
    survival_labels = clinical_df["survival_class"].values.astype(float)
    os_time = pd.to_numeric(clinical_df.get("OS.time"), errors="coerce").values
    os_event = pd.to_numeric(clinical_df.get("OS"), errors="coerce").values

    # Extract tumor stages for auxiliary task
    tumor_stages = extract_tumor_stages(clinical_df)

    # Create CV splits
    splits = create_cv_splits(survival_labels, n_folds=config["training"]["cv_folds"], seed=seed)

    # Patient IDs
    patient_ids = expr_df.columns.tolist()

    # Build dataset
    dataset = BreastCancerGraphDataset(
        patient_embeddings=patient_embeddings,
        edge_index=gene_gene_edges,
        edge_weights=gene_gene_weights,
        survival_labels=survival_labels,
        clinical_features=clinical_features,
        os_time=os_time,
        os_event=os_event,
        tumor_stages=tumor_stages,
        patient_ids=patient_ids,
    )

    logger.info(f"Dataset built: {len(dataset)} patients, {len(gene_list)} genes, {gene_embeddings.shape[1]}-dim embeddings")

    return {
        "dataset": dataset,
        "splits": splits,
        "gene_list": gene_list,
        "clinical_dim": clinical_features.shape[1],
        "n_genes": len(gene_list),
        "patient_ids": patient_ids,
    }


def get_dataloaders(
    dataset: BreastCancerGraphDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 32,
    smote: bool = True,
    smote_strategy: str = "auto",
    seed: int = 42,
) -> tuple:
    """Create train and validation DataLoaders for a single CV fold.

    SMOTE is applied ONLY to training data (Vaida et al., 2025).

    Returns:
        (train_loader, val_loader)
    """
    # Extract training data
    train_emb = dataset.patient_embeddings[train_idx]
    train_labels = dataset.survival_labels[train_idx]
    train_clinical = dataset.clinical_features[train_idx]

    # Apply SMOTE on training set only
    if smote:
        train_emb, train_labels, train_clinical = apply_smote(
            train_emb, train_labels, train_clinical, strategy=smote_strategy, seed=seed
        )

    # Build training dataset
    train_os_time = dataset.os_time[train_idx] if dataset.os_time is not None else None
    train_os_event = dataset.os_event[train_idx] if dataset.os_event is not None else None
    train_stages = dataset.tumor_stages[train_idx] if dataset.tumor_stages is not None else None

    # For SMOTE-augmented data, extend OS time/event/stages with NaN for synthetic samples
    n_original = len(train_idx)
    n_synthetic = len(train_labels) - n_original

    if n_synthetic > 0 and train_os_time is not None:
        train_os_time = np.concatenate([train_os_time, np.full(n_synthetic, np.nan)])
        train_os_event = np.concatenate([train_os_event, np.full(n_synthetic, np.nan)])
        if train_stages is not None:
            # Use majority-class stage for synthetic samples
            train_stages = np.concatenate([
                train_stages, np.full(n_synthetic, int(np.bincount(train_stages.astype(int)).argmax()))
            ])

    train_dataset = BreastCancerGraphDataset(
        patient_embeddings=train_emb,
        edge_index=dataset.edge_index,
        edge_weights=dataset.edge_weights,
        survival_labels=train_labels,
        clinical_features=train_clinical,
        os_time=train_os_time,
        os_event=train_os_event,
        tumor_stages=train_stages,
    )

    # Validation dataset (no SMOTE)
    val_dataset = BreastCancerGraphDataset(
        patient_embeddings=dataset.patient_embeddings[val_idx],
        edge_index=dataset.edge_index,
        edge_weights=dataset.edge_weights,
        survival_labels=dataset.survival_labels[val_idx],
        clinical_features=dataset.clinical_features[val_idx],
        os_time=dataset.os_time[val_idx] if dataset.os_time is not None else None,
        os_event=dataset.os_event[val_idx] if dataset.os_event is not None else None,
        tumor_stages=dataset.tumor_stages[val_idx] if dataset.tumor_stages is not None else None,
        patient_ids=[dataset.patient_ids[i] for i in val_idx],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"DataLoaders: train={len(train_dataset)} ({len(train_loader)} batches), val={len(val_dataset)} ({len(val_loader)} batches)")

    return train_loader, val_loader


if __name__ == "__main__":
    config = load_config()
    data = build_dataset(config)

    dataset = data["dataset"]
    splits = data["splits"]

    # Test first fold
    train_idx, val_idx = splits[0]
    train_loader, val_loader = get_dataloaders(
        dataset, train_idx, val_idx,
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
    )

    # Print a sample batch
    for batch in train_loader:
        print(f"Batch x shape: {batch.x.shape}")
        print(f"Batch edge_index shape: {batch.edge_index.shape}")
        print(f"Batch y: {batch.y}")
        print(f"Batch clinical shape: {batch.clinical.shape}")
        break
