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
        feature_mode: str = "biobert",
    ):
        """
        Args:
            patient_embeddings: shape depends on feature_mode
                - "biobert":       [num_patients, num_genes, embedding_dim]
                - "gene_id_expr":  [num_patients, num_genes] (expression scalar only)
            edge_index: [2, num_edges] shared topology
            edge_weights: [num_edges] edge weights
            survival_labels: [num_patients] discrete survival class (0-3)
            clinical_features: [num_patients, num_clinical_features]
            os_time: [num_patients] continuous overall survival time
            os_event: [num_patients] event indicator (1=death, 0=censored)
            tumor_stages: [num_patients] tumor stage labels for auxiliary task
            patient_ids: list of patient identifiers
            feature_mode: which input representation to use at __getitem__.
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
        self.feature_mode = feature_mode

        # Shared per-node gene-id tensor: same indices (0..N_genes-1) for every
        # patient. PyG concatenates this attribute along with `x` at batch time
        # so the model's gene_embedding lookup works on [B*N_genes] directly.
        if feature_mode == "gene_id_expr":
            n_genes = patient_embeddings.shape[1]
            self.gene_idx = torch.arange(n_genes, dtype=torch.long)
        else:
            self.gene_idx = None

    def __len__(self):
        return len(self.survival_labels)

    def __getitem__(self, idx):
        # Node features depend on feature_mode
        if self.feature_mode == "gene_id_expr":
            # patient_embeddings[idx] is [num_genes] -- lift to [num_genes, 1]
            # so the model can broadcast-multiply the gene_embed lookup by it.
            expr = np.asarray(self.patient_embeddings[idx], dtype=np.float32)
            x = torch.tensor(expr, dtype=torch.float).unsqueeze(-1)
            gene_idx = self.gene_idx  # already [num_genes] long
        else:
            # Legacy BioBERT path: patient_embeddings[idx] is [num_genes, embed_dim]
            x = torch.tensor(self.patient_embeddings[idx], dtype=torch.float)
            gene_idx = None

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

        if gene_idx is not None:
            # PyG sees this as a per-node attribute (leading dim == num_nodes)
            # and concatenates across graphs automatically at batch time.
            data.gene_idx = gene_idx

        # Optional fields for C-index evaluation
        if self.os_time is not None:
            data.os_time = torch.tensor(self.os_time[idx], dtype=torch.float)
        if self.os_event is not None:
            # Stored as float so NaN (sentinel for SMOTE-synthetic samples that
            # have no real survival event) can round-trip. train.py / evaluate.py
            # already treat this as float and mask with ~np.isnan.
            data.os_event = torch.tensor(self.os_event[idx], dtype=torch.float)

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
        embeddings: [num_patients, num_genes, embedding_dim]  (BioBERT mode), or
                    [num_patients, num_genes]                 (gene_id_expr mode).
        labels: [num_patients]
        clinical_features: [num_patients, num_features]
        strategy: SMOTE sampling strategy

    Returns:
        (resampled_embeddings, resampled_labels, resampled_clinical)
    """
    orig_shape = embeddings.shape
    if embeddings.ndim == 3:
        n_patients, n_genes, emb_dim = orig_shape
    elif embeddings.ndim == 2:
        n_patients, n_genes = orig_shape
        emb_dim = 1
    else:
        raise ValueError(f"Unexpected embeddings.ndim={embeddings.ndim}")
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

    # Reshape back to the original per-patient layout (3D for BioBERT, 2D for gene_id_expr).
    emb_flat_dim = n_genes * emb_dim
    if embeddings.ndim == 3:
        emb_resampled = combined_resampled[:, :emb_flat_dim].reshape(-1, n_genes, emb_dim)
    else:
        emb_resampled = combined_resampled[:, :emb_flat_dim].reshape(-1, n_genes)
    clinical_resampled = combined_resampled[:, emb_flat_dim:]

    logger.info(f"SMOTE: {len(labels_valid)} -> {len(labels_resampled)} samples")
    logger.info(f"  Before: {dict(Counter(labels_valid))}")
    logger.info(f"  After:  {dict(Counter(labels_resampled))}")

    return emb_resampled, labels_resampled, clinical_resampled


def create_cv_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    events: np.ndarray = None,
) -> list:
    """Create stratified k-fold cross-validation splits.

    When `events` is provided, the strata are (survival_bin, event_indicator)
    combined -- this balances censoring across folds, which is critical for
    stable C-index estimation. Without this, rare "dead in <1yr" patients
    can all land in the same fold and blow up variance. This fix directly
    targets the 0.56-0.72 C-index range observed across folds in Phase 0.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    valid_mask = ~np.isnan(labels)
    if events is not None:
        events = np.asarray(events, dtype=float)
        valid_mask = valid_mask & ~np.isnan(events)

    valid_indices = np.where(valid_mask)[0]
    valid_labels = labels[valid_mask].astype(int)

    if events is not None:
        valid_events = events[valid_mask].astype(int)
        # Combined stratum: bin*10 + event  (bin in [0,3], event in {0,1})
        strat = valid_labels * 10 + valid_events
        # Guard: StratifiedKFold needs >= n_folds samples per stratum. If a
        # (bin,event) combination is too rare, drop it back to labels only.
        from collections import Counter
        min_stratum = min(Counter(strat).values())
        if min_stratum < n_folds:
            logger.warning(
                f"Some (bin, event) strata have only {min_stratum} samples (< {n_folds} folds). "
                "Falling back to label-only stratification."
            )
            strat = valid_labels
        else:
            logger.info(
                f"Stratifying on (bin, event); stratum distribution: "
                f"{dict(Counter(strat.tolist()))}"
            )
    else:
        strat = valid_labels

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(valid_indices, strat):
        splits.append((valid_indices[train_idx], valid_indices[val_idx]))

    logger.info(f"Created {n_folds}-fold CV splits (seed={seed})")
    for i, (train, val) in enumerate(splits):
        logger.info(f"  Fold {i}: train={len(train)}, val={len(val)}")

    return splits


def extract_tumor_stages(clinical_df: pd.DataFrame) -> np.ndarray:
    """Extract numerical tumor stage labels for auxiliary task.

    Reference: Rahaman et al., 2023 — AuxNet predicts tumor stage.

    Returns:
        Array of stage codes 0-3 (I-IV), or None if nothing is parseable.
        Unparseable rows are filled with the median of the valid ones so the
        auxiliary CE loss can still train (CE with ignore_index=-1 is used
        downstream to skip unresolved rows when there aren't enough valid ones).
    """
    from src.preprocessing import _parse_stage_string

    stage_col = None
    for col in [
        "tumor_stage", "pathologic_stage", "ajcc_pathologic_stage",
        "clinical_stage", "ajcc_pathologic_tumor_stage",
    ]:
        if col in clinical_df.columns:
            stage_col = col
            break

    if stage_col is None:
        logger.warning("No tumor stage column found; auxiliary task disabled.")
        return None

    stages_raw = clinical_df[stage_col].astype(str)
    result = np.array([_parse_stage_string(s) for s in stages_raw], dtype=np.int64)

    valid = result[result >= 0]
    n_valid = len(valid)
    pct_valid = 100.0 * n_valid / max(len(result), 1)
    logger.info(
        f"Tumor stage ({stage_col}): {n_valid}/{len(result)} parseable ({pct_valid:.1f}%)"
    )

    # Require >=70% parseability before enabling the auxiliary task, consistent
    # with the >=70% guard in preprocessing.extract_clinical_features. Below
    # that, the signal is too sparse to help the main survival objective.
    STAGE_AUX_MIN_PCT = 70.0
    if pct_valid < STAGE_AUX_MIN_PCT:
        logger.warning(
            f"Only {pct_valid:.1f}% of stages parseable (< {STAGE_AUX_MIN_PCT}%) -- "
            "disabling auxiliary tumor-stage task."
        )
        return None

    # Keep -1 for unparseable rows. train_one_epoch masks these out via
    # `valid_aux = batch.tumor_stage >= 0` (and CE uses ignore_index=-1),
    # so unresolved rows contribute zero gradient rather than a biased median.
    uniq, counts = np.unique(result, return_counts=True)
    logger.info(f"Tumor stage distribution: {dict(zip(uniq.tolist(), counts.tolist()))}")
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

    # Load knowledge graph edges
    kg_edges = torch.load(os.path.join(processed_dir, "kg_edges.pt"), weights_only=True)
    gene_gene_edges = kg_edges["gene_gene_edges"]
    gene_gene_weights = kg_edges["gene_gene_weights"]

    # Load gene list
    with open(os.path.join(processed_dir, "selected_genes.txt")) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    # Feature mode drives the per-node representation. `gene_id_expr` is the
    # Phase 1.2 shrinkage path: each patient contributes only one scalar per
    # gene; the learnable gene_embedding table lives inside the model.
    feature_mode = config.get("data", {}).get("feature_mode", "biobert")

    if feature_mode == "gene_id_expr":
        # Per-patient expression scalars only: [n_patients, n_genes]
        # expr_df is [n_genes x n_patients], so transpose.
        patient_embeddings = expr_df.values.T.astype(np.float32)
        logger.info(
            f"feature_mode=gene_id_expr: patient tensor shape={patient_embeddings.shape} "
            f"(was {patient_embeddings.shape[0]} x {patient_embeddings.shape[1]} x 768 "
            "with BioBERT -- ~{:.0f}x smaller)".format(768)
        )
    else:
        # Legacy BioBERT path: weight frozen 768-d embeddings by expression.
        gene_embeddings = np.load(os.path.join(emb_dir, "gene_embeddings.npy"))
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

    # Create CV splits stratified on (bin, event) for balanced censoring
    splits = create_cv_splits(
        survival_labels,
        n_folds=config["training"]["cv_folds"],
        seed=seed,
        events=os_event,
    )

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
        feature_mode=feature_mode,
    )

    if feature_mode == "gene_id_expr":
        logger.info(
            f"Dataset built: {len(dataset)} patients, {len(gene_list)} genes, "
            f"expression-scalar mode (patient tensor dim = {len(gene_list)})"
        )
    else:
        logger.info(
            f"Dataset built: {len(dataset)} patients, {len(gene_list)} genes, "
            f"{patient_embeddings.shape[-1]}-dim embeddings"
        )

    return {
        "dataset": dataset,
        "splits": splits,
        "gene_list": gene_list,
        "clinical_dim": clinical_features.shape[1],
        "n_genes": len(gene_list),
        "patient_ids": patient_ids,
        "feature_mode": feature_mode,
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
    n_before_smote = len(train_labels)

    # Apply SMOTE on training set only.
    # We track `smote_applied` rather than the user's `smote=True` request
    # because SMOTE silently no-ops when the feature tensor is too large
    # (>MAX_SMOTE_FEATURES) or when a class has <2 samples. The caller needs
    # to know which branch happened so it can choose between uniform class
    # weights (balanced via SMOTE) and inverse-frequency weights.
    smote_applied = False
    if smote:
        train_emb, train_labels, train_clinical = apply_smote(
            train_emb, train_labels, train_clinical, strategy=smote_strategy, seed=seed
        )
        smote_applied = len(train_labels) > n_before_smote

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
            # Use majority-class stage for synthetic samples. Exclude the
            # sentinel value -1 (unparseable stage) before bincount, which
            # rejects negative values. If every training sample is -1, fall
            # back to -1 so downstream masks still recognise them as missing.
            stages_int = train_stages.astype(int)
            valid = stages_int[stages_int >= 0]
            if valid.size > 0:
                fill_stage = int(np.bincount(valid).argmax())
            else:
                fill_stage = -1
            train_stages = np.concatenate([
                train_stages, np.full(n_synthetic, fill_stage)
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
        feature_mode=dataset.feature_mode,
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
        feature_mode=dataset.feature_mode,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        f"DataLoaders: train={len(train_dataset)} ({len(train_loader)} batches), "
        f"val={len(val_dataset)} ({len(val_loader)} batches), "
        f"smote_applied={smote_applied}"
    )

    return train_loader, val_loader, smote_applied


if __name__ == "__main__":
    config = load_config()
    data = build_dataset(config)

    dataset = data["dataset"]
    splits = data["splits"]

    # Test first fold
    train_idx, val_idx = splits[0]
    train_loader, val_loader, smote_applied = get_dataloaders(
        dataset, train_idx, val_idx,
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
    )
    logger.info(f"Smoke test smote_applied={smote_applied}")

    # Print a sample batch
    for batch in train_loader:
        print(f"Batch x shape: {batch.x.shape}")
        print(f"Batch edge_index shape: {batch.edge_index.shape}")
        print(f"Batch y: {batch.y}")
        print(f"Batch clinical shape: {batch.clinical.shape}")
        break
