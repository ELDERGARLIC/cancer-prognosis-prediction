"""
Stage 4: Model Architecture

Implements:
    1. BioKG_GAT: Multi-layer Graph Attention Network with residual connections
    2. HybridModel: GAT + clinical feature fusion + auxiliary task head
    3. GATEncoder: Standalone encoder for embedding extraction

References:
    - GAT architecture: Alharbi et al., 2025 (3 layers, 8/4/1 heads)
    - Residual connections: Choudhry et al., 2025
    - Clinical fusion: Gao et al., 2021 (concatenation approach)
    - Auxiliary task: Rahaman et al., 2023 (AuxNet for tumor stage)
    - Hybrid GNN+RF: Palmal et al., 2024
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

logger = logging.getLogger(__name__)


class BioKG_GAT(nn.Module):
    """Multi-layer Graph Attention Network for biological knowledge graphs.

    Architecture (Alharbi et al., 2025):
        Layer 1: GATConv(768, 128, heads=8) -> ELU -> Dropout
        Layer 2: GATConv(1024, 128, heads=4) -> ELU -> Dropout
        Layer 3: GATConv(512, 128, heads=1) -> graph pooling

    With residual connections (Choudhry et al., 2025).
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        heads: list = None,
        dropout: float = 0.4,
        pooling: str = "mean",
    ):
        super().__init__()
        if heads is None:
            heads = [8, 4, 1]

        self.dropout_rate = dropout

        # Layer 1
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads[0], dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads[0])

        # Layer 2
        self.gat2 = GATConv(hidden_dim * heads[0], hidden_dim, heads=heads[1], dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads[1])

        # Layer 3
        self.gat3 = GATConv(hidden_dim * heads[1], hidden_dim, heads=heads[2], dropout=dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Residual projections (Choudhry et al., 2025)
        self.res1 = nn.Linear(in_dim, hidden_dim * heads[0])
        self.res2 = nn.Linear(hidden_dim * heads[0], hidden_dim * heads[1])

        self.dropout = nn.Dropout(dropout)

        # Graph-level pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Layer 1 with residual
        residual = self.res1(x)
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x + residual)
        x = self.dropout(x)

        # Layer 2 with residual
        residual = self.res2(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x + residual)
        x = self.dropout(x)

        # Layer 3 (no residual on final layer)
        x = self.gat3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # Pool to patient-level embedding
        x = self.pool(x, batch)  # [batch_size, hidden_dim]
        return x


class HybridModel(nn.Module):
    """GAT + Clinical Feature Fusion + Auxiliary Head.

    Architecture (Gao et al., 2021 concatenation + Rahaman et al., 2023 AuxNet):
        GNN -> patient embedding (128-dim)
        Concatenate with clinical features
        FC layers -> survival class prediction (main task)
        Separate head -> tumor stage prediction (auxiliary task)
    """

    def __init__(
        self,
        gnn: BioKG_GAT,
        clinical_dim: int,
        num_classes: int = 4,
        num_stages: int = 4,
        fc_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gnn = gnn
        gnn_out_dim = gnn.output_dim

        # Main classification head (survival prediction)
        self.fc = nn.Sequential(
            nn.Linear(gnn_out_dim + clinical_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, num_classes),
        )

        # Auxiliary head: tumor stage prediction (Rahaman et al., 2023)
        self.aux_head = nn.Sequential(
            nn.Linear(gnn_out_dim, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, num_stages),
        )

        # Store dimensions for later use
        self.gnn_out_dim = gnn_out_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes

    def forward(self, x, edge_index, batch, clinical_features, edge_attr=None):
        # GNN forward pass
        gnn_emb = self.gnn(x, edge_index, batch, edge_attr)  # [batch_size, 128]

        # Auxiliary task: predict tumor stage from GNN embedding
        aux_out = self.aux_head(gnn_emb)

        # Main task: fuse with clinical features
        fused = torch.cat([gnn_emb, clinical_features], dim=1)
        main_out = self.fc(fused)

        return main_out, aux_out, gnn_emb

    def extract_embeddings(self, x, edge_index, batch, clinical_features=None, edge_attr=None):
        """Extract patient embeddings without classification head.

        Used for:
            - Hybrid GAT -> RF pipeline (Palmal et al., 2024)
            - t-SNE/UMAP visualization
            - GNNExplainer
        """
        gnn_emb = self.gnn(x, edge_index, batch, edge_attr)
        if clinical_features is not None:
            return torch.cat([gnn_emb, clinical_features], dim=1)
        return gnn_emb


class GATClassifier(nn.Module):
    """Simplified GAT model for ablation studies (without KG or clinical features).

    Reference: Gogoshin & Rodin, 2023 — baseline GCN/GAT comparison.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 4,
        heads: list = None,
        dropout: float = 0.4,
    ):
        super().__init__()
        if heads is None:
            heads = [8, 4, 1]

        self.gat = BioKG_GAT(in_dim, hidden_dim, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        emb = self.gat(x, edge_index, batch, edge_attr)
        out = self.classifier(emb)
        return out, emb


def build_model(config: dict, clinical_dim: int, device: str = None) -> HybridModel:
    """Build the full hybrid model from config.

    Args:
        config: Configuration dictionary.
        clinical_dim: Number of clinical features.
        device: Compute device.

    Returns:
        HybridModel on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_cfg = config["model"]

    gnn = BioKG_GAT(
        in_dim=model_cfg["llm_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        heads=model_cfg["gat_heads"],
        dropout=model_cfg["dropout"],
    )

    model = HybridModel(
        gnn=gnn,
        clinical_dim=clinical_dim,
        num_classes=4,  # 4 survival bins
        num_stages=4,   # 4 tumor stages (I-IV)
        dropout=model_cfg["dropout"],
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built on {device}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Test model creation
    model = build_model(config, clinical_dim=10)

    # Test forward pass with dummy data
    batch_size = 4
    n_genes = 100
    emb_dim = config["model"]["llm_embedding_dim"]

    x = torch.randn(batch_size * n_genes, emb_dim)
    edge_index = torch.randint(0, n_genes, (2, 500))
    # Make edges valid within batch
    batch = torch.repeat_interleave(torch.arange(batch_size), n_genes)
    clinical = torch.randn(batch_size, 10)

    device = next(model.parameters()).device
    x, edge_index, batch, clinical = x.to(device), edge_index.to(device), batch.to(device), clinical.to(device)

    main_out, aux_out, emb = model(x, edge_index, batch, clinical)
    print(f"Main output shape: {main_out.shape}")   # [4, 4]
    print(f"Aux output shape: {aux_out.shape}")      # [4, 4]
    print(f"Embedding shape: {emb.shape}")           # [4, 128]
