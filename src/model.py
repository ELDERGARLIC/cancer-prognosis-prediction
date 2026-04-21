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
        [Optional] Linear projection 768 -> projection_dim (BioBERT -> task subspace)
        Layer 1: GATConv(in_dim, 128, heads=8) -> ELU -> Dropout
        Layer 2: GATConv(1024, 128, heads=4) -> ELU -> Dropout
        Layer 3: GATConv(512, 128, heads=1) -> mean+max pool

    With residual connections (Choudhry et al., 2025) and graph-level
    mean||max pooling concatenation for richer downstream features.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        heads: list = None,
        dropout: float = 0.4,
        pooling: str = "meanmax",
        projection_dim: int = None,
        num_genes: int = None,
        gene_embed_dim: int = 0,
    ):
        """Args:
            in_dim: per-node input feature dim. For BioBERT embeddings it's 768;
                for the lightweight `gene_id_expr` path it's 1 (expression scalar).
            gene_embed_dim: if > 0, a learnable `Embedding(num_genes, gene_embed_dim)`
                is applied and multiplied by the incoming expression scalar. This
                is the Phase 1.2 shrinkage path -- patient tensors become
                (N_genes, 1) instead of (N_genes, 768), so SMOTE is tractable
                and the network has far fewer parameters to overfit with ~1k
                training samples.
            num_genes: required when `gene_embed_dim > 0` so the embedding table
                can be sized correctly.
        """
        super().__init__()
        if heads is None:
            heads = [8, 4, 1]

        self.dropout_rate = dropout

        # Lightweight gene-id-expression path: learnable embedding per gene
        # index, scaled by expression. Output dim per node = gene_embed_dim.
        self.gene_embed_dim = int(gene_embed_dim) if gene_embed_dim else 0
        if self.gene_embed_dim > 0:
            if num_genes is None or num_genes <= 0:
                raise ValueError(
                    "gene_embed_dim > 0 requires num_genes to be a positive int"
                )
            self.gene_embedding = nn.Embedding(num_genes, self.gene_embed_dim)
            # Initialize near zero so the model starts from "expression magnitude
            # only" and gradually learns per-gene structure.
            nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.1)
            gat_in = self.gene_embed_dim
            self.projection = None  # Skip BioBERT projection in this mode
            logger.info(
                f"BioKG_GAT: using learnable gene-id embedding "
                f"(num_genes={num_genes}, embed_dim={self.gene_embed_dim})"
            )
        # Optional projection of BioBERT embeddings to a lower-dim, learnable
        # subspace before GAT layers. Shrinks patient tensors and gives the
        # network a task-specific compression of the LLM features.
        elif projection_dim is not None and projection_dim > 0 and projection_dim != in_dim:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gene_embedding = None
            gat_in = projection_dim
        else:
            self.projection = None
            self.gene_embedding = None
            gat_in = in_dim

        # Layer 1
        self.gat1 = GATConv(gat_in, hidden_dim, heads=heads[0], dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads[0])

        # Layer 2
        self.gat2 = GATConv(hidden_dim * heads[0], hidden_dim, heads=heads[1], dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads[1])

        # Layer 3
        self.gat3 = GATConv(hidden_dim * heads[1], hidden_dim, heads=heads[2], dropout=dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Residual projections (Choudhry et al., 2025)
        self.res1 = nn.Linear(gat_in, hidden_dim * heads[0])
        self.res2 = nn.Linear(hidden_dim * heads[0], hidden_dim * heads[1])

        self.dropout = nn.Dropout(dropout)

        # Graph-level pooling. 'meanmax' concatenates mean+max -> 2x hidden_dim
        self.pooling_mode = pooling
        if pooling == "mean":
            self.output_dim = hidden_dim
        elif pooling == "max":
            self.output_dim = hidden_dim
        elif pooling == "add":
            self.output_dim = hidden_dim
        else:  # meanmax
            self.output_dim = hidden_dim * 2

    def _pool(self, x, batch):
        if self.pooling_mode == "mean":
            return global_mean_pool(x, batch)
        if self.pooling_mode == "max":
            return global_max_pool(x, batch)
        if self.pooling_mode == "add":
            return global_add_pool(x, batch)
        # meanmax: concat mean and max for richer representation
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

    def forward(self, x, edge_index, batch, edge_attr=None, gene_idx=None):
        """
        Args:
            x: node features. Shape depends on mode:
               - BioBERT path:  [num_nodes, 768]
               - gene-id path:  [num_nodes, 1] (expression scalar)
            gene_idx: [num_nodes] long tensor of gene indices into the
               embedding table. Required when `self.gene_embedding` is active.
        """
        if self.gene_embedding is not None:
            if gene_idx is None:
                raise ValueError(
                    "gene_idx must be provided when gene_embedding is active"
                )
            # x is [N, 1] expression scalars; broadcast-multiply the looked-up
            # gene embedding. Result is a per-patient-per-gene dense feature.
            embedded = self.gene_embedding(gene_idx)  # [N, gene_embed_dim]
            x = embedded * x  # broadcast [N, gene_embed_dim] * [N, 1]
        elif self.projection is not None:
            x = self.projection(x)

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
        return self._pool(x, batch)  # [batch_size, output_dim]


class HybridModel(nn.Module):
    """GAT + Clinical Feature Fusion + Multi-task Heads.

    Architecture (Gao et al., 2021 concat + Rahaman et al., 2023 AuxNet
    + Katzman et al., 2018 DeepSurv Cox head):

        GNN                       -> patient embedding (gnn_out_dim)
        concat with clinical      -> fused vector
        main FC head              -> survival class logits (4 classes)
        tumor-stage aux head      -> stage logits (4 classes)
        Cox survival head         -> scalar log-hazard (DeepSurv-style)
        ordinal regression head   -> single continuous score in [0,3]

    The Cox head is the key addition after the first run showed that
    CoxPH baseline achieved 0.748 C-index while the GAT got 0.38 -- a
    differentiable Cox objective lets the GNN optimize C-index directly.
    """

    def __init__(
        self,
        gnn: BioKG_GAT,
        clinical_dim: int,
        num_classes: int = 4,
        num_stages: int = 4,
        fc_hidden: int = 128,
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

        # Cox-PH survival head (DeepSurv, Katzman 2018) -- scalar log hazard.
        # Uses fused (gnn + clinical) features so it can leverage age/stage.
        self.cox_head = nn.Sequential(
            nn.Linear(gnn_out_dim + clinical_dim, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
        )

        # Ordinal regression head -- single scalar, trained with MSE against
        # the integer survival class. Encourages monotone ordering.
        self.ordinal_head = nn.Sequential(
            nn.Linear(gnn_out_dim + clinical_dim, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
        )

        # Store dimensions for later use
        self.gnn_out_dim = gnn_out_dim
        self.clinical_dim = clinical_dim
        self.num_classes = num_classes

    def forward(self, x, edge_index, batch, clinical_features, edge_attr=None, gene_idx=None):
        """Returns (main_logits, aux_logits, gnn_emb, cox_log_hazard, ordinal_score).

        `gene_idx` is forwarded to the GNN so the learnable gene-id embedding
        path (Phase 1.2) can look up per-gene embeddings. Callers using the
        BioBERT path can omit it.
        """
        # GNN forward pass
        gnn_emb = self.gnn(x, edge_index, batch, edge_attr, gene_idx=gene_idx)  # [B, gnn_out_dim]

        # Auxiliary task: tumor stage from pure GNN embedding
        aux_out = self.aux_head(gnn_emb)

        # Fuse with clinical features for the heads that care about patient-level covariates
        fused = torch.cat([gnn_emb, clinical_features], dim=1)
        main_out = self.fc(fused)
        cox_out = self.cox_head(fused).squeeze(-1)       # [B]
        ordinal_out = self.ordinal_head(fused).squeeze(-1)  # [B]

        return main_out, aux_out, gnn_emb, cox_out, ordinal_out

    def extract_embeddings(self, x, edge_index, batch, clinical_features=None, edge_attr=None, gene_idx=None):
        """Extract patient embeddings without classification head.

        Used for:
            - Hybrid GAT -> RF pipeline (Palmal et al., 2024)
            - t-SNE/UMAP visualization
            - GNNExplainer
        """
        gnn_emb = self.gnn(x, edge_index, batch, edge_attr, gene_idx=gene_idx)
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
        projection_dim: int = None,
    ):
        super().__init__()
        if heads is None:
            heads = [8, 4, 1]

        self.gat = BioKG_GAT(
            in_dim, hidden_dim, heads, dropout,
            projection_dim=projection_dim,
        )
        # Use gat.output_dim so meanmax pooling (2x hidden_dim) works as input.
        self.classifier = nn.Sequential(
            nn.Linear(self.gat.output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        emb = self.gat(x, edge_index, batch, edge_attr)
        out = self.classifier(emb)
        return out, emb


def build_model(config: dict, clinical_dim: int, device: str = None, num_genes: int = None) -> HybridModel:
    """Build the full hybrid model from config.

    Args:
        config: Configuration dictionary.
        clinical_dim: Number of clinical features.
        device: Compute device.
        num_genes: Number of genes in the KG; required when the lightweight
            gene-id-embedding feature mode is enabled.

    Returns:
        HybridModel on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_cfg = config["model"]
    data_cfg = config.get("data", {})

    # Phase 1.2: feature_mode selects the per-gene input dimensionality.
    #   "biobert" (default): patient-weighted BioBERT-768 per gene (legacy path)
    #   "gene_id_expr": per-gene expression scalar + learnable gene-id embedding
    feature_mode = data_cfg.get("feature_mode", "biobert")
    if feature_mode == "gene_id_expr":
        gene_embed_dim = int(model_cfg.get("gene_embed_dim", 32))
        in_dim = 1  # expression scalar per gene node
    else:
        gene_embed_dim = 0
        in_dim = model_cfg["llm_embedding_dim"]

    gnn = BioKG_GAT(
        in_dim=in_dim,
        hidden_dim=model_cfg["hidden_dim"],
        heads=model_cfg["gat_heads"],
        dropout=model_cfg["dropout"],
        projection_dim=model_cfg.get("projection_dim"),
        pooling=model_cfg.get("pooling", "meanmax"),
        num_genes=num_genes,
        gene_embed_dim=gene_embed_dim,
    )

    model = HybridModel(
        gnn=gnn,
        clinical_dim=clinical_dim,
        num_classes=4,  # 4 survival bins
        num_stages=4,   # 4 tumor stages (I-IV)
        fc_hidden=model_cfg.get("fc_hidden", 128),
        dropout=model_cfg["dropout"],
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built on {device}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(
        f"  feature_mode={feature_mode} "
        f"GAT in_dim={in_dim} gene_embed_dim={gene_embed_dim} "
        f"proj_dim={model_cfg.get('projection_dim')} "
        f"pool_out_dim={gnn.output_dim} clinical_dim={clinical_dim}"
    )

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

    main_out, aux_out, emb, cox, ordinal = model(x, edge_index, batch, clinical)
    print(f"Main output shape: {main_out.shape}")   # [4, 4]
    print(f"Aux output shape: {aux_out.shape}")      # [4, 4]
    print(f"Embedding shape: {emb.shape}")           # [4, hidden or 2*hidden]
    print(f"Cox log-hazard shape: {cox.shape}")      # [4]
    print(f"Ordinal score shape: {ordinal.shape}")   # [4]
