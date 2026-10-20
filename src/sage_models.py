"""GraphSAGE patient-survival models for Stage 2/3.

Models:
  MinimalSAGE       - Stage 2: 2-layer SAGE + global mean pool + MLP head, no clinical
  SAGEClinical      - Stage 3 (knob D): same as MinimalSAGE + late-fusion of clinical
                      features after global pool, before MLP head

Both produce a scalar log-hazard per patient and are trained with Cox partial
likelihood. Pre-fusion (or pooled) embeddings are returned alongside log-hazard
for the R1 cosine-similarity sentinel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class MinimalSAGE(nn.Module):
    """2-layer GraphSAGE -> global mean pool -> 2-layer MLP -> scalar log-hazard."""

    def __init__(self, in_dim=1, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, clinical=None, return_emb=False):
        h = F.relu(self.sage1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.sage2(h, edge_index))
        emb = global_mean_pool(h, batch)
        log_h = self.mlp(emb).squeeze(-1)
        if return_emb:
            return log_h, emb
        return log_h


class SAGEClinical(nn.Module):
    """Knob D: late-fusion of clinical features after gene-graph pooling.

    The gene-derived embedding (post-pool, hidden_dim) is concatenated with the
    patient's clinical vector (clinical_dim) and fed into the MLP head. This
    matches Gao 2021's late-fusion ablation: gene-only 0.893 -> gene+clinical 0.954.

    The R1 sentinel (cosine similarity) is computed on the *pre-fusion* pooled
    embedding so it stays comparable to Stage 2.
    """

    def __init__(self, in_dim=1, hidden_dim=128, clinical_dim=7, dropout=0.4):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        fused_dim = hidden_dim + clinical_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.clinical_dim = clinical_dim

    def forward(self, x, edge_index, batch, clinical, return_emb=False):
        h = F.relu(self.sage1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.sage2(h, edge_index))
        emb = global_mean_pool(h, batch)                # (B, hidden_dim)
        fused = torch.cat([emb, clinical], dim=-1)      # (B, hidden_dim + clinical_dim)
        log_h = self.mlp(fused).squeeze(-1)
        if return_emb:
            return log_h, emb  # return pre-fusion emb (R1 sentinel comparability)
        return log_h


class SAGEPathwayClinical(nn.Module):
    """Knob B: 2-layer SAGE -> per-pathway mean pool -> single-head attention
    over pathways -> concat clinical -> MLP -> scalar log-hazard.

    Architectural choices, deliberately simple (per design discipline):
      - Pathway pool = mean of gene embeddings within each pathway.
      - Attention over pathways = single learnable query vector, dot-product
        scores, softmax. Query initialized to zeros so all pathways start
        equally weighted (softmax(0)=uniform). No MLP on the keys, no value
        projection -- just attention-weighted mean of pathway-pooled vectors.
      - MLP head identical to knob D / knob A.

    Per-fold pathway-membership matrix is registered as a buffer (not a
    parameter); each fold rebuilds the model with its own membership.

    The R1 cosine sentinel is computed on the *post-attention patient
    embedding* `emb` (pre-clinical-concat), comparable to knob D / knob A's
    pooled embedding.
    """

    def __init__(self, in_dim=1, hidden_dim=128, clinical_dim=7, dropout=0.4,
                 membership: "torch.Tensor" = None):
        super().__init__()
        assert membership is not None and membership.dim() == 2, \
            "membership must be a (n_pathways, n_genes) tensor"
        self.sage1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("membership", membership)  # (P, G), row-normalized to mean
        self.attn_query = nn.Parameter(torch.zeros(hidden_dim))  # uniform-init attention
        fused_dim = hidden_dim + clinical_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.clinical_dim = clinical_dim
        self.n_pathways = membership.shape[0]
        self.n_genes_per_patient = membership.shape[1]

    def forward(self, x, edge_index, batch, clinical, return_emb=False, return_attn=False):
        h = F.relu(self.sage1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.sage2(h, edge_index))                  # (B*G, hidden)

        B = clinical.shape[0]
        G = self.n_genes_per_patient
        # All patients in this fold have the same gene set, so reshape works.
        h_reshaped = h.view(B, G, -1)                          # (B, G, hidden)
        path_reps = self.membership @ h_reshaped               # (B, P, hidden)

        scores = path_reps @ self.attn_query                   # (B, P)
        weights = F.softmax(scores, dim=-1)                    # (B, P)
        emb = (weights.unsqueeze(-1) * path_reps).sum(dim=1)   # (B, hidden)

        fused = torch.cat([emb, clinical], dim=-1)
        log_h = self.mlp(fused).squeeze(-1)
        if return_attn:
            return log_h, emb, weights
        if return_emb:
            return log_h, emb
        return log_h


def cox_partial_likelihood_loss(log_h, T, E):
    """Cox partial likelihood (Breslow), numerically stable via logcumsumexp.

    Sort by descending T -> at index i in sorted array, the risk set is
    {j : T_j >= T_i} = {0, 1, ..., i}. logcumsumexp(log_h, 0) gives
    log(sum(exp(log_h_j)) for j in risk set).
    """
    n_events = E.sum()
    if n_events.item() < 1:
        return torch.tensor(0.0, device=log_h.device, requires_grad=True)
    order = torch.argsort(-T)
    log_h_s = log_h[order]
    E_s = E[order]
    log_risk = torch.logcumsumexp(log_h_s, dim=0)
    nll = -((log_h_s - log_risk) * E_s).sum() / n_events
    return nll


def mean_pairwise_cosine(emb):
    """R1 sentinel: mean pairwise cosine of patient embeddings. > 0.99 = catastrophic."""
    if emb.shape[0] < 2:
        return float("nan")
    e = F.normalize(emb, dim=-1)
    sim = e @ e.T
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    return float(sim[mask].mean())
