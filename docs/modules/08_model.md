# 08 — `src/model.py` (Model Architectures)

| Property | Value |
|----------|-------|
| **File** | `src/model.py` |
| **Lines** | 283 |
| **Pipeline Stage** | Stage 4 (architecture definition) |
| **Execution Order** | 8th — model built at start of each training fold |
| **Runtime** | 22:58:49 (first instantiation) |
| **Duration** | <1s (model construction) |

---

## Purpose

Defines the neural network architectures used in the pipeline:

1. **BioKG_GAT**: 3-layer Graph Attention Network backbone with residual connections
2. **HybridModel**: GAT + clinical feature fusion + multi-task heads
3. **GATClassifier**: Simplified GAT for ablation/baseline experiments
4. **build_model()**: Factory function to instantiate from config

---

## Runtime Log Trace

```
22:58:49  [INFO] Model built on mps
              Total parameters: 2,712,072
              Trainable parameters: 2,712,072
```

---

## Imports

```python
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
```

---

## Class: `BioKG_GAT(nn.Module)`

**Lines 27–104** | 3-layer GAT backbone with batch normalization, ELU activation, dropout, and residual connections.

### Architecture

```
Input: x [batch_nodes, 768]
       edge_index [2, E]
       batch [batch_nodes]
            │
            ▼
┌──────────────────────────┐
│ GAT Layer 1              │
│ GATConv(768, 128, heads=8│  Concat 8 heads → 128 (averaged internally)
│   , concat=False)        │
│ BatchNorm(128)           │
│ ELU                      │
│ Dropout(0.4)             │
│ + Residual (Linear 768→  │  Linear projection for dimension matching
│             128)         │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ GAT Layer 2              │
│ GATConv(128, 128, heads=4│
│   , concat=False)        │
│ BatchNorm(128)           │
│ ELU                      │
│ Dropout(0.4)             │
│ + Residual (Linear 128→  │
│             128)         │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ GAT Layer 3              │
│ GATConv(128, 128, heads=1│
│   , concat=False)        │
│ BatchNorm(128)           │
│ ELU                      │
│ Dropout(0.4)             │
│ (no residual)            │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Graph Pooling            │
│ mean_pool ∥ max_pool     │  Concatenation of mean and max
│ → [batch_size, 256]      │
└──────────────────────────┘
```

### `forward(self, x, edge_index, batch, edge_attr=None) -> Tensor`
Returns `[batch_size, 256]` graph-level embeddings.

---

## Class: `HybridModel(nn.Module)`

**Lines 107–179** | Combines GAT backbone with clinical features and multi-task learning.

### Architecture

```
                 GNN Embedding [B, 256]
                        │
    Clinical [B, 6] ────┤
                        │
                        ▼
            ┌───────────────────────┐
            │ Main Head (Survival)  │
            │ Linear(262, 64)       │
            │ BatchNorm(64)         │
            │ ELU                   │
            │ Dropout(0.4)          │
            │ Linear(64, 32)        │
            │ ELU                   │
            │ Dropout(0.4)          │
            │ Linear(32, 4)         │  4 survival classes
            └───────────┬───────────┘
                        │
            ┌───────────────────────┐
            │ Aux Head (Staging)    │
            │ Linear(256, 32)       │  GNN embedding only (no clinical)
            │ ReLU                  │
            │ Dropout(0.4)          │
            │ Linear(32, 4)         │  4 tumor stages
            └───────────────────────┘
```

### `forward(self, x, edge_index, batch, clinical_features, edge_attr=None)`
Returns `(main_logits, aux_logits, gnn_embedding)`.

### `extract_embeddings(self, x, edge_index, batch, clinical_features=None, ...)`
Returns GNN embeddings (optionally concatenated with clinical) for downstream RF training.

---

## Class: `GATClassifier(nn.Module)`

**Lines 182–211** | Simplified model for ablation: GAT + 2-layer MLP classifier. No clinical fusion, no auxiliary task.

---

## Function: `build_model(config, clinical_dim, device=None) -> HybridModel`

**Lines 214–254** | Factory that reads config and instantiates the full model:

```python
gnn = BioKG_GAT(
    in_channels=768,        # BioBERT embedding dim
    hidden_channels=128,    # config.model.hidden_dim
    out_channels=128,
    heads=[8, 4, 1],        # config.model.gat_heads
    dropout=0.4             # config.model.dropout
)

model = HybridModel(
    gnn=gnn,
    clinical_dim=6,          # Number of clinical features
    gnn_output_dim=256,      # mean+max pool concatenation
    num_classes=4,           # Survival classes
    num_stages=4             # Tumor stages (aux task)
)
```

---

## Parameter Count Breakdown

| Component | Parameters |
|-----------|-----------|
| GAT Layer 1 (768→128, 8 heads) | ~790K |
| GAT Layer 2 (128→128, 4 heads) | ~66K |
| GAT Layer 3 (128→128, 1 head) | ~16K |
| BatchNorm layers (×3) | ~768 |
| Residual projections (×2) | ~99K |
| Main FC head (262→64→32→4) | ~19K |
| Auxiliary head (256→32→4) | ~8K |
| **Total** | **2,712,072** |

All parameters are trainable. The model is relatively lightweight — the bottleneck is the input dimension (768) at the first GAT layer.
