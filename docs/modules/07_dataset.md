# 07 — `src/dataset.py` (Dataset & DataLoader)

| Property | Value |
|----------|-------|
| **File** | `src/dataset.py` |
| **Lines** | 445 |
| **Pipeline Stage** | Stage 3 (and reused in Stages 4–5) |
| **Execution Order** | 7th — called by `run_stage3()` |
| **Runtime** | 22:58:45 → 22:58:48 (Stage 3), reused at 22:58:48 (Stage 4) |
| **Duration** | ~3s per invocation |

---

## Purpose

Defines the PyTorch Geometric dataset, SMOTE augmentation logic, cross-validation splitting, and DataLoader creation. This module is called multiple times:

1. **Stage 3**: Initial dataset build (patient embeddings + CV splits)
2. **Stage 4**: Rebuilt inside `run_training()` for each training run
3. **Stage 5**: Rebuilt for baseline GCN training in evaluation

---

## Runtime Log Trace

```
22:58:47  [INFO] Creating patient-weighted embeddings: 1217 patients x 200 genes x 768 dims
22:58:48  [INFO] Patient-weighted embeddings shape: (1217, 200, 768)
22:58:48  [INFO] Tumor stage distribution: {-1: 1217}
22:58:48  [INFO] Created 5-fold CV splits
              Fold 0: train=956, val=239
              Fold 1: train=956, val=239
              Fold 2: train=956, val=239
              Fold 3: train=956, val=239
              Fold 4: train=956, val=239
22:58:48  [INFO] Dataset built: 1217 patients, 200 genes, 768-dim embeddings

(During Stage 4, per-fold DataLoader creation):
22:58:49  [WARN] Feature dim too large for SMOTE (153,606 > 50,000).
                 Skipping SMOTE; use class-weighted loss instead.
22:58:50  [INFO] DataLoaders: train=956 (30 batches), val=239 (8 batches)
```

---

## Imports

```python
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
```

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_SMOTE_FEATURES` | 50,000 | Safety threshold — SMOTE skipped if feature dim exceeds this |

---

## Class: `BreastCancerGraphDataset`

**Lines 37–114** | Custom PyTorch Dataset wrapping patient data as PyG `Data` objects.

### `__init__(self, embeddings, labels, edge_index, clinical_features, ...)`
Stores:
- `embeddings`: `(N, 200, 768)` patient-weighted gene embeddings
- `labels`: `(N,)` survival class labels
- `edge_index`: `(2, E)` shared KG topology
- `clinical_features`: `(N, 6)` clinical feature vectors
- Optional: `os_time`, `os_event`, `tumor_stage`

### `__getitem__(self, idx) -> Data`
Returns one patient graph:
```python
Data(
    x=[200, 768],           # Node features (expression × BioBERT)
    edge_index=[2, E],       # Shared KG edges
    clinical=[1, 6],         # Clinical features (unsqueezed for batching)
    y=scalar,                # Survival class (0–3)
    os_time=scalar,          # Overall survival time (days)
    os_event=scalar,         # Event indicator (0/1)
    tumor_stage=scalar       # Tumor stage for aux task (-1 if unknown)
)
```

**Critical detail**: `clinical` is unsqueezed to `[1, 6]` so PyG's DataLoader correctly batches it into `[batch_size, 6]` instead of concatenating into a 1D tensor.

---

## Functions

### `compute_class_weights(labels) -> np.ndarray`
**Lines 120–130** | Computes inverse-frequency class weights:
```
weight[c] = n_samples / (n_classes × count[c])
```
Used when SMOTE is skipped to balance the cross-entropy loss.

### `apply_smote(embeddings, labels, clinical_features, strategy="auto", seed=42) -> tuple`
**Lines 133–194** | SMOTE oversampling attempt:
1. Flattens `(N, 200, 768)` → `(N, 153,600)`
2. Concatenates with clinical features → `(N, 153,606)`
3. **Checks**: `153,606 > MAX_SMOTE_FEATURES (50,000)` → **SMOTE SKIPPED**
4. If skipped: logs warning, returns original data unchanged
5. If applied: runs SMOTE, reshapes back to `(N', 200, 768)`, returns augmented data

### `create_cv_splits(labels, n_folds=5, seed=42) -> list`
**Lines 197–220** | Creates 5-fold stratified train/val index pairs using `StratifiedKFold(shuffle=True, random_state=42)`.

### `extract_tumor_stages(clinical_df) -> np.ndarray`
**Lines 223–265** | Parses tumor stage strings:
- "Stage I" / "Stage IA" / "Stage IB" → 0
- "Stage II" / "Stage IIA" / "Stage IIB" → 1
- "Stage III" / "Stage IIIA" / "Stage IIIB" / "Stage IIIC" → 2
- "Stage IV" → 3
- Unknown / missing → -1

In this run: all tumor stages are -1 (column was entirely NaN → filled with 0 in preprocessing, then parse fails → -1).

### `build_dataset(config) -> dict`
**Lines 268–344** | Full dataset construction:
1. Loads `selected_genes.txt`, `expression_selected.tsv`, `survival_labels.tsv`, `clinical_features.tsv`
2. Loads KG edges from `kg_edges.pt`
3. Loads gene embeddings from `gene_embeddings.npy`
4. Calls `create_patient_weighted_embeddings()` → `(1217, 200, 768)`
5. Extracts tumor stages
6. Creates CV splits
7. Wraps in `BreastCancerGraphDataset`
8. Returns `{"dataset", "cv_splits", "gene_list", "config", ...}`

### `get_dataloaders(dataset, train_idx, val_idx, batch_size=32, smote=True, ...) -> tuple`
**Lines 346–420** | Creates train/val DataLoaders:
1. Extracts train/val subsets by index
2. Attempts SMOTE on training data (skipped in this run)
3. Creates new `BreastCancerGraphDataset` instances for train/val
4. Wraps in PyG `DataLoader(shuffle=True)` for train, `DataLoader(shuffle=False)` for val
5. Returns `(train_loader, val_loader)`

---

## Output (In-Memory)

| Object | Shape | Description |
|--------|-------|-------------|
| `dataset` | 1,217 items | `BreastCancerGraphDataset` |
| `cv_splits` | 5 × (train, val) | Index pairs for cross-validation |
| `train_loader` | 30 batches | batch_size=32, shuffled |
| `val_loader` | 8 batches | batch_size=32, unshuffled |

---

## SMOTE Decision Flow

```
Total features = 200 × 768 + 6 = 153,606

153,606 > MAX_SMOTE_FEATURES (50,000)?
    │
    YES → Skip SMOTE, log warning
    │     Return original data
    │     Use class-weighted loss instead
    │
    (NO) → Flatten embeddings + clinical
           Apply SMOTE(strategy="auto")
           Reshape back to (N', genes, emb_dim)
           Return augmented data
```
