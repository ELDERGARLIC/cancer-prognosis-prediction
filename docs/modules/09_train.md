# 09 — `src/train.py` (Training Loop)

| Property | Value |
|----------|-------|
| **File** | `src/train.py` |
| **Lines** | 497 |
| **Pipeline Stage** | Stage 4 |
| **Execution Order** | 9th — the core training module |
| **Runtime** | 22:58:48 → 00:22:09 |
| **Duration** | ~1h 23m 21s |

---

## Purpose

Implements the complete training pipeline: 5-fold stratified cross-validation of the GAT model with early stopping, learning rate scheduling, class-weighted loss, and hybrid Random Forest training on GNN embeddings. This is the most time-consuming module in the pipeline.

---

## Runtime Log Trace

```
22:58:48  [INFO] Using device: mps
22:58:48  [INFO] Building dataset...
22:58:49  [INFO] Model built on mps (2,712,072 params)
22:58:49  [WARN] Feature dim too large for SMOTE (153,606 > 50,000)
22:58:50  [INFO] DataLoaders: train=956 (30 batches), val=239 (8 batches)
22:58:50  [INFO] Class weights: {0: 1.695, 1: 0.587, 2: 1.440, 3: 0.988}

22:58:50  [INFO] --- Fold 1/5 ---
22:59:15  [INFO] Epoch   1 | Loss: 1.385 | Val AUC: 0.546
23:08:29  [INFO] Epoch  50 | Loss: 1.328 | Val AUC: 0.609
23:17:52  [INFO] Epoch 100 | Loss: 1.306 | Val AUC: 0.668
23:27:15  [INFO] Epoch 150 | Loss: 1.249 | Val AUC: 0.701
23:28:22  [INFO] Early stopping at epoch 156
23:28:23  [INFO] Best val: loss=1.245, acc=0.372, auc=0.695, c_idx=0.455
23:28:23  [INFO] Training Calibrated Random Forest on GNN embeddings...
23:28:31  [INFO] RF metrics: acc=0.515, auc=0.678

23:28:33  [INFO] --- Fold 2/5 ---
23:49:06  [INFO] Early stopping at epoch 110
23:49:07  [INFO] Best val: loss=1.282, acc=0.318, auc=0.658, c_idx=0.337
23:49:15  [INFO] RF metrics: acc=0.460, auc=0.640

23:49:16  [INFO] --- Fold 3/5 ---
00:12:52  [INFO] Early stopping at epoch 127
00:12:53  [INFO] Best val: loss=1.269, acc=0.335, auc=0.676, c_idx=0.376
00:13:01  [INFO] RF metrics: acc=0.452, auc=0.633

00:13:02  [INFO] --- Fold 4/5 ---
00:17:43  [INFO] Early stopping at epoch 25
00:17:44  [INFO] Best val: loss=1.337, acc=0.347, auc=0.574, c_idx=0.410
00:17:52  [INFO] RF metrics: acc=0.427, auc=0.519

00:17:53  [INFO] --- Fold 5/5 ---
00:22:00  [INFO] Early stopping at epoch 22
00:22:01  [INFO] Best val: loss=1.360, acc=0.289, auc=0.532, c_idx=0.320
00:22:09  [INFO] RF metrics: acc=0.418, auc=0.533

00:22:09  [INFO] Cross-Validation Results (GAT):
              loss: 1.2985 ± 0.0430
              accuracy: 0.3322 ± 0.0281
              auc_roc: 0.6271 ± 0.0628
              c_index: 0.3798 ± 0.0490
00:22:09  [INFO] Cross-Validation Results (Calibrated RF):
              accuracy: 0.4544 ± 0.0338
              auc_roc: 0.6006 ± 0.0630
```

---

## Imports

```python
import os
import copy
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index
from src.model import build_model, HybridModel
from src.dataset import BreastCancerGraphDataset, get_dataloaders, build_dataset, compute_class_weights
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 45–47**

### `set_seed(seed=42)`
**Lines 50–59** | Sets seeds for Python `random`, NumPy, PyTorch, CUDA. Enables `cudnn.deterministic`.

### `get_device() -> torch.device`
**Lines 62–67** | Auto-detects `cuda` > `mps` > `cpu`.

### `train_one_epoch(model, loader, optimizer, device, aux_loss_weight=0.3, class_weights=None) -> dict`
**Lines 70–131** | Single training epoch:

```
For each batch:
    1. Move data to device
    2. Forward pass → (main_logits, aux_logits, gnn_emb)
    3. Main loss: CrossEntropyLoss(weight=class_weights)(main_logits, labels)
    4. Aux loss: CrossEntropyLoss(aux_logits, tumor_stage)
       - Only computed for samples with valid stage (≥ 0)
       - Weighted by aux_loss_weight (0.3)
    5. Total loss = main_loss + 0.3 × aux_loss
    6. Backward pass + gradient clipping (max_norm=1.0)
    7. Optimizer step
```

Returns `{"loss": float, "accuracy": float}`.

### `evaluate(model, loader, device) -> dict`
**Lines 135–211** | Validation pass (no gradients):
- **Loss**: CE loss (unweighted)
- **Accuracy**: Top-1 classification accuracy
- **AUC-ROC**: Macro one-vs-rest using softmax probabilities
- **C-index**: Concordance index with risk score = `Σ class_weight × P(class)`

Risk score computation: `risk = Σ (class_idx × P(class_idx))` — higher predicted class → higher risk.

### `extract_embeddings(model, loader, device) -> tuple`
**Lines 215–239** | Runs model in eval mode, concatenates GNN embeddings + clinical features for all samples. Returns `(embeddings, labels, clinical)` as numpy arrays.

### `train_calibrated_rf(train_emb, train_labels, val_emb, val_labels, config) -> tuple`
**Lines 242–286** | Trains the hybrid Random Forest:
```python
rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)

if config["hybrid"]["rf_calibrated"]:
    rf = CalibratedClassifierCV(rf, cv=3, method="isotonic")
```
Returns `(model, {"accuracy": float, "auc_roc": float})`.

### `train_fold(fold, model, dataset, train_idx, val_idx, config, device) -> dict`
**Lines 289–402** | Complete single-fold training:

```
1. Create DataLoaders (SMOTE attempt → skipped)
2. Compute class weights from training labels
3. Optimizer: Adam(lr=0.001, weight_decay=1e-4)
4. Scheduler: ReduceLROnPlateau(factor=0.5, patience=10)
5. Training loop:
   ├── train_one_epoch() with class weights
   ├── evaluate() on validation
   ├── Scheduler step on val_loss
   ├── Early stopping check (patience=20)
   └── Save best model (deepcopy)
6. Restore best model weights
7. Extract GNN embeddings
8. Train calibrated RF on embeddings
9. Return metrics + history + models
```

### `run_training(config) -> dict`
**Lines 405–491** | Full 5-fold CV orchestrator:
1. Build dataset
2. For each fold: fresh model → `train_fold()`
3. Aggregate metrics: mean ± std across folds
4. Save `training_results.json` and `model_fold{N}.pt`
5. Return complete results dictionary

---

## Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 0.001 | `config.training.lr` |
| Weight decay | 1e-4 | `config.training.weight_decay` |
| Max epochs | 200 | `config.training.epochs` |
| Early stopping patience | 20 | `config.training.patience` |
| Batch size | 32 | `config.training.batch_size` |
| CV folds | 5 | `config.training.cv_folds` |
| Aux loss weight | 0.3 | `config.training.aux_loss_weight` |
| Gradient clip | 1.0 | Hardcoded `max_norm` |
| LR scheduler factor | 0.5 | Hardcoded |
| LR scheduler patience | 10 | Hardcoded |
| RF estimators | 500 | `config.hybrid.rf_n_estimators` |
| RF calibration | Isotonic, 3-fold | `config.hybrid.rf_calibrated` |

---

## Per-Fold Summary

| Fold | Epochs | Duration | GAT Acc | GAT AUC | RF Acc | RF AUC |
|------|--------|----------|---------|---------|--------|--------|
| 1 | 156 | ~30m | 0.372 | 0.695 | 0.515 | 0.678 |
| 2 | 110 | ~21m | 0.318 | 0.658 | 0.460 | 0.640 |
| 3 | 127 | ~24m | 0.335 | 0.676 | 0.452 | 0.633 |
| 4 | 25 | ~5m | 0.347 | 0.574 | 0.427 | 0.519 |
| 5 | 22 | ~4m | 0.289 | 0.532 | 0.418 | 0.533 |

---

## Output Artifacts

| File | Size | Description |
|------|------|-------------|
| `results/training_results.json` | ~104KB | Full metrics + training history |
| `results/model_fold0.pt` | ~10.8MB | Fold 0 best model checkpoint |
| `results/model_fold1.pt` | ~10.8MB | Fold 1 best model checkpoint |
| `results/model_fold2.pt` | ~10.8MB | Fold 2 best model checkpoint |
| `results/model_fold3.pt` | ~10.8MB | Fold 3 best model checkpoint |
| `results/model_fold4.pt` | ~10.8MB | Fold 4 best model checkpoint |
