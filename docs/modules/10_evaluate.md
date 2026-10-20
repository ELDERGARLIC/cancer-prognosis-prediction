# 10 — `src/evaluate.py` (Evaluation & Baselines)

| Property | Value |
|----------|-------|
| **File** | `src/evaluate.py` |
| **Lines** | 498 |
| **Pipeline Stage** | Stage 5a |
| **Execution Order** | 10th — first module in `run_stage5()` |
| **Runtime** | 00:22:09 → 00:25:25 |
| **Duration** | ~3m 16s |

---

## Purpose

Computes comprehensive evaluation metrics, trains baseline models for comparison, and runs ablation studies. This module provides the statistical evidence needed to assess whether the GAT + KG + BioBERT approach adds value over simpler methods.

Three core tasks:
1. **Full metrics computation** (per-class precision/recall/F1, macro AUC, C-index)
2. **Baseline comparison** (Cox PH, Random Forest, MLP, Vanilla GCN)
3. **Ablation study** (expression-only GAT, expression+clinical GAT, vanilla GCN)

---

## Runtime Log Trace

```
00:22:09  [INFO] Running evaluation...
00:22:10  [INFO] Evaluation metrics computed for 5 folds
00:22:10  [INFO] Running baseline comparison (fold 0)
00:22:11  [INFO] Cox PH baseline: C-index=0.509, AUC=N/A
00:22:12  [INFO] Random Forest baseline: Acc=0.356, AUC=0.548
00:22:12  [INFO] MLP baseline: Acc=0.318, AUC=0.514
00:24:10  [INFO] Vanilla GCN baseline: Acc=0.347, AUC=0.582
00:24:10  [INFO] Running ablation study
00:24:10  [INFO] Ablation: expression only (no clinical)
00:25:15  [INFO] Expression-only AUC: 0.623
00:25:18  [INFO] Ablation: expression + clinical (GAT)
00:25:20  [INFO] Expr+clinical AUC: 0.695
00:25:22  [INFO] Ablation: vanilla GCN
00:25:25  [INFO] GCN-only AUC: 0.582
00:25:25  [INFO] Evaluation report saved to results/evaluation_report.json
```

---

## Imports

```python
import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 44–46**

### `compute_full_metrics(all_labels, all_probs, all_os_time=None, all_os_event=None) -> dict`
**Lines 49–122** | Comprehensive metric computation:

| Metric | Method |
|--------|--------|
| Accuracy | `accuracy_score` |
| AUC-ROC (macro) | `roc_auc_score(multi_class='ovr', average='macro')` |
| AUC-ROC (weighted) | `roc_auc_score(multi_class='ovr', average='weighted')` |
| Per-class precision | `precision_score(average=None)` |
| Per-class recall | `recall_score(average=None)` |
| Per-class F1 | `f1_score(average=None)` |
| Confusion matrix | `confusion_matrix` |
| C-index | `lifelines.utils.concordance_index` |

Uses optimized single-call per-class metric computation instead of 3 separate calls per class.

### `train_cox_baseline(dataset, train_idx, val_idx) -> dict`
**Lines 125–175** | Cox Proportional Hazards model:
- Fits on clinical features (6 dims) + overall survival time + event indicator
- Reports: C-index, AIC, partial log-likelihood
- **Result**: C-index = 0.509

### `train_rf_baseline(dataset, train_idx, val_idx, seed=42) -> dict`
**Lines 178–210** | Random Forest on flattened graph features:
- Flattens node features `(200, 768)` → `(153,600)` per patient
- `RandomForestClassifier(n_estimators=200, max_features="sqrt")`
- **Result**: Accuracy = 0.356, AUC = 0.548

### `train_mlp_baseline(dataset, train_idx, val_idx, seed=42) -> dict`
**Lines 213–251** | Multi-layer Perceptron:
- Same flattened features as RF
- Architecture: (200, 100) hidden layers, ReLU, Adam optimizer, 200 epochs, early stopping
- **Result**: Accuracy = 0.318, AUC = 0.514

### `train_vanilla_gcn_baseline(dataset, train_idx, val_idx, config, device, seed=42) -> dict`
**Lines 254–338** | Vanilla 2-layer GCN (no attention):
- `GCNConv(768, 128)` → `GCNConv(128, 64)` → `global_mean_pool` → `Linear(64, 4)`
- 50 epochs, Adam, no LR scheduling
- **Result**: Accuracy = 0.347, AUC = 0.582

Inner class `VanillaGCN(nn.Module)` defined at lines 270–283.

### `run_ablation_study(dataset, train_idx, val_idx, config, device) -> dict`
**Lines 341–392** | Tests three ablation variants:
1. **Expression only**: Full GAT model but zeros out clinical features
2. **Expression + Clinical**: Full GAT model (normal operation)
3. **Vanilla GCN**: Replaces GAT with GCN baseline

Returns dict with AUC and accuracy for each variant.

### `run_baseline_comparison(dataset, train_idx, val_idx, config, device) -> dict`
**Lines 395–452** | Orchestrates all four baselines. Returns dict keyed by model name.

### `run_evaluation(config, training_results=None) -> dict`
**Lines 455–492** | Main evaluation entry point:
1. Loads or computes per-fold metrics
2. Saves `evaluation_report.json`
3. Returns evaluation dictionary

---

## Baseline Comparison Summary

| Model | Accuracy | AUC-ROC | C-index | Notes |
|-------|----------|---------|---------|-------|
| **GAT + Clinical (ours)** | 0.332 | **0.627** | 0.380 | Full pipeline |
| **Calibrated RF (hybrid)** | **0.454** | 0.601 | — | On GAT embeddings |
| Cox PH | — | — | 0.509 | Clinical features only |
| Random Forest | 0.356 | 0.548 | — | Flattened node features |
| MLP | 0.318 | 0.514 | — | Flattened node features |
| Vanilla GCN | 0.347 | 0.582 | — | No attention mechanism |

---

## Ablation Study Summary

| Variant | AUC-ROC | Accuracy | Delta vs Full |
|---------|---------|----------|---------------|
| Expression only (no clinical) | 0.623 | 0.326 | -0.072 AUC |
| Expression + Clinical (full) | 0.695 | 0.372 | baseline |
| Vanilla GCN | 0.582 | 0.347 | -0.113 AUC |

**Key finding**: Clinical features add +7.2% AUC, and GAT attention adds +11.3% AUC vs GCN.

---

## Output Artifacts

| File | Description |
|------|-------------|
| `results/evaluation_report.json` | Full metrics, baselines, ablation results |
