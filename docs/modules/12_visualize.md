# 12 — `src/visualize.py` (Visualization)

| Property | Value |
|----------|-------|
| **File** | `src/visualize.py` |
| **Lines** | 488 |
| **Pipeline Stage** | Stage 5c |
| **Execution Order** | 12th — final module in the pipeline |
| **Runtime** | 00:29:40 → 00:29:51 |
| **Duration** | ~11s |

---

## Purpose

Generates all publication-quality visualizations for the pipeline's results. Produces plots for model evaluation (ROC curves, confusion matrices, training dynamics), embedding analysis (t-SNE, UMAP), survival analysis (Kaplan-Meier), and comparative studies (baselines, ablation).

Uses a consistent visual theme: Agg backend (headless), 100 DPI, `tab10` color palette, and standardized class names.

---

## Runtime Log Trace

```
00:29:40  [INFO] Generating Kaplan-Meier survival curves
00:29:41  [INFO] Saved: results/figures/kaplan_meier.png
00:29:41  [INFO] Generating t-SNE embedding plot
00:29:44  [INFO] Saved: results/figures/tsne_embeddings.png
00:29:44  [INFO] Generating UMAP embedding plot
00:29:45  [INFO] Saved: results/figures/umap_embeddings.png
00:29:45  [INFO] Generating ROC curves
00:29:45  [INFO] Saved: results/figures/roc_curves.png
00:29:45  [INFO] Generating confusion matrix
00:29:46  [INFO] Saved: results/figures/confusion_matrix.png
00:29:46  [INFO] Generating training curves
00:29:47  [INFO] Saved: results/figures/training_curves.png
00:29:47  [INFO] Generating model comparison chart
00:29:47  [INFO] Saved: results/figures/model_comparison.png
00:29:47  [INFO] Generating ablation study chart
00:29:48  [INFO] Saved: results/figures/ablation_study.png
00:29:51  [INFO] Generated 8 visualization files
```

---

## Imports

```python
import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score)
from sklearn.preprocessing import label_binarize
from lifelines import KaplanMeierFitter
```

Optional import inside `plot_umap_embeddings`:
```python
import umap
```

---

## Constants

```python
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CLASS_NAMES = ["<1yr", "1-3yr", "3-5yr", ">5yr"]
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 46–48**

### `plot_kaplan_meier(os_time, os_event, labels, output_path) -> str`
**Lines 51–94** | Kaplan-Meier survival curves stratified by predicted class:
- One curve per class (4 classes)
- Log-rank implied by visual separation
- X-axis: days, Y-axis: survival probability
- Median survival line at 0.5

### `plot_tsne_embeddings(embeddings, labels, output_path, perplexity=30, seed=42) -> str`
**Lines 97–136** | 2D t-SNE projection of GNN embeddings:
- `TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)`
- Points colored by survival class
- Shows cluster separation quality

### `plot_umap_embeddings(embeddings, labels, output_path, n_neighbors=15, min_dist=0.1, seed=42) -> str`
**Lines 139–182** | 2D UMAP projection (optional dependency):
- `UMAP(n_neighbors=15, min_dist=0.1, n_components=2)`
- Falls back gracefully if `umap-learn` not installed
- Generally better global structure preservation than t-SNE

### `plot_roc_curves(all_labels, all_probs, output_path) -> str`
**Lines 185–234** | One-vs-Rest ROC curves:
- One curve per class + micro/macro average
- AUC values in legend
- Diagonal reference line

### `plot_confusion_matrix(all_labels, all_preds, output_path) -> str`
**Lines 237–257** | Normalized confusion matrix heatmap:
- `ConfusionMatrixDisplay` with `normalize="true"`
- Class names on axes

### `plot_training_curves(training_history, output_path) -> str`
**Lines 260–303** | 2×2 subplot grid:
- **Top-left**: Train + val loss vs epoch
- **Top-right**: Validation accuracy vs epoch
- **Bottom-left**: Validation AUC-ROC vs epoch
- **Bottom-right**: Learning rate schedule

### `plot_model_comparison(baselines, our_model_metrics, output_path) -> str`
**Lines 306–361** | Grouped bar chart comparing all models:
- Models: GAT (ours), Cal RF (hybrid), Cox PH, RF, MLP, GCN
- Metrics: Accuracy, AUC-ROC (side by side)
- Error bars where available

### `plot_ablation_study(ablation_results, output_path) -> str`
**Lines 364–407** | Bar chart for ablation variants:
- Expression only vs Expression+Clinical vs Vanilla GCN
- AUC-ROC comparison
- Highlights contribution of each component

### `generate_all_visualizations(config, training_results) -> list`
**Lines 410–473** | Orchestrator:
1. Creates `results/figures/` directory
2. Collects embeddings, labels, probabilities from training results
3. Calls all 8 plot functions in sequence
4. Returns list of saved file paths

---

## Output Artifacts

| File | Description | Key Insight |
|------|-------------|-------------|
| `results/figures/kaplan_meier.png` | Survival curves by class | Clear separation between <1yr and >5yr groups |
| `results/figures/tsne_embeddings.png` | t-SNE of GNN embeddings | Moderate clustering by survival class |
| `results/figures/umap_embeddings.png` | UMAP of GNN embeddings | Better global structure than t-SNE |
| `results/figures/roc_curves.png` | Per-class ROC curves | >5yr class has highest AUC |
| `results/figures/confusion_matrix.png` | Normalized confusion matrix | Most confusion between adjacent classes |
| `results/figures/training_curves.png` | Loss, accuracy, AUC, LR | Early stopping triggered at different epochs |
| `results/figures/model_comparison.png` | All models bar chart | GAT > GCN > RF > MLP in AUC |
| `results/figures/ablation_study.png` | Ablation bar chart | Clinical features + attention both contribute |

---

## Visualization Pipeline

```
training_results
    │
    ├─ fold_results[0].training_history ──→ plot_training_curves
    │
    ├─ fold_results[*].val_labels + val_probs ──→ plot_roc_curves
    │                                          ──→ plot_confusion_matrix
    │
    ├─ fold_results[0].val_embeddings ──→ plot_tsne_embeddings
    │                                  ──→ plot_umap_embeddings
    │
    ├─ fold_results[0].dataset (OS.time, OS.event) ──→ plot_kaplan_meier
    │
    ├─ eval_results.baselines ──→ plot_model_comparison
    │
    └─ eval_results.ablation ──→ plot_ablation_study
```
