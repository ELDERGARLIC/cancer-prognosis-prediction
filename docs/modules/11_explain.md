# 11 — `src/explain.py` (Explainability)

| Property | Value |
|----------|-------|
| **File** | `src/explain.py` |
| **Lines** | 407 |
| **Pipeline Stage** | Stage 5b |
| **Execution Order** | 11th — called after evaluation in `run_stage5()` |
| **Runtime** | 00:25:29 → 00:29:40 |
| **Duration** | ~4m 11s |

---

## Purpose

Implements two complementary explainability methods to understand which genes and edges drive the model's survival predictions:

1. **GNNExplainer**: Identifies the most important edges and node features for each patient's prediction. Operates on the graph structure.
2. **SHAP (TreeExplainer)**: Explains the hybrid Random Forest's predictions using Shapley values on the GNN embeddings.
3. **Risk Heatmap**: Visualizes gene importance patterns across survival classes using attention-weighted features.

---

## Runtime Log Trace

```
00:25:29  [INFO] Running GNNExplainer on 50 patients...
00:27:31  [INFO] GNNExplainer completed
              Top genes by node importance: {
                  "MT1E": 0.127,
                  "SCGB2A2": 0.089,
                  "SLC39A6": 0.076,
                  "GATA3": 0.068,
                  "FOXA1": 0.055
              }
00:27:31  [INFO] Running SHAP analysis on Random Forest...
00:28:43  [INFO] SHAP analysis completed
              Top features by mean |SHAP|: {
                  "emb_dim_42": 0.034,
                  "emb_dim_127": 0.029,
                  "emb_dim_89": 0.027,
                  "emb_dim_3": 0.024,
                  "emb_dim_201": 0.022
              }
00:28:43  [INFO] Creating risk heatmap...
00:29:40  [INFO] Risk heatmap saved
00:29:40  [INFO] Explainability results saved
```

---

## Imports

```python
import os
import json
import logging
import numpy as np
import torch
import yaml
import shap
import matplotlib.pyplot as plt
```

Deferred imports inside functions:
```python
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
from src.model import build_model
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 29–31**

### `run_gnn_explainer(model, dataset, gene_list, device, n_samples=50) -> dict`
**Lines 34–148** | Applies GNNExplainer to identify important graph structures:

**Inner class: `GNNWrapper(nn.Module)`** (Lines 62–78)

Wraps `HybridModel` to expose a simpler `forward(x, edge_index)` interface expected by PyG's Explainer API. Uses a fixed zero-tensor for clinical features and creates batch indices from `edge_index`.

**Algorithm**:
```
For each of 50 randomly sampled patients:
    1. Create Explainer(model=GNNWrapper, algorithm=GNNExplainer())
    2. Generate explanation for the patient's graph
    3. Extract node_mask (gene importance) and edge_mask (edge importance)
    4. Aggregate: accumulate node importance per gene
    
Return:
    - top_genes: sorted by mean importance across patients
    - top_edges: most frequently important edges
    - gene_importance_scores: full dict {gene: score}
```

**Identified top genes**:
- **MT1E** (metallothionein 1E): highest importance (0.127) — known stress response marker
- **SCGB2A2** (mammaglobin): 0.089 — breast-specific secretoglobin
- **SLC39A6**: 0.076 — zinc transporter linked to estrogen receptor
- **GATA3**: 0.068 — transcription factor, ER+ breast cancer marker
- **FOXA1**: 0.055 — pioneer factor in hormone-positive breast cancer

### `run_shap_analysis(rf_model, train_embeddings, val_embeddings, val_labels, gene_list) -> dict`
**Lines 151–228** | SHAP TreeExplainer on the calibrated RF:

```
1. Extract base RF from CalibratedClassifierCV
2. Create TreeExplainer(rf_model)
3. Compute SHAP values for validation set
4. Rank embedding dimensions by mean |SHAP|
5. Map top dimensions back to approximate gene contributions
```

**Key details**:
- The RF operates on GNN embeddings (256-dim from mean+max pool), not raw gene features
- SHAP values are computed per embedding dimension, not per gene
- Gene-level attribution requires approximation

### `create_risk_heatmap(model, dataset, gene_list, device) -> str`
**Lines 231–319** | Generates gene × survival-class importance heatmap:

```
1. Group patients by survival class
2. For each class, collect GAT node-level attention patterns
3. Average attention across patients in each class
4. Plot heatmap: x=survival class, y=top 30 genes
5. Save to results/risk_heatmap.png
```

### `run_explainability(config, training_results) -> dict`
**Lines 322–400** | Orchestrator:
1. Loads best fold model
2. Runs GNNExplainer on 50 patients
3. Runs SHAP on RF model
4. Creates risk heatmap
5. Saves `explainability_results.json`

---

## Key Findings

### GNNExplainer: Top 10 Important Genes

| Rank | Gene | Importance | Biological Relevance |
|------|------|-----------|---------------------|
| 1 | MT1E | 0.127 | Stress response, chemoresistance |
| 2 | SCGB2A2 | 0.089 | Breast-specific marker (mammaglobin) |
| 3 | SLC39A6 | 0.076 | Zinc transport, ER signaling |
| 4 | GATA3 | 0.068 | ER+ breast cancer TF |
| 5 | FOXA1 | 0.055 | Luminal breast cancer pioneer factor |
| 6 | TFF3 | 0.051 | Trefoil factor, mucosal protection |
| 7 | MLPH | 0.048 | Melanosome transport, luminal marker |
| 8 | AGR2 | 0.044 | ER-regulated, tumor progression |
| 9 | ANKRD30A | 0.041 | NY-BR-1, breast-specific antigen |
| 10 | CA12 | 0.038 | Carbonic anhydrase, ER target |

Notable: 8/10 top genes are established breast cancer biomarkers, validating that the model attends to biologically meaningful features.

---

## Output Artifacts

| File | Description |
|------|-------------|
| `results/explainability_results.json` | GNNExplainer + SHAP results |
| `results/risk_heatmap.png` | Gene × class importance heatmap |
| `results/shap_summary.png` | SHAP beeswarm plot |
