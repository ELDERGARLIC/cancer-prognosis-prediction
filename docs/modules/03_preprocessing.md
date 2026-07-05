# 03 — `src/preprocessing.py` (Data Preprocessing)

| Property | Value |
|----------|-------|
| **File** | `src/preprocessing.py` |
| **Lines** | 620 |
| **Pipeline Stage** | Stage 1b |
| **Execution Order** | 3rd — called by `run_stage1()` after data download |
| **Runtime** | 22:34:24 → 22:52:44 |
| **Duration** | ~18m 20s (dominated by LASSO: ~18m) |

---

## Purpose

Transforms raw TCGA data into analysis-ready features. This is the most computationally expensive preprocessing step due to LASSO feature selection with cross-validation over 17,343 genes. The module handles:

1. Loading and matching expression/clinical data
2. Low-expression gene filtering
3. Expression normalization (log + z-score)
4. Clinical data imputation (KNN)
5. Survival label discretization
6. LASSO feature selection
7. DisGeNET cross-referencing
8. Clinical feature extraction

---

## Runtime Log Trace

```
22:34:24  [INFO] Loading expression and clinical data
22:34:25  [INFO] Loaded expression matrix: 20530 genes x 1218 samples
22:34:25  [INFO] Loaded clinical data: 1098 patients, 11 columns
22:34:25  [INFO] Matched 1217/1218 expression samples to clinical records
22:34:25  [INFO] Gene filtering: 20530 -> 17343 genes (threshold: >1000)
22:34:25  [INFO] Normalizing gene expression
22:34:25  [INFO] Data appears to be pre-normalized (max=20.98), skipping log
22:34:26  [INFO] Applied z-score standardization across samples
22:34:26  [INFO] Creating survival labels
22:34:26  [WARN] Column 'tumor_stage' is entirely NaN — filled with 0
22:34:26  [INFO] KNN-imputed 2262 missing numeric values
22:34:26  [INFO] Dropped 21 patients with missing/invalid survival time
22:34:26  [INFO] Survival class distribution:
              Class 0 (<1yr): 166 | Class 1 (1-3yr): 476
              Class 2 (3-5yr): 174 | Class 3 (>5yr): 261
22:34:26  [INFO] Matched 1217 patients between expression and clinical
22:34:26  [INFO] Running LASSO feature selection
22:34:26  [INFO] Running LASSO on 17343 genes, 1195 samples...
22:52:44  [INFO] LASSO selected 93 genes with non-zero coefficients
22:52:44  [INFO] Supplemented with 107 high-variance genes
22:52:44  [INFO] Total selected genes: 200
22:52:44  [INFO] DisGeNET filter: 200 genes -> 0 genes (KG has 100 neoplastic)
22:52:44  [WARN] Intersection too small (0). Keeping all 200 genes.
22:52:44  [INFO] Extracted 6 clinical features
22:52:44  [INFO] Preprocessing complete: 200 genes, 1217 patients, 6 clinical
```

---

## Imports

```python
import os
import logging
import gzip
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 36–38**

### `load_expression_data(expr_path) -> pd.DataFrame`
**Lines 41–54** | Loads gene × sample expression TSV (or `.gz`). Sets gene symbols as index.

### `load_clinical_data(clinical_path) -> pd.DataFrame`
**Lines 57–61** | Loads clinical TSV with appropriate index.

### `filter_low_expression_genes(expr_df, min_total_counts=1000) -> pd.DataFrame`
**Lines 64–76** | Removes genes with total expression across all samples ≤ threshold. Reduction: 20,530 → 17,343 genes.

### `normalize_expression(expr_df) -> pd.DataFrame`
**Lines 79–102** | Two-step normalization:
1. **Log transform**: `log2(x+1)` — skipped if data already normalized (max < 30)
2. **Z-score**: Per-gene standardization (mean=0, std=1)

### `impute_clinical_data(clinical_df) -> pd.DataFrame`
**Lines 105–139** | KNN imputation (k=5) on numeric columns. Critical edge-case handling:
- Identifies columns that are entirely NaN (e.g., `tumor_stage`)
- Imputes only columns with at least one valid value
- All-NaN columns filled with 0 and logged as warnings
- Skips ID-like and non-numeric columns

### `select_features_lasso(expr_df, labels, n_genes=1500, seed=42) -> list`
**Lines 142–189** | **Most expensive operation** (~18 minutes):
- Runs `LassoCV` with 10-fold inner CV and 100 alpha candidates
- Selects genes with non-zero coefficients (93 genes)
- If count < target, supplements with highest-variance remaining genes
- Final count: 200 genes (93 LASSO + 107 high-variance)

### `select_features_rfe(expr_df, labels, n_genes=1500, seed=42) -> list`
**Lines 192–220** | Alternative feature selection using Recursive Feature Elimination with Random Forest. Not used in the default pipeline (LASSO is preferred).

### `filter_with_disgenet(selected_genes, disgenet_path, disease_semantic_type="Neoplastic Process") -> list`
**Lines 223–268** | Cross-references LASSO genes with DisGeNET breast cancer associations:
- If intersection ≥ 10% of selected genes: keeps only the intersection
- If intersection too small (as in this run: 0 overlap): keeps all selected genes
- Adds DisGeNET genes as supplementary context for the KG

### `create_survival_labels(clinical_df, bins=None, labels_names=None) -> pd.DataFrame`
**Lines 272–338** | Discretizes continuous `OS.time` into 4 classes:
- Default bins: [365, 1095, 1825] days (1yr, 3yr, 5yr boundaries)
- Drops 21 patients with missing/invalid survival time
- Returns DataFrame with `survival_class`, `OS.time`, `OS` columns

### `extract_clinical_features(clinical_df) -> tuple`
**Lines 341–404** | Extracts and encodes 6 clinical features:
- `age`: normalized age at diagnosis
- `stage_I` through `stage_IV`: one-hot encoded tumor stage
- `is_female`: binary gender indicator

### `get_patient_sample_mapping(expr_df, clinical_df) -> dict`
**Lines 407–438** | Maps TCGA sample barcodes to 12-character patient IDs.

### `run_preprocessing(config, expr_path, clinical_path, disgenet_path=None) -> dict`
**Lines 441–598** | Full preprocessing orchestrator:
1. Load expression + clinical
2. Match patients across datasets
3. Filter genes → normalize → impute clinical
4. Create survival labels
5. LASSO feature selection
6. DisGeNET cross-reference
7. Extract clinical features
8. Save all artifacts to `data/processed/`

---

## Output Artifacts

| File | Shape | Description |
|------|-------|-------------|
| `data/processed/expression_selected.tsv` | 200 × 1,217 | Expression matrix (selected genes) |
| `data/processed/selected_genes.txt` | 200 lines | Gene symbols |
| `data/processed/survival_labels.tsv` | 1,217 × 4 | Survival class + OS data |
| `data/processed/clinical_features.tsv` | 1,217 × 6 | Clinical feature matrix |

---

## Key Data Transformations

```
Raw expression (20530 × 1218)
    │ filter_low_expression_genes (min_total > 1000)
    ▼
Filtered (17343 × 1218)
    │ patient matching (1217/1218 matched)
    ▼
Matched (17343 × 1217)
    │ normalize_expression (z-score, skip log)
    ▼
Normalized (17343 × 1217)
    │ select_features_lasso (93 + 107 = 200)
    ▼
Selected (200 × 1217)
    │ filter_with_disgenet (0 overlap, kept all)
    ▼
Final (200 × 1217) + labels (1217) + clinical (1217 × 6)
```
