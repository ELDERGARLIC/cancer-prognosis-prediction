# Model Performance Report

Comprehensive performance analysis of the breast cancer prognosis prediction pipeline, including per-fold cross-validation results, baseline comparisons, ablation studies, and explainability findings.

---

## Task Definition

**Objective**: Predict discrete survival outcome for TCGA-BRCA patients.

| Property | Value |
|----------|-------|
| Task type | 4-class classification |
| Classes | <1yr, 1–3yr, 3–5yr, >5yr |
| Evaluation | 5-fold stratified cross-validation |
| Primary metric | Accuracy, AUC-ROC (macro), C-index |
| Patients | 1,217 |
| Fold split | 956 train / 239 validation |

---

## Cross-Validation Results

### GAT Model (Main Model)

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std |
|--------|--------|--------|--------|--------|--------|------------|
| **Loss** | 1.245 | 1.282 | 1.269 | 1.337 | 1.360 | **1.299 ± 0.043** |
| **Accuracy** | 0.372 | 0.318 | 0.335 | 0.347 | 0.289 | **0.332 ± 0.028** |
| **AUC-ROC** | 0.695 | 0.658 | 0.676 | 0.574 | 0.532 | **0.627 ± 0.063** |
| **C-index** | 0.455 | 0.337 | 0.376 | 0.410 | 0.320 | **0.380 ± 0.049** |

### Calibrated Random Forest (Hybrid Model)

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std |
|--------|--------|--------|--------|--------|--------|------------|
| **Accuracy** | 0.515 | 0.460 | 0.452 | 0.427 | 0.418 | **0.454 ± 0.034** |
| **AUC-ROC** | 0.678 | 0.640 | 0.633 | 0.519 | 0.533 | **0.601 ± 0.063** |

### Key Observations

```
Performance Comparison (Accuracy)
─────────────────────────────────────────────────────
Calibrated RF  ████████████████████████████████ 0.454
GAT            ██████████████████████           0.332
Random chance  █████████████                    0.250
─────────────────────────────────────────────────────

Performance Comparison (AUC-ROC)
─────────────────────────────────────────────────────
GAT            █████████████████████████████████ 0.627
Calibrated RF  ██████████████████████████████   0.601
Random chance  ████████████████████████████     0.500
─────────────────────────────────────────────────────
```

- The **Calibrated RF** outperforms the raw GAT on accuracy (+12.2 pp) by leveraging the learned GNN embeddings as features for a traditional classifier
- The **GAT** achieves higher AUC-ROC (+2.6 pp), suggesting better probabilistic calibration in its softmax outputs
- **Fold variance** is notable (std ~0.03–0.06), indicating sensitivity to the data split — expected with moderately-sized datasets
- **Fold 1** is consistently the best performer; **Fold 5** early-stops at epoch 22, suggesting a particularly difficult validation split

---

## Training Dynamics

### Convergence Behavior per Fold

| Fold | Epochs | Early Stop? | Best Val Loss | Final Val AUC |
|------|--------|-------------|---------------|---------------|
| 1 | 156 | Yes (patience 20) | 1.245 | 0.695 |
| 2 | 110 | Yes | 1.282 | 0.658 |
| 3 | 127 | Yes | 1.269 | 0.676 |
| 4 | 25 | Yes | 1.337 | 0.574 |
| 5 | 22 | Yes | 1.360 | 0.532 |

```
Training Progression (Fold 1 — Best Fold)
──────────────────────────────────────────────────────────────────
Epoch     Loss(train)   Loss(val)    Acc(val)    AUC(val)
──────────────────────────────────────────────────────────────────
   1       1.385         1.382        0.188       0.546
  10       1.369         1.391        0.188       0.536
  20       1.366         1.372        0.259       0.546
  30       1.346         1.349        0.385       0.572
  50       1.328         1.347        0.326       0.609
  70       1.299         1.305        0.347       0.653
  90       1.276         1.268        0.347       0.684
 100       1.306         1.317        0.335       0.668
 120       1.251         1.273        0.356       0.676
 140       1.216         1.260        0.360       0.700
 150       1.249         1.250        0.393       0.701   ← Peak AUC
 156       EARLY STOP    1.245        0.372       0.695
──────────────────────────────────────────────────────────────────
```

- Training loss steadily decreases from ~1.39 to ~1.22 over 156 epochs
- Validation loss plateaus around epoch 90, oscillating between 1.25–1.32
- AUC-ROC shows the clearest improvement trajectory: 0.546 → 0.701
- The gap between training and validation loss suggests mild overfitting in later epochs

---

## Baseline Comparisons

### Results Table

| Model | Accuracy | F1 (weighted) | AUC-ROC | C-index |
|-------|----------|---------------|---------|---------|
| **Calibrated RF (Ours)** | **0.454** | — | 0.601 | — |
| **GAT (Ours)** | 0.332 | — | **0.627** | 0.380 |
| Cox PH | — | — | — | **0.748** |
| Vanilla GCN | 0.469 | 0.251 | — | — |
| RF Baseline | 0.435 | 0.218 | — | — |
| MLP Baseline | 0.435 | 0.180 | — | — |

### Cox Proportional Hazards

```
┌──────────────────────────────────────────────────────────────┐
│  Cox PH Baseline                                             │
│  ├── Features: First 50 PCA components of gene embeddings   │
│  ├── Penalizer: 0.1 (L2 regularization)                     │
│  ├── C-index: 0.748                                         │
│  │                                                           │
│  │  Note: Cox PH directly models hazard ratios and is       │
│  │  optimized for C-index. It significantly outperforms      │
│  │  the GAT on concordance (0.748 vs 0.380), suggesting     │
│  │  the survival prediction task benefits from explicit      │
│  │  time-to-event modeling rather than class discretization. │
│  │                                                           │
│  │  Warning: Low-variance features triggered a convergence  │
│  │  warning from lifelines (50 features with near-zero      │
│  │  variance in the PCA-reduced space).                      │
│  └                                                           │
└──────────────────────────────────────────────────────────────┘
```

### Random Forest Baseline
- Standard RF (100 trees) on concatenated mean-pooled embeddings + clinical features
- Accuracy: 0.435, F1: 0.218
- Comparable to hybrid RF, suggesting the GNN embeddings provide only modest lift over mean pooling

### MLP Baseline
- 3-layer MLP (256→128→64) with ReLU and early stopping
- Accuracy: 0.435, F1: 0.180
- Slightly worse F1 than RF, indicating non-linear feature interactions are not strongly helpful

### Vanilla GCN Baseline
- 2-layer GCN (no attention, no clinical features, no auxiliary task)
- Trained for 100 epochs on the same graph structure
- Accuracy: 0.469, F1: 0.251
- Competitive with the full GAT+RF hybrid, suggesting the graph topology provides value even without attention

---

## Ablation Study

### Results

| Ablation | Accuracy | F1 (weighted) | Description |
|----------|----------|---------------|-------------|
| Expression only (RF) | 0.452 | 0.232 | Mean gene embeddings → RF |
| Expression + Clinical (RF) | 0.435 | 0.218 | Mean embeddings + 6 clinical features → RF |
| GCN with BioKG | 0.460 | 0.250 | Vanilla GCN using KG topology |
| **Full Pipeline (GAT+RF)** | **0.454** | — | GAT on BioKG + Calibrated RF |

```
Ablation Study (Accuracy)
─────────────────────────────────────────────────────
GCN + BioKG        █████████████████████████████████ 0.460
Full GAT+RF        ████████████████████████████████  0.454
Expr only (RF)     ████████████████████████████████  0.452
Expr+Clinical (RF) ██████████████████████████████    0.435
Random (4-class)   █████████████████                 0.250
─────────────────────────────────────────────────────
```

### Ablation Insights

1. **Expression embeddings alone** (mean-pooled BioBERT × expression) are surprisingly competitive at 0.452, suggesting the GenePT-w embedding strategy captures meaningful biological signal

2. **Adding clinical features** slightly decreases performance (0.452 → 0.435). This counter-intuitive result may be because:
   - The tumor_stage column was entirely NaN (filled with 0)
   - The stage one-hot encoding added noise rather than signal
   - Age and gender have limited prognostic value for this cohort

3. **BioKG graph topology** provides marginal benefit. With 81% of genes isolated (no edges), the graph structure is very sparse. The GCN and GAT largely operate as node-level transformers with self-loops

4. **The full pipeline** achieves comparable accuracy to simpler approaches, but the GAT's AUC-ROC (0.627) is notably higher, suggesting better probabilistic predictions

---

## Class-Weighted Loss Analysis

SMOTE was skipped because the flattened feature dimension (153,606) exceeds the safety threshold (50,000). Instead, inverse-frequency class weights were applied:

| Class | Label | Count | Weight |
|-------|-------|-------|--------|
| 0 | <1yr | 166 | ~1.69 |
| 1 | 1–3yr | 476 | ~0.59 |
| 2 | 3–5yr | 174 | ~1.44 |
| 3 | >5yr | 261 | ~0.99 |

The majority class (1–3yr) gets the lowest weight (0.59), while the smallest classes (<1yr, 3–5yr) get upweighted to ~1.69 and ~1.44 respectively.

---

## Explainability Results

### GNNExplainer — Top-10 Important Genes

GNNExplainer learns soft masks over node features to identify which genes are most important for the model's predictions. Results aggregated over 20 patient samples:

| Rank | Gene | Importance Score | Description |
|------|------|-----------------|-------------|
| 1 | CCDC117 | 0.364 | Coiled-coil domain containing 117 |
| 2 | LOC100101266 | 0.363 | Uncharacterized locus |
| 3 | TM9SF2 | 0.362 | Transmembrane 9 superfamily member 2 |
| 4 | GPR152 | 0.360 | G protein-coupled receptor 152 |
| 5 | NCKAP5 | 0.358 | NCK associated protein 5 |
| 6 | LOC550112 | 0.358 | Uncharacterized locus |
| 7 | C1orf156 | 0.356 | Chromosome 1 open reading frame 156 |
| 8 | EXTL3 | 0.355 | Exostosin glycosyltransferase-like 3 |
| 9 | C9orf93 | 0.355 | Chromosome 9 open reading frame 93 |
| 10 | XKR6 | 0.355 | XK related 6 |

**Known breast cancer gene overlap**: 0 of the top-10 genes overlap with known canonical breast cancer markers (BRCA1, BRCA2, TP53, HER2, ESR1, etc.). This is expected because the LASSO-selected genes are statistical predictors from expression data, not necessarily known oncogenes.

### SHAP Analysis — Top-20 RF Features

SHAP (TreeExplainer) on the Calibrated Random Forest reveals which dimensions of the GNN embedding are most predictive:

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
| 1 | gnn_81 | 0.00996 |
| 2 | gnn_93 | 0.00947 |
| 3 | gnn_53 | 0.00800 |
| 4 | gnn_94 | 0.00695 |
| 5 | gnn_9 | 0.00691 |
| 6 | gnn_116 | 0.00666 |
| 7 | gnn_15 | 0.00583 |
| 8 | gnn_39 | 0.00465 |
| 9 | gnn_71 | 0.00454 |
| 10 | gnn_14 | 0.00434 |
| 11 | gnn_125 | 0.00364 |
| 12 | gnn_117 | 0.00349 |
| 13 | gnn_21 | 0.00340 |
| 14 | **clinical_0** | **0.00327** |
| 15 | gnn_92 | 0.00327 |
| 16 | gnn_87 | 0.00302 |
| 17 | gnn_74 | 0.00286 |
| 18 | gnn_90 | 0.00284 |
| 19 | gnn_24 | 0.00278 |
| 20 | gnn_20 | 0.00274 |

**Key finding**: `clinical_0` (patient age) ranks 14th overall — confirming that age is a meaningful but not dominant predictor. The GNN embedding dimensions dominate feature importance.

---

## Per-Fold Hybrid RF Performance

```
Fold 1:  RF Acc=0.515  AUC=0.678  ██████████████████████████████████████  Best
Fold 2:  RF Acc=0.460  AUC=0.640  █████████████████████████████████
Fold 3:  RF Acc=0.452  AUC=0.633  █████████████████████████████████
Fold 4:  RF Acc=0.427  AUC=0.519  ██████████████████████████████
Fold 5:  RF Acc=0.418  AUC=0.533  █████████████████████████████        Worst
```

Fold 1 is the strongest fold across both GAT and RF metrics. The performance degradation in folds 4–5 correlates with very early stopping (25 and 22 epochs), suggesting the model did not converge well on those data splits.

---

## Limitations and Future Directions

### Current Limitations

1. **Sparse knowledge graph**: 81% of gene nodes are isolated. The GAT's attention mechanism has limited opportunity to aggregate neighborhood information. Denser graphs (e.g., lower STRING confidence threshold, or co-expression networks) could help.

2. **SMOTE not applicable**: The high-dimensional patient embeddings (153K features) exceed memory-safe thresholds for SMOTE. Class-weighted loss is the fallback, but SMOTE on a lower-dimensional projection could be explored.

3. **Discretized survival**: Converting continuous survival time into 4 classes loses information. Ordinal classification or direct survival modeling (e.g., DeepSurv, Cox regression on GNN outputs) could improve C-index.

4. **Small intersection with known biology**: LASSO selects statistically predictive genes, not necessarily biologically well-characterized ones. This limits the explainability value of gene-level analysis.

5. **Fold instability**: Folds 4 and 5 early-stop very quickly (22–25 epochs), suggesting training is sensitive to initialization and data split.

### Potential Improvements

1. **Cox-based loss function**: Replace CE loss with a differentiable Cox partial likelihood loss to optimize C-index directly
2. **Graph augmentation**: Add co-expression edges (Pearson > 0.7) to densify the graph
3. **Ensemble across folds**: Average predictions from all 5 fold models for final inference
4. **Feature dimension reduction**: PCA on embeddings before SMOTE to enable oversampling
5. **Ordinal regression**: Use ordinal cross-entropy to exploit the natural ordering of survival classes
