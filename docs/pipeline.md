# Pipeline Documentation

This document describes each stage of the breast cancer prognosis prediction pipeline, including data flow, intermediate artifacts, and architectural decisions.

---

## End-to-End Flow

```
                          ┌──────────────────┐
                          │   main.py        │
                          │   (Orchestrator) │
                          └────────┬─────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
   │   Stage 1    │      │   Stage 2    │      │   Stage 3        │
   │   Data Acq.  │─────▶│   KG + LLM   │─────▶│   Dataset Build  │
   │   & Preproc  │      │   Embeddings │      │   (PyG + CV)     │
   └──────────────┘      └──────────────┘      └────────┬─────────┘
                                                        │
                                   ┌────────────────────┤
                                   ▼                    ▼
                          ┌──────────────┐      ┌──────────────────┐
                          │   Stage 4    │      │   Stage 5        │
                          │   Training   │─────▶│   Evaluation &   │
                          │   (5-Fold CV)│      │   Explainability │
                          └──────────────┘      └──────────────────┘
```

---

## Stage 1: Data Acquisition & Preprocessing

**Entry point**: `run_stage1(config)` → calls `run_data_download()` then `run_preprocessing()`

### 1a. Data Download (`src/data_download.py`)

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│ GDC REST API │────▶│ TCGA-BRCA RNA-Seq (HiSeqV2, 20530 genes)   │
│ (or Xena)   │     │ → data/raw/tcga_brca_expression.tsv.gz       │
└─────────────┘     └──────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────────────────────────────────┐
│ GDC Cases   │────▶│ Clinical data (1098 patients, 11 columns)    │
│ (or Xena)   │     │ → data/raw/tcga_brca_clinical.tsv            │
└─────────────┘     └──────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────────────────────────────────┐
│ STRING v12  │────▶│ Protein-Protein Interactions (9606.links)    │
│             │     │ → data/knowledge_graph/string_ppi.tsv         │
└─────────────┘     └──────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────────────────────────────────┐
│ DisGeNET    │────▶│ Gene-Disease Associations (breast cancer)    │
│ API         │     │ → data/knowledge_graph/disgenet_gene_disease  │
└─────────────┘     └──────────────────────────────────────────────┘

┌─────────────┐     ┌──────────────────────────────────────────────┐
│ NCBI Entrez │────▶│ Gene functional summaries (1750 genes)       │
│             │     │ → data/embeddings/gene_summaries.json         │
└─────────────┘     └──────────────────────────────────────────────┘
```

**Fallback strategy**: GDC is the primary source. If the GDC bulk download fails (e.g., 500 error), the pipeline falls back to UCSC Xena mirrors automatically.

### 1b. Preprocessing (`src/preprocessing.py`)

```
Expression Matrix (20530 genes × 1218 samples)
       │
       ▼
┌──────────────────┐
│ Low-Expression   │  Removes genes with total counts ≤ 1000
│ Gene Filter      │  20530 → 17343 genes
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Normalization    │  log2(x+1) if raw counts, then z-score per gene
│ (z-score)        │  Skips log if already pre-normalized (max ≤ 30)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Clinical         │  KNN imputation (k=5) for numeric columns
│ Imputation       │  All-NaN columns (e.g. tumor_stage) filled with 0
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Survival Label   │  Bins OS.time into 4 classes:
│ Discretization   │  <365d (Class 0), 365–1095d (Class 1),
│                  │  1095–1825d (Class 2), >1825d (Class 3)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ LASSO Feature    │  LassoCV (10-fold, 100 alphas) selects genes
│ Selection        │  93 genes with non-zero coefficients
│                  │  + 107 high-variance supplement = 200 total
└────────┬─────────┘
         ▼
┌──────────────────┐
│ DisGeNET Filter  │  Intersects with known neoplastic genes
│ (optional)       │  If intersection too small, keeps all + adds KG genes
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Clinical Feature │  Extracts: age, stage (one-hot I–IV), gender
│ Extraction       │  → 6 features per patient
└────────┬─────────┘
         ▼
  Outputs:
  ├── data/processed/expression_selected.tsv   (200 genes × 1217 patients)
  ├── data/processed/selected_genes.txt         (200 gene symbols)
  ├── data/processed/survival_labels.tsv        (labels + OS.time + OS)
  └── data/processed/clinical_features.tsv      (6 features)
```

**Patient matching**: TCGA sample barcodes (e.g., `TCGA-XX-XXXX-01`) are truncated to 12-character patient IDs and matched across expression and clinical data. 1,217 of 1,218 samples match successfully.

**Class distribution** after label discretization:

| Class | Time Range | Patients | Proportion |
|-------|-----------|----------|------------|
| 0 | <1 year | 166 | 13.6% |
| 1 | 1–3 years | 476 | 39.1% |
| 2 | 3–5 years | 174 | 14.3% |
| 3 | >5 years | 261 | 21.4% |

---

## Stage 2: Knowledge Graph & LLM Embeddings

**Entry point**: `run_stage2(config)` → calls `build_knowledge_graph()`, `run_embedding_generation()`, `build_gene_vector_store()`

### 2a. Knowledge Graph Construction (`src/kg_construction.py`)

```
┌───────────────────────────────────────────────────────────────────┐
│                     Knowledge Graph                               │
│                                                                   │
│   Gene ◆──────────◆ Gene         (STRING PPI, 64 edges)          │
│         \                                                         │
│          \                                                        │
│           ◆──────── Pathway      (KEGG, 173 edges, 126 pathways) │
│          /                                                        │
│         /                                                         │
│   Gene ◆──────────◆ Disease      (DisGeNET, 0 edges*)            │
│                                                                   │
│   * No overlap between LASSO-selected genes and DisGeNET entries  │
│                                                                   │
│   Statistics:                                                     │
│   ├── 200 gene nodes                                              │
│   ├── 64 gene-gene edges (PPI, confidence ≥ 700)                 │
│   ├── 173 gene-pathway edges (126 KEGG pathways)                 │
│   ├── Gene-gene density: 0.0016                                  │
│   ├── Average degree: 0.32                                       │
│   ├── Max degree: 6                                              │
│   └── Isolated genes: 162 (81% — sparse graph)                   │
└───────────────────────────────────────────────────────────────────┘
```

**Edge construction details**:
- **STRING PPI**: Loads 13.7M human protein links, filters to confidence ≥ 700, maps protein IDs to gene symbols, intersects with 200 selected genes → 64 bidirectional edges
- **DisGeNET**: Filters for "Neoplastic Process" semantic type, but no selected genes appear in the breast cancer association set → 0 edges
- **KEGG**: REST API batch query per gene; finds 126 pathways containing at least one selected gene → 173 gene-pathway membership edges

### 2b. LLM Embeddings (`src/llm_embeddings.py`)

```
┌────────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│ Gene Summaries     │     │ BioBERT v1.2      │     │ Gene Embeddings │
│ (NCBI Entrez text) │────▶│ [CLS] pooling     │────▶│ (200 × 768)     │
│ 1750 gene entries  │     │ max_length=512     │     │ float32 .npy    │
└────────────────────┘     └───────────────────┘     └─────────────────┘
```

- **Model**: `dmis-lab/biobert-base-cased-v1.2` (PubMed + PMC pre-trained BERT)
- **Input**: Gene functional summary text from NCBI (e.g., "This gene encodes a member of the...")
- **Output**: 768-dimensional `[CLS]` token embedding per gene
- **Pathway/disease embeddings**: Generated from synthetic descriptions (e.g., "KEGG pathway: Cell cycle — genes involved in cell cycle regulation")

### 2c. FAISS Vector Store (`src/vector_store.py`)

```
┌─────────────────┐     ┌────────────────────────┐
│ Gene Embeddings │────▶│ FAISS IndexFlatL2      │
│ (200 × 768)     │     │ Exact L2 nearest       │
│                 │     │ neighbor search         │
└─────────────────┘     │                        │
                        │ Sanity check:          │
                        │ RPL13AP6 neighbors:    │
                        │ → RPL19P12 (d=4.98)    │
                        │ → MRPL42P5 (d=10.33)   │
                        └────────────────────────┘
```

---

## Stage 3: Dataset Construction

**Entry point**: `run_stage3(config)` → calls `build_dataset()`

### Patient-Weighted Embeddings (GenePT-w)

```
Gene Embeddings          Expression Matrix        Patient Embeddings
(200 × 768)              (1217 × 200)             (1217 × 200 × 768)
     │                        │                         │
     │    For each patient p: │                         │
     │    emb[p,g,:] = expr[p,g] × gene_emb[g,:]       │
     └────────────────────────┘─────────────────────────┘
```

Each patient gets a **personalized graph** where node features are the gene's BioBERT embedding scaled by that patient's expression level for that gene. This encodes both functional meaning (from the LLM) and patient-specific expression signal.

### PyTorch Geometric Dataset

```
┌─────────────────────────────────────────────────────────┐
│  BreastCancerGraphDataset (1217 patients)                │
│                                                         │
│  Per patient (one Data object):                         │
│  ├── x: [200, 768]        node features (weighted emb) │
│  ├── edge_index: [2, 237]  shared KG topology           │
│  ├── clinical: [1, 6]      clinical features            │
│  ├── y: scalar              survival class (0–3)        │
│  ├── os_time: scalar        overall survival (days)     │
│  └── tumor_stage: scalar    stage for aux task          │
│                                                         │
│  5-Fold Stratified CV:                                  │
│  └── Each fold: 956 train / 239 validation              │
└─────────────────────────────────────────────────────────┘
```

**SMOTE handling**: The flattened feature dimension (200 × 768 + 6 = 153,606) exceeds the `MAX_SMOTE_FEATURES` threshold of 50,000. SMOTE is skipped and **class-weighted cross-entropy loss** is used instead for class imbalance.

---

## Stage 4: Model Training

**Entry point**: `run_stage4(config)` → calls `run_training()`

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HybridModel                                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    BioKG_GAT (GNN Backbone)                 │    │
│  │                                                             │    │
│  │  Input: x [B×200, 768]                                     │    │
│  │         edge_index [2, E] (shared across batch)             │    │
│  │                                                             │    │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────┐             │    │
│  │  │ GAT Layer1│   │ GAT Layer2│   │ GAT Layer3│             │    │
│  │  │ 768→128   │──▶│ 128→128   │──▶│ 128→128   │             │    │
│  │  │ 8 heads   │   │ 4 heads   │   │ 1 head    │             │    │
│  │  │ +BN +ELU  │   │ +BN +ELU  │   │ +BN +ELU  │             │    │
│  │  │ +Dropout  │   │ +Dropout  │   │ +Dropout  │             │    │
│  │  │ +Residual │   │ +Residual │   │           │             │    │
│  │  └───────────┘   └───────────┘   └───────────┘             │    │
│  │         │                                                   │    │
│  │         ▼                                                   │    │
│  │  Graph Pooling (mean + max, concatenated)                   │    │
│  │  → [B, 256]                                                 │    │
│  └─────────────────────────────────────┬───────────────────────┘    │
│                                        │                            │
│  ┌──────────────────┐                  │                            │
│  │ Clinical Features│                  │                            │
│  │ [B, 6]           │──────┐           │                            │
│  └──────────────────┘      │           │                            │
│                            ▼           ▼                            │
│                     ┌──────────────────────┐                        │
│                     │ FC Head (Main Task)  │                        │
│                     │ 262 → 64 → BN → ELU │                        │
│                     │ → Dropout → 64 → 32  │                        │
│                     │ → 4 (survival class) │                        │
│                     └──────────┬───────────┘                        │
│                                │                                    │
│                     ┌──────────┴───────────┐                        │
│                     │ Aux Head (Staging)   │                        │
│                     │ 256 → 32 → 4 stages │                        │
│                     └──────────────────────┘                        │
│                                                                     │
│  Total Parameters: 2,712,072 (all trainable)                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Loop

```
For each fold (1..5):
    │
    ├── Initialize model (fresh weights)
    ├── Compute class weights (inverse frequency)
    ├── Create DataLoaders (batch_size=32)
    │
    ├── For each epoch (max 200):
    │   ├── Train: weighted CE loss + 0.3 × aux loss
    │   │          + gradient clipping (max_norm=1.0)
    │   ├── Validate: loss, accuracy, macro AUC, C-index
    │   └── ReduceLROnPlateau (factor=0.5, patience=10)
    │
    ├── Early stopping (patience=20 on val loss)
    ├── Save best model checkpoint
    │
    └── Hybrid RF Training:
        ├── Extract GNN embeddings from best model
        ├── Train RF (500 trees, calibrated, isotonic)
        └── Evaluate RF on validation fold
```

### Training Progress (Fold 1 Example)

```
Epoch    1 → Train Loss: 1.385  Val AUC: 0.546  Val Acc: 0.188
Epoch   50 → Train Loss: 1.328  Val AUC: 0.609  Val Acc: 0.326
Epoch  100 → Train Loss: 1.306  Val AUC: 0.668  Val Acc: 0.335
Epoch  150 → Train Loss: 1.249  Val AUC: 0.701  Val Acc: 0.393
Epoch  156 → Early stop  Val AUC: 0.695  Val Acc: 0.372  (best)

RF Hybrid → Accuracy: 0.515  AUC: 0.678
```

---

## Stage 5: Evaluation & Explainability

**Entry point**: `run_stage5(config, training_results)` → calls `run_evaluation()`, `run_explainability()`, `generate_all_visualizations()`

### 5a. Evaluation (`src/evaluate.py`)

- **CV summary** with mean ± std across folds
- **Baseline comparisons**: Cox PH, Random Forest, MLP, Vanilla GCN
- **Ablation study**: Expression-only RF, Expression+Clinical RF, GCN with BioKG

### 5b. Explainability (`src/explain.py`)

- **GNNExplainer**: Identifies important nodes (genes) for individual predictions by learning soft masks over node features and edges
- **SHAP TreeExplainer**: Feature importance for the RF hybrid model, revealing which GNN embedding dimensions and clinical features drive predictions
- **Risk heatmap**: NetworkX graph visualization of top genes and their KG connections

### 5c. Visualization (`src/visualize.py`)

Generates 5 publication-ready figures:
1. `training_curves.png` — Loss, accuracy, and AUC/C-index curves per fold
2. `kaplan_meier.png` — Kaplan-Meier survival curves by predicted risk group
3. `tsne_embeddings.png` — 2D t-SNE projection of patient GNN embeddings
4. `umap_embeddings.png` — 2D UMAP projection of patient GNN embeddings
5. `model_comparison.png` — Bar chart comparing all models

### Output Artifacts

```
results/
├── training_results.json       # Full fold-level metrics + training history
├── evaluation_report.json      # Baselines, ablation, model states
├── explainability_results.json # GNNExplainer + SHAP top features
├── model_fold{0..4}.pt         # Saved model checkpoints (~10.8 MB each)
├── training_curves.png         # Training dynamics visualization
├── kaplan_meier.png            # Survival analysis plot
├── tsne_embeddings.png         # t-SNE embedding visualization
├── umap_embeddings.png         # UMAP embedding visualization
├── model_comparison.png        # Cross-model comparison chart
└── risk_heatmap.png            # Gene importance network graph
```

---

## Configuration Reference

All pipeline behavior is controlled by `configs/config.yaml`:

```yaml
data:
  tcga_project: TCGA-BRCA
  min_total_counts: 1000          # Gene filtering threshold
  n_genes_lasso: 200              # Target gene count after LASSO
  string_confidence_threshold: 700 # STRING PPI confidence cutoff
  survival_bins: [365, 1095, 1825] # Survival class boundaries (days)

model:
  gat_layers: 3                   # Number of GAT layers
  gat_heads: [8, 4, 1]           # Attention heads per layer
  hidden_dim: 128                 # GNN hidden dimension
  dropout: 0.4                    # Dropout rate
  llm_embedding_dim: 768          # BioBERT output dimension
  llm_model: "dmis-lab/biobert-base-cased-v1.2"

training:
  lr: 0.001                       # Initial learning rate
  weight_decay: 1.0e-4            # L2 regularization
  epochs: 200                     # Maximum training epochs
  patience: 20                    # Early stopping patience
  batch_size: 32                  # Mini-batch size
  cv_folds: 5                     # Cross-validation folds
  aux_loss_weight: 0.3            # Auxiliary task loss weight

hybrid:
  rf_n_estimators: 500            # Random Forest trees
  rf_calibrated: true             # Use isotonic calibration
  rf_min_samples_split: 5         # Min samples per RF split
```

---

## Reproducibility

- **Random seeds**: Set to 42 across Python, NumPy, PyTorch, and CUDA
- **Deterministic mode**: `torch.backends.cudnn.deterministic = True`
- **CV splits**: `StratifiedKFold` with fixed `random_state=42`
- **Device**: Supports CUDA, Apple MPS, and CPU (auto-detected)
