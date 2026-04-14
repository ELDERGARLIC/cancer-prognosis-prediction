# Pipeline Documentation

This document describes each stage of the breast cancer prognosis prediction pipeline, including data flow, intermediate artifacts, and architectural decisions.

---

## End-to-End Flow

```mermaid
flowchart TD
    MAIN["main.py<br/>(Orchestrator)"] --> S1 & S2 & S3

    S1["Stage 1<br/>Data Acq. & Preproc"] --> S2["Stage 2<br/>KG + LLM Embeddings"]
    S2 --> S3["Stage 3<br/>Dataset Build (PyG + CV)"]
    S3 --> S4["Stage 4<br/>Training (5-Fold CV)"]
    S4 --> S5["Stage 5<br/>Evaluation & Explainability"]

    style S1 fill:#e8f5e9,stroke:#2e7d32
    style S2 fill:#e3f2fd,stroke:#1565c0
    style S3 fill:#fff3e0,stroke:#e65100
    style S4 fill:#fce4ec,stroke:#c62828
    style S5 fill:#f3e5f5,stroke:#6a1b9a
```

---

## Stage 1: Data Acquisition & Preprocessing

**Entry point**: `run_stage1(config)` → calls `run_data_download()` then `run_preprocessing()`

### 1a. Data Download (`src/data_download.py`)

```mermaid
flowchart LR
    GDC["GDC REST API<br/>(or Xena)"] -->|RNA-Seq| EXPR["TCGA-BRCA Expression<br/>20,530 genes × 1,218 samples<br/>→ data/raw/tcga_brca_expression.tsv.gz"]
    GDC_C["GDC Cases API<br/>(or Xena)"] -->|Clinical| CLIN["Clinical Data<br/>1,098 patients, 11 columns<br/>→ data/raw/tcga_brca_clinical.tsv"]
    STRING["STRING v12.0"] -->|PPI| PPI["Protein-Protein Interactions<br/>→ data/knowledge_graph/string_ppi.tsv"]
    DISGENET["DisGeNET API"] -->|Gene-Disease| GDA["Gene-Disease Associations<br/>→ data/knowledge_graph/disgenet_gene_disease.tsv"]
    NCBI["NCBI Entrez"] -->|Summaries| SUM["Gene Functional Summaries (1,750 genes)<br/>→ data/embeddings/gene_summaries.json"]
```

**Fallback strategy**: GDC is the primary source. If the GDC bulk download fails (e.g., 500 error), the pipeline falls back to UCSC Xena mirrors automatically.

### 1b. Preprocessing (`src/preprocessing.py`)

```mermaid
flowchart TD
    RAW["Expression Matrix<br/>20,530 genes × 1,218 samples"] --> FILT["Low-Expression Gene Filter<br/>Removes total counts ≤ 1,000<br/>20,530 → 17,343 genes"]
    FILT --> NORM["Normalization (z-score)<br/>log2(x+1) if raw, then z-score per gene<br/>Skips log if pre-normalized (max ≤ 30)"]
    NORM --> IMP["Clinical Imputation<br/>KNN (k=5) for numeric columns<br/>All-NaN columns filled with 0"]
    IMP --> SURV["Survival Label Discretization<br/>< 365d (Class 0), 365–1095d (Class 1)<br/>1095–1825d (Class 2), >1825d (Class 3)"]
    SURV --> LASSO["LASSO Feature Selection<br/>LassoCV (10-fold, 100 alphas)<br/>93 non-zero + 107 high-variance = 200"]
    LASSO --> DGN["DisGeNET Filter (optional)<br/>Intersection too small → keep all 200"]
    DGN --> CLIN["Clinical Feature Extraction<br/>age, stage I–IV (one-hot), gender → 6 features"]
    CLIN --> OUT1["expression_selected.tsv<br/>200 × 1,217"]
    CLIN --> OUT2["selected_genes.txt<br/>200 genes"]
    CLIN --> OUT3["survival_labels.tsv"]
    CLIN --> OUT4["clinical_features.tsv<br/>6 features"]

    style LASSO fill:#ffcdd2,stroke:#c62828
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

```mermaid
graph TD
    subgraph KG["Knowledge Graph (200 gene nodes)"]
        G1["Gene"] ---|"STRING PPI<br/>64 edges, conf ≥ 700"| G2["Gene"]
        G1 ---|"KEGG<br/>173 edges, 126 pathways"| P["Pathway"]
        G1 -.-|"DisGeNET<br/>0 edges*"| D["Disease"]
    end

    KG --- STATS["Statistics:<br/>Density: 0.0016 | Avg degree: 0.32<br/>Max degree: 6 | Isolated: 162 (81%)"]

    style D fill:#ffcdd2,stroke:#c62828
    style STATS fill:#fff9c4,stroke:#f57f17
```

\* No overlap between LASSO-selected genes and DisGeNET entries.

**Edge construction details**:
- **STRING PPI**: Loads 13.7M human protein links, filters to confidence ≥ 700, maps protein IDs to gene symbols, intersects with 200 selected genes → 64 bidirectional edges
- **DisGeNET**: Filters for "Neoplastic Process" semantic type, but no selected genes appear in the breast cancer association set → 0 edges
- **KEGG**: REST API batch query per gene; finds 126 pathways containing at least one selected gene → 173 gene-pathway membership edges

### 2b. LLM Embeddings (`src/llm_embeddings.py`)

```mermaid
flowchart LR
    SUM["Gene Summaries<br/>(NCBI Entrez text)<br/>1,750 entries"] --> BIO["BioBERT v1.2<br/>[CLS] pooling<br/>max_length=512"]
    BIO --> EMB["Gene Embeddings<br/>(200 × 768)<br/>float32 .npy"]

    style BIO fill:#e3f2fd,stroke:#1565c0
```

- **Model**: `dmis-lab/biobert-base-cased-v1.2` (PubMed + PMC pre-trained BERT)
- **Input**: Gene functional summary text from NCBI (e.g., "This gene encodes a member of the...")
- **Output**: 768-dimensional `[CLS]` token embedding per gene
- **Pathway/disease embeddings**: Generated from synthetic descriptions (e.g., "KEGG pathway: Cell cycle — genes involved in cell cycle regulation")

### 2c. FAISS Vector Store (`src/vector_store.py`)

```mermaid
flowchart LR
    EMB["Gene Embeddings<br/>(200 × 768)"] --> FAISS["FAISS IndexFlatL2<br/>Exact L2 nearest neighbor"]
    FAISS --> CHECK["Sanity check:<br/>RPL13AP6 neighbors:<br/>→ RPL19P12 (d=4.98)<br/>→ MRPL42P5 (d=10.33)"]

    style FAISS fill:#c8e6c9,stroke:#2e7d32
```

---

## Stage 3: Dataset Construction

**Entry point**: `run_stage3(config)` → calls `build_dataset()`

### Patient-Weighted Embeddings (GenePT-w)

```mermaid
flowchart LR
    GE["Gene Embeddings<br/>(200 × 768)"] --> MUL["⊗ Element-wise multiply<br/>emb[p,g,:] = expr[p,g] × gene_emb[g,:]"]
    EX["Expression Matrix<br/>(1,217 × 200)"] --> MUL
    MUL --> PE["Patient Embeddings<br/>(1,217 × 200 × 768)"]

    style MUL fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

Each patient gets a **personalized graph** where node features are the gene's BioBERT embedding scaled by that patient's expression level for that gene. This encodes both functional meaning (from the LLM) and patient-specific expression signal.

### PyTorch Geometric Dataset

```mermaid
flowchart TD
    subgraph DATASET["BreastCancerGraphDataset (1,217 patients)"]
        direction TB
        subgraph PATIENT["Per patient (one Data object)"]
            X["x: [200, 768] — node features"]
            EI["edge_index: [2, 237] — shared KG topology"]
            CL["clinical: [1, 6] — clinical features"]
            Y["y: scalar — survival class (0–3)"]
            OS["os_time: scalar — overall survival (days)"]
            TS["tumor_stage: scalar — stage for aux task"]
        end
        CV["5-Fold Stratified CV<br/>Each fold: 956 train / 239 validation"]
    end

    style DATASET fill:#fff3e0,stroke:#e65100
```

**SMOTE handling**: The flattened feature dimension (200 × 768 + 6 = 153,606) exceeds the `MAX_SMOTE_FEATURES` threshold of 50,000. SMOTE is skipped and **class-weighted cross-entropy loss** is used instead for class imbalance.

---

## Stage 4: Model Training

**Entry point**: `run_stage4(config)` → calls `run_training()`

### Model Architecture

```mermaid
flowchart TD
    IN["Input: x [B×200, 768]<br/>edge_index [2, E]"] --> GAT1

    subgraph GAT["BioKG_GAT (GNN Backbone)"]
        GAT1["GAT Layer 1<br/>768→128, 8 heads<br/>+BN +ELU +Dropout +Residual"] --> GAT2["GAT Layer 2<br/>128→128, 4 heads<br/>+BN +ELU +Dropout +Residual"]
        GAT2 --> GAT3["GAT Layer 3<br/>128→128, 1 head<br/>+BN +ELU +Dropout"]
        GAT3 --> POOL["Graph Pooling<br/>mean ‖ max → [B, 256]"]
    end

    POOL --> CAT
    CLIN["Clinical Features [B, 6]"] --> CAT["Concatenate → [B, 262]"]

    CAT --> MAIN["FC Head (Main Task)<br/>262 → 64 → BN → ELU → Drop<br/>→ 32 → 4 (survival class)"]

    POOL --> AUX["Aux Head (Staging)<br/>256 → 32 → ReLU → Drop → 4 stages"]

    MAIN --> PARAMS["Total Parameters: 2,712,072 (all trainable)"]
    AUX --> PARAMS

    style GAT fill:#e3f2fd,stroke:#1565c0
    style MAIN fill:#c8e6c9,stroke:#2e7d32
    style AUX fill:#fff9c4,stroke:#f57f17
```

### Training Loop

```mermaid
flowchart TD
    FOLD{{"For each fold (1..5)"}} --> INIT["Initialize model (fresh weights)"]
    INIT --> WEIGHTS["Compute class weights (inverse frequency)"]
    WEIGHTS --> DL["Create DataLoaders (batch_size=32)"]

    DL --> EPOCH{{"For each epoch (max 200)"}}
    EPOCH --> TRAIN["Train: weighted CE + 0.3 × aux loss<br/>+ gradient clipping (max_norm=1.0)"]
    TRAIN --> VAL["Validate: loss, accuracy, macro AUC, C-index"]
    VAL --> SCHED["ReduceLROnPlateau (factor=0.5, patience=10)"]

    SCHED --> BEST{"New best<br/>val_loss?"}
    BEST -->|Yes| SAVE["Save best model checkpoint"]
    SAVE --> EPOCH
    BEST -->|No| PAT{"Patience<br/>exhausted?"}
    PAT -->|"No (< 20)"| EPOCH
    PAT -->|"Yes (≥ 20)"| STOP["Early stopping"]

    STOP --> RF_TRAIN["Hybrid RF Training:<br/>Extract GNN embeddings → Train RF (500 trees)<br/>→ Calibrate (isotonic) → Evaluate on val fold"]

    RF_TRAIN --> FOLD

    style STOP fill:#ffcdd2,stroke:#c62828
    style SAVE fill:#c8e6c9,stroke:#2e7d32
    style RF_TRAIN fill:#e3f2fd,stroke:#1565c0
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

```mermaid
flowchart TD
    subgraph RES["results/"]
        direction TB
        JSON1["training_results.json — Full fold-level metrics + history"]
        JSON2["evaluation_report.json — Baselines, ablation, model states"]
        JSON3["explainability_results.json — GNNExplainer + SHAP"]
        MODELS["model_fold0..4.pt — Model checkpoints (~10.8 MB each)"]
        subgraph FIGS["figures/"]
            FIG1["training_curves.png"]
            FIG2["kaplan_meier.png"]
            FIG3["tsne_embeddings.png"]
            FIG4["umap_embeddings.png"]
            FIG5["model_comparison.png"]
            FIG6["risk_heatmap.png"]
        end
    end

    style RES fill:#f3e5f5,stroke:#6a1b9a
    style FIGS fill:#e1bee7,stroke:#8e24aa
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
