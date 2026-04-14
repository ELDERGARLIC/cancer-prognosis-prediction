# Data Documentation

This document describes all data sources, the preprocessing pipeline, intermediate data artifacts, and data-level visualizations.

---

## Data Sources

### 1. TCGA-BRCA Gene Expression

| Property | Value |
|----------|-------|
| **Source** | GDC (primary), UCSC Xena (fallback) |
| **Assay** | RNA-Seq (STAR-Counts / HiSeqV2 RSEM) |
| **Project** | TCGA-BRCA (Breast Invasive Carcinoma) |
| **Raw dimensions** | 20,530 genes × 1,218 samples |
| **Format** | TSV (gzipped) |
| **File** | `data/raw/tcga_brca_expression.tsv.gz` |

The expression matrix contains normalized RNA-Seq values (RSEM from Xena, or STAR-Counts from GDC). Each row is a gene (HUGO symbol), each column is a TCGA sample.

### 2. TCGA-BRCA Clinical Data

| Property | Value |
|----------|-------|
| **Source** | GDC Cases API (primary), UCSC Xena (fallback) |
| **Patients** | 1,098 |
| **Columns** | 11 |
| **File** | `data/raw/tcga_brca_clinical.tsv` |

**Clinical columns**:

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | TCGA barcode (12-char) |
| `OS.time` | float | Overall survival time (days) |
| `OS` | int | Event indicator (1=death, 0=censored) |
| `age_at_diagnosis` | float | Age in years |
| `gender` | string | `female` / `male` |
| `tumor_stage` | string | AJCC stage (e.g., "Stage IIA") |
| `vital_status` | string | `Alive` / `Dead` |
| `days_to_death` | float | Days from diagnosis to death (if applicable) |
| `days_to_last_follow_up` | float | Days to last known alive date |
| `er_status` | string | Estrogen receptor status |
| `pr_status` | string | Progesterone receptor status |

### 3. STRING Protein-Protein Interactions

| Property | Value |
|----------|-------|
| **Source** | STRING v12.0 |
| **Species** | Human (taxonomy 9606) |
| **Raw edges** | ~13.7 million |
| **Confidence filter** | ≥ 700 (high confidence) |
| **After filter** | 473,860 edges |
| **After gene filter** | 64 edges (between 200 selected genes) |
| **File** | `data/knowledge_graph/string_ppi.tsv` |

### 4. DisGeNET Gene-Disease Associations

| Property | Value |
|----------|-------|
| **Source** | DisGeNET v7+ (API or curated fallback) |
| **Disease** | Breast Carcinoma (UMLS CUI: C0006142) |
| **Semantic type** | Neoplastic Process |
| **Associations** | 100 gene-disease pairs |
| **Overlap with selected genes** | 0 (no intersection) |
| **File** | `data/knowledge_graph/disgenet_gene_disease.tsv` |

### 5. KEGG Pathway Memberships

| Property | Value |
|----------|-------|
| **Source** | KEGG REST API |
| **Pathways found** | 126 (for selected 200 genes) |
| **Gene-pathway edges** | 173 |
| **Query method** | Per-gene batch lookup |

### 6. NCBI Gene Summaries

| Property | Value |
|----------|-------|
| **Source** | NCBI Entrez (esearch + esummary) |
| **Genes with summaries** | 1,750 |
| **Format** | JSON (`{gene_symbol: summary_text}`) |
| **File** | `data/embeddings/gene_summaries.json` |

---

## Preprocessing Pipeline

### Step 1: Low-Expression Gene Filtering

```mermaid
flowchart LR
    BEFORE["20,530 genes"] -->|"Filter: total expression > 1,000"| AFTER["17,343 genes<br/>(removed 3,187)"]

    style BEFORE fill:#ffcdd2
    style AFTER fill:#c8e6c9
```

This removes genes that are either not expressed or expressed at very low levels in the cohort, reducing noise in downstream analyses.

### Step 2: Expression Normalization

```mermaid
flowchart TD
    IN["Input: Raw expression matrix<br/>17,343 × 1,217"] --> CHECK{"max(expr) > 30?"}
    CHECK -->|"No (20.98)"| SKIP["Skip log2(x+1)<br/>Already pre-normalized"]
    CHECK -->|Yes| LOG["Apply log2(x+1)"]
    SKIP --> ZSCORE["Z-score per gene<br/>expr[g] = (expr[g] − mean) / std"]
    LOG --> ZSCORE
    ZSCORE --> OUT["Standardized matrix<br/>zero mean, unit variance"]

    style SKIP fill:#c8e6c9
```

### Step 3: Clinical Data Imputation

```mermaid
flowchart TD
    RAW["Numeric clinical columns"] --> CHK{"All-NaN<br/>column?"}
    CHK -->|Yes| FILL["Fill with 0<br/>e.g. tumor_stage<br/>(logged as warning)"]
    CHK -->|No| KNN["KNN Imputation (k=5)<br/>2,262 missing values filled"]
    ID["ID columns"] --> SKIP["Excluded from imputation"]
    FILL --> OUT["Imputed clinical DataFrame"]
    KNN --> OUT

    style FILL fill:#fff9c4,stroke:#f57f17
    style KNN fill:#c8e6c9,stroke:#2e7d32
```

### Step 4: Survival Label Discretization

Continuous overall survival time is binned into 4 discrete classes:

```mermaid
flowchart LR
    C0["Class 0<br/><365 days (<1yr)<br/>166 pts (13.6%)"] --- C1["Class 1<br/>365–1095 days (1–3yr)<br/>476 pts (39.1%)"]
    C1 --- C2["Class 2<br/>1095–1825 days (3–5yr)<br/>174 pts (14.3%)"]
    C2 --- C3["Class 3<br/>>1825 days (>5yr)<br/>261 pts (21.4%)"]

    style C0 fill:#ffcdd2,stroke:#c62828
    style C1 fill:#c8e6c9,stroke:#2e7d32
    style C2 fill:#fff9c4,stroke:#f57f17
    style C3 fill:#bbdefb,stroke:#1565c0
```

```mermaid
pie title Survival Class Distribution (1,217 patients)
    "<1 year (Class 0)" : 166
    "1-3 years (Class 1)" : 476
    "3-5 years (Class 2)" : 174
    ">5 years (Class 3)" : 261
```

**Class imbalance**: Class 1 (1–3 years) is the majority class at 39.1%, while Class 2 (3–5 years) is the smallest at 14.3%. This ~2.9:1 imbalance ratio is addressed through class-weighted cross-entropy loss.

### Step 5: LASSO Feature Selection

```mermaid
flowchart TD
    IN["Input: 17,343 genes, 1,195 samples"] --> LASSO["LassoCV<br/>10-fold inner CV, 100 alpha values<br/>Runtime: ~18 minutes"]
    LASSO --> NZ["93 genes with non-zero<br/>LASSO coefficients"]
    LASSO --> HV["107 supplementary<br/>high-variance genes"]
    NZ & HV --> TOTAL["Total: 200 selected genes"]

    TOTAL --> NOTE["Config: n_genes_lasso=200<br/>Originally 1500, reduced to prevent OOM<br/>during SMOTE and embedding construction"]

    style LASSO fill:#ffcdd2,stroke:#c62828
    style TOTAL fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

### Step 6: DisGeNET Cross-Reference

```mermaid
flowchart LR
    LASSO["200 LASSO genes"] --> INT{"Intersection<br/>≥ 10%?"}
    DGN["100 DisGeNET genes"] --> INT
    INT -->|"No (0 overlap)"| KEEP["Keep all 200 LASSO genes<br/>DisGeNET genes added as<br/>extra KG context"]

    style INT fill:#fff9c4,stroke:#f57f17
    style KEEP fill:#c8e6c9,stroke:#2e7d32
```

### Step 7: Clinical Feature Extraction

```mermaid
flowchart LR
    subgraph FEATURES["6 Clinical Features"]
        direction TB
        F1["1. age — continuous, normalized"]
        F2["2. stage_I — binary, one-hot"]
        F3["3. stage_II — binary, one-hot"]
        F4["4. stage_III — binary, one-hot"]
        F5["5. stage_IV — binary, one-hot"]
        F6["6. is_female — binary"]
    end

    style FEATURES fill:#e3f2fd,stroke:#1565c0
```

---

## Processed Data Artifacts

| File | Shape / Size | Description |
|------|-------------|-------------|
| `data/processed/expression_selected.tsv` | 200 × 1,217 | Expression matrix (selected genes only) |
| `data/processed/selected_genes.txt` | 200 lines | Gene symbols (one per line) |
| `data/processed/survival_labels.tsv` | 1,217 × 4 | Labels + OS.time + OS + survival_class |
| `data/processed/clinical_features.tsv` | 1,217 × 6 | Extracted clinical features |
| `data/knowledge_graph/kg_edges.pt` | PyTorch tensor | Combined KG edge_index |
| `data/knowledge_graph/kg_metadata.json` | JSON | Gene/pathway/disease index mappings |
| `data/embeddings/gene_embeddings.npy` | (200, 768) float32 | BioBERT gene embeddings |
| `data/embeddings/gene_summaries.json` | 1,750 entries | NCBI gene functional summaries |
| `data/embeddings/faiss_index.bin` | FAISS index | L2 nearest neighbor index |
| `data/embeddings/gene_names.json` | 200 entries | Gene names for FAISS index |

---

## Data Visualizations

The pipeline generates several data-level visualizations saved to `results/`.

### Kaplan-Meier Survival Curves

**File**: `results/kaplan_meier.png`

Shows survival probability over time, stratified by model-predicted risk group. Patients are grouped by their predicted survival class, and the KM estimator computes the empirical survival function for each group. Shaded regions indicate 95% confidence intervals.

```mermaid
xychart-beta
    title "Kaplan-Meier Survival Curves (Schematic)"
    x-axis "Time (days)" [0, 500, 1000, 1500, 2000, 2500]
    y-axis "Survival Probability" 0 --> 1
    line ">5yr (best)" [1.0, 0.95, 0.88, 0.80, 0.72, 0.65]
    line "3-5yr" [1.0, 0.85, 0.70, 0.55, 0.42, 0.35]
    line "1-3yr" [1.0, 0.75, 0.55, 0.40, 0.30, 0.22]
    line "<1yr (worst)" [1.0, 0.50, 0.28, 0.15, 0.08, 0.05]
```

### t-SNE Embedding Projection

**File**: `results/tsne_embeddings.png`

2D t-SNE projection (perplexity=30) of patient GNN embeddings from the best fold-0 model. Each point is a patient, colored by true survival class. Shows how well the learned representations separate different prognosis groups.

### UMAP Embedding Projection

**File**: `results/umap_embeddings.png`

2D UMAP projection of the same patient embeddings. UMAP tends to better preserve global structure compared to t-SNE, revealing broader cluster patterns.

### Training Curves

**File**: `results/training_curves.png`

Three-panel figure showing:
1. **Loss curves**: Training loss (decreasing) and validation loss (with early-stopping point marked)
2. **Accuracy**: Validation accuracy progression across epochs
3. **AUC & C-index**: Validation AUC-ROC and concordance index over training

### Risk Heatmap (Gene Importance Network)

**File**: `results/risk_heatmap.png`

Network graph visualization of the top 30 most important genes identified by GNNExplainer:
- **Node size**: Proportional to gene importance score
- **Node color**: Red (high importance) to blue (lower importance)
- **Edges**: Knowledge graph connections (PPI + pathway) between the visualized genes

### Model Comparison Bar Chart

**File**: `results/model_comparison.png`

Bar chart comparing all evaluated models on accuracy and other metrics:
- GAT (ours)
- Calibrated Random Forest (hybrid)
- Cox PH baseline
- Random Forest baseline
- MLP baseline
- Vanilla GCN

---

## Knowledge Graph Statistics

```mermaid
flowchart TD
    subgraph NODES["Nodes"]
        N1["200 genes"]
        N2["126 KEGG pathways"]
        N3["0 diseases (no overlap)"]
    end

    subgraph EDGES["Edges"]
        E1["64 gene-gene (STRING PPI, conf ≥ 700)"]
        E2["173 gene-pathway (KEGG membership)"]
        E3["0 gene-disease (DisGeNET)"]
    end

    subgraph PROPS["Graph Properties"]
        P1["Density: 0.0016"]
        P2["Avg degree: 0.32"]
        P3["Max degree: 6"]
        P4["Isolated genes: 162 (81%)"]
    end

    NODES --- EDGES --- PROPS

    style NODES fill:#e3f2fd,stroke:#1565c0
    style EDGES fill:#fff3e0,stroke:#e65100
    style PROPS fill:#fff9c4,stroke:#f57f17
```

**Note**: The graph is very sparse because LASSO-selected genes are statistically informative but not necessarily physically interacting. The GAT still learns useful representations through self-loops and the global pooling mechanism.

---

## Patient Embedding Construction

The GenePT-w (weighted) approach creates patient-specific node features by scaling BioBERT gene embeddings with expression levels:

```mermaid
flowchart TD
    EXPR["expression[p, g]<br/>z-score normalized value<br/>(1,217 × 200)"] --> MUL["⊗ Multiply<br/>node_feature[p,g,:] = expression[p,g] × biobert[g,:]"]
    BIO["biobert[g, :]<br/>768-dim BioBERT [CLS] embedding<br/>(200 × 768)"] --> MUL
    MUL --> RESULT["Patient Embedding Tensor<br/>(1,217 × 200 × 768)<br/>Memory: ~712 MB"]

    style MUL fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style RESULT fill:#c8e6c9,stroke:#2e7d32
```

This ensures each patient's graph has node features that encode both:
1. **Functional meaning** of the gene (from BioBERT, trained on biomedical literature)
2. **Patient-specific signal** (how much that gene is expressed in this patient)

---

## Selected Genes (Top 20 by LASSO)

The first 93 genes were selected by LASSO (non-zero coefficients), followed by 107 high-variance supplement genes. The first 20 LASSO-selected genes:

| # | Gene | Description |
|---|------|-------------|
| 1 | RPL13AP6 | Ribosomal protein L13a pseudogene 6 |
| 2 | CLEC9A | C-type lectin domain family 9 member A |
| 3 | ST7OT1 | ST7 overlapping transcript 1 |
| 4 | KLF7 | Kruppel-like factor 7 |
| 5 | C11orf17 | Chromosome 11 open reading frame 17 |
| 6 | RPL39 | Ribosomal protein L39 |
| 7 | C6orf81 | Chromosome 6 open reading frame 81 |
| 8 | POU2F1 | POU class 2 homeobox 1 |
| 9 | HOXA11 | Homeobox A11 |
| 10 | FLT3 | Fms-related receptor tyrosine kinase 3 |
| 11 | C9orf163 | Chromosome 9 open reading frame 163 |
| 12 | STEAP2 | STEAP2 metalloreductase |
| 13 | HSPA8 | Heat shock protein family A member 8 |
| 14 | IRF2 | Interferon regulatory factor 2 |
| 15 | EFHB | EF-hand domain family member B |
| 16 | IGFBP5 | Insulin-like growth factor binding protein 5 |
| 17 | PELO | Pelota mRNA surveillance and ribosome rescue factor |
| 18 | TUBGCP3 | Tubulin gamma complex associated protein 3 |
| 19 | C9orf93 | Chromosome 9 open reading frame 93 |
| 20 | PTGS1 | Prostaglandin-endoperoxide synthase 1 |
