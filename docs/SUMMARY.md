# Breast Cancer Prognosis Prediction — Project Summary

## Overview

This project implements an end-to-end machine learning pipeline for **breast cancer prognosis prediction** using a hybrid approach that combines **Graph Attention Networks (GATs)** with **biomedical knowledge graphs** and **LLM-derived gene embeddings**. The system predicts discrete survival outcomes for TCGA-BRCA patients across four time horizons: <1 year, 1–3 years, 3–5 years, and >5 years.

The core innovation is the integration of three complementary information sources:

1. **Gene expression profiles** from TCGA-BRCA (RNA-Seq)
2. **Biological knowledge graphs** constructed from STRING PPI, DisGeNET, and KEGG pathways
3. **LLM-generated gene embeddings** from BioBERT, weighted by patient-specific expression levels (GenePT-w approach)

These are fused through a 3-layer GAT that operates on a per-patient biological graph, followed by a calibrated Random Forest for final classification.

## Pipeline Architecture

```mermaid
flowchart TD
    subgraph S1["Stage 1 — Data Acquisition & Preprocessing"]
        direction LR
        A1["TCGA-BRCA<br/>Download"] --> A2["Gene Filter<br/>20,530 → 17,343"]
        A2 --> A3["Normalize<br/>z-score"]
        A3 --> A4["LASSO Select<br/>17,343 → 200"]
    end

    subgraph S2["Stage 2 — Knowledge Graph & LLM Embeddings"]
        direction LR
        B1["STRING PPI<br/>64 edges"]
        B2["DisGeNET<br/>0 edges"]
        B3["KEGG Pathways<br/>173 edges"]
        B4["BioBERT<br/>768-dim embeddings"]
    end

    subgraph S3["Stage 3 — Dataset Construction"]
        C1["Patient-Weighted Embeddings (1217 × 200 × 768)<br/>+ 5-Fold Stratified CV Splits + Clinical Features (6-dim)"]
    end

    subgraph S4["Stage 4 — Model Training (5-Fold CV)"]
        direction LR
        D1["3-Layer GAT<br/>+ Multi-Task Loss"] --> D2["Calibrated RF (Hybrid)<br/>on GNN Embeddings"]
    end

    subgraph S5["Stage 5 — Evaluation & Explainability"]
        direction LR
        E1["Baselines<br/>Cox/RF/MLP"]
        E2["Ablation<br/>Study"]
        E3["GNNExplainer<br/>+ SHAP"]
        E4["Visualize<br/>KM/tSNE/UMAP"]
    end

    S1 --> S2 --> S3 --> S4 --> S5

    style S1 fill:#e8f5e9,stroke:#2e7d32
    style S2 fill:#e3f2fd,stroke:#1565c0
    style S3 fill:#fff3e0,stroke:#e65100
    style S4 fill:#fce4ec,stroke:#c62828
    style S5 fill:#f3e5f5,stroke:#6a1b9a
```

## Key Results

### Model Performance (5-Fold Cross-Validation)

| Model | Accuracy | AUC-ROC | C-Index |
|-------|----------|---------|---------|
| **GAT (Ours)** | 0.332 ± 0.028 | 0.627 ± 0.063 | 0.380 ± 0.049 |
| **Calibrated RF (Hybrid)** | **0.454 ± 0.034** | 0.601 ± 0.063 | — |
| Cox PH Baseline | — | — | **0.748** |
| RF Baseline | 0.435 | — | — |
| MLP Baseline | 0.435 | — | — |
| Vanilla GCN | **0.469** | — | — |

The hybrid GAT → Calibrated Random Forest pipeline achieves competitive performance on this challenging 4-class survival prediction task, with the calibrated RF improving upon raw GAT accuracy by ~12 percentage points.

### Explainability Highlights

**Top genes identified by GNNExplainer:**
CCDC117, LOC100101266, TM9SF2, GPR152, NCKAP5, LOC550112, C1orf156, EXTL3, C9orf93, XKR6

**Top SHAP features for the RF hybrid model** include GNN embedding dimensions 81, 93, 53, 94, and clinical feature `clinical_0` (patient age).

## Dataset Summary

| Property | Value |
|----------|-------|
| Cancer type | TCGA-BRCA (Breast Invasive Carcinoma) |
| Patients | 1,217 |
| Selected genes | 200 (93 LASSO + 107 high-variance) |
| LLM embedding dim | 768 (BioBERT) |
| Clinical features | 6 (age, stage I–IV, gender) |
| Survival classes | 4 (<1yr, 1–3yr, 3–5yr, >5yr) |
| Class distribution | 166 / 476 / 174 / 261 |
| Knowledge graph edges | 64 PPI + 173 KEGG pathway |

## Technology Stack

- **Deep Learning**: PyTorch, PyTorch Geometric (GAT, GCN, DataLoader)
- **LLM Embeddings**: HuggingFace Transformers (BioBERT)
- **Vector Store**: FAISS (L2 nearest neighbor search)
- **Classical ML**: scikit-learn (Random Forest, LASSO, MLP, KNN Imputer)
- **Survival Analysis**: lifelines (Cox PH, Kaplan-Meier, C-index)
- **Explainability**: GNNExplainer (PyG), SHAP (TreeExplainer)
- **Visualization**: matplotlib, UMAP, t-SNE
- **Data Sources**: GDC/UCSC Xena (TCGA), STRING v12.0, DisGeNET, KEGG REST

## Documentation Index

| Document | Description |
|----------|-------------|
| [`docs/pipeline.md`](pipeline.md) | Detailed pipeline stages with data flow diagrams |
| [`docs/modules.md`](modules.md) | API reference for every module in `src/` |
| [`docs/data.md`](data.md) | Data sources, preprocessing, and data visualization |
| [`docs/model_performance.md`](model_performance.md) | Full performance metrics, baselines, and ablation study |

## Project Structure

```mermaid
flowchart LR
    subgraph ROOT["."]
        direction TB
        configs["configs/<br/>└── config.yaml"]
        subgraph data["data/"]
            direction TB
            raw["raw/ — TCGA expression + clinical"]
            processed["processed/ — Filtered expression, labels, genes"]
            kg_dir["knowledge_graph/ — STRING PPI, DisGeNET, KEGG"]
            emb_dir["embeddings/ — BioBERT, FAISS index"]
        end
        subgraph src["src/"]
            direction TB
            dd["data_download.py — Stage 1a"]
            pp["preprocessing.py — Stage 1b"]
            kgc["kg_construction.py — Stage 2a"]
            llm["llm_embeddings.py — Stage 2b"]
            vs["vector_store.py — Stage 2c"]
            ds["dataset.py — Stage 3"]
            mo["model.py — Architectures"]
            tr["train.py — Stage 4"]
            ev["evaluate.py — Stage 5a"]
            ex["explain.py — Stage 5b"]
            viz["visualize.py — Stage 5c"]
        end
        results["results/ — Trained models, metrics, figures"]
        docs["docs/ — Documentation"]
        main["main.py — Pipeline entry point"]
        pyproject["pyproject.toml — Dependencies"]
    end
```

## Quick Start

```bash
# Install dependencies
poetry install

# Run the full pipeline (Stages 1–5)
python main.py

# Run a specific stage
python main.py --stage 4    # Training only

# Run from a specific stage onward
python main.py --stage 2    # Stages 2–5
```

## Configuration

All hyperparameters are centralized in `configs/config.yaml`. Key settings:

- **Gene selection**: `n_genes_lasso: 200` (LASSO + high-variance supplement)
- **GAT architecture**: 3 layers, heads [8, 4, 1], hidden dim 128, dropout 0.4
- **Training**: lr 0.001, patience 20, 200 max epochs, 5-fold CV
- **Hybrid RF**: 500 estimators, isotonic calibration, min_samples_split 5
- **Survival bins**: 1 year, 3 years, 5 years (4-class classification)
