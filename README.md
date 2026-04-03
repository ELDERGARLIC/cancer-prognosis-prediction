# Breast Cancer Prognosis Prediction using Knowledge Graph-Enhanced Graph Neural Networks

This repository contains the implementation of a cutting-edge hybrid framework for breast cancer prognosis prediction. The project fuses **Graph Neural Networks (GNNs)** with **Retrieval-Augmented Generation (RAG)** and **Large Language Model (LLM)** embeddings to create a biologically aware, high-precision survival analysis model.

## 🏥 Overview

Breast cancer prognosis often faces the **p >> n problem**, where genetic features vastly outnumber patient samples. Traditional models treat genes as independent variables, ignoring the complex regulatory networks (Graph Topology) and deep biological functions (Semantic Context) revealed in millions of biomedical documents.

This project addresses the "Prognosis Gap" (Liang et al., 2025) by integrating **The Cancer Genome Atlas (TCGA)** multi-omic data with structured **Knowledge Graphs (BioKG, DisGeNET)** and unstructured biomedical text using a **RAG-GNN** architecture.

## 🚀 Core Innovations (2026-Level Architecture)

1.  **RAG-GNN Integration**: Moving beyond topology-only GNNs. We utilize **Retrieval-Augmented Generation** to pull functional context from PubMed and NCBI, addressing the failure of standard GNNs to capture biological function (Hays & Richardson, 2026).
2.  **Semantic Gene Embeddings (GenePT)**: Instead of random initialization, nodes are initialized with **1536-dimensional semantic vectors** generated from NCBI summaries using LLMs (e.g., GPT-3.5/BioBERT). This captures deep interaction contexts (Chen & Zou, 2023).
3.  **Patient-Specific Dynamic Fusion**: We implement a revolutionary weighting system where static LLM gene embeddings are multiplied by the patient's individual **TCGA mRNA expression levels**. This ensures **patient-specific feature emphasis** rather than population averages (Vavekanand, 2026).
4.  **Vector Database Optimization**: Use of **FAISS/ChromaDB** for high-speed retrieval of BioKG edges and semantic literature, preventing RAM overhead during large-scale graph processing.

## 📊 Methodology

### 1. Data Prep & Semantic Embedding
*   **Transcriptomics**: TCGA-BRCA mRNA-seq counts.
*   **Knowledge Retrieval**: Scraping NCBI summaries for 1000+ key genes.
*   **Embedding Pipeline**: Converting text to dense vectors via **BioLinkBERT** or **OpenAI API**.
*   **Vector DB Storage**: Indexing embeddings in a Vector Database for on-the-fly retrieval.

### 2. Hybrid Model Development (RAG-GNN)
*   **Dynamic Node Initialization**: `Initialized_Feature = LLM_Embedding * Patient_Expression`.
*   **Architecture**: **Graph Attention Networks (GAT)** to learn pathway importance.
*   **Contrastive Learning**: A joint embedding space where graph topology and LLM-retrieved text are aligned using a Contrastive Loss function.

### 3. Interpretable Survival Prediction
*   **Classification**: A **Calibrated Random Forest** ensemble processing the GNN-latent space to predict High vs. Low-risk survival.
*   **XAI (Explainable AI)**: Using **GNNExplainer** to map predictions back to specific genes (BRCA1, ERBB2) and textual evidence retrieved via RAG.

## 🛠️ Technology Stack

*   **Graph Framework**: PyTorch Geometric
*   **LLM Tools**: Transformers (HuggingFace), LangChain (for RAG chains)
*   **Vector Database**: FAISS / ChromaDB
*   **Bioinformatics**: Biopython, TCGAbiolinks
*   **Compute**: CUDA-optimized for TRUBA/GPU clusters

## 📈 Expected Impact

*   **Functional Accuracy**: Overcoming the "Structural vs. Functional" dichotomy where pure GNNs fail.
*   **Hallucination-Resistance**: Grounding LLM semantic power in deterministic biological graphs (BioKG).
*   **Clinical Readiness**: Bridging the gap between "Black Box" AI and interpretable clinical decision support.

## 📚 References

*   **Vavekanand (2026)**: A Comprehensive Review of Multimodal LLMs for Medical Imaging and Omics. *Archives of Computational Methods in Engineering*.
*   **Hays & Richardson (2026)**: RAG-GNN: Integrating Retrieved Knowledge with GNNs for Precision Medicine. *Pre-print*.
*   **Liang et al. (2025)**: The potential of large language models to advance precision oncology. *eBioMedicine / The Lancet*.
*   **Chen & Zou (2023)**: GenePT: A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT. *Stanford University*.
*   **Alharbi & Vakanski (2025)**: Multi-Omics Integration using GNNs. *IEEE Access*.

---

*This project represents the cutting edge of Generative AI in Oncology, merging the structural power of Graphs with the semantic depth of Language Models.*
