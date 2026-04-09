# Module Reference

Complete API documentation for every module in the `src/` package. Each section describes the module's role, public interface, and key implementation details.

---

## Table of Contents

1. [main.py](#mainpy) — Pipeline Orchestrator
2. [src/data_download.py](#srcdata_downloadpy) — Data Acquisition
3. [src/preprocessing.py](#srcpreprocessingpy) — Data Preprocessing
4. [src/kg_construction.py](#srckg_constructionpy) — Knowledge Graph Construction
5. [src/llm_embeddings.py](#srcllm_embeddingspy) — LLM Embedding Generation
6. [src/vector_store.py](#srcvector_storepy) — FAISS Vector Store
7. [src/dataset.py](#srcdatasetpy) — Dataset & DataLoader
8. [src/model.py](#srcmodelpy) — Model Architectures
9. [src/train.py](#srctrainpy) — Training Loop
10. [src/evaluate.py](#srcevaluatepy) — Evaluation & Baselines
11. [src/explain.py](#srcexplainpy) — Explainability
12. [src/visualize.py](#srcvisualizepy) — Visualization

---

## `main.py`

**Role**: Pipeline entry point and orchestrator. Parses CLI arguments, loads configuration, and executes stages sequentially.

### Functions

#### `load_config(config_path: str = "configs/config.yaml") -> dict`
Loads and returns the YAML configuration dictionary.

#### `set_seed(seed: int = 42)`
Sets random seeds for reproducibility across `numpy`, `torch`, and CUDA backends.

#### `run_stage1(config: dict) -> dict`
Executes data download and preprocessing. Returns a dictionary with paths to downloaded and processed files.

#### `run_stage2(config: dict) -> dict`
Builds the knowledge graph, generates BioBERT embeddings, and creates the FAISS vector store. Returns embedding and KG artifact paths.

#### `run_stage3(config: dict) -> dict`
Constructs the PyTorch Geometric dataset with patient-weighted embeddings and CV splits.

#### `run_stage4(config: dict) -> dict`
Runs 5-fold cross-validation training of the GAT + Hybrid RF model. Returns training results including per-fold metrics and histories.

#### `run_stage5(config: dict, training_results: dict = None) -> dict`
Runs evaluation (baselines, ablation), explainability (GNNExplainer, SHAP), and generates visualizations.

#### `main()`
CLI interface with `--config` and `--stage` arguments. When `--stage N` is given, runs stages N through 5.

### CLI Usage

```bash
python main.py                    # Run all stages
python main.py --stage 4          # Run stage 4 onward
python main.py --config alt.yaml  # Use alternative config
```

---

## `src/data_download.py`

**Role**: Downloads all raw data from external APIs (GDC, UCSC Xena, STRING, DisGeNET, NCBI Entrez).

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GDC_FILES_ENDPOINT` | `https://api.gdc.cancer.gov/files` | GDC file search API |
| `GDC_DATA_ENDPOINT` | `https://api.gdc.cancer.gov/data` | GDC bulk data download |
| `GDC_CASES_ENDPOINT` | `https://api.gdc.cancer.gov/cases` | GDC clinical cases API |
| `XENA_HUB` | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com` | UCSC Xena mirror |
| `DISGENET_API_BASE` | `https://www.disgenet.org/api` | DisGeNET REST API |

### Functions

#### `download_tcga_htseq_counts_gdc(output_dir: str, project: str = "TCGA-BRCA") -> str`
Primary expression download. Queries GDC for STAR-Counts files, downloads as tarball, merges into a single gene × sample TSV. Falls back to Xena if GDC returns an error.

#### `download_tcga_expression_xena(output_dir: str) -> str`
Fallback expression download from UCSC Xena (HiSeqV2 RSEM normalized).

#### `download_tcga_clinical_gdc(output_dir: str, project: str = "TCGA-BRCA") -> str`
Fetches clinical fields (`diagnoses.days_to_death`, `diagnoses.vital_status`, `demographic.gender`, etc.) from the GDC Cases API. Computes `OS.time` and `OS` (event indicator) columns.

#### `download_tcga_clinical_xena(output_dir: str) -> str`
Fallback clinical download from Xena `BRCA_clinicalMatrix.gz`.

#### `download_string_ppi(output_dir: str, species: int = 9606) -> str`
Downloads STRING v12.0 human protein links (~14M rows). Strips the `9606.` taxonomy prefix from protein IDs.

#### `download_string_id_mapping(output_dir: str, species: int = 9606) -> str`
Downloads STRING protein → gene name mapping table.

#### `download_disgenet(output_dir: str, api_key: str = None, disease_id: str = "UMLS_C0006142") -> str`
Queries DisGeNET for gene–disease associations related to breast carcinoma (CUI: C0006142). Includes a curated fallback list of 100 breast cancer-associated genes if the API is unavailable.

#### `download_ncbi_gene_summaries(gene_list: list, output_dir: str) -> str`
Batch NCBI Entrez esearch + esummary for gene functional descriptions. Processes in batches of 100 with 0.4s delays to respect rate limits. Saves to `gene_summaries.json`.

#### `run_data_download(config: dict) -> dict`
Orchestrates all downloads. Returns `{"expression_path", "clinical_path", "string_path", "mapping_path", "disgenet_path", "summaries_path"}`.

---

## `src/preprocessing.py`

**Role**: Transforms raw data into analysis-ready features: filters genes, normalizes expression, imputes clinical data, selects features via LASSO, and creates survival labels.

### Functions

#### `load_expression_data(expr_path: str) -> pd.DataFrame`
Loads gene × sample expression matrix from TSV or `.tsv.gz`. Returns DataFrame with genes as rows, samples as columns.

#### `load_clinical_data(clinical_path: str) -> pd.DataFrame`
Loads clinical TSV. Sets appropriate index column.

#### `filter_low_expression_genes(expr_df: pd.DataFrame, min_total_counts: int = 1000) -> pd.DataFrame`
Removes genes whose total expression across all samples falls below `min_total_counts`. Reduces from ~20K to ~17K genes.

#### `normalize_expression(expr_df: pd.DataFrame) -> pd.DataFrame`
Two-step normalization:
1. `log2(x + 1)` if data appears to be raw counts (max > 30)
2. Z-score standardization per gene across samples

#### `impute_clinical_data(clinical_df: pd.DataFrame) -> pd.DataFrame`
KNN imputation (k=5) on numeric columns. Handles edge cases:
- Skips non-numeric and ID-like columns
- Columns that are entirely NaN are filled with 0 (with a warning)
- Only imputes columns that have at least one non-NaN value

#### `select_features_lasso(expr_df, labels, n_genes: int = 1500, seed: int = 42) -> list`
Runs `LassoCV` with 10-fold inner CV and 100 alpha values. Selects genes with non-zero coefficients. If fewer than `n_genes` are selected, supplements with highest-variance genes to reach the target count.

#### `select_features_rfe(expr_df, labels, n_genes: int = 1500, seed: int = 42) -> list`
Alternative feature selection using Recursive Feature Elimination with a Random Forest estimator.

#### `filter_with_disgenet(selected_genes, disgenet_path, disease_semantic_type: str = "Neoplastic Process") -> list`
Intersects LASSO-selected genes with DisGeNET breast cancer gene associations. If the intersection is too small (<10% of selected genes), keeps all selected genes and adds KG genes as supplementary context.

#### `create_survival_labels(clinical_df, bins=None, labels_names=None) -> pd.DataFrame`
Discretizes continuous `OS.time` (overall survival in days) into categorical classes using the configured bins. Default: <1yr, 1–3yr, 3–5yr, >5yr. Drops patients with missing or invalid survival time.

#### `extract_clinical_features(clinical_df) -> tuple`
Extracts and encodes clinical features:
- **Age**: Numeric (at diagnosis)
- **Stage**: One-hot encoded (I, II, III, IV)
- **Gender**: Binary (`is_female`)

Returns a tuple of `(features_df, feature_names)`.

#### `run_preprocessing(config, expr_path, clinical_path, disgenet_path=None) -> dict`
Full preprocessing pipeline. Saves all intermediate and final files to `data/processed/`.

---

## `src/kg_construction.py`

**Role**: Constructs a heterogeneous biological knowledge graph from STRING PPI, DisGeNET gene-disease associations, and KEGG pathway memberships.

### Functions

#### `load_string_ppi(string_path, mapping_path, gene_list, confidence_threshold: int = 700) -> tuple`
Loads STRING protein-protein interaction network:
1. Reads protein links TSV (~13.7M rows for human)
2. Filters to confidence ≥ threshold
3. Maps protein IDs to gene symbols via the STRING ID mapping file
4. Filters to edges between selected genes
5. Creates bidirectional edge_index and edge weights

Returns `(edge_index, edge_weights, gene_to_idx)`.

#### `load_disgenet_edges(disgenet_path, gene_list, gene_to_idx, disease_semantic_type: str = "Neoplastic Process") -> tuple`
Loads gene-disease associations filtered by semantic type. Maps gene symbols to graph indices. Returns `(edge_index, disease_to_idx, idx_to_disease)`.

#### `fetch_kegg_pathways(gene_list: list) -> dict`
Queries the KEGG REST API for pathway membership of each gene. Uses batch requests with rate limiting (0.35s delay). Returns `{pathway_id: {"name": str, "genes": [str]}}`. Falls back to hardcoded common cancer pathways if the API fails.

#### `build_pathway_edges(pathways, gene_list, gene_to_idx) -> tuple`
Converts pathway membership dict into a bipartite edge_index (gene ↔ pathway). Returns `(edge_index, pathway_to_idx, idx_to_pathway)`.

#### `compute_graph_statistics(gene_gene_edges, gene_disease_edges, gene_pathway_edges, n_genes: int) -> dict`
Computes summary statistics: edge counts, density, mean/max degree, number of isolated genes.

#### `build_knowledge_graph(config, gene_list: list) -> dict`
Orchestrates KG construction. Saves `kg_edges.pt` (PyTorch tensor) and `kg_metadata.json` to `data/knowledge_graph/`.

---

## `src/llm_embeddings.py`

**Role**: Generates BioBERT embeddings for genes, pathways, and diseases. Implements the GenePT-w patient embedding strategy.

### Functions

#### `load_gene_summaries(summaries_path: str) -> dict`
Loads NCBI gene summaries JSON. Returns `{gene_symbol: summary_text}`.

#### `generate_gene_embeddings(gene_list, summaries, model_name=..., batch_size=32, max_length=512, device=None) -> np.ndarray`
Generates 768-dim BioBERT embeddings for each gene:
1. Constructs input text: gene symbol + NCBI summary
2. Tokenizes with BioBERT tokenizer (max 512 tokens)
3. Extracts `[CLS]` token embedding from the last hidden state
4. Processes in batches of 32 for memory efficiency

Returns `(n_genes, 768)` float32 array. Genes without summaries get a zero vector.

#### `create_patient_weighted_embeddings(gene_embeddings, expression_matrix, gene_list) -> np.ndarray`
Implements **GenePT-w** (weighted) approach:
- For each patient, scales each gene's LLM embedding by the patient's expression level for that gene
- `patient_emb[p, g, :] = expression[p, g] × gene_emb[g, :]`

Returns `(n_patients, n_genes, 768)` array.

#### `generate_pathway_embeddings(pathway_names, model_name=..., device=None) -> np.ndarray`
Generates embeddings for KEGG pathway names using synthetic descriptions.

#### `generate_disease_embeddings(disease_names, model_name=..., device=None) -> np.ndarray`
Generates embeddings for disease names using synthetic descriptions.

#### `run_embedding_generation(config, gene_list, summaries_path) -> dict`
Manages embedding generation with caching. Loads from `.npy` files if they exist, generates otherwise.

---

## `src/vector_store.py`

**Role**: Provides FAISS-based nearest neighbor search over gene embeddings for similarity queries and retrieval.

### Class: `GeneEmbeddingStore`

#### `__init__(self, embedding_dim: int = 768)`
Initializes an empty store with the given embedding dimension.

#### `build_index(self, embeddings, gene_names, metadata=None, use_ivf=False, nlist=10)`
Builds the FAISS index from a matrix of embeddings:
- **Flat L2** (default): Exact search, used when n_genes ≤ 1000
- **IVF Flat**: Approximate search with inverted file index, for larger datasets

#### `search(self, query_embedding, k: int = 10) -> list`
k-nearest neighbor search by L2 distance. Returns `[(gene_name, distance), ...]`.

#### `search_by_gene(self, gene_name: str, k: int = 10) -> list`
Convenience method: retrieves a gene's stored embedding and searches for its neighbors.

#### `get_embedding(self, gene_name: str) -> np.ndarray`
Retrieves a single gene's embedding vector via FAISS `reconstruct`.

#### `get_embeddings_batch(self, gene_names: list) -> np.ndarray`
Batch retrieval. Returns `(len(gene_names), dim)` array. Missing genes get zero vectors.

#### `save(self, output_dir: str)` / `load(self, input_dir: str)`
Serializes/deserializes the FAISS index and gene name list to/from disk.

### Functions

#### `build_gene_vector_store(gene_embeddings, gene_list, output_dir: str) -> GeneEmbeddingStore`
Builds, saves, and sanity-checks the vector store. Logs nearest neighbors of the first gene as a verification step.

---

## `src/dataset.py`

**Role**: Defines the PyTorch Geometric dataset, handles SMOTE augmentation, cross-validation splitting, and DataLoader creation.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_SMOTE_FEATURES` | `50,000` | Feature dim threshold; SMOTE skipped if exceeded |

### Class: `BreastCancerGraphDataset(torch.utils.data.Dataset)`

Wraps patient data into PyTorch Geometric `Data` objects with shared graph topology.

#### `__init__(self, embeddings, labels, edge_index, clinical_features, ...)`
Stores patient embeddings `(N, n_genes, emb_dim)`, labels, shared edges, clinical features, and optional survival metadata.

#### `__getitem__(self, idx) -> Data`
Returns a single patient's graph:
- `x`: `[n_genes, emb_dim]` — node features (expression-weighted embeddings)
- `edge_index`: `[2, n_edges]` — shared KG topology
- `clinical`: `[1, n_clinical]` — clinical features (unsqueezed for correct batching)
- `y`: survival class label
- `os_time`, `os_event`, `tumor_stage`: optional metadata

### Functions

#### `compute_class_weights(labels: np.ndarray) -> np.ndarray`
Computes inverse-frequency class weights: `weight[c] = N / (n_classes × count[c])`. Used when SMOTE is skipped to balance the cross-entropy loss.

#### `apply_smote(embeddings, labels, clinical_features, strategy="auto", seed=42) -> tuple`
Applies SMOTE oversampling on the training set:
1. Flattens `(N, genes, dim)` embeddings to `(N, genes×dim)`
2. Concatenates with clinical features
3. Checks total feature dimension against `MAX_SMOTE_FEATURES`
4. If too large, logs a warning and returns the original data unchanged
5. Otherwise, applies SMOTE and reshapes back

#### `create_cv_splits(labels, n_folds: int = 5, seed: int = 42) -> list`
Creates stratified K-fold train/validation index pairs using `StratifiedKFold`.

#### `extract_tumor_stages(clinical_df: pd.DataFrame) -> np.ndarray`
Parses tumor stage strings (e.g., "Stage IIA") into integer codes 0–3 for the auxiliary classification task. Unrecognized stages map to -1.

#### `build_dataset(config: dict) -> dict`
Full dataset construction:
1. Loads processed expression, labels, clinical features, KG edges, gene embeddings
2. Creates patient-weighted embeddings
3. Extracts tumor stages
4. Creates CV splits
5. Returns `{"dataset", "cv_splits", "gene_list", "config", ...}`

#### `get_dataloaders(dataset, train_idx, val_idx, batch_size=32, smote=True, ...) -> tuple`
Creates train and validation `DataLoader` objects. Applies SMOTE to training data if enabled and feature dimension permits.

---

## `src/model.py`

**Role**: Defines the neural network architectures — the BioKG-GAT backbone, the HybridModel with clinical fusion and multi-task heads, and a vanilla GAT classifier for ablation.

### Class: `BioKG_GAT(nn.Module)`

3-layer Graph Attention Network with residual connections, batch normalization, and ELU activations.

#### Architecture

| Layer | Input Dim | Output Dim | Heads | Features |
|-------|-----------|------------|-------|----------|
| GAT 1 | 768 | 128 | 8 | + BatchNorm + ELU + Dropout + Residual (linear proj) |
| GAT 2 | 128 | 128 | 4 | + BatchNorm + ELU + Dropout + Residual (linear proj) |
| GAT 3 | 128 | 128 | 1 | + BatchNorm + ELU + Dropout |
| Pool | 128 | 256 | — | mean ∥ max concatenation |

#### `forward(self, x, edge_index, batch, edge_attr=None) -> Tensor`
Returns graph-level embeddings `[batch_size, 256]`.

### Class: `HybridModel(nn.Module)`

Combines GNN backbone with clinical features for multi-task learning.

#### `__init__(self, gnn, clinical_dim, gnn_output_dim=256, num_classes=4, num_stages=4)`
- **Main head**: `Linear(256+clinical) → BN → ELU → Dropout → Linear → Linear → 4 classes`
- **Auxiliary head**: `Linear(256) → ReLU → Linear → 4 stages`

#### `forward(self, x, edge_index, batch, clinical_features, edge_attr=None)`
Returns `(main_logits, aux_logits, gnn_embedding)`.

#### `extract_embeddings(self, x, edge_index, batch, clinical_features=None, ...)`
Returns concatenated GNN + optional clinical embeddings for downstream use (e.g., RF training).

### Class: `GATClassifier(nn.Module)`

Simplified GAT + classifier for baseline/ablation experiments. No clinical fusion or auxiliary task.

### Function: `build_model(config, clinical_dim, device=None) -> HybridModel`

Factory function that instantiates `BioKG_GAT` + `HybridModel` from config and moves to the target device. Logs parameter counts.

---

## `src/train.py`

**Role**: Implements the training loop with 5-fold cross-validation, early stopping, learning rate scheduling, and hybrid RF training.

### Functions

#### `set_seed(seed: int = 42)`
Sets random seeds for Python `random`, NumPy, PyTorch, and CUDA. Enables `cudnn.deterministic`.

#### `get_device() -> torch.device`
Auto-detects and returns `cuda`, `mps`, or `cpu`.

#### `train_one_epoch(model, loader, optimizer, device, aux_loss_weight=0.3, class_weights=None) -> dict`
Single training epoch:
- **Loss**: `CrossEntropyLoss(weight=class_weights)` for main task + `CrossEntropyLoss` for auxiliary task (tumor staging), weighted by `aux_loss_weight`
- **Gradient clipping**: `max_norm=1.0`
- Auxiliary loss only computed for samples with valid stage labels (≥ 0)

Returns `{"loss", "accuracy"}`.

#### `evaluate(model, loader, device) -> dict`
Validation pass (no gradients):
- **Loss**: CE loss (unweighted)
- **Accuracy**: Top-1
- **AUC-ROC**: Macro one-vs-rest using softmax probabilities
- **C-index**: Concordance index using predicted risk score (`Σ class_weight × P(class)`)

Returns `{"loss", "accuracy", "auc_roc", "c_index"}`.

#### `extract_embeddings(model, loader, device) -> tuple`
Runs the model in eval mode to extract GNN embeddings for all samples. Returns `(embeddings, labels, clinical_features)` as numpy arrays.

#### `train_calibrated_rf(train_emb, train_labels, val_emb, val_labels, config) -> tuple`
Trains a Random Forest on GNN embeddings:
- 500 estimators, min_samples_split=5
- Optionally wrapped in `CalibratedClassifierCV` (isotonic, 3-fold inner CV)
- Evaluated on validation set: accuracy and macro AUC-ROC

Returns `(rf_model, metrics_dict)`.

#### `train_fold(fold, model, dataset, train_idx, val_idx, config, device) -> dict`
Complete single-fold training:
1. Creates DataLoaders (with SMOTE attempt)
2. Computes class weights from training labels
3. Trains GAT with Adam + ReduceLROnPlateau
4. Early stopping on validation loss
5. Restores best model, extracts embeddings
6. Trains calibrated RF on GNN embeddings

Returns fold results including metrics, history, and models.

#### `run_training(config: dict) -> dict`
Orchestrates 5-fold CV:
1. Builds dataset
2. Loops through folds, calling `train_fold`
3. Aggregates cross-fold statistics (mean ± std)
4. Saves `training_results.json` and `model_fold*.pt`

---

## `src/evaluate.py`

**Role**: Comprehensive model evaluation with multiple baselines, ablation studies, and detailed per-class metrics.

### Functions

#### `compute_full_metrics(y_true, y_pred, y_probs, os_time=None, os_event=None, num_classes=4) -> dict`
Computes a complete metrics suite:
- Accuracy, macro precision/recall/F1
- Per-class precision, recall, F1
- Confusion matrix
- Macro AUC-ROC (one-vs-rest)
- C-index (if survival data provided)
- Time-binned C-index

#### `train_cox_baseline(X_train, y_train_time, y_train_event, X_val, y_val_time, y_val_event) -> dict`
Fits a Cox Proportional Hazards model (lifelines) on the first 50 PCA features. Returns validation C-index. Uses penalizer=0.1 for regularization.

#### `train_rf_baseline(X_train, y_train, X_val, y_val, seed=42) -> dict`
Trains a Random Forest classifier (100 estimators) on raw features. Returns accuracy and weighted F1.

#### `train_mlp_baseline(X_train, y_train, X_val, y_val, seed=42) -> dict`
Trains a scikit-learn MLP (256→128→64, early stopping) on raw features. Returns accuracy and weighted F1.

#### `train_vanilla_gcn_baseline(dataset, train_idx, val_idx, config, device) -> dict`
Trains a 2-layer GCN (no attention, no clinical features, no auxiliary task) for 100 epochs. Provides a fair comparison to show the benefit of GAT attention and clinical fusion.

#### `run_ablation_study(dataset, train_idx, val_idx, config, device) -> dict`
Three ablation experiments:
1. **Expression only (RF)**: Mean of gene embeddings → RF classifier
2. **Expression + Clinical (RF)**: Mean embeddings + clinical features → RF
3. **GCN with BioKG**: Vanilla GCN on the same graph topology

#### `run_baseline_comparison(dataset, train_idx, val_idx, config, device) -> dict`
Compares the GAT+RF hybrid against: Cox PH, Random Forest, MLP, Vanilla GCN.

#### `run_evaluation(config, training_results=None) -> dict`
Main evaluation entry point. Logs summary, runs baselines and ablation, saves `evaluation_report.json`.

---

## `src/explain.py`

**Role**: Provides post-hoc interpretability through GNNExplainer (node-level importance) and SHAP (feature importance for the RF hybrid).

### Functions

#### `run_gnn_explainer(model, data, gene_list, device, top_k=10, num_samples=20) -> dict`
Runs PyTorch Geometric's `GNNExplainer` on the GNN backbone:
1. Wraps `HybridModel.gnn` in a `GNNWrapper` class that accepts the `GNNExplainer` interface
2. Samples `num_samples` patients from the dataset
3. For each patient, learns soft node feature masks (200 epochs, lr=0.01)
4. Aggregates importance scores across patients
5. Returns top-k genes by mean importance, plus overlap with known breast cancer genes

#### `run_shap_analysis(rf_model, embeddings, feature_names=None, top_k=20, max_samples=100) -> dict`
SHAP feature importance for the Random Forest:
1. Uses `TreeExplainer` (fast, exact for tree models)
2. Falls back to `KernelExplainer` if TreeExplainer fails
3. Computes mean absolute SHAP values across all classes
4. Returns top-k features ranked by importance

Feature names follow the pattern `gnn_0` through `gnn_255` for GNN embedding dimensions, and `clinical_0` through `clinical_5` for clinical features.

#### `create_risk_heatmap(gene_importance, kg_edges, gene_list, top_k=30, output_path=...) -> str`
Visualizes gene importance as a network graph:
- Nodes: Top-k most important genes
- Edges: KG connections between those genes
- Node size: Proportional to importance score
- Color: Red-to-blue gradient by importance
- Saves as `results/risk_heatmap.png`

#### `run_explainability(config, training_results) -> dict`
Orchestrates all explainability analyses. Loads fold-0 model, runs GNNExplainer and SHAP, generates risk heatmap, saves `explainability_results.json`.

---

## `src/visualize.py`

**Role**: Generates publication-ready matplotlib figures for training dynamics, survival analysis, embedding projections, and model comparisons.

### Constants

| Constant | Value |
|----------|-------|
| `COLORS` | Predefined color palette for consistent plots |
| `CLASS_NAMES` | `["<1yr", "1-3yr", "3-5yr", ">5yr"]` |

### Functions

#### `plot_training_curves(history, output_path=...)`
Three-panel figure:
1. Training and validation loss over epochs
2. Validation accuracy over epochs
3. Validation AUC-ROC and C-index over epochs

#### `plot_kaplan_meier(os_time, os_event, risk_groups, output_path=...)`
Kaplan-Meier survival curves stratified by model-predicted risk group. Uses the lifelines `KaplanMeierFitter` with confidence intervals.

#### `plot_tsne_embeddings(embeddings, labels, output_path=..., perplexity=30, seed=42)`
2D t-SNE projection of patient GNN embeddings, colored by survival class. Includes class-labeled legend.

#### `plot_umap_embeddings(embeddings, labels, output_path=..., seed=42)`
2D UMAP projection (if `umap-learn` is installed). Similar layout to t-SNE.

#### `plot_roc_curves(y_true, y_probs, num_classes=4, output_path=...)`
Per-class and macro-average ROC curves with AUC annotations.

#### `plot_confusion_matrix(y_true, y_pred, output_path=...)`
Confusion matrix heatmap using sklearn's `ConfusionMatrixDisplay`.

#### `plot_model_comparison(model_metrics, metric_name="accuracy", output_path=...)`
Bar chart comparing accuracy (or other metrics) across all models: GAT, Calibrated RF, baselines, and ablations.

#### `plot_ablation_study(ablation_results, output_path=...)`
Bar chart of ablation study results (expression-only, +clinical, GCN with BioKG).

#### `generate_all_visualizations(config, training_results) -> list`
Orchestrates all visualizations using fold-0 results. Returns list of generated file paths.

---

## Module Dependency Graph

```
main.py
  ├── src/data_download.py
  ├── src/preprocessing.py
  ├── src/kg_construction.py
  ├── src/llm_embeddings.py
  │     └── (HuggingFace transformers)
  ├── src/vector_store.py
  │     └── (faiss)
  ├── src/dataset.py
  │     ├── src/llm_embeddings.py
  │     └── (torch_geometric, imblearn)
  ├── src/model.py
  │     └── (torch_geometric.nn)
  ├── src/train.py
  │     ├── src/model.py
  │     ├── src/dataset.py
  │     └── (sklearn, lifelines)
  ├── src/evaluate.py
  │     ├── src/model.py
  │     ├── src/dataset.py
  │     └── (sklearn, lifelines)
  ├── src/explain.py
  │     └── (shap, torch_geometric.explain)
  └── src/visualize.py
        └── (matplotlib, umap, sklearn.manifold)
```
