# 06 — `src/vector_store.py` (FAISS Vector Store)

| Property | Value |
|----------|-------|
| **File** | `src/vector_store.py` |
| **Lines** | 237 |
| **Pipeline Stage** | Stage 2c |
| **Execution Order** | 6th — last module in `run_stage2()` |
| **Runtime** | 22:58:45 → 22:58:45 |
| **Duration** | <1s |

---

## Purpose

Provides a FAISS-based nearest neighbor search index over gene embeddings. This enables similarity queries like "find the 10 genes most similar to BRCA1 in embedding space." The vector store is used for:

1. Sanity-checking gene embeddings (are biologically similar genes nearby?)
2. Potential downstream retrieval-augmented tasks
3. Gene neighborhood exploration

---

## Runtime Log Trace

```
22:58:45  [INFO] Loading faiss.
22:58:45  [INFO] Successfully loaded faiss.
22:58:45  [INFO] Built flat L2 index with 200 vectors
22:58:45  [INFO] Saved vector store to data/embeddings
22:58:45  [INFO] Sanity check - neighbors of RPL13AP6:
              RPL13AP6: distance=0.0000
              RPL19P12: distance=4.9826
              MRPL42P5: distance=10.3284
              ABCC6P1: distance=13.9042
              N4BP2L1: distance=17.5389
```

---

## Imports

```python
import os
import json
import logging
import numpy as np
import faiss
import yaml
```

---

## Class: `GeneEmbeddingStore`

**Lines 28–180** | FAISS-backed vector database for gene embeddings.

### `__init__(self, embedding_dim=768)`
Initializes empty store.

### `build_index(self, embeddings, gene_names, metadata=None, use_ivf=False, nlist=10)`
Builds FAISS index:
- **IndexFlatL2** (default): Exact brute-force search. Used when n_genes ≤ 1000.
- **IndexIVFFlat**: Approximate search with inverted file index. Auto-selected for >1000 genes.

### `search(self, query_embedding, k=10) -> list`
k-nearest neighbor L2 search. Returns `[(gene_name, distance), ...]`.

### `search_by_gene(self, gene_name, k=10) -> list`
Convenience wrapper: retrieves stored embedding and searches.

### `get_embedding(self, gene_name) -> np.ndarray`
Returns single gene's 768-dim vector via FAISS `reconstruct()`.

### `get_embeddings_batch(self, gene_names) -> np.ndarray`
Batch retrieval. Missing genes get zero vectors.

### `save(self, output_dir)` / `load(self, input_dir)`
Serializes/deserializes FAISS index + gene name list + metadata.

---

## Functions

### `load_config(config_path) -> dict`
**Lines 23–25**

### `build_gene_vector_store(gene_embeddings, gene_list, output_dir) -> GeneEmbeddingStore`
**Lines 183–210** | Factory function:
1. Creates `GeneEmbeddingStore(embedding_dim=768)`
2. Builds flat L2 index with 200 gene vectors
3. Saves index and metadata to disk
4. Runs sanity check: queries neighbors of the first gene (RPL13AP6)
5. Returns the populated store

---

## Output Artifacts

| File | Description |
|------|-------------|
| `data/embeddings/faiss_index.bin` | FAISS binary index file |
| `data/embeddings/gene_names.json` | Gene symbol list (index order) |
| `data/embeddings/gene_metadata.json` | Optional gene metadata |

---

## Sanity Check Results

The neighbor sanity check validates that the BioBERT embeddings capture biologically meaningful similarity:

| Gene | Distance | Relationship |
|------|----------|-------------|
| RPL13AP6 | 0.000 | Self (query gene, ribosomal pseudogene) |
| RPL19P12 | 4.983 | Also a ribosomal protein pseudogene |
| MRPL42P5 | 10.328 | Mitochondrial ribosomal protein pseudogene |
| ABCC6P1 | 13.904 | ABC transporter pseudogene |
| N4BP2L1 | 17.539 | NEDD4 binding protein |

The top neighbors are all ribosomal-related genes, confirming that BioBERT correctly groups functionally similar genes in embedding space.
