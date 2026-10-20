# 05 — `src/llm_embeddings.py` (LLM Embedding Generation)

| Property | Value |
|----------|-------|
| **File** | `src/llm_embeddings.py` |
| **Lines** | 315 |
| **Pipeline Stage** | Stage 2b |
| **Execution Order** | 5th — called after KG construction in `run_stage2()` |
| **Runtime** | 22:58:45 → 22:58:45 (cached) |
| **Duration** | <1s (loaded from cache; first run takes ~2–5 min) |

---

## Purpose

Generates 768-dimensional BioBERT embeddings for genes, pathways, and diseases. Implements the **GenePT-w** (weighted) patient embedding strategy, where each patient's gene embeddings are scaled by their expression levels to create personalized graph node features.

In this run, gene embeddings were cached from a previous execution and loaded from `.npy` files.

---

## Runtime Log Trace

```
22:58:45  [INFO] Generating gene embeddings with BioBERT
22:58:45  [INFO] Loading cached gene embeddings from data/embeddings/gene_embeddings.npy
22:58:45  [INFO] Generating pathway embeddings
22:58:45  [INFO] Embedding generation complete
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
from pathlib import Path
```

Deferred imports inside functions:
```python
from transformers import AutoTokenizer, AutoModel
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 26–28**

### `load_gene_summaries(summaries_path) -> dict`
**Lines 31–34** | Loads NCBI gene summaries JSON. Returns `{gene_symbol: summary_text}`.

### `generate_gene_embeddings(gene_list, summaries, model_name=..., batch_size=32, max_length=512, device=None) -> np.ndarray`
**Lines 37–107** | Core embedding generation:
1. Loads BioBERT model and tokenizer (`dmis-lab/biobert-base-cased-v1.2`)
2. For each gene, constructs input: `"{gene_symbol}: {NCBI_summary}"`
3. Tokenizes with `max_length=512`, `padding=True`, `truncation=True`
4. Extracts `[CLS]` token from the last hidden state
5. Processes in batches of 32 for memory efficiency
6. Genes without summaries get zero-vector embeddings

Returns `(n_genes, 768)` float32 array.

### `create_patient_weighted_embeddings(gene_embeddings, expression_matrix, gene_list) -> np.ndarray`
**Lines 110–149** | **GenePT-w implementation**:

```
For each patient p (1,217 patients):
    For each gene g (200 genes):
        patient_emb[p, g, :] = expression[p, g] × gene_emb[g, :]
```

- `expression[p, g]`: z-score normalized expression value (can be negative)
- `gene_emb[g, :]`: 768-dim BioBERT `[CLS]` embedding
- Result: `(1217, 200, 768)` tensor (~712 MB)

This is called both in Stage 3 (dataset construction) and Stage 4 (training dataset rebuild).

### `generate_pathway_embeddings(pathway_names, model_name=..., device=None) -> np.ndarray`
**Lines 152–183** | Generates embeddings for KEGG pathway names using synthetic descriptions like `"KEGG pathway: {name} — genes involved in {name} regulation"`.

### `generate_disease_embeddings(disease_names, model_name=..., device=None) -> np.ndarray`
**Lines 186–216** | Generates embeddings for disease names using synthetic descriptions.

### `run_embedding_generation(config, gene_list, summaries_path) -> dict`
**Lines 219–293** | Orchestrator with caching:
1. Checks for `gene_embeddings.npy` — loads if exists, generates if not
2. Generates pathway embeddings if KG metadata has pathways
3. Generates disease embeddings if KG metadata has diseases
4. Returns `{"gene_embeddings": np.ndarray, "pathway_embeddings": ..., ...}`

---

## Model Details

| Property | Value |
|----------|-------|
| Model | `dmis-lab/biobert-base-cased-v1.2` |
| Pre-training | PubMed abstracts + PMC full-text |
| Architecture | BERT-base (12 layers, 768 hidden, 12 heads) |
| Parameters | ~110M |
| Embedding dim | 768 |
| Max sequence length | 512 tokens |
| Pooling strategy | `[CLS]` token from last hidden state |

---

## Output Artifacts

| File | Shape | Description |
|------|-------|-------------|
| `data/embeddings/gene_embeddings.npy` | (200, 768) | BioBERT gene embeddings |
| (in memory) | (1217, 200, 768) | Patient-weighted embeddings |

---

## Caching Strategy

```
If gene_embeddings.npy exists:
    → Load from disk (<1s)
Else:
    → Load BioBERT model (~5s)
    → Generate embeddings for 200 genes (~1–3 min)
    → Save to .npy for future runs
```

The `HF_HOME` environment variable can be set to control where HuggingFace caches the model weights (relevant for sandboxed environments).
