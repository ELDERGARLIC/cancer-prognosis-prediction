# 04 — `src/kg_construction.py` (Knowledge Graph Construction)

| Property | Value |
|----------|-------|
| **File** | `src/kg_construction.py` |
| **Lines** | 433 |
| **Pipeline Stage** | Stage 2a |
| **Execution Order** | 4th — first module called in `run_stage2()` |
| **Runtime** | 22:52:45 → 22:58:45 |
| **Duration** | ~6m (dominated by KEGG API calls) |

---

## Purpose

Constructs a heterogeneous biological knowledge graph from three external data sources. The KG provides the graph topology (edge_index) that the GAT operates over — every patient shares the same graph structure, but has different node features (expression-weighted embeddings).

The three edge types are:
1. **Gene–Gene** (STRING PPI): Physical/functional protein interactions
2. **Gene–Disease** (DisGeNET): Gene associations with breast cancer
3. **Gene–Pathway** (KEGG): Pathway membership relationships

---

## Runtime Log Trace

```
22:52:45  [INFO] Building gene-gene edges from STRING PPI
22:52:45  [INFO] Loading STRING PPI network...
22:52:50  [INFO] STRING edges after confidence filter (>=700): 473860
22:52:50  [INFO] STRING edges after gene filter: 64 (from 200 genes)
22:52:50  [INFO] Final gene-gene edges: 64
22:52:50  [INFO] Building gene-disease edges from DisGeNET
22:52:50  [INFO] Loading DisGeNET gene-disease associations...
22:52:50  [INFO] DisGeNET associations after semantic type filter: 100
22:52:50  [INFO] Gene-disease edges: 0, Diseases: 0
22:52:50  [INFO] Building gene-pathway edges from KEGG
22:52:50  [INFO] Fetching KEGG pathway data...
22:58:45  [INFO] Found 126 KEGG pathways for selected genes
22:58:45  [INFO] Gene-pathway edges: 173, Pathways: 126
22:58:45  [INFO] Knowledge Graph Statistics:
              n_genes: 200
              n_gene_gene_edges: 64
              n_gene_disease_edges: 0
              n_gene_pathway_edges: 173
              gene_gene_density: 0.0016
              avg_degree: 0.32, max_degree: 6
              isolated_genes: 162
22:58:45  [INFO] Knowledge graph saved successfully
```

---

## Imports

```python
import os
import json
import logging
import pandas as pd
import yaml
import torch
import requests
from collections import defaultdict
```

---

## Functions

### `load_config(config_path) -> dict`
**Lines 29–31**

### `load_string_ppi(string_path, mapping_path, gene_list, confidence_threshold=700) -> tuple`
**Lines 34–114** | Loads and filters STRING PPI network:
1. Reads the full STRING links file (~13.7M rows, ~5s)
2. Filters to `combined_score >= 700` → 473,860 edges
3. Loads protein-to-gene mapping, maps protein IDs to gene symbols
4. Filters to edges where both endpoints are in the 200 selected genes → 64 edges
5. Creates bidirectional PyTorch edge_index tensor
6. Returns `(edge_index, edge_weights, gene_to_idx)`

### `load_disgenet_edges(disgenet_path, gene_list, gene_to_idx, disease_semantic_type="Neoplastic Process") -> tuple`
**Lines 117–167** | Loads gene-disease associations:
- Filters DisGeNET entries by semantic type ("Neoplastic Process")
- Maps gene symbols to graph node indices
- In this run: **0 edges** (no overlap between LASSO genes and DisGeNET)
- Returns `(edge_index, disease_to_idx, idx_to_disease)`

### `fetch_kegg_pathways(gene_list) -> dict`
**Lines 170–227** | Queries KEGG REST API (**most time-consuming**: ~6 min):
- Sends per-gene requests to `https://rest.kegg.jp/link/pathway/{gene}`
- 0.35s delay between requests to respect rate limits
- Fetches pathway names via `https://rest.kegg.jp/get/{pathway_id}`
- Falls back to `_get_fallback_pathways()` if API fails
- Returns `{pathway_id: {"name": str, "genes": [str]}}`

### `_get_fallback_pathways() -> dict`
**Lines 230–256** | Returns hardcoded common cancer pathways (Cell cycle, Apoptosis, p53, PI3K-Akt, etc.) as fallback.

### `build_pathway_edges(pathways, gene_list, gene_to_idx) -> tuple`
**Lines 259–289** | Converts pathway dict to bipartite gene↔pathway edge_index.

### `compute_graph_statistics(gene_gene_edges, gene_disease_edges, gene_pathway_edges, n_genes) -> dict`
**Lines 292–327** | Computes: edge counts, density, mean/max degree, isolated gene count.

### `build_knowledge_graph(config, gene_list) -> dict`
**Lines 330–417** | Main orchestrator:
1. Calls `load_string_ppi()` → gene-gene edges
2. Calls `load_disgenet_edges()` → gene-disease edges
3. Calls `fetch_kegg_pathways()` + `build_pathway_edges()` → gene-pathway edges
4. Computes statistics
5. Saves `kg_edges.pt` and `kg_metadata.json`
6. Returns full KG data dictionary

---

## Output Artifacts

| File | Description |
|------|-------------|
| `data/knowledge_graph/kg_edges.pt` | PyTorch tensor with combined gene-gene edge_index |
| `data/knowledge_graph/kg_metadata.json` | Gene/pathway/disease index mappings + statistics |

---

## Graph Topology

```
Gene Nodes: 200
                    ┌─── STRING PPI ───┐
                    │   64 edges        │
  Gene ◆────────────◆ Gene              │
       │                                │
       │   ┌─── DisGeNET ───┐          │
       │   │   0 edges      │          │
       ◆───◆ Disease         │          │
       │                    │          │
       │   ┌─── KEGG ──────┐          │
       │   │   173 edges    │          │
       ◆───◆ Pathway        │          │
            (126 pathways)              │
                                       │
  Total edges used by GAT: 64 (gene-gene only)
  Pathway/disease edges: stored in metadata
  for auxiliary use but not in main edge_index
```

**Note**: The GAT primarily operates on gene-gene PPI edges. With only 64 edges across 200 nodes, the graph is extremely sparse (density 0.0016). 162 of 200 genes are isolated (no PPI connections). The GAT still learns useful node representations through self-loops and the global pooling mechanism.
