# Module Documentation — Chronological Runtime Order

Per-file documentation for every Python module in the breast cancer prognosis prediction pipeline, ordered by their execution sequence during a full pipeline run.

---

## Pipeline Timeline

```
TIME         MODULE                    STAGE   DURATION     WHAT HAPPENS
─────────────────────────────────────────────────────────────────────────
22:31:06     01_main.py                —       (orchestrator) Parse args, load config, set seed
22:31:07     02_data_download.py       1a      ~3m 16s      Download TCGA, STRING, DisGeNET
22:34:24     03_preprocessing.py       1b      ~18m 20s     Filter, normalize, LASSO, labels
22:52:45     04_kg_construction.py     2a      ~6m 00s      Build PPI + KEGG knowledge graph
22:58:45     05_llm_embeddings.py      2b      <1s          Load/generate BioBERT embeddings
22:58:45     06_vector_store.py        2c      <1s          Build FAISS gene similarity index
22:58:45     07_dataset.py             3       ~3s          Patient embeddings + CV splits
22:58:49     08_model.py               4       <1s          Instantiate GAT + HybridModel
22:58:50     09_train.py               4       ~1h 23m      5-fold CV training + RF hybrid
00:22:09     10_evaluate.py            5a      ~3m 16s      Metrics, baselines, ablation
00:25:29     11_explain.py             5b      ~4m 11s      GNNExplainer + SHAP
00:29:40     12_visualize.py           5c      ~11s         8 publication-quality plots
─────────────────────────────────────────────────────────────────────────
TOTAL                                          ~1h 59m
```

---

## Documents

| # | File | Module | Lines | Doc |
|---|------|--------|-------|-----|
| 01 | `main.py` | Pipeline orchestrator | 211 | [01_main.md](01_main.md) |
| 02 | `src/data_download.py` | Data acquisition | 707 | [02_data_download.md](02_data_download.md) |
| 03 | `src/preprocessing.py` | Data preprocessing | 620 | [03_preprocessing.md](03_preprocessing.md) |
| 04 | `src/kg_construction.py` | Knowledge graph | 433 | [04_kg_construction.md](04_kg_construction.md) |
| 05 | `src/llm_embeddings.py` | BioBERT embeddings | 315 | [05_llm_embeddings.md](05_llm_embeddings.md) |
| 06 | `src/vector_store.py` | FAISS vector store | 237 | [06_vector_store.md](06_vector_store.md) |
| 07 | `src/dataset.py` | Dataset & DataLoader | 445 | [07_dataset.md](07_dataset.md) |
| 08 | `src/model.py` | GAT architecture | 283 | [08_model.md](08_model.md) |
| 09 | `src/train.py` | Training loop | 497 | [09_train.md](09_train.md) |
| 10 | `src/evaluate.py` | Evaluation & baselines | 498 | [10_evaluate.md](10_evaluate.md) |
| 11 | `src/explain.py` | Explainability | 407 | [11_explain.md](11_explain.md) |
| 12 | `src/visualize.py` | Visualization | 488 | [12_visualize.md](12_visualize.md) |

**Total**: 5,141 lines of Python across 12 files.

---

## Data Flow Across Modules

```
02_data_download ──→ Raw data (expression, clinical, PPI, DisGeNET, summaries)
       │
       ▼
03_preprocessing ──→ Processed data (200 genes, 1217 patients, labels, clinical)
       │
       ├──→ 04_kg_construction ──→ Edge index + KG metadata
       │         │
       │         ▼
       ├──→ 05_llm_embeddings ──→ Gene embeddings (200 × 768)
       │         │
       │         ▼
       └──→ 06_vector_store ──→ FAISS index
                  │
                  ▼
           07_dataset ──→ Patient graph dataset (1217 × Data objects)
                  │
                  ▼
           08_model ──→ GAT + HybridModel architecture
                  │
                  ▼
           09_train ──→ Trained models + RF hybrids (5 folds)
                  │
                  ├──→ 10_evaluate ──→ Metrics + baselines + ablation
                  ├──→ 11_explain  ──→ Gene importance + SHAP
                  └──→ 12_visualize ──→ 8 publication plots
```
