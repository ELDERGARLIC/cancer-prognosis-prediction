# 01 — `main.py` (Pipeline Orchestrator)

| Property | Value |
|----------|-------|
| **File** | `main.py` |
| **Lines** | 211 |
| **Pipeline Stage** | All stages (entry point) |
| **Execution Order** | 1st — runs at pipeline start |
| **Runtime** | 22:31:06 → 00:29:51 (~1h 59m total pipeline) |

---

## Purpose

`main.py` is the **single entry point** for the entire breast cancer prognosis prediction pipeline. It parses CLI arguments, loads the YAML configuration, sets random seeds, and orchestrates the execution of all 5 pipeline stages in sequence. Each stage is encapsulated in a `run_stageN()` function that lazily imports the required module to avoid circular dependencies and reduce startup time.

---

## Execution Flow

```
python main.py [--config configs/config.yaml] [--stage N]
       │
       ├── Parse CLI args
       ├── load_config()
       ├── set_seed(42)
       ├── Log device info (CUDA / MPS / CPU)
       │
       ├── run_stage1(config)   → data_download + preprocessing
       ├── run_stage2(config)   → kg_construction + llm_embeddings + vector_store
       ├── run_stage3(config)   → dataset construction
       ├── run_stage4(config)   → model training (5-fold CV)
       └── run_stage5(config)   → evaluate + explain + visualize
```

When `--stage N` is provided, stages 1 through N-1 are skipped. Stages N through 5 are executed. This allows re-running later stages without repeating earlier expensive operations (e.g., data download, LASSO feature selection).

---

## Imports

```python
import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np
```

Stage-specific imports are deferred inside each `run_stageN()` function.

---

## Functions

### `load_config(config_path: str = "configs/config.yaml") -> dict`
**Lines 40–42**

Reads and returns the YAML configuration dictionary. This single config file controls all pipeline parameters.

### `set_seed(seed: int = 42)`
**Lines 45–51**

Sets deterministic random seeds across all libraries:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)` (if CUDA available)

### `run_stage1(config) -> dict`
**Lines 54–75** | Runtime: **22:31:07 → 22:52:45** (~21m 38s)

Calls:
1. `src.data_download.run_data_download(config)` — downloads expression, clinical, STRING, DisGeNET data
2. `src.preprocessing.run_preprocessing(config, ...)` — filters, normalizes, LASSO, survival labels
3. `src.data_download.download_ncbi_gene_summaries(...)` — fetches gene functional summaries from NCBI

### `run_stage2(config) -> dict`
**Lines 78–105** | Runtime: **22:52:45 → 22:58:45** (~6m)

Calls:
1. `src.kg_construction.build_knowledge_graph(config, gene_list)` — STRING PPI + DisGeNET + KEGG
2. `src.llm_embeddings.run_embedding_generation(config, gene_list, ...)` — BioBERT embeddings
3. `src.vector_store.build_gene_vector_store(...)` — FAISS index

### `run_stage3(config) -> dict`
**Lines 108–115** | Runtime: **22:58:45 → 22:58:48** (~3s)

Calls `src.dataset.build_dataset(config)` — constructs patient-weighted embeddings and CV splits.

### `run_stage4(config) -> dict`
**Lines 119–126** | Runtime: **22:58:48 → 00:22:09** (~1h 23m)

Calls `src.train.run_training(config)` — 5-fold cross-validation training.

### `run_stage5(config, training_results=None) -> dict`
**Lines 130–171** | Runtime: **00:22:09 → 00:29:51** (~7m 42s)

Calls:
1. `src.evaluate.run_evaluation(config, training_results)` — summary metrics
2. `src.evaluate.run_baseline_comparison(...)` — Cox PH, RF, MLP, GCN baselines
3. `src.evaluate.run_ablation_study(...)` — expression-only, +clinical, GCN ablations
4. `src.explain.run_explainability(config, training_results)` — GNNExplainer + SHAP
5. `src.visualize.generate_all_visualizations(config, training_results)` — plots

### `main()`
**Lines 175–206**

CLI entry point. Parses `--config` and `--stage`, configures logging to stdout + `pipeline.log`.

---

## Logging Configuration

```
Format:  %(asctime)s [%(levelname)s] %(name)s: %(message)s
Outputs: stdout + pipeline.log
Level:   INFO
```

---

## Runtime Timeline

```
22:31:06  ─── Pipeline starts ───────────────────────────────
22:31:07  │  STAGE 1: Data Acquisition & Preprocessing
22:34:23  │  ├── Data download complete (GDC + fallback)
22:52:44  │  └── Preprocessing complete (LASSO: 18m)
22:52:45  │  STAGE 2: KG & LLM Embeddings
22:58:45  │  └── KG + BioBERT + FAISS complete
22:58:45  │  STAGE 3: Dataset Construction
22:58:48  │  └── Patient embeddings + CV splits
22:58:48  │  STAGE 4: Model Training (5-Fold CV)
23:28:23  │  ├── Fold 1: 156 epochs (best AUC 0.695)
23:49:07  │  ├── Fold 2: 110 epochs
00:12:53  │  ├── Fold 3: 127 epochs
00:17:44  │  ├── Fold 4: 25 epochs
00:22:09  │  └── Fold 5: 22 epochs
00:22:09  │  STAGE 5: Evaluation & Explainability
00:25:25  │  ├── Baselines + Ablation
00:29:40  │  ├── GNNExplainer + SHAP
00:29:51  │  └── Visualizations
00:29:51  ─── PIPELINE COMPLETE ─────────────────────────────
```
