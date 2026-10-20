# 02 — `src/data_download.py` (Data Acquisition)

| Property | Value |
|----------|-------|
| **File** | `src/data_download.py` |
| **Lines** | 707 |
| **Pipeline Stage** | Stage 1a |
| **Execution Order** | 2nd — called by `run_stage1()` |
| **Runtime** | 22:31:07 → 22:34:23 (data) + 22:52:44 → 22:52:45 (gene summaries) |
| **Duration** | ~3m 16s (downloads) + ~1s (summaries) |

---

## Purpose

Downloads all raw data from external APIs and databases required by the pipeline:
- TCGA-BRCA gene expression (GDC or UCSC Xena)
- TCGA-BRCA clinical data (GDC or UCSC Xena)
- STRING protein-protein interactions (v12.0)
- DisGeNET gene-disease associations
- NCBI gene functional summaries (for BioBERT input)

Implements a **fallback strategy**: GDC is the primary data source; if it fails (e.g., 500 Server Error), the pipeline automatically falls back to UCSC Xena mirrors.

---

## Runtime Log Trace

```
22:31:07  [INFO] Stage 1: Downloading TCGA-BRCA expression data
22:31:07  [INFO] Querying GDC API for TCGA-BRCA expression files...
22:31:10  [INFO] Found 1231 expression files
22:31:10  [INFO] Downloading expression files from GDC...
22:34:23  [WARN] GDC download failed (500 Server Error), falling back to UCSC Xena
22:34:23  [INFO] Expression file already exists: data/raw/tcga_brca_expression.tsv.gz
22:34:23  [INFO] Stage 2: Downloading TCGA-BRCA clinical data
22:34:23  [INFO] Clinical file already exists: data/raw/tcga_brca_clinical.tsv
22:34:23  [INFO] Stage 3: Downloading STRING PPI network
22:34:23  [INFO] STRING PPI file already exists
22:34:23  [INFO] Stage 4: Downloading DisGeNET associations
22:34:23  [INFO] DisGeNET file already exists
22:34:23  [INFO] All data downloads complete!
   ...
22:52:44  [INFO] Gene summaries file already exists (1750 genes)
22:52:44  [INFO] Downloading summaries for 11 additional genes...
22:52:45  [INFO] Saved summaries for 1750 genes
```

---

## Imports

```python
import os
import json
import gzip
import logging
import requests
import pandas as pd
import yaml
from io import BytesIO
```

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `GDC_FILES_ENDPOINT` | `https://api.gdc.cancer.gov/files` | GDC file search API |
| `GDC_DATA_ENDPOINT` | `https://api.gdc.cancer.gov/data` | GDC bulk data download |
| `GDC_CASES_ENDPOINT` | `https://api.gdc.cancer.gov/cases` | GDC clinical cases API |
| `XENA_HUB` | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com` | UCSC Xena S3 mirror |
| `DISGENET_API_BASE` | `https://www.disgenet.org/api` | DisGeNET REST API |
| `_BREAST_CANCER_CUIS` | Set of UMLS CUIs | Breast cancer concept identifiers |

---

## Functions

### `load_config(config_path) -> dict`
**Lines 32–34** | Loads YAML config.

### `download_tcga_expression_xena(output_dir) -> str`
**Lines 37–62** | Downloads HiSeqV2 RSEM expression from UCSC Xena. Skips if file exists.

### `download_tcga_htseq_counts_gdc(output_dir, project="TCGA-BRCA") -> str`
**Lines 65–167** | Primary expression download path:
1. Queries GDC `/files` API for STAR-Counts with `TCGA-BRCA` project filter
2. Posts file UUIDs to `/data` endpoint for bulk tarball download
3. Extracts and merges individual sample files into a single gene × sample TSV
4. On failure (HTTP error), falls back to `download_tcga_expression_xena()`

### `download_tcga_clinical_gdc(output_dir, project="TCGA-BRCA") -> str`
**Lines 170–249** | Fetches clinical data from GDC Cases API:
- Fields: `diagnoses.days_to_death`, `diagnoses.vital_status`, `diagnoses.days_to_last_follow_up`, `diagnoses.age_at_diagnosis`, `diagnoses.tumor_stage`, `demographic.gender`
- Computes `OS.time` (max of days_to_death and days_to_last_follow_up) and `OS` (1 if Dead, 0 if Alive)
- Falls back to Xena on failure

### `download_tcga_clinical_xena(output_dir) -> str`
**Lines 252–292** | Fallback clinical download from Xena `BRCA_clinicalMatrix.gz`.

### `download_string_ppi(output_dir, species=9606) -> str`
**Lines 295–327** | Downloads STRING v12.0 human protein links (~360MB compressed). Strips `9606.` prefix from protein IDs.

### `download_string_id_mapping(output_dir, species=9606) -> str`
**Lines 330–359** | Downloads STRING protein → gene symbol mapping table.

### `download_disgenet(output_dir, api_key=None, disease_id="UMLS_C0006142") -> str`
**Lines 433–519** | Queries DisGeNET for breast cancer gene-disease associations:
1. Tries official API with API key (if provided)
2. Tries multiple breast cancer CUIs
3. Falls back to curated list of 100 known breast cancer genes

### `download_ncbi_gene_summaries(gene_list, output_dir) -> str`
**Lines 554–640** | Batch NCBI Entrez query for gene functional descriptions:
- Uses `esearch` + `esummary` from NCBI E-utilities
- Processes in batches of 100 genes
- 0.4s delay between batches (NCBI rate limit)
- Merges with existing summaries file if present
- Output: `gene_summaries.json` with 1,750 entries

### `run_data_download(config) -> dict`
**Lines 643–698** | Orchestrator function called by `main.py`:
1. Creates output directories
2. Downloads expression → clinical → STRING → DisGeNET
3. Returns dict of all file paths

---

## Output Artifacts

| File | Size | Description |
|------|------|-------------|
| `data/raw/tcga_brca_expression.tsv.gz` | ~25MB | Gene × sample expression matrix |
| `data/raw/tcga_brca_clinical.tsv` | ~150KB | Clinical data (1098 patients) |
| `data/knowledge_graph/string_ppi.tsv` | ~360MB | STRING PPI network (13.7M edges) |
| `data/knowledge_graph/string_id_mapping.tsv` | ~3MB | Protein → gene name mapping |
| `data/knowledge_graph/disgenet_gene_disease.tsv` | ~5KB | Gene-disease associations (100 entries) |
| `data/embeddings/gene_summaries.json` | ~1MB | NCBI gene functional summaries |

---

## Error Handling

- **GDC 500 errors**: Caught via `requests.raise_for_status()`, falls back to Xena
- **DisGeNET API unavailable**: Falls back to curated gene list via `_disgenet_fallback()`
- **File already exists**: All download functions check for existing files and skip if present (idempotent)
- **NCBI rate limits**: 0.4s delay between batch requests
