"""Stage 0: fetch METABRIC from cBioPortal + 769-gene overlap report (R4 sentinel).

The thesis claim depends on external validation: train on TCGA-BRCA, run inference on
METABRIC, report C-index drop. METABRIC inference uses the same per-patient gene graph,
subset to genes present in METABRIC. If the gene-symbol overlap is < 60%, the design's
"inductive transfer via gene-symbol indexing" leg breaks and the architecture has to
adapt before training (e.g., retrain on the intersection set from the start).

Source: https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz (cBioPortal datahub).

Outputs:
  - data/external/metabric/                : extracted study files (clinical + expression)
  - data/external/metabric_overlap.json    : 769-gene overlap report
  - appends a section to results/stage_0_summary.md
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
RESULTS = ROOT / "results"

METABRIC_DIR = EXT / "brca_metabric"
SELECTED_GENES_PATH = DATA / "selected_genes.txt"
OVERLAP_JSON = EXT / "metabric_overlap.json"
SUMMARY_MD = RESULTS / "stage_0_summary.md"

# cBioPortal datahub stores files via Git LFS; fetch via media.githubusercontent.com.
LFS_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric"
METABRIC_FILES = [
    "data_clinical_patient.txt",
    "data_clinical_sample.txt",
    "data_mrna_illumina_microarray.txt",
    # z-scores file is large (~1GB); we only need it for downstream cross-cohort
    # comparison if we keep TCGA z-scoring. Skip in v1; pull on demand.
]

OVERLAP_FAIL_THRESHOLD = 0.60  # R4 sentinel from design doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stage_0_metabric")


def download_if_missing():
    """Pull individual METABRIC files from cBioPortal datahub LFS storage."""
    METABRIC_DIR.mkdir(parents=True, exist_ok=True)
    for fname in METABRIC_FILES:
        local = METABRIC_DIR / fname
        if local.exists() and local.stat().st_size > 1000:
            log.info(f"  already present: {fname} ({local.stat().st_size/1e6:.1f} MB)")
            continue
        url = f"{LFS_BASE}/{fname}"
        log.info(f"  downloading {fname} from {url}")
        cmd = ["curl", "-sSL", "-A", "Mozilla/5.0", "-o", str(local), url]
        rc = subprocess.run(cmd).returncode
        if rc != 0 or not local.exists() or local.stat().st_size < 1000:
            raise RuntimeError(f"curl failed for {fname} (rc={rc}, size={local.stat().st_size if local.exists() else 0})")
        log.info(f"    -> {local.stat().st_size/1e6:.1f} MB")


def extract_if_needed():
    """No-op: individual files already in place after download_if_missing()."""
    log.info(f"METABRIC files in {METABRIC_DIR}:")
    for p in sorted(METABRIC_DIR.iterdir()):
        log.info(f"  {p.name} ({p.stat().st_size/1e6:.2f} MB)")


def load_metabric_expression():
    """Find and load METABRIC mRNA expression. cBioPortal study layout uses
    `data_mrna_*.txt` files; we prefer the z-scored variant for compatibility
    with the TCGA expression matrix that was z-scored on TCGA cohort.
    """
    candidates = sorted(METABRIC_DIR.glob("data_mrna*.txt"))
    log.info(f"  candidate mRNA files: {[p.name for p in candidates]}")
    # Prefer the median-zscores variant (illumina-microarray-zscores-ref-all-samples is the reference one)
    preferred = [
        "data_mrna_illumina_microarray_zscores_ref_all_samples.txt",
        "data_mrna_illumina_microarray.txt",
        "data_mrna_agilent_microarray_zscores_ref_all_samples.txt",
        "data_mrna_agilent_microarray.txt",
    ]
    chosen = None
    for name in preferred:
        p = METABRIC_DIR / name
        if p.exists():
            chosen = p
            break
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is None:
        raise FileNotFoundError("No data_mrna_*.txt file found in METABRIC distribution")

    log.info(f"  loading expression from {chosen.name} (only header for gene-symbol overlap)")
    # 24k genes x ~2k samples = full read is heavy; for the overlap audit we only
    # need the gene-symbol column. Read with usecols.
    head = pd.read_csv(chosen, sep="\t", nrows=2)
    sym_col_candidate = [c for c in head.columns if c.lower() in ("hugo_symbol", "gene", "gene_symbol")]
    use = sym_col_candidate or [head.columns[0]]
    df = pd.read_csv(chosen, sep="\t", usecols=use, low_memory=False)
    log.info(f"  loaded gene-symbol col only; rows: {len(df)}, cols: {list(df.columns)}")
    return chosen.name, df


def load_metabric_clinical():
    """Find and load METABRIC clinical (sample + patient files)."""
    sample_path = METABRIC_DIR / "data_clinical_sample.txt"
    patient_path = METABRIC_DIR / "data_clinical_patient.txt"
    out = {}
    for label, p in (("sample", sample_path), ("patient", patient_path)):
        if not p.exists():
            log.warning(f"  missing {p.name}")
            continue
        # cBioPortal clinical files have 4-line metadata header; data starts row 5
        df = pd.read_csv(p, sep="\t", comment="#", low_memory=False)
        log.info(f"  {p.name}: {df.shape[0]} rows, {df.shape[1]} cols, sample cols: {list(df.columns[:8])}")
        out[label] = df
    return out


def overlap_report():
    selected = [g.strip() for g in SELECTED_GENES_PATH.read_text().splitlines() if g.strip()]
    selected_set = set(selected)
    n_kg = len(selected_set)
    log.info(f"KG genes: {n_kg}")

    fname, expr = load_metabric_expression()
    # gene symbols typically in 'Hugo_Symbol' col
    sym_col = None
    for c in ("Hugo_Symbol", "HUGO_SYMBOL", "Gene", "gene_symbol"):
        if c in expr.columns:
            sym_col = c
            break
    if sym_col is None:
        log.warning(f"No standard gene-symbol column; first 5 cols: {list(expr.columns[:5])}")
        # fall back to first column
        sym_col = expr.columns[0]
    metabric_genes = set(expr[sym_col].astype(str).dropna().unique())
    log.info(f"METABRIC genes (from '{sym_col}'): {len(metabric_genes)}")

    overlap = selected_set & metabric_genes
    pct = len(overlap) / max(n_kg, 1)
    missing = sorted(selected_set - metabric_genes)[:20]

    log.info(f"Overlap: {len(overlap)} / {n_kg} ({pct*100:.1f}%)")
    log.info(f"First 20 missing genes: {missing}")

    payload = {
        "metabric_expression_file": fname,
        "metabric_gene_symbol_col": sym_col,
        "n_metabric_genes": len(metabric_genes),
        "n_kg_genes": n_kg,
        "overlap_count": len(overlap),
        "overlap_pct": pct,
        "missing_sample": missing,
        "fail_threshold": OVERLAP_FAIL_THRESHOLD,
        "passes_threshold": pct >= OVERLAP_FAIL_THRESHOLD,
    }
    OVERLAP_JSON.write_text(json.dumps(payload, indent=2))
    log.info(f"Overlap report: {OVERLAP_JSON}")

    if pct < OVERLAP_FAIL_THRESHOLD:
        log.error(
            f"FAIL: METABRIC overlap {pct*100:.1f}% < {OVERLAP_FAIL_THRESHOLD*100:.0f}% — "
            f"design's inductive-transfer leg breaks. Either retrain on intersection set "
            f"or revisit gene-symbol normalization."
        )
    else:
        log.info(f"PASS: METABRIC overlap {pct*100:.1f}% >= {OVERLAP_FAIL_THRESHOLD*100:.0f}%")

    return payload


def append_summary(payload, clinical):
    n_samples = (
        clinical["sample"].shape[0] if "sample" in clinical else None
    )
    status = "PASS" if payload["passes_threshold"] else "FAIL"
    lines = [
        "",
        "## METABRIC External Cohort (R4 sentinel)",
        "",
        f"Source: cBioPortal datahub LFS, files: {METABRIC_FILES}",
        f"Expression file used: `{payload['metabric_expression_file']}`",
        f"Gene-symbol column: `{payload['metabric_gene_symbol_col']}`",
        f"Sample count (clinical): {n_samples}" if n_samples else "Sample count: (clinical file missing)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| METABRIC genes | {payload['n_metabric_genes']} |",
        f"| TCGA KG genes | {payload['n_kg_genes']} |",
        f"| Overlap | **{payload['overlap_count']} ({payload['overlap_pct']*100:.1f}%)** |",
        f"| Threshold (R4 sentinel) | {payload['fail_threshold']*100:.0f}% |",
        f"| Status | **{status}** |",
        "",
    ]
    if not payload["passes_threshold"]:
        lines += [
            "**FAIL CONSEQUENCE:** the design's inductive-transfer leg breaks. Either:",
            "1. Retrain TCGA model on the intersection set (gene universe = TCGA ∩ METABRIC) "
            "from the start; or",
            "2. Investigate gene-symbol normalization (HGNC alias resolution, deprecated symbols, "
            "ENSG vs symbol).",
            "",
        ]
    else:
        lines += [
            "Stage 5 external validation can use the existing TCGA-trained model and subset "
            "the per-patient graph to the overlap genes at inference time.",
            "",
        ]
    lines.append(f"First 20 missing genes: `{payload['missing_sample']}`")
    lines.append("")

    existing = SUMMARY_MD.read_text() if SUMMARY_MD.exists() else ""
    SUMMARY_MD.write_text(existing + "\n".join(lines))
    log.info(f"Summary appended to {SUMMARY_MD}")


def main():
    download_if_missing()
    extract_if_needed()
    clinical = load_metabric_clinical()
    payload = overlap_report()
    append_summary(payload, clinical)


if __name__ == "__main__":
    main()
