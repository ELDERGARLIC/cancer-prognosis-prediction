"""FIX 1: Recover ER and PR status from raw TCGA clinical and patch
clinical_features.tsv.

The inherited preprocessing produced all-zero `er_signed` and `pr_signed`
columns (Stage 0 finding; Methods §1.3 limitation). The cBioPortal legacy
brca_tcga study's patient-level clinical file carries ER_STATUS_BY_IHC and
PR_STATUS_BY_IHC fields with values {Positive, Negative, Indeterminate,
[Not Available], etc.}. This script:

  1. Parses data/external/brca_tcga/data_clinical_patient.txt
  2. Extracts ER + PR + HER2 IHC status per patient
  3. Maps to signed values: Positive → +1, Negative → −1, otherwise → 0
  4. Patches data/processed/clinical_features.tsv in place (preserving row
     order, which aligns with expression_selected.tsv columns)
  5. Reports coverage statistics and missing-data audit

After this, scripts/00_baseline.py and the Stage 3 knob-A pipeline must be
re-run to reflect the recovered features in C-index numbers.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_TCGA_PATIENT = ROOT / "data" / "external" / "brca_tcga" / "data_clinical_patient.txt"
CLIN_FEATURES = ROOT / "data" / "processed" / "clinical_features.tsv"
SAMPLE_MAP = ROOT / "data" / "processed" / "sample_patient_mapping.tsv"
SURVIVAL = ROOT / "data" / "processed" / "clinical_processed.tsv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("recover_er_pr")


def map_status(s) -> int:
    """Positive → +1, Negative → −1, otherwise → 0."""
    if pd.isna(s):
        return 0
    s = str(s).strip().lower()
    if s == "positive":
        return 1
    if s == "negative":
        return -1
    return 0


def main():
    log.info(f"Loading raw TCGA clinical from {RAW_TCGA_PATIENT}")
    # cBioPortal patient file: 4 metadata rows + 1 header (column-name) row
    df = pd.read_csv(RAW_TCGA_PATIENT, sep="\t", skiprows=4)
    log.info(f"  raw shape: {df.shape}")
    log.info(f"  patient ID column: PATIENT_ID (first values: "
             f"{df['PATIENT_ID'].head(3).tolist()})")

    # Recover ER, PR, HER2 status
    er_col = "ER_STATUS_BY_IHC"
    pr_col = "PR_STATUS_BY_IHC"
    her2_col = "IHC_HER2"
    for c in (er_col, pr_col, her2_col):
        assert c in df.columns, f"missing column {c}"

    er_dist = df[er_col].value_counts(dropna=False).to_dict()
    pr_dist = df[pr_col].value_counts(dropna=False).to_dict()
    her2_dist = df[her2_col].value_counts(dropna=False).to_dict()
    log.info(f"  raw ER distribution: {er_dist}")
    log.info(f"  raw PR distribution: {pr_dist}")
    log.info(f"  raw HER2 distribution: {her2_dist}")

    df["er_signed_recovered"] = df[er_col].map(map_status)
    df["pr_signed_recovered"] = df[pr_col].map(map_status)
    df["her2_signed_recovered"] = df[her2_col].map(map_status)

    # Build patient_id → (er, pr, her2) mapping
    rec = df.set_index("PATIENT_ID")[
        ["er_signed_recovered", "pr_signed_recovered", "her2_signed_recovered"]
    ]
    log.info(f"  unique patients in raw: {len(rec)}")

    # Now load the existing clinical_features.tsv. It has no patient-id column,
    # so we use sample_patient_mapping.tsv (or clinical_processed) to align.
    log.info(f"Loading clinical_features.tsv: {CLIN_FEATURES}")
    feat = pd.read_csv(CLIN_FEATURES, sep="\t")
    log.info(f"  shape: {feat.shape}, cols: {list(feat.columns)}")

    # Get patient IDs in the same row order as expression_selected.tsv columns.
    # Use clinical_processed.tsv which has case_id and aligns with expression by row.
    surv = pd.read_csv(SURVIVAL, sep="\t")
    case_ids = surv["case_id"].tolist()
    assert len(case_ids) == feat.shape[0], (
        f"row count mismatch: clinical_features.tsv={feat.shape[0]}, "
        f"clinical_processed.tsv={len(case_ids)}"
    )
    log.info(f"  aligned via {len(case_ids)} case_ids from clinical_processed.tsv")

    # Map case_ids to recovered values
    er_recovered = []
    pr_recovered = []
    her2_recovered = []
    n_unmapped = 0
    n_unknown_status = {"er": 0, "pr": 0, "her2": 0}
    for cid in case_ids:
        if cid not in rec.index:
            n_unmapped += 1
            er_recovered.append(0)
            pr_recovered.append(0)
            her2_recovered.append(0)
            continue
        er_v = int(rec.loc[cid, "er_signed_recovered"])
        pr_v = int(rec.loc[cid, "pr_signed_recovered"])
        her2_v = int(rec.loc[cid, "her2_signed_recovered"])
        er_recovered.append(er_v)
        pr_recovered.append(pr_v)
        her2_recovered.append(her2_v)
        if er_v == 0: n_unknown_status["er"] += 1
        if pr_v == 0: n_unknown_status["pr"] += 1
        if her2_v == 0: n_unknown_status["her2"] += 1

    log.info(f"  patients not found in raw TCGA clinical: {n_unmapped}/{len(case_ids)}")
    log.info(f"  patients with unknown ER status (mapped to 0): {n_unknown_status['er']}")
    log.info(f"  patients with unknown PR status (mapped to 0): {n_unknown_status['pr']}")
    log.info(f"  patients with unknown HER2 status (mapped to 0): {n_unknown_status['her2']}")

    # Patch the clinical_features.tsv. Replace the existing all-zero er_signed/pr_signed
    # columns with the recovered values, and add her2_signed as a new column.
    er_arr = np.array(er_recovered, dtype=np.float32)
    pr_arr = np.array(pr_recovered, dtype=np.float32)
    her2_arr = np.array(her2_recovered, dtype=np.float32)

    log.info(f"  ER signed distribution after recovery: "
             f"{pd.Series(er_arr).value_counts().to_dict()}")
    log.info(f"  PR signed distribution after recovery: "
             f"{pd.Series(pr_arr).value_counts().to_dict()}")
    log.info(f"  HER2 signed distribution after recovery: "
             f"{pd.Series(her2_arr).value_counts().to_dict()}")

    # Backup original
    backup = CLIN_FEATURES.with_suffix(".tsv.bak_pre_erpr_fix")
    if not backup.exists():
        log.info(f"  backing up original to {backup}")
        feat.to_csv(backup, sep="\t", index=False)

    feat["er_signed"] = er_arr
    feat["pr_signed"] = pr_arr
    feat["her2_signed"] = her2_arr
    feat.to_csv(CLIN_FEATURES, sep="\t", index=False)
    log.info(f"  wrote patched clinical_features.tsv (now has columns: "
             f"{list(feat.columns)})")

    # Variance check (this was the diagnostic that revealed the bug)
    for c in ("er_signed", "pr_signed", "her2_signed"):
        v = float(feat[c].var())
        log.info(f"  variance({c}) = {v:.4f}  "
                 f"({'PASS' if v >= 0.01 else 'FAIL'} the 0.01 low-variance threshold)")


if __name__ == "__main__":
    main()
