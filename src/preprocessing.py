"""
Stage 1.2-1.5: Gene Expression Preprocessing, Feature Selection, and Survival Labels

Pipeline:
    1. Filter low-expression genes (Rahaman et al., 2023)
    2. Normalize: log2(TPM+1) or TMM-style normalization
    3. Z-score standardize
    4. LASSO feature selection (Alharbi et al., 2025; Saadh et al., 2025)
    5. DisGeNET intersection filtering (Qumsiyeh et al., 2022)
    6. Survival label discretization (Zheng et al., 2024)
    7. Clinical feature extraction (Gao et al., 2021)
    8. SMOTE class balancing (Vaida et al., 2025)
"""

import os
import logging
import gzip
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_expression_data(expr_path: str) -> pd.DataFrame:
    """Load gene expression matrix from TSV or compressed TSV.

    Returns:
        DataFrame with genes as rows, samples as columns.
    """
    if expr_path.endswith(".gz"):
        with gzip.open(expr_path, "rt") as f:
            df = pd.read_csv(f, sep="\t", index_col=0)
    else:
        df = pd.read_csv(expr_path, sep="\t", index_col=0)

    logger.info(f"Loaded expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df


def load_clinical_data(clinical_path: str) -> pd.DataFrame:
    """Load clinical data."""
    df = pd.read_csv(clinical_path, sep="\t")
    logger.info(f"Loaded clinical data: {df.shape[0]} patients, {df.shape[1]} columns")
    return df


def filter_low_expression_genes(expr_df: pd.DataFrame, min_total_counts: int = 1000) -> pd.DataFrame:
    """Filter genes with low total expression across all samples.

    Reference: Rahaman et al., 2023 — keep genes with counts > threshold.
    """
    total_counts = expr_df.sum(axis=1)
    mask = total_counts > min_total_counts
    filtered = expr_df[mask]
    logger.info(
        f"Gene filtering: {expr_df.shape[0]} -> {filtered.shape[0]} genes "
        f"(threshold: total counts > {min_total_counts})"
    )
    return filtered


def normalize_expression(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(x+1) normalization followed by z-score standardization.

    For Xena data (already log2(norm_count+1)), we just z-score standardize.
    For raw counts, we apply log2(x+1) first.
    """
    # Check if data appears to be already log-transformed
    max_val = expr_df.max().max()
    if max_val > 100:
        # Likely raw counts, apply log2(x+1)
        logger.info("Applying log2(x+1) transformation (detected raw counts)")
        expr_df = np.log2(expr_df + 1)
    else:
        logger.info("Data appears to be pre-normalized (max={:.2f}), skipping log transform".format(max_val))

    # Z-score standardize each gene across patients
    scaler = StandardScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(expr_df.T).T,
        index=expr_df.index,
        columns=expr_df.columns,
    )
    logger.info("Applied z-score standardization across samples")
    return normalized


def impute_clinical_data(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing clinical values using KNN imputation.

    Reference: sklearn.impute.KNNImputer

    IMPORTANT: Categorical/label columns (tumor_stage, pathologic_stage,
    ER/PR/HER2 status, etc.) are preserved as-is. Filling them with 0 breaks
    downstream parsing (see the tumor-stage-all-minus-one bug from the first
    run). Only truly numeric columns get KNN-imputed.
    """
    # Separate numeric and categorical columns
    numeric_cols = clinical_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = clinical_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Skip ID columns for imputation
    id_cols = [c for c in categorical_cols if "id" in c.lower() or "submitter" in c.lower()]
    categorical_cols = [c for c in categorical_cols if c not in id_cols]

    # Protect clinical-label columns that are semantically categorical even if
    # they happen to be numeric-typed (some TCGA extracts store stage as int).
    PROTECTED_KEYWORDS = (
        "stage", "status", "grade", "subtype", "er_", "pr_", "her2",
        "hormone", "metastasis", "metastatic", "recurrence", "histolog",
    )
    protected_numeric = [
        c for c in numeric_cols
        if any(kw in c.lower() for kw in PROTECTED_KEYWORDS)
    ]
    numeric_cols = [c for c in numeric_cols if c not in protected_numeric]

    if numeric_cols:
        n_missing_before = clinical_df[numeric_cols].isnull().sum().sum()
        if n_missing_before > 0:
            imputable_cols = [c for c in numeric_cols if clinical_df[c].notna().any()]
            all_nan_cols = [c for c in numeric_cols if c not in imputable_cols]

            if imputable_cols:
                imputer = KNNImputer(n_neighbors=5)
                imputed = pd.DataFrame(
                    imputer.fit_transform(clinical_df[imputable_cols]),
                    columns=imputable_cols,
                    index=clinical_df.index,
                )
                clinical_df[imputable_cols] = imputed

            # Leave all-NaN numeric columns alone -- filling with 0 masquerades
            # as signal and can corrupt downstream feature extraction. Drop
            # them instead so nothing silently uses a constant column.
            for col in all_nan_cols:
                logger.warning(f"Column '{col}' is entirely NaN — dropping from clinical frame")
                clinical_df = clinical_df.drop(columns=[col])

            logger.info(f"KNN-imputed {n_missing_before} missing numeric values")

    if protected_numeric:
        logger.info(
            f"Preserved {len(protected_numeric)} clinical label columns "
            f"without imputation: {protected_numeric}"
        )

    return clinical_df


def select_features_lasso(
    expr_df: pd.DataFrame,
    labels: np.ndarray,
    n_genes: int = 1500,
    seed: int = 42,
) -> list:
    """Select top genes using LASSO regression.

    Reference: Alharbi et al., 2025; Saadh et al., 2025
    """
    np.random.seed(seed)

    X = expr_df.T.values  # samples x genes
    y = labels

    # Remove samples with NaN labels
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Running LASSO feature selection on {X.shape[1]} genes, {X.shape[0]} samples...")

    # Use LassoCV for automatic alpha selection
    lasso = LassoCV(cv=5, random_state=seed, max_iter=10000, n_jobs=-1)
    lasso.fit(X, y)

    # Get absolute coefficients
    coef_abs = np.abs(lasso.coef_)
    gene_importance = pd.Series(coef_abs, index=expr_df.index)
    gene_importance = gene_importance.sort_values(ascending=False)

    # Select top n_genes with non-zero coefficients
    nonzero_genes = gene_importance[gene_importance > 0]
    logger.info(f"LASSO selected {len(nonzero_genes)} genes with non-zero coefficients")

    if len(nonzero_genes) >= n_genes:
        selected = nonzero_genes.head(n_genes).index.tolist()
    else:
        # If LASSO selects fewer genes, supplement with highest-variance genes
        remaining = n_genes - len(nonzero_genes)
        variance = expr_df.var(axis=1)
        variance = variance.drop(nonzero_genes.index, errors="ignore")
        top_var = variance.nlargest(remaining).index.tolist()
        selected = nonzero_genes.index.tolist() + top_var
        logger.info(f"Supplemented with {remaining} high-variance genes")

    logger.info(f"Total selected genes: {len(selected)}")
    return selected


def select_features_rfe(
    expr_df: pd.DataFrame,
    labels: np.ndarray,
    n_genes: int = 1500,
    seed: int = 42,
) -> list:
    """Alternative feature selection using Recursive Feature Elimination."""
    np.random.seed(seed)

    X = expr_df.T.values
    y = labels

    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)

    logger.info(f"Running RFE feature selection on {X.shape[1]} genes...")

    # Use a fast estimator for RFE
    estimator = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    # Step through features more aggressively for speed
    step = max(1, (X.shape[1] - n_genes) // 10)
    rfe = RFE(estimator, n_features_to_select=n_genes, step=step)
    rfe.fit(X, y)

    selected = expr_df.index[rfe.support_].tolist()
    logger.info(f"RFE selected {len(selected)} genes")
    return selected


def augment_with_disgenet_priors(
    selected_genes: list,
    disgenet_path: str,
    expression_gene_index: pd.Index,
    disease_semantic_type: str = "Neoplastic Process",
    max_priors: int = 500,
) -> list:
    """Augment the LASSO-selected universe with DisGeNET BRCA-associated genes.

    Reference: Qumsiyeh et al., 2022.

    Phase 1 motivation: the old `filter_with_disgenet` did an *intersection*,
    which with LASSO-1000 gave exactly 11 overlap -- not enough to build any
    meaningful gene-disease subgraph. The fix is a *union*: keep all LASSO
    picks (they maximize survival-predictive signal) AND pull in DisGeNET's
    curated BRCA-associated genes (they maximize biological prior) if they
    exist in the expression matrix. Duplicates are deduped by symbol.

    Cap at `max_priors` DisGeNET additions so the universe doesn't blow up
    past what downstream steps were sized for.

    Args:
        selected_genes: LASSO-picked symbols (already the main universe).
        disgenet_path: TSV dump.
        expression_gene_index: symbols available in the expression matrix;
            DisGeNET priors are only added if expression data exists for them.
        disease_semantic_type: filter for "Neoplastic Process".
        max_priors: cap on DisGeNET-sourced additions.
    """
    if not os.path.exists(disgenet_path):
        logger.warning(f"DisGeNET file not found: {disgenet_path}. Skipping augmentation.")
        return selected_genes

    disgenet = pd.read_csv(disgenet_path, sep="\t")

    if "diseaseSemanticType" in disgenet.columns:
        disgenet = disgenet[
            disgenet["diseaseSemanticType"].str.contains(disease_semantic_type, na=False)
        ]

    gene_col = "geneSymbol" if "geneSymbol" in disgenet.columns else "gene_symbol"
    if gene_col not in disgenet.columns:
        logger.warning("Could not identify gene column in DisGeNET file")
        return selected_genes

    # Rank DisGeNET genes by score (if present) so we pick the strongest
    # associations first when hitting the max_priors cap.
    if "score" in disgenet.columns:
        disgenet = disgenet.sort_values("score", ascending=False)

    disgenet_symbols_upper = [
        str(g).upper() for g in disgenet[gene_col].dropna().tolist()
    ]

    # Dedupe while preserving order (score-sorted above)
    seen = set()
    disgenet_ranked = []
    for g in disgenet_symbols_upper:
        if g in seen:
            continue
        seen.add(g)
        disgenet_ranked.append(g)

    # Filter to genes actually in the expression matrix
    expression_universe_upper = {str(g).upper(): g for g in expression_gene_index}
    disgenet_in_expr = [
        expression_universe_upper[g]
        for g in disgenet_ranked
        if g in expression_universe_upper
    ]

    # Existing LASSO set (uppercase for dedupe)
    selected_upper = {g.upper() for g in selected_genes}

    priors_to_add = []
    for g in disgenet_in_expr:
        if g.upper() in selected_upper:
            continue
        priors_to_add.append(g)
        if len(priors_to_add) >= max_priors:
            break

    unioned = list(selected_genes) + priors_to_add

    logger.info(
        f"DisGeNET augmentation: LASSO={len(selected_genes)}, "
        f"DisGeNET candidates={len(disgenet_in_expr)}, "
        f"added_priors={len(priors_to_add)} (cap={max_priors}), "
        f"final union={len(unioned)}"
    )
    return unioned


# Backward-compat shim: existing call sites use `filter_with_disgenet`. The
# union semantics are a strict improvement, but we preserve the old signature
# so this one change doesn't cascade into every downstream caller.
def filter_with_disgenet(
    selected_genes: list,
    disgenet_path: str,
    disease_semantic_type: str = "Neoplastic Process",
    expression_gene_index: pd.Index = None,
    max_priors: int = 500,
) -> list:
    """DEPRECATED name; forwards to augment_with_disgenet_priors (union)."""
    if expression_gene_index is None:
        # Without expression knowledge we can't safely add priors, so fall
        # back to the no-op that the old code hit when intersection was small.
        logger.warning(
            "filter_with_disgenet called without expression_gene_index; "
            "returning LASSO set unchanged."
        )
        return selected_genes
    return augment_with_disgenet_priors(
        selected_genes=selected_genes,
        disgenet_path=disgenet_path,
        expression_gene_index=expression_gene_index,
        disease_semantic_type=disease_semantic_type,
        max_priors=max_priors,
    )


def create_survival_labels(
    clinical_df: pd.DataFrame,
    bins: list = None,
    labels_names: list = None,
) -> pd.DataFrame:
    """Discretize survival time into classes.

    Reference: Zheng et al., 2024
        Class 0: < 1 year (365 days)
        Class 1: 1-3 years (365-1095 days)
        Class 2: 3-5 years (1095-1825 days)
        Class 3: > 5 years

    Also keeps continuous OS.time + event status for C-index.
    """
    if bins is None:
        bins = [365, 1095, 1825]
    if labels_names is None:
        labels_names = ["<1yr", "1-3yr", "3-5yr", ">5yr"]

    df = clinical_df.copy()

    # Ensure we have OS.time
    if "OS.time" not in df.columns:
        # Try to compute from available columns
        for col in ["days_to_death", "days_to_last_follow_up", "OS_time", "_OS_time"]:
            if col in df.columns:
                if "OS.time" not in df.columns:
                    df["OS.time"] = df[col]
                else:
                    df["OS.time"] = df["OS.time"].fillna(df[col])

    if "OS.time" not in df.columns:
        raise ValueError("Cannot find OS.time or equivalent column in clinical data")

    # Convert to numeric
    df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")

    # Drop patients with missing survival time
    valid_mask = df["OS.time"].notna() & (df["OS.time"] > 0)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} patients with missing/invalid survival time")

    # Create discrete survival bins
    bin_edges = [0] + bins + [float("inf")]
    df["survival_class"] = pd.cut(
        df["OS.time"],
        bins=bin_edges,
        labels=list(range(len(labels_names))),
        right=True,
    )
    df["survival_label"] = pd.cut(
        df["OS.time"],
        bins=bin_edges,
        labels=labels_names,
        right=True,
    )

    # Log class distribution
    class_dist = df["survival_class"].value_counts().sort_index()
    logger.info("Survival class distribution:")
    for cls, count in class_dist.items():
        label = labels_names[int(cls)] if pd.notna(cls) else "NaN"
        logger.info(f"  Class {cls} ({label}): {count} patients")

    return df


import re

# Compiled once at import to avoid re-parsing on every call. These capture:
#   - "Stage IV"/"Stage IIA"/"iv"/"iiia"   ->  roman numeral
#   - "T3N1M0" / "T2N0" / "pT1c"            ->  T-stage proxy
#   - "1"/"2.0"/"3B"                        ->  numeric / numeric + letter
# Note on the regexes:
#  - ROMAN: we must tolerate substage letters (IIA/IIIC/IB). The alternation
#    order matters: greedy longest-first (iv, iii, ii, i), then [abc]?
#    swallows the substage before the \b closes the word.
#  - LOOSE: a bare roman like "iiia" has no word boundary between "iii" and
#    "a" (both word chars), so use anchored fullmatch against ^...$ with
#    optional substage tail.
#  - TNM: codes like "T3N1M0" have no separator between digit and "N", so
#    the old trailing [a-z]?\b couldn't close. Drop the tail and only
#    require \b before "T" -- the T-stage digit is a sufficient proxy.
_STAGE_ROMAN_RE = re.compile(r"\bstage\s*(iv|iii|ii|i)[abc]?\b", re.IGNORECASE)
_STAGE_ROMAN_LOOSE_RE = re.compile(r"^(iv|iii|ii|i)[abc]?$", re.IGNORECASE)
_STAGE_TNM_RE = re.compile(r"\bp?t(\d)", re.IGNORECASE)
_STAGE_NUMERIC_RE = re.compile(r"^\s*([1-4])[abc]?\s*$", re.IGNORECASE)

_STAGE_UNKNOWN = {
    "", "nan", "none", "not reported", "unknown", "[not available]",
    "[not applicable]", "[unknown]", "stage x", "tx", "not evaluated",
}


def _parse_stage_string(s) -> int:
    """Parse a tumor-stage string to ordinal 0-3 (I-IV). Returns -1 if unknown.

    Handles:
      - "Stage IIA", "Stage IIIC", "iiia"          -> roman numeral
      - AJCC TNM codes: "T2N1M0", "pT1c"           -> T-stage as a proxy
      - Numeric codes from older extracts: "1.0", "3B" -> int
      - Sentinel values (NaN, "[Not Available]", "Stage X") -> -1

    Unit tests (enforced via assertions at module import):
      _parse_stage_string("Stage IIA") == 1
      _parse_stage_string("Stage IV") == 3
      _parse_stage_string("T3N1M0") == 2
      _parse_stage_string("[Not Available]") == -1
      _parse_stage_string(float('nan')) == -1
    """
    if s is None:
        return -1
    # numpy / pandas NaN scalars
    try:
        if isinstance(s, float) and pd.isna(s):
            return -1
    except (TypeError, ValueError):
        pass

    raw = str(s).strip().lower()
    if raw in _STAGE_UNKNOWN:
        return -1

    # 1) "Stage IIA" / "Stage IV" -- most reliable
    m = _STAGE_ROMAN_RE.search(raw)
    if m:
        roman = m.group(1).lower()
        return {"i": 0, "ii": 1, "iii": 2, "iv": 3}[roman]

    # 2) Pure roman without "Stage" prefix
    stripped = raw.replace("stage", "").strip()
    m = _STAGE_ROMAN_LOOSE_RE.fullmatch(stripped.split()[0] if stripped.split() else "")
    if m:
        return {"i": 0, "ii": 1, "iii": 2, "iv": 3}[m.group(1).lower()]

    # 3) Bare numeric with optional subgrade letter ("3B", "2.0")
    m = _STAGE_NUMERIC_RE.match(stripped)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 4:
            return n - 1
    try:
        n = int(float(stripped.rstrip("abc")))
        if 1 <= n <= 4:
            return n - 1
    except (ValueError, TypeError):
        pass

    # 4) AJCC TNM code -- use T-stage as ordinal proxy
    m = _STAGE_TNM_RE.search(raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 4:
            return n - 1

    return -1


# Self-test -- catches regressions at import time rather than in training.
assert _parse_stage_string("Stage IIA") == 1, "stage parser: Stage IIA should be 1"
assert _parse_stage_string("Stage IIIC") == 2, "stage parser: Stage IIIC should be 2"
assert _parse_stage_string("Stage IV") == 3, "stage parser: Stage IV should be 3"
assert _parse_stage_string("iiia") == 2, "stage parser: lowercase roman"
assert _parse_stage_string("T3N1M0") == 2, "stage parser: TNM -> T-stage"
assert _parse_stage_string("pT1c") == 0, "stage parser: pT1c -> 0"
assert _parse_stage_string("[Not Available]") == -1
assert _parse_stage_string("Stage X") == -1, "stage parser: Stage X -> -1"
assert _parse_stage_string(float("nan")) == -1
assert _parse_stage_string(None) == -1
assert _parse_stage_string("2.0") == 1


def extract_clinical_features(clinical_df: pd.DataFrame) -> tuple:
    """Extract and encode clinical features for model fusion.

    Reference: Gao et al., 2021 — concatenation approach.

    Returns:
        (feature_matrix, feature_names, patient_ids)
    """
    df = clinical_df.copy()

    features = {}

    # Age at diagnosis (convert from days to years if needed)
    age_col = None
    for col in ["age_at_diagnosis", "age_at_initial_pathologic_diagnosis", "age"]:
        if col in df.columns:
            age_col = col
            break
    if age_col:
        age = pd.to_numeric(df[age_col], errors="coerce")
        if age.median() and age.median() > 200:  # Likely in days
            age = age / 365.25
        # Normalize age to z-score so it doesn't dominate small binary features
        age_mean = age.mean()
        age_std = age.std()
        if age_std and age_std > 0:
            features["age"] = (age - age_mean) / age_std
        else:
            features["age"] = age.fillna(0)

    # Tumor stage -- encode as ordinal (0-3) AND one-hot so models can use both.
    stage_col = None
    for col in [
        "tumor_stage", "pathologic_stage", "ajcc_pathologic_stage",
        "clinical_stage", "ajcc_pathologic_tumor_stage",
    ]:
        if col in df.columns:
            stage_col = col
            break

    if stage_col:
        # Pre-dump value_counts to aid debugging future schema drifts.
        raw_top = df[stage_col].value_counts(dropna=False).head(10).to_dict()
        logger.info(f"Tumor stage raw value counts (top 10): {raw_top}")

        stage_ordinal = df[stage_col].apply(_parse_stage_string)
        valid_mask = stage_ordinal >= 0
        n_valid = int(valid_mask.sum())
        pct_valid = 100 * n_valid / max(len(stage_ordinal), 1)
        logger.info(
            f"Tumor stage ({stage_col}): {n_valid}/{len(stage_ordinal)} parsed "
            f"({pct_valid:.1f}%)"
        )

        # Require >=70% parseability before we trust stage as a feature. This
        # guard prevents a mostly-empty column from being encoded as 'all median'
        # which injects a constant signal and confuses the model.
        STAGE_FEATURE_MIN_PCT = 70.0
        if pct_valid >= STAGE_FEATURE_MIN_PCT:
            # Fill unknowns with median stage so one-hot/ordinal is defined
            median_stage = int(stage_ordinal[valid_mask].median())
            stage_filled = stage_ordinal.where(valid_mask, median_stage)
            features["stage_ordinal"] = stage_filled.astype(float) / 3.0  # scale to [0,1]
            for s_idx, s_name in enumerate(["I", "II", "III", "IV"]):
                features[f"stage_{s_name}"] = (stage_filled == s_idx).astype(float)
            logger.info(f"Stage features enabled (parsed {pct_valid:.1f}% >= {STAGE_FEATURE_MIN_PCT}%).")
        else:
            logger.warning(
                f"Stage column '{stage_col}' only {pct_valid:.1f}% parseable "
                f"(< {STAGE_FEATURE_MIN_PCT}%); skipping stage features."
            )

    # ER/PR/HER2 status (look for these in column names)
    for marker in ["er", "pr", "her2"]:
        marker_col = None
        # Prefer explicit *_status columns, fall back to marker-containing column
        for col in df.columns:
            lc = col.lower()
            if marker == "er" and "estrogen" in lc:
                marker_col = col
                break
            if marker == "pr" and "progesterone" in lc:
                marker_col = col
                break
            if marker in lc and "status" in lc:
                marker_col = col
                break
        if marker_col is None:
            for col in df.columns:
                if marker in col.lower():
                    marker_col = col
                    break

        if marker_col:
            status = df[marker_col].astype(str).str.lower()
            pos = status.str.contains(r"positive|pos|\+", na=False, regex=True)
            neg = status.str.contains(r"negative|neg|-", na=False, regex=True)
            # Three-state: +1 positive, -1 negative, 0 unknown
            features[f"{marker}_signed"] = pos.astype(float) - neg.astype(float)

    # Gender
    if "gender" in df.columns:
        features["is_female"] = (df["gender"].astype(str).str.lower() == "female").astype(float)

    feature_df = pd.DataFrame(features)

    # Impute remaining NaN with 0 (these are truly missing binary flags)
    feature_df = feature_df.fillna(0)

    logger.info(f"Extracted {feature_df.shape[1]} clinical features: {list(feature_df.columns)}")
    return feature_df


def get_patient_sample_mapping(expr_df: pd.DataFrame, clinical_df: pd.DataFrame) -> dict:
    """Map between expression sample IDs and clinical patient IDs.

    TCGA barcodes: TCGA-XX-XXXX-01A = tumor sample from patient TCGA-XX-XXXX
    """
    # Extract patient ID from sample barcode (first 12 characters for TCGA)
    sample_to_patient = {}
    for sample_id in expr_df.columns:
        if sample_id.startswith("TCGA"):
            patient_id = "-".join(sample_id.split("-")[:3])  # TCGA-XX-XXXX
            sample_to_patient[sample_id] = patient_id
        else:
            sample_to_patient[sample_id] = sample_id

    # Find matching patients
    clinical_ids = set()
    id_col = None
    for col in ["case_id", "submitter_id", "sampleID", "bcr_patient_barcode"]:
        if col in clinical_df.columns:
            id_col = col
            clinical_ids = set(clinical_df[col].astype(str))
            break

    if id_col is None:
        # Try index
        clinical_ids = set(clinical_df.index.astype(str))

    matched = {s: p for s, p in sample_to_patient.items() if p in clinical_ids}
    logger.info(
        f"Matched {len(matched)}/{len(sample_to_patient)} expression samples to clinical records"
    )
    return sample_to_patient


def run_preprocessing(config: dict, expr_path: str, clinical_path: str, disgenet_path: str = None) -> dict:
    """Run the complete preprocessing pipeline.

    Returns:
        Dictionary with processed data arrays and metadata.
    """
    seed = config["training"]["seed"]
    np.random.seed(seed)

    processed_dir = config["paths"]["processed_data"]
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load data
    logger.info("=" * 60)
    logger.info("Loading expression and clinical data")
    logger.info("=" * 60)
    expr_df = load_expression_data(expr_path)
    clinical_df = load_clinical_data(clinical_path)

    # 2. Map samples to patients
    sample_to_patient = get_patient_sample_mapping(expr_df, clinical_df)

    # 3. Filter low-expression genes
    min_counts = config["data"]["min_total_counts"]
    expr_df = filter_low_expression_genes(expr_df, min_total_counts=min_counts)

    # 4. Normalize
    logger.info("=" * 60)
    logger.info("Normalizing gene expression")
    logger.info("=" * 60)
    expr_df = normalize_expression(expr_df)

    # 5. Create survival labels
    logger.info("=" * 60)
    logger.info("Creating survival labels")
    logger.info("=" * 60)
    clinical_df = impute_clinical_data(clinical_df)
    clinical_df = create_survival_labels(
        clinical_df,
        bins=config["data"]["survival_bins"],
        labels_names=config["data"]["survival_labels"],
    )

    # 6. Match patients between expression and clinical data
    # Determine which ID column to use
    id_col = None
    for col in ["case_id", "submitter_id", "sampleID", "bcr_patient_barcode"]:
        if col in clinical_df.columns:
            id_col = col
            break

    if id_col is None:
        id_col = clinical_df.columns[0]

    # Map expression columns to patient IDs and filter
    matched_samples = []
    matched_patients = []
    for sample_id in expr_df.columns:
        patient_id = sample_to_patient.get(sample_id, sample_id)
        if id_col and patient_id in clinical_df[id_col].values:
            matched_samples.append(sample_id)
            matched_patients.append(patient_id)
        elif patient_id in clinical_df.index.astype(str).values:
            matched_samples.append(sample_id)
            matched_patients.append(patient_id)

    if not matched_samples:
        # Fall back: use expression columns directly if they match clinical IDs
        logger.warning("No direct matches found. Attempting index-based matching...")
        expr_samples = set(expr_df.columns)
        clin_ids = set(clinical_df[id_col].astype(str)) if id_col else set(clinical_df.index.astype(str))
        matched_samples = list(expr_samples & clin_ids)
        matched_patients = matched_samples

    logger.info(f"Matched {len(matched_samples)} patients between expression and clinical data")

    expr_matched = expr_df[matched_samples]

    # Get corresponding clinical data with survival labels
    if id_col:
        clinical_matched = clinical_df[clinical_df[id_col].isin(matched_patients)].copy()
        # Ensure same order
        clinical_matched = clinical_matched.set_index(id_col)
        clinical_matched = clinical_matched.loc[[sample_to_patient.get(s, s) for s in matched_samples]]
        clinical_matched = clinical_matched.reset_index()
    else:
        clinical_matched = clinical_df.loc[matched_patients]

    # 7. LASSO feature selection
    logger.info("=" * 60)
    logger.info("Running LASSO feature selection")
    logger.info("=" * 60)

    survival_labels = clinical_matched["survival_class"].values.astype(float)
    n_genes = config["data"]["n_genes_lasso"]

    selected_genes = select_features_lasso(expr_matched, survival_labels, n_genes=n_genes, seed=seed)

    # 8. DisGeNET augmentation (Phase 1 fix: union, not intersection).
    # Bringing in curated BRCA-associated genes gives the gene-disease edge
    # set somewhere to attach to -- the LASSO universe by itself only
    # overlapped DisGeNET by 11 genes, which is too few to build any
    # meaningful gene-disease subgraph.
    if disgenet_path and os.path.exists(disgenet_path):
        logger.info("=" * 60)
        logger.info("Augmenting LASSO universe with DisGeNET BRCA priors")
        logger.info("=" * 60)
        max_priors = int(config.get("data", {}).get("disgenet_max_priors", 500))
        selected_genes = augment_with_disgenet_priors(
            selected_genes=selected_genes,
            disgenet_path=disgenet_path,
            expression_gene_index=expr_matched.index,
            disease_semantic_type=config["data"]["disease_semantic_type"],
            max_priors=max_priors,
        )

    # 9. Extract clinical features
    logger.info("=" * 60)
    logger.info("Extracting clinical features")
    logger.info("=" * 60)
    clinical_features = extract_clinical_features(clinical_matched)

    # 10. Prepare final expression matrix with selected genes
    # Keep only genes that are in our expression matrix
    available_genes = [g for g in selected_genes if g in expr_matched.index]
    if len(available_genes) < len(selected_genes):
        logger.warning(
            f"Only {len(available_genes)}/{len(selected_genes)} selected genes found in expression matrix"
        )
    expr_selected = expr_matched.loc[available_genes]

    # 11. Save processed data
    logger.info("=" * 60)
    logger.info("Saving processed data")
    logger.info("=" * 60)

    expr_selected.to_csv(os.path.join(processed_dir, "expression_selected.tsv"), sep="\t")
    clinical_matched.to_csv(os.path.join(processed_dir, "clinical_processed.tsv"), sep="\t", index=False)
    clinical_features.to_csv(os.path.join(processed_dir, "clinical_features.tsv"), sep="\t", index=False)

    # Save gene list
    with open(os.path.join(processed_dir, "selected_genes.txt"), "w") as f:
        f.write("\n".join(available_genes))

    # Save sample-to-patient mapping
    pd.Series(sample_to_patient).to_csv(
        os.path.join(processed_dir, "sample_patient_mapping.tsv"), sep="\t"
    )

    results = {
        "expression": expr_selected,
        "clinical": clinical_matched,
        "clinical_features": clinical_features,
        "selected_genes": available_genes,
        "survival_labels": survival_labels,
        "sample_ids": matched_samples,
        "patient_ids": matched_patients,
    }

    logger.info(f"Preprocessing complete:")
    logger.info(f"  Genes: {len(available_genes)}")
    logger.info(f"  Patients: {len(matched_samples)}")
    logger.info(f"  Clinical features: {clinical_features.shape[1]}")

    return results


if __name__ == "__main__":
    config = load_config()

    raw_dir = config["paths"]["raw_data"]
    kg_dir = config["paths"]["knowledge_graph"]

    expr_path = os.path.join(raw_dir, "tcga_brca_expression.tsv.gz")
    clinical_path = os.path.join(raw_dir, "tcga_brca_clinical.tsv")
    disgenet_path = os.path.join(kg_dir, "disgenet_gene_disease.tsv")

    # Check files exist
    if not os.path.exists(expr_path):
        alt = os.path.join(raw_dir, "tcga_brca_htseq_counts.tsv")
        if os.path.exists(alt):
            expr_path = alt
        else:
            raise FileNotFoundError(f"Run data_download.py first. Missing: {expr_path}")

    results = run_preprocessing(config, expr_path, clinical_path, disgenet_path)
