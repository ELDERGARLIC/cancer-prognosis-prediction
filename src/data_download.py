"""
Stage 1.1: TCGA-BRCA Data Acquisition

Downloads gene expression (HTSeq-Counts) and clinical data from TCGA-BRCA
via the GDC API. Also downloads STRING PPI and DisGeNET data for knowledge
graph construction.

References:
    - GDC API: https://api.gdc.cancer.gov
    - UCSC Xena as fallback: https://xenabrowser.net/
"""

import os
import json
import gzip
import logging
import time
import tarfile
import requests
import pandas as pd
import yaml
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
XENA_HUB = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com"
DISGENET_API_BASE = "https://api.disgenet.com/api/v1"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_tcga_expression_xena(output_dir: str) -> str:
    """Download TCGA-BRCA gene expression from UCSC Xena (recommended fallback).

    Downloads the TCGA-BRCA HiSeqV2 (log2(norm_count+1)) expression matrix.
    This is more reliable than the GDC API for bulk downloads.
    """
    os.makedirs(output_dir, exist_ok=True)
    expr_file = os.path.join(output_dir, "tcga_brca_expression.tsv.gz")

    if os.path.exists(expr_file):
        logger.info(f"Expression file already exists: {expr_file}")
        return expr_file

    # TCGA-BRCA gene expression (HiSeqV2, log2(norm_count+1))
    url = f"{XENA_HUB}/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"
    logger.info("Downloading TCGA-BRCA expression from UCSC Xena...")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(expr_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Saved expression data to {expr_file}")
    return expr_file


def _gdc_session() -> requests.Session:
    """Build a requests session with retry/backoff for GDC's flaky bulk endpoint.

    GDC's ``/data`` endpoint frequently returns 500/502/503/504 or drops the
    connection when asked for large batches. We use urllib3's Retry with
    exponential backoff so transient failures self-heal before being raised.
    """
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2.0,  # sleeps 0, 2, 4, 8, 16 s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": "cancer-prognosis-prediction/1.0 (+github)",
        "Accept": "application/json, application/x-tar",
    })
    return s


def _parse_gdc_counts_tar(tar_bytes: bytes, case_map: dict) -> dict:
    """Extract per-sample counts from a GDC STAR-Counts tarball (in-memory).

    GDC STAR-Counts TSV columns are:
        0: gene_id  (ENSG.version)
        1: gene_name  (HGNC symbol)
        2: gene_type
        3: unstranded
        4: stranded_first
        5: stranded_second
        6: tpm_unstranded
        7: fpkm_unstranded
        8: fpkm_uq_unstranded

    We index by ``gene_name`` (symbol) because the rest of the pipeline
    (BioBERT prompts, STRING PPI, KEGG, DisGeNET) keys on symbols.
    Unstranded (column 3) is the canonical count field.
    """
    all_counts = {}
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:*") as tar:
        for member in tar.getmembers():
            if not (member.name.endswith(".counts") or member.name.endswith(".tsv")):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf-8", errors="replace")
            file_uuid = member.name.split("/")[0]
            sample_id = case_map.get(file_uuid, file_uuid)

            counts = {}
            for line in content.strip().split("\n"):
                if not line or line.startswith("#") or line.startswith("gene_id"):
                    continue
                if line.startswith("N_") or line.startswith("__"):  # STAR stats
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                gene_symbol = parts[1].strip()
                if not gene_symbol or gene_symbol == "-":
                    # Some ENSG entries lack a symbol; fall back to Ensembl ID
                    gene_symbol = parts[0].split(".")[0]
                try:
                    count = int(float(parts[3]))  # unstranded count
                except (ValueError, IndexError):
                    continue
                # If a symbol appears twice in the same file (rare; multiple
                # ENSG IDs map to one symbol), take the max so we retain signal.
                prev = counts.get(gene_symbol)
                counts[gene_symbol] = count if prev is None else max(prev, count)

            if counts:
                all_counts[sample_id] = counts
    return all_counts


def _download_gdc_batch(
    session: requests.Session,
    file_ids: list,
    batch_idx: int,
    n_batches: int,
    max_attempts: int = 4,
) -> bytes:
    """POST one batch of file_ids to /data and return the tarball bytes.

    Raises the last exception if all attempts fail. Implements per-attempt
    backoff on top of the session-level Retry because session retries do not
    cover chunked-transfer ``ConnectionResetError``s mid-stream.
    """
    payload = {"ids": file_ids}
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.post(
                GDC_DATA_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(30, 900),  # (connect, read)
                stream=True,
            )
            resp.raise_for_status()
            # Stream to a bytearray so a reset mid-transfer raises here, not later
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    buf.extend(chunk)
            return bytes(buf)
        except (requests.exceptions.RequestException, ConnectionResetError) as e:
            last_exc = e
            if attempt == max_attempts:
                break
            sleep_s = 3 * (2 ** (attempt - 1))  # 3, 6, 12, 24
            logger.warning(
                f"  Batch {batch_idx + 1}/{n_batches} attempt {attempt}/{max_attempts} "
                f"failed ({type(e).__name__}: {e}). Retrying in {sleep_s}s..."
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"GDC batch {batch_idx + 1}/{n_batches} failed after {max_attempts} attempts: {last_exc}"
    )


def download_tcga_htseq_counts_gdc(
    output_dir: str,
    project: str = "TCGA-BRCA",
    batch_size: int = 50,
) -> str:
    """Download TCGA-BRCA STAR-Counts gene expression via the GDC API.

    The GDC ``/data`` endpoint frequently fails on very large single-payload
    requests (``ConnectionResetError`` / 500). We therefore:

    * split the file list into ``batch_size`` chunks,
    * stream each POST with a retry-backoff session,
    * parse each tarball in memory and discard it,
    * checkpoint parsed counts to ``tcga_brca_htseq_counts.partial.json`` so a
      crashed run can resume.

    Returns the path to the combined counts matrix TSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts_file = os.path.join(output_dir, "tcga_brca_htseq_counts.tsv")
    checkpoint_file = os.path.join(output_dir, "tcga_brca_htseq_counts.partial.json")

    if os.path.exists(counts_file):
        logger.info(f"HTSeq counts file already exists: {counts_file}")
        return counts_file

    session = _gdc_session()

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id",
        "format": "JSON",
        "size": 5000,
    }

    logger.info("Querying GDC API for TCGA-BRCA expression files...")
    resp = session.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hits = data["data"]["hits"]
    file_ids = [hit["file_id"] for hit in hits]
    logger.info(f"Found {len(file_ids)} expression files")

    if not file_ids:
        raise RuntimeError("GDC returned zero expression files for the query.")

    case_map = {}
    for hit in hits:
        for case in hit.get("cases", []):
            case_map[hit["file_id"]] = case.get("submitter_id", case.get("case_id"))

    all_counts = {}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                all_counts = json.load(f)
            logger.info(f"Resuming from checkpoint: {len(all_counts)} samples already parsed")
        except Exception:
            all_counts = {}

    already_done = set(all_counts.keys())
    pending_ids = [fid for fid in file_ids if case_map.get(fid, fid) not in already_done]
    if len(pending_ids) < len(file_ids):
        logger.info(f"Skipping {len(file_ids) - len(pending_ids)} files already parsed")

    n_batches = (len(pending_ids) + batch_size - 1) // batch_size
    logger.info(
        f"Downloading expression files from GDC in {n_batches} batches of up to {batch_size} "
        f"(retries with backoff on transient failures)..."
    )

    for b in range(n_batches):
        batch = pending_ids[b * batch_size : (b + 1) * batch_size]
        t0 = time.time()
        tar_bytes = _download_gdc_batch(session, batch, b, n_batches)
        parsed = _parse_gdc_counts_tar(tar_bytes, case_map)
        all_counts.update(parsed)
        dt = time.time() - t0
        logger.info(
            f"  Batch {b + 1}/{n_batches}: {len(parsed)} samples parsed "
            f"({len(all_counts)}/{len(file_ids)} total) in {dt:.1f}s"
        )
        # Checkpoint every 5 batches so a mid-run crash can resume
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            with open(checkpoint_file, "w") as f:
                json.dump(all_counts, f)

    if not all_counts:
        raise RuntimeError("No counts parsed from any GDC batch.")

    df = pd.DataFrame(all_counts).fillna(0).astype(int)
    df.index.name = "gene_id"
    df.to_csv(counts_file, sep="\t")
    logger.info(f"Saved combined counts matrix: {df.shape[0]} genes x {df.shape[1]} samples")

    # Cleanup checkpoint on success
    try:
        os.remove(checkpoint_file)
    except OSError:
        pass

    return counts_file


def download_tcga_clinical_gdc(output_dir: str, project: str = "TCGA-BRCA") -> str:
    """Download clinical data from GDC API."""
    os.makedirs(output_dir, exist_ok=True)
    clinical_file = os.path.join(output_dir, "tcga_brca_clinical.tsv")

    if os.path.exists(clinical_file):
        logger.info(f"Clinical file already exists: {clinical_file}")
        return clinical_file

    # NOTE: `diagnoses.tumor_stage` was deprecated by GDC in 2021 and now returns
    # all nulls. The modern schema exposes AJCC stage under multiple fields --
    # request all of them so downstream parsing can pick whichever is populated.
    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.days_to_last_follow_up",
        "demographic.gender",
        "demographic.race",
        "diagnoses.age_at_diagnosis",
        # Modern stage fields (GDC schema post-2021)
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_clinical_stage",
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_m",
        # Legacy field, kept for older extracts that still populate it
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.morphology",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_death",
        "exposures.bmi",
    ]

    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": [project]},
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 2000,
    }

    logger.info("Downloading TCGA-BRCA clinical data from GDC API...")
    response = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    records = []
    for case in data["data"]["hits"]:
        record = {"case_id": case.get("submitter_id", case.get("id"))}

        demo = case.get("demographic", {})
        record["vital_status"] = demo.get("vital_status")
        record["days_to_death"] = demo.get("days_to_death")
        record["days_to_last_follow_up"] = demo.get("days_to_last_follow_up")
        record["gender"] = demo.get("gender")
        record["race"] = demo.get("race")

        diagnoses = case.get("diagnoses", [{}])
        if diagnoses:
            diag = diagnoses[0]
            record["age_at_diagnosis"] = diag.get("age_at_diagnosis")
            # Prefer AJCC pathologic stage, fall back through clinical stage,
            # then T-stage, then the deprecated tumor_stage field.
            record["ajcc_pathologic_stage"] = diag.get("ajcc_pathologic_stage")
            record["ajcc_clinical_stage"] = diag.get("ajcc_clinical_stage")
            record["ajcc_pathologic_t"] = diag.get("ajcc_pathologic_t")
            record["ajcc_pathologic_n"] = diag.get("ajcc_pathologic_n")
            record["ajcc_pathologic_m"] = diag.get("ajcc_pathologic_m")
            # Unified tumor_stage: first non-null of the above in preference order
            stage_candidates = [
                diag.get("ajcc_pathologic_stage"),
                diag.get("ajcc_clinical_stage"),
                diag.get("tumor_stage"),
                diag.get("ajcc_pathologic_t"),  # T-stage as last-resort proxy
            ]
            record["tumor_stage"] = next(
                (s for s in stage_candidates if s and str(s).lower() not in ("not reported", "unknown")),
                None,
            )
            record["primary_diagnosis"] = diag.get("primary_diagnosis")
            # Use diagnosis-level days if demographic-level missing
            if record["days_to_death"] is None:
                record["days_to_death"] = diag.get("days_to_death")
            if record["days_to_last_follow_up"] is None:
                record["days_to_last_follow_up"] = diag.get("days_to_last_follow_up")

        records.append(record)

    df = pd.DataFrame(records)

    # Compute OS.time and event status
    df["OS.time"] = df.apply(
        lambda r: r["days_to_death"] if pd.notna(r["days_to_death"]) else r["days_to_last_follow_up"],
        axis=1,
    )
    df["OS"] = (df["vital_status"] == "Dead").astype(int)

    df.to_csv(clinical_file, sep="\t", index=False)
    logger.info(f"Saved clinical data: {len(df)} patients to {clinical_file}")
    return clinical_file


def download_tcga_clinical_xena(output_dir: str) -> str:
    """Download TCGA-BRCA clinical data from UCSC Xena (fallback)."""
    os.makedirs(output_dir, exist_ok=True)
    clinical_file = os.path.join(output_dir, "tcga_brca_clinical.tsv")

    if os.path.exists(clinical_file):
        logger.info(f"Clinical file already exists: {clinical_file}")
        return clinical_file

    url = f"{XENA_HUB}/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix.gz"
    logger.info("Downloading clinical data from UCSC Xena...")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    raw = gzip.decompress(response.content)
    df = pd.read_csv(BytesIO(raw), sep="\t")

    # Standardize column names
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "overall_survival" in col_lower or col == "OS":
            rename_map[col] = "OS"
        elif "os.time" in col_lower or col == "OS.time" or "_time" in col_lower:
            rename_map[col] = "OS.time"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure OS and OS.time columns exist
    if "OS" not in df.columns:
        if "_EVENT" in str(df.columns):
            event_col = [c for c in df.columns if "EVENT" in c.upper()][0]
            df["OS"] = df[event_col]
        elif "vital_status" in df.columns:
            df["OS"] = (df["vital_status"].str.lower() == "dead").astype(int)

    df.to_csv(clinical_file, sep="\t", index=False)
    logger.info(f"Saved clinical data: {df.shape} to {clinical_file}")
    return clinical_file


def download_string_ppi(output_dir: str, species: int = 9606) -> str:
    """Download STRING protein-protein interaction network for human.

    Args:
        output_dir: Directory to save the file.
        species: NCBI taxonomy ID (9606 = human).

    Returns:
        Path to the downloaded STRING file.
    """
    os.makedirs(output_dir, exist_ok=True)
    string_file = os.path.join(output_dir, "string_ppi.tsv")

    if os.path.exists(string_file):
        logger.info(f"STRING PPI file already exists: {string_file}")
        return string_file

    url = f"https://stringdb-downloads.org/download/protein.links.v12.0/{species}.protein.links.v12.0.txt.gz"
    logger.info(f"Downloading STRING PPI network for species {species}...")

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    raw = gzip.decompress(response.content)
    df = pd.read_csv(BytesIO(raw), sep=" ")

    # Remove species prefix from protein IDs (e.g., "9606.ENSP00000000233" -> "ENSP00000000233")
    df["protein1"] = df["protein1"].str.replace(f"{species}.", "", regex=False)
    df["protein2"] = df["protein2"].str.replace(f"{species}.", "", regex=False)

    df.to_csv(string_file, sep="\t", index=False)
    logger.info(f"Saved STRING PPI: {len(df)} interactions to {string_file}")
    return string_file


def download_string_id_mapping(output_dir: str, species: int = 9606) -> str:
    """Download STRING protein ID to gene name mapping."""
    os.makedirs(output_dir, exist_ok=True)
    mapping_file = os.path.join(output_dir, "string_id_mapping.tsv")

    if os.path.exists(mapping_file):
        logger.info(f"STRING ID mapping already exists: {mapping_file}")
        return mapping_file

    url = f"https://stringdb-downloads.org/download/protein.info.v12.0/{species}.protein.info.v12.0.txt.gz"
    logger.info("Downloading STRING ID mapping...")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    raw = gzip.decompress(response.content)
    df = pd.read_csv(BytesIO(raw), sep="\t")

    # Clean protein IDs
    if "#string_protein_id" in df.columns:
        df = df.rename(columns={"#string_protein_id": "string_protein_id"})
    df["string_protein_id"] = df["string_protein_id"].str.replace(f"{species}.", "", regex=False)

    # Keep only essential columns
    keep_cols = [c for c in ["string_protein_id", "preferred_name", "protein_size", "annotation"] if c in df.columns]
    df = df[keep_cols]

    df.to_csv(mapping_file, sep="\t", index=False)
    logger.info(f"Saved ID mapping: {len(df)} proteins to {mapping_file}")
    return mapping_file


def _query_disgenet_gda(
    headers: dict,
    disease_id: str,
    source: str = "CURATED",
) -> list[dict]:
    """Query a single disease from the DisGeNET GDA summary endpoint with pagination.

    Default source is ``CURATED`` because DisGeNET ACADEMIC profiles (the free
    tier most users have) are forbidden from querying ``source=ALL`` and the API
    returns HTTP 403 with "Academic accounts may only access CURATED sources."
    If a caller explicitly requests ``ALL`` and hits 403, we auto-downgrade to
    ``CURATED`` for that disease so the run can proceed.
    """
    records = []
    page = 0
    active_source = source

    while page < 100:
        params = {
            "disease": disease_id,
            "source": active_source,
            "type": "disease",
            "page_number": page,
        }
        try:
            resp = requests.get(
                f"{DISGENET_API_BASE}/gda/summary",
                params=params,
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 403 and active_source != "CURATED":
                logger.warning(
                    f"DisGeNET 403 for {disease_id} with source={active_source} "
                    "(ACADEMIC profiles can only query CURATED). Retrying with CURATED."
                )
                active_source = "CURATED"
                continue
            if resp.status_code in (401, 403, 429):
                detail = ""
                try:
                    j = resp.json()
                    detail = f" — {j.get('payload', {}).get('details') or j.get('status')}"
                except Exception:
                    pass
                logger.warning(
                    f"DisGeNET API returned {resp.status_code} for {disease_id}{detail}"
                )
                break
            resp.raise_for_status()
            data = resp.json()
            payload = data.get("payload", data if isinstance(data, list) else [])
            if not payload:
                break

            for item in payload:
                semantic_types = item.get("diseaseClasses_UMLS_ST", [])
                raw_dtype = item.get("diseaseType", "disease")
                dtype = raw_dtype.strip("[]") if isinstance(raw_dtype, str) else "disease"
                records.append({
                    "geneSymbol": item.get("symbolOfGene", ""),
                    "geneNcbiID": item.get("geneNcbiID", ""),
                    "diseaseName": item.get("diseaseName", ""),
                    "diseaseId": item.get("diseaseUMLSCUI", ""),
                    "diseaseType": dtype,
                    "diseaseSemanticType": "; ".join(semantic_types) if semantic_types else "Neoplastic Process",
                    "score": item.get("score", 0),
                    "ei": item.get("ei", 0),
                    "el": item.get("el", ""),
                    "numPMIDs": item.get("numPMIDs", 0),
                    "source": "DisGeNET_API",
                })

            if len(payload) < 100:
                break
            page += 1

        except requests.exceptions.RequestException as e:
            logger.warning(f"DisGeNET API request failed for {disease_id} page {page}: {e}")
            break

    return records


# Related breast cancer UMLS CUIs for broader gene coverage
_BREAST_CANCER_CUIS = [
    "UMLS_C0006142",  # Malignant neoplasm of breast
    "UMLS_C0678222",  # Breast Carcinoma
    "UMLS_C1458155",  # Invasive Breast Carcinoma
    "UMLS_C0279605",  # In-situ Breast Carcinoma
    "UMLS_C0007104",  # Female Breast Carcinoma
]


def download_disgenet(
    output_dir: str,
    api_key: str = None,
    disease_id: str = "UMLS_C0006142",
) -> str:
    """Download gene-disease associations from the DisGeNET REST API.

    Queries multiple breast cancer-related CUIs via the GDA summary endpoint
    to maximize gene coverage (TRIAL accounts return max 30 results per query).
    Supplements API results with a curated literature-based gene list.

    Args:
        output_dir: Directory to save the output file.
        api_key: DisGeNET API key. Falls back to DISGENET_API_KEY env var.
        disease_id: Primary disease identifier (default: breast cancer C0006142).
    """
    os.makedirs(output_dir, exist_ok=True)
    disgenet_file = os.path.join(output_dir, "disgenet_gene_disease.tsv")

    if os.path.exists(disgenet_file):
        logger.info(f"DisGeNET file already exists: {disgenet_file}")
        return disgenet_file

    api_key = api_key or os.environ.get("DISGENET_API_KEY")
    if not api_key:
        logger.warning("No DisGeNET API key provided. Set DISGENET_API_KEY env var.")
        return _disgenet_fallback(disgenet_file)

    logger.info("Downloading gene-disease associations from DisGeNET REST API...")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    # Query multiple related CUIs for broader coverage
    disease_ids = [disease_id] if disease_id in _BREAST_CANCER_CUIS else [disease_id] + _BREAST_CANCER_CUIS
    if disease_id not in disease_ids:
        disease_ids = [disease_id] + _BREAST_CANCER_CUIS
    else:
        disease_ids = _BREAST_CANCER_CUIS

    all_records = []
    seen_genes = set()

    for did in disease_ids:
        records = _query_disgenet_gda(headers, did)
        new_count = 0
        for r in records:
            gene = r["geneSymbol"]
            if gene not in seen_genes:
                seen_genes.add(gene)
                new_count += 1
            all_records.append(r)
        logger.info(f"  {did}: {len(records)} associations ({new_count} new genes)")
        if not records:
            continue

    # Merge: keep highest score per gene (across all disease queries)
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.sort_values("score", ascending=False).drop_duplicates(subset="geneSymbol", keep="first")
        api_genes = set(df["geneSymbol"].str.upper())
        logger.info(f"DisGeNET API: {len(df)} unique gene associations")

        # Supplement with curated fallback genes not already from the API
        fallback_genes = _get_fallback_genes()
        supplemental = [g for g in fallback_genes if g.upper() not in api_genes]
        if supplemental:
            supp_df = pd.DataFrame({
                "geneSymbol": supplemental,
                "geneNcbiID": "",
                "diseaseName": "Breast Carcinoma",
                "diseaseId": "C0006142",
                "diseaseType": "disease",
                "diseaseSemanticType": "Neoplastic Process",
                "score": 0.5,
                "ei": 0,
                "el": "",
                "numPMIDs": 0,
                "source": "curated_supplement",
            })
            df = pd.concat([df, supp_df], ignore_index=True)
            logger.info(f"  Added {len(supplemental)} supplemental genes from curated list")

        df.to_csv(disgenet_file, sep="\t", index=False)
        logger.info(f"Saved DisGeNET data: {len(df)} total associations ({df['geneSymbol'].nunique()} unique genes)")
        return disgenet_file

    logger.warning("No data retrieved from DisGeNET API. Using fallback gene list.")
    return _disgenet_fallback(disgenet_file)


def _get_fallback_genes() -> list[str]:
    """Curated breast cancer gene list from literature."""
    return [
        "BRCA1", "BRCA2", "TP53", "ERBB2", "ESR1", "PGR", "PTEN", "PIK3CA",
        "AKT1", "CDH1", "GATA3", "MAP3K1", "RB1", "CDKN2A", "MYC", "CCND1",
        "FOXA1", "KMT2C", "TBX3", "RUNX1", "CBFB", "SF3B1", "NCOR1", "AFF2",
        "NF1", "ARID1A", "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D", "BARD1",
        "BRIP1", "CDK12", "FGFR1", "FGFR2", "NOTCH1", "NOTCH2", "MDM2",
        "VEGFA", "EGFR", "MKI67", "BCL2", "BAX", "CASP3", "CASP8", "CASP9",
        "BIRC5", "AURKA", "AURKB", "TOP2A", "MMP9", "MMP2", "CTNNB1",
        "WNT1", "APC", "AXIN1", "GSK3B", "LEF1", "TCF7L2", "KRAS", "BRAF",
        "MAP2K1", "MAPK1", "JAK2", "STAT3", "IL6", "TNF", "TGFB1", "SMAD4",
    ]


def _disgenet_fallback(disgenet_file: str) -> str:
    """Create a fallback breast cancer gene list from literature."""
    bc_genes = _get_fallback_genes()
    fallback_df = pd.DataFrame({
        "geneSymbol": bc_genes,
        "diseaseName": "Breast Carcinoma",
        "diseaseId": "C0006142",
        "diseaseType": "disease",
        "diseaseSemanticType": "Neoplastic Process",
        "score": 0.8,
        "source": "curated_fallback",
    })
    fallback_df.to_csv(disgenet_file, sep="\t", index=False)
    logger.info(f"Created fallback gene list with {len(bc_genes)} breast cancer genes")
    return disgenet_file


def download_ncbi_gene_summaries(gene_list: list, output_dir: str) -> str:
    """Download gene summary texts from NCBI Entrez for LLM embedding.

    Args:
        gene_list: List of gene symbols.
        output_dir: Directory to save the summaries.

    Returns:
        Path to the gene summaries JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    summaries_file = os.path.join(output_dir, "gene_summaries.json")

    if os.path.exists(summaries_file):
        logger.info(f"Gene summaries file already exists: {summaries_file}")
        with open(summaries_file) as f:
            existing = json.load(f)
        # Check if we need to download more
        missing = [g for g in gene_list if g not in existing]
        if not missing:
            return summaries_file
        logger.info(f"Downloading summaries for {len(missing)} additional genes...")
        summaries = existing
    else:
        summaries = {}
        missing = gene_list

    # Batch query NCBI Entrez
    batch_size = 100
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        query = " OR ".join([f"{g}[Gene Name]" for g in batch])

        # Search for gene IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "gene",
            "term": f'({query}) AND "Homo sapiens"[Organism]',
            "retmax": len(batch),
            "retmode": "json",
        }

        try:
            resp = requests.get(search_url, params=search_params, timeout=30)
            resp.raise_for_status()
            search_data = resp.json()
            gene_ids = search_data.get("esearchresult", {}).get("idlist", [])

            if not gene_ids:
                continue

            # Fetch gene summaries
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            fetch_params = {
                "db": "gene",
                "id": ",".join(gene_ids),
                "retmode": "json",
            }
            resp = requests.get(fetch_url, params=fetch_params, timeout=30)
            resp.raise_for_status()
            fetch_data = resp.json()

            for gid in gene_ids:
                gene_info = fetch_data.get("result", {}).get(str(gid), {})
                gene_name = gene_info.get("name", "")
                description = gene_info.get("description", "")
                summary = gene_info.get("summary", "")
                nomenclature_name = gene_info.get("nomenclaturename", "")

                if gene_name:
                    text = f"{gene_name}: {nomenclature_name}. {description}. {summary}"
                    summaries[gene_name.upper()] = text.strip()

        except Exception as e:
            logger.warning(f"Error fetching batch {i // batch_size}: {e}")
            continue

    # Also generate fallback summaries for genes without NCBI entries
    for gene in gene_list:
        if gene.upper() not in summaries:
            summaries[gene.upper()] = f"{gene}: A human gene associated with cellular processes."

    with open(summaries_file, "w") as f:
        json.dump(summaries, f, indent=2)

    logger.info(f"Saved summaries for {len(summaries)} genes to {summaries_file}")
    return summaries_file


def run_data_download(config: dict) -> dict:
    """Run the complete data download pipeline.

    Returns:
        Dictionary of file paths for each downloaded dataset.
    """
    paths = config.get("paths", {})
    raw_dir = paths.get("raw_data", "data/raw")
    kg_dir = paths.get("knowledge_graph", "data/knowledge_graph")

    file_paths = {}

    # 1. TCGA expression data -- GDC only (batched + retried). No silent Xena
    # fallback because Xena uses a different sample-ID convention and a
    # log2(norm_count+1) encoding that breaks downstream clinical joins and
    # normalization heuristics (see prior run analysis).
    logger.info("=" * 60)
    logger.info("Stage 1: Downloading TCGA-BRCA expression data")
    logger.info("=" * 60)
    file_paths["expression"] = download_tcga_htseq_counts_gdc(
        raw_dir, config["data"]["tcga_project"]
    )

    # 2. TCGA clinical data
    logger.info("=" * 60)
    logger.info("Stage 2: Downloading TCGA-BRCA clinical data")
    logger.info("=" * 60)
    try:
        file_paths["clinical"] = download_tcga_clinical_gdc(raw_dir, config["data"]["tcga_project"])
    except Exception as e:
        logger.warning(f"GDC clinical download failed ({e}), falling back to UCSC Xena")
        file_paths["clinical"] = download_tcga_clinical_xena(raw_dir)

    # 3. STRING PPI network
    logger.info("=" * 60)
    logger.info("Stage 3: Downloading STRING PPI network")
    logger.info("=" * 60)
    file_paths["string_ppi"] = download_string_ppi(kg_dir)
    file_paths["string_mapping"] = download_string_id_mapping(kg_dir)

    # 4. DisGeNET
    logger.info("=" * 60)
    logger.info("Stage 4: Downloading DisGeNET associations")
    logger.info("=" * 60)
    disease_cui = config.get("data", {}).get("disease_cui", "C0006142")
    disgenet_api_key = os.environ.get("DISGENET_API_KEY")
    file_paths["disgenet"] = download_disgenet(
        kg_dir,
        api_key=disgenet_api_key,
        disease_id=f"UMLS_{disease_cui}",
    )

    logger.info("=" * 60)
    logger.info("All data downloads complete!")
    logger.info("=" * 60)

    return file_paths


if __name__ == "__main__":
    config = load_config()
    paths = run_data_download(config)
    print("\nDownloaded files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
