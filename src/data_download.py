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
import requests
import pandas as pd
import yaml
from io import BytesIO

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


def download_tcga_htseq_counts_gdc(output_dir: str, project: str = "TCGA-BRCA") -> str:
    """Download TCGA-BRCA HTSeq-Counts from GDC API.

    Queries for STAR-Counts gene expression files and downloads them.
    Returns path to the combined counts matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts_file = os.path.join(output_dir, "tcga_brca_htseq_counts.tsv")

    if os.path.exists(counts_file):
        logger.info(f"HTSeq counts file already exists: {counts_file}")
        return counts_file

    # Query for STAR-Counts files
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
        "size": 2000,
    }

    logger.info("Querying GDC API for TCGA-BRCA expression files...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    file_ids = [hit["file_id"] for hit in data["data"]["hits"]]
    logger.info(f"Found {len(file_ids)} expression files")

    if not file_ids:
        logger.warning("No files found via GDC API. Falling back to UCSC Xena.")
        return download_tcga_expression_xena(output_dir)

    # Build case_id -> submitter_id mapping
    case_map = {}
    for hit in data["data"]["hits"]:
        for case in hit.get("cases", []):
            case_map[hit["file_id"]] = case.get("submitter_id", case.get("case_id"))

    # Download files in a single tarball
    logger.info("Downloading expression files from GDC (this may take several minutes)...")
    payload = {"ids": file_ids}
    response = requests.post(GDC_DATA_ENDPOINT, json=payload, headers={"Content-Type": "application/json"}, timeout=600)
    response.raise_for_status()

    # Save the raw tarball
    tar_path = os.path.join(output_dir, "gdc_download.tar.gz")
    with open(tar_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded {len(file_ids)} files. Parsing counts...")

    # Parse the downloaded tar to extract counts
    import tarfile
    all_counts = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".counts") or member.name.endswith(".tsv"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read().decode("utf-8")
                # Determine the sample ID from the directory name (which is the file UUID)
                file_uuid = member.name.split("/")[0]
                sample_id = case_map.get(file_uuid, file_uuid)

                counts = {}
                for line in content.strip().split("\n"):
                    if line.startswith("__"):  # skip __no_feature, __ambiguous, etc.
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        gene_id = parts[0].split(".")[0]  # Remove Ensembl version
                        # For STAR-Counts, use unstranded (column index 1)
                        try:
                            count = int(float(parts[1]))
                        except (ValueError, IndexError):
                            continue
                        counts[gene_id] = count

                if counts:
                    all_counts[sample_id] = counts

    if all_counts:
        df = pd.DataFrame(all_counts).fillna(0).astype(int)
        df.index.name = "gene_id"
        df.to_csv(counts_file, sep="\t")
        logger.info(f"Saved combined counts matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    else:
        logger.warning("No counts parsed from GDC download. Falling back to UCSC Xena.")
        return download_tcga_expression_xena(output_dir)

    return counts_file


def download_tcga_clinical_gdc(output_dir: str, project: str = "TCGA-BRCA") -> str:
    """Download clinical data from GDC API."""
    os.makedirs(output_dir, exist_ok=True)
    clinical_file = os.path.join(output_dir, "tcga_brca_clinical.tsv")

    if os.path.exists(clinical_file):
        logger.info(f"Clinical file already exists: {clinical_file}")
        return clinical_file

    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.days_to_last_follow_up",
        "demographic.gender",
        "demographic.race",
        "diagnoses.age_at_diagnosis",
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
            record["tumor_stage"] = diag.get("tumor_stage")
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
    source: str = "ALL",
) -> list[dict]:
    """Query a single disease from the DisGeNET GDA summary endpoint with pagination."""
    records = []
    page = 0

    while page < 100:
        params = {
            "disease": disease_id,
            "source": source,
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
            if resp.status_code in (401, 403, 429):
                logger.warning(f"DisGeNET API returned {resp.status_code} for {disease_id}")
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

    # 1. TCGA expression data (try GDC first, fallback to Xena)
    logger.info("=" * 60)
    logger.info("Stage 1: Downloading TCGA-BRCA expression data")
    logger.info("=" * 60)
    try:
        file_paths["expression"] = download_tcga_htseq_counts_gdc(raw_dir, config["data"]["tcga_project"])
    except Exception as e:
        logger.warning(f"GDC download failed ({e}), falling back to UCSC Xena")
        file_paths["expression"] = download_tcga_expression_xena(raw_dir)

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
