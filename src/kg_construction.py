"""
Stage 2.1-2.2: Knowledge Graph Construction

Builds a heterogeneous biological knowledge graph from:
    - STRING PPI network (gene-gene interactions)
    - DisGeNET (gene-disease associations)
    - KEGG/Reactome pathways (gene-pathway memberships)

References:
    - BioKG heterogeneous graph: Vaida et al., 2025
    - DisGeNET filtering: Qumsiyeh et al., 2022
    - Graph sparsity: Ling et al., 2022; Chowa et al., 2023
    - STRING confidence threshold > 700: Ling et al., 2022
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
import yaml
import torch
import requests
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_string_ppi(
    string_path: str,
    mapping_path: str,
    gene_list: list,
    confidence_threshold: int = 700,
) -> tuple:
    """Load STRING PPI edges and filter for selected genes.

    Reference: Ling et al., 2022 — only keep edges with combined_score > 700.

    Returns:
        (edge_index, edge_weights, gene_to_idx mapping)
    """
    logger.info("Loading STRING PPI network...")
    ppi = pd.read_csv(string_path, sep="\t")

    # Load protein-to-gene-name mapping
    mapping = pd.read_csv(mapping_path, sep="\t")
    protein_to_gene = dict(zip(mapping["string_protein_id"], mapping["preferred_name"]))

    # Map protein IDs to gene names
    ppi["gene1"] = ppi["protein1"].map(protein_to_gene)
    ppi["gene2"] = ppi["protein2"].map(protein_to_gene)

    # Filter by confidence threshold
    ppi = ppi[ppi["combined_score"] >= confidence_threshold]
    logger.info(f"STRING edges after confidence filter (>={confidence_threshold}): {len(ppi)}")

    # Filter for genes in our selected set
    gene_set = set(g.upper() for g in gene_list)
    gene_name_map = {g.upper(): g for g in gene_list}

    ppi["gene1_upper"] = ppi["gene1"].str.upper()
    ppi["gene2_upper"] = ppi["gene2"].str.upper()

    ppi_filtered = ppi[ppi["gene1_upper"].isin(gene_set) & ppi["gene2_upper"].isin(gene_set)]

    logger.info(f"STRING edges after gene filter: {len(ppi_filtered)} (from {len(gene_list)} genes)")

    # Build gene-to-index mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Build edge lists (bidirectional)
    src, dst, weights = [], [], []
    for _, row in ppi_filtered.iterrows():
        g1 = gene_name_map.get(row["gene1_upper"])
        g2 = gene_name_map.get(row["gene2_upper"])
        if g1 and g2 and g1 in gene_to_idx and g2 in gene_to_idx:
            idx1 = gene_to_idx[g1]
            idx2 = gene_to_idx[g2]
            # Add bidirectional edges
            src.extend([idx1, idx2])
            dst.extend([idx2, idx1])
            w = row["combined_score"] / 1000.0  # Normalize to [0, 1]
            weights.extend([w, w])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weights = torch.tensor(weights, dtype=torch.float)

    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_weights = edge_weights[mask]

    # Remove duplicate edges
    if edge_index.size(1) > 0:
        edge_set = set()
        unique_mask = []
        for i in range(edge_index.size(1)):
            edge = (edge_index[0, i].item(), edge_index[1, i].item())
            if edge not in edge_set:
                edge_set.add(edge)
                unique_mask.append(True)
            else:
                unique_mask.append(False)
        unique_mask = torch.tensor(unique_mask)
        edge_index = edge_index[:, unique_mask]
        edge_weights = edge_weights[unique_mask]

    logger.info(f"Final gene-gene edges: {edge_index.size(1)}")
    return edge_index, edge_weights, gene_to_idx


def build_coexpression_edges(
    expr_df: pd.DataFrame,
    gene_list: list,
    gene_to_idx: dict,
    threshold: float = 0.7,
    top_k: int = 8,
) -> tuple:
    """Build gene-gene co-expression edges from the expression matrix.

    Motivation: the STRING PPI network leaves ~80% of LASSO-selected genes
    isolated (first-run analysis), starving the GAT of neighborhood signal.
    Co-expression edges densify the graph using the same data the model
    already sees, connecting genes that move together across patients.

    Args:
        expr_df: Expression matrix (genes x samples), already normalized.
        gene_list: Gene ordering used by the dataset.
        gene_to_idx: Mapping from gene symbol to node index.
        threshold: Minimum |Pearson correlation| to consider.
        top_k: Keep at most this many edges per gene to avoid dense hubs.

    Returns:
        (edge_index, edge_weights) with bidirectional edges, no self-loops,
        no duplicates.
    """
    logger.info(
        f"Building co-expression edges (|r|>={threshold}, top_k={top_k})..."
    )

    # Subset to selected genes and align order
    available = [g for g in gene_list if g in expr_df.index]
    if len(available) < 5:
        logger.warning("Not enough genes for co-expression; skipping.")
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float)

    expr_sub = expr_df.loc[available]
    # Expression matrix is genes x samples; correlate across samples
    # by treating each row as a variable.
    values = expr_sub.values.astype(np.float32)

    # Standardize row-wise in case normalization drift
    mu = values.mean(axis=1, keepdims=True)
    sd = values.std(axis=1, keepdims=True) + 1e-8
    z = (values - mu) / sd
    n_samples = z.shape[1]
    corr = (z @ z.T) / max(n_samples - 1, 1)
    np.fill_diagonal(corr, 0.0)

    src, dst, weights = [], [], []
    abs_corr = np.abs(corr)
    # For each gene, pick its top_k strongest correlates above threshold
    for i, g1 in enumerate(available):
        if g1 not in gene_to_idx:
            continue
        # Argsort descending by |corr|
        neighbor_order = np.argsort(-abs_corr[i])
        added = 0
        for j in neighbor_order:
            if added >= top_k:
                break
            if abs_corr[i, j] < threshold:
                break
            g2 = available[j]
            if g2 not in gene_to_idx:
                continue
            idx1 = gene_to_idx[g1]
            idx2 = gene_to_idx[g2]
            if idx1 == idx2:
                continue
            # Add bidirectional; edge weight = |correlation| in [0,1]
            w = float(abs_corr[i, j])
            src.extend([idx1, idx2])
            dst.extend([idx2, idx1])
            weights.extend([w, w])
            added += 1

    if not src:
        logger.info("No co-expression edges found above threshold.")
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weights = torch.tensor(weights, dtype=torch.float)

    # Deduplicate (gene pair may appear multiple times from both sides' top_k)
    edge_set = {}
    for i in range(edge_index.size(1)):
        key = (edge_index[0, i].item(), edge_index[1, i].item())
        w = edge_weights[i].item()
        # Keep max weight across duplicates
        if key not in edge_set or edge_set[key] < w:
            edge_set[key] = w

    if edge_set:
        keys = list(edge_set.keys())
        values_list = list(edge_set.values())
        edge_index = torch.tensor(list(zip(*keys)), dtype=torch.long)
        edge_weights = torch.tensor(values_list, dtype=torch.float)

    logger.info(f"Co-expression edges: {edge_index.size(1)}")
    return edge_index, edge_weights


def merge_edge_sets(
    edges_a: torch.Tensor,
    weights_a: torch.Tensor,
    edges_b: torch.Tensor,
    weights_b: torch.Tensor,
) -> tuple:
    """Union of two edge sets. Duplicate edges keep the max weight."""
    if edges_a.size(1) == 0:
        return edges_b, weights_b
    if edges_b.size(1) == 0:
        return edges_a, weights_a

    edge_set = {}
    for i in range(edges_a.size(1)):
        key = (edges_a[0, i].item(), edges_a[1, i].item())
        edge_set[key] = max(edge_set.get(key, 0.0), weights_a[i].item())
    for i in range(edges_b.size(1)):
        key = (edges_b[0, i].item(), edges_b[1, i].item())
        edge_set[key] = max(edge_set.get(key, 0.0), weights_b[i].item())

    keys = list(edge_set.keys())
    idx = torch.tensor(list(zip(*keys)), dtype=torch.long)
    w = torch.tensor([edge_set[k] for k in keys], dtype=torch.float)
    return idx, w


def load_disgenet_edges(
    disgenet_path: str,
    gene_list: list,
    gene_to_idx: dict,
    disease_semantic_type: str = "Neoplastic Process",
    disease_cuis: list = None,
    min_score: float = 0.05,
) -> tuple:
    """Load DisGeNET gene-disease associations and build edges.

    Reference: Qumsiyeh et al., 2022.

    Phase 1 fixes vs the original:
      * Accept a list of UMLS CUIs (config.data.disease_cuis) to filter the
        table. Original behavior only filtered by semantic type, which is too
        broad ("Neoplastic Process" covers every cancer in DisGeNET).
      * Lower default score threshold to 0.05 (was implicitly 0.3 when we
        pre-filtered the TSV, now configurable). DisGeNET scores cluster
        below 0.3 for tissue-specific associations like BRCA.
      * Diagnostic logging when the gene-universe intersection is small.

    Args:
        disgenet_path: path to the TSV dump.
        gene_list: HGNC symbol universe (1000 LASSO genes).
        gene_to_idx: gene symbol -> node index.
        disease_semantic_type: legacy semantic-type filter (kept).
        disease_cuis: optional list of UMLS CUIs to restrict the table to.
        min_score: minimum GDA score.
    """
    logger.info("Loading DisGeNET gene-disease associations...")
    disgenet = pd.read_csv(disgenet_path, sep="\t")
    logger.info(f"DisGeNET raw rows: {len(disgenet)}")

    # Filter 1: UMLS CUIs (preferred -- more specific than semantic type)
    cui_col = None
    for candidate in ("diseaseId", "cui", "UMLS_CUI", "disease_cui"):
        if candidate in disgenet.columns:
            cui_col = candidate
            break
    if disease_cuis and cui_col:
        disgenet = disgenet[disgenet[cui_col].isin(list(disease_cuis))]
        logger.info(
            f"DisGeNET after CUI filter ({len(disease_cuis)} CUIs): {len(disgenet)}"
        )
    elif "diseaseSemanticType" in disgenet.columns:
        disgenet = disgenet[
            disgenet["diseaseSemanticType"].str.contains(disease_semantic_type, na=False)
        ]
        logger.info(
            f"DisGeNET after semantic type filter: {len(disgenet)}"
        )

    # Filter 2: GDA score
    if "score" in disgenet.columns and min_score is not None:
        before = len(disgenet)
        disgenet = disgenet[disgenet["score"] >= float(min_score)]
        logger.info(
            f"DisGeNET after score >= {min_score}: {len(disgenet)} (dropped {before - len(disgenet)})"
        )

    # Get gene-disease pairs for our selected genes
    gene_set_upper = set(g.upper() for g in gene_list)
    gene_upper_to_original = {g.upper(): g for g in gene_list}

    gene_col = "geneSymbol" if "geneSymbol" in disgenet.columns else "gene_symbol"
    disease_col = "diseaseName" if "diseaseName" in disgenet.columns else "disease_name"

    disgenet["gene_upper"] = disgenet[gene_col].astype(str).str.upper()

    # Diagnostic: quantify the gene-universe intersection before the join.
    # A small intersection is the leading cause of the "1236 associations ->
    # 11 edges" regression.
    disgenet_universe = set(disgenet["gene_upper"].dropna())
    inter = disgenet_universe & gene_set_upper
    logger.info(f"  DisGeNET unique genes in filtered table: {len(disgenet_universe)}")
    logger.info(f"  LASSO universe: {len(gene_set_upper)}")
    logger.info(f"  Intersection: {len(inter)}")
    if len(inter) < 50:
        logger.warning(
            "DisGeNET/LASSO intersection is tiny -- likely symbol-case or "
            "alias mismatch. Sample DisGeNET symbols: "
            f"{list(disgenet_universe)[:8]}; sample LASSO: "
            f"{list(gene_set_upper)[:8]}"
        )

    filtered = disgenet[disgenet["gene_upper"].isin(gene_set_upper)]

    # Build disease node mapping
    diseases = filtered[disease_col].unique().tolist()
    disease_to_idx = {d: i for i, d in enumerate(diseases)}

    # Build gene -> disease edges
    src, dst = [], []
    for _, row in filtered.iterrows():
        gene = gene_upper_to_original.get(row["gene_upper"])
        disease = row[disease_col]
        if gene in gene_to_idx and disease in disease_to_idx:
            src.append(gene_to_idx[gene])
            dst.append(disease_to_idx[disease])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    logger.info(f"Gene-disease edges: {edge_index.size(1)}, Diseases: {len(diseases)}")
    return edge_index, diseases, disease_to_idx


def _kegg_entrez_to_symbol(entrez_ids: list, cache_path: str) -> dict:
    """Map KEGG's hsa:<entrez> IDs to HGNC symbols, cached on disk.

    Why this function exists: the previous KEGG loader called
    `/find/genes/<symbol>+hsa` per gene which (a) is rate-limited, (b) returned
    at most one hsa: hit regardless of pathway membership, and (c) produced
    ~1 edge per pathway in practice.

    The new flow: we pull the full `/link/pathway/hsa` dump (one request),
    which is a big list of (pathway_id, entrez_id). Entrez IDs need to be
    converted to HGNC symbols before joining with our LASSO universe -- that's
    what this helper does.

    Args:
        entrez_ids: list of strings shaped "hsa:7157" or "7157".
        cache_path: JSON file to cache the mapping between runs.
    """
    ids_clean = sorted({str(x).replace("hsa:", "").strip() for x in entrez_ids if x})
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception as e:
            logger.warning(f"KEGG cache unreadable ({e}); rebuilding.")

    missing = [i for i in ids_clean if i not in cache]
    if not missing:
        return cache

    logger.info(
        f"KEGG: looking up {len(missing)} new Entrez IDs "
        f"({len(cache)} already cached)."
    )

    def _persist() -> None:
        # Checkpoint cache so partial progress survives a crash / ctrl-C and the
        # next run only retries IDs that are still missing.
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cache, f, indent=2, sort_keys=True)
            os.replace(tmp, cache_path)
        except OSError as e:
            logger.warning(f"Could not write KEGG cache: {e}")

    # ---- Pass 1: mygene.info in batches with retry -------------------------
    try:
        import mygene  # lazy import -- optional dep
        mg = mygene.MyGeneInfo()
        # Smaller batches are far less likely to hit the mygene 120s gateway
        # timeout we saw in production logs. Retry each batch with exponential
        # backoff; if it still fails we fall through to NCBI EUtils below.
        batch_size = 200
        max_attempts = 4
        base_backoff = 5.0
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            got = False
            for attempt in range(1, max_attempts + 1):
                try:
                    res = mg.getgenes(chunk, fields="symbol", species="human")
                    for r in res:
                        eid = str(r.get("_id") or r.get("query") or "").strip()
                        sym = r.get("symbol")
                        if eid and sym:
                            cache[eid] = sym.upper()
                    got = True
                    break
                except Exception as e:
                    msg = str(e)
                    transient = any(
                        code in msg for code in ("504", "502", "503", "429", "timeout", "Timeout")
                    )
                    if attempt < max_attempts and transient:
                        sleep_s = base_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            f"mygene batch {i}-{i + len(chunk)} failed "
                            f"(attempt {attempt}/{max_attempts}): {e}; "
                            f"retrying in {sleep_s:.0f}s."
                        )
                        time.sleep(sleep_s)
                        continue
                    logger.warning(
                        f"mygene batch {i}-{i + len(chunk)} gave up after "
                        f"{attempt} attempt(s): {e}"
                    )
                    break
            if got:
                _persist()
    except ImportError:
        logger.warning(
            "mygene not installed; skipping to NCBI EUtils. Install with "
            "`pip install mygene` for much faster builds."
        )

    # ---- Pass 2: NCBI EUtils esummary fallback for anything still missing --
    still_missing = [eid for eid in missing if eid not in cache]
    if still_missing:
        logger.info(
            f"KEGG: {len(still_missing)} Entrez IDs unresolved by mygene; "
            f"falling back to NCBI EUtils esummary."
        )
        # esummary.fcgi accepts comma-separated ids (~200 is safe for GET).
        eutils_batch = 200
        session = requests.Session()
        for i in range(0, len(still_missing), eutils_batch):
            chunk = still_missing[i : i + eutils_batch]
            ids_param = ",".join(chunk)
            resolved_any = False
            for attempt in range(1, 4):
                try:
                    r = session.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                        params={"db": "gene", "id": ids_param, "retmode": "json"},
                        timeout=30,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        result = data.get("result", {}) or {}
                        for eid in chunk:
                            entry = result.get(eid)
                            if not isinstance(entry, dict):
                                continue
                            # Prefer official symbol; fall back to "name".
                            sym = entry.get("nomenclaturesymbol") or entry.get("name")
                            if sym:
                                cache[eid] = str(sym).upper()
                                resolved_any = True
                        break
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(2 * attempt)
                        continue
                    logger.warning(f"EUtils esummary returned {r.status_code}; skipping batch.")
                    break
                except Exception as e:
                    if attempt < 3:
                        time.sleep(2 * attempt)
                        continue
                    logger.warning(f"EUtils esummary batch failed: {e}")
                    break
            if resolved_any:
                _persist()
            # NCBI is happy with <= 3 req/s without an API key; stay safe.
            time.sleep(0.4)

    _persist()
    return cache


def fetch_kegg_pathways(gene_list: list, kg_dir: str = None) -> dict:
    """Fetch KEGG pathway memberships for selected genes.

    Strategy (fixed Phase 1):
      1) `/list/pathway/hsa` -- one request, returns all human pathways.
      2) `/link/pathway/hsa` -- one request, returns (pathway_id, entrez_id)
         for every gene-pathway membership in KEGG human.
      3) Map Entrez IDs -> HGNC symbols via mygene (cached to disk).
      4) Filter to LASSO gene universe.

    Args:
        gene_list: HGNC symbol universe to filter against.
        kg_dir: directory for the Entrez->symbol cache (default: cwd).

    Returns:
        {pathway_name: [matching_hgnc_symbols]}
    """
    logger.info("Fetching KEGG pathway data...")
    pathways = defaultdict(list)

    try:
        # Step 1: human pathway list
        resp = requests.get("https://rest.kegg.jp/list/pathway/hsa", timeout=30)
        if resp.status_code != 200:
            logger.warning("KEGG API unavailable. Using fallback pathways.")
            return _get_fallback_pathways(gene_list)

        pathway_ids = {}
        for line in resp.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                pid = parts[0].replace("path:", "")
                pname = parts[1].split(" - ")[0].strip()
                pathway_ids[pid] = pname
        logger.info(f"KEGG: fetched {len(pathway_ids)} human pathways")

        # Step 2: bulk gene-pathway links
        link_resp = requests.get("https://rest.kegg.jp/link/pathway/hsa", timeout=60)
        if link_resp.status_code != 200:
            logger.warning("KEGG /link/pathway/hsa failed. Using fallback.")
            return _get_fallback_pathways(gene_list)

        # Each line: "hsa:<entrez>\tpath:hsa<id>"
        pairs = []
        all_entrez = set()
        for line in link_resp.text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            hsa_id = parts[0].replace("hsa:", "").strip()
            path_id = parts[1].replace("path:", "").strip()
            if hsa_id and path_id in pathway_ids:
                pairs.append((hsa_id, path_id))
                all_entrez.add(hsa_id)
        logger.info(
            f"KEGG: {len(pairs)} raw gene-pathway pairs covering "
            f"{len(all_entrez)} Entrez IDs"
        )

        # Step 3: Entrez -> symbol mapping (cached)
        cache_path = os.path.join(
            kg_dir or "data/knowledge_graph",
            "kegg_entrez_to_symbol.json",
        )
        entrez_to_symbol = _kegg_entrez_to_symbol(sorted(all_entrez), cache_path)
        n_mapped = sum(1 for e in all_entrez if e in entrez_to_symbol)
        logger.info(
            f"KEGG: mapped {n_mapped}/{len(all_entrez)} Entrez IDs to HGNC symbols"
        )

        # Diagnostic: sample to confirm format alignment with LASSO universe
        sample_keggs = list(entrez_to_symbol.items())[:5]
        sample_universe = list(gene_list)[:5]
        logger.info(f"  Sample KEGG symbols: {sample_keggs}")
        logger.info(f"  Sample universe symbols: {sample_universe}")

        # Step 4: filter to universe, build pathways dict
        universe = {g.upper() for g in gene_list}
        for entrez, path_id in pairs:
            sym = entrez_to_symbol.get(entrez)
            if sym and sym in universe:
                pathways[pathway_ids[path_id]].append(sym)

        # Deduplicate within each pathway
        pathways = {k: sorted(set(v)) for k, v in pathways.items()}

    except Exception as e:
        logger.warning(f"KEGG fetch failed: {e}. Using fallback pathways.")
        return _get_fallback_pathways(gene_list)

    total_edges = sum(len(v) for v in pathways.values())
    logger.info(
        f"KEGG: {len(pathways)} pathways, {total_edges} total gene-pathway edges "
        f"after filter to LASSO universe"
    )
    return dict(pathways)


def _get_fallback_pathways(gene_list: list) -> dict:
    """Fallback pathway assignments based on known breast cancer pathways."""
    known_pathways = {
        "PI3K-Akt signaling": ["PIK3CA", "AKT1", "PTEN", "MTOR", "PIK3R1", "AKT2", "TSC1", "TSC2"],
        "p53 signaling": ["TP53", "MDM2", "CDKN2A", "BAX", "BCL2", "CASP3", "CASP8", "CASP9", "BIRC5"],
        "MAPK signaling": ["KRAS", "BRAF", "MAP2K1", "MAPK1", "MAP3K1", "EGFR", "ERBB2", "FGFR1", "FGFR2"],
        "Wnt signaling": ["CTNNB1", "WNT1", "APC", "AXIN1", "GSK3B", "LEF1", "TCF7L2"],
        "Cell cycle": ["RB1", "CCND1", "CDK4", "CDK6", "CDKN2A", "CDKN1A", "CCNB1", "AURKA", "AURKB"],
        "DNA repair": ["BRCA1", "BRCA2", "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D", "BARD1", "BRIP1"],
        "Estrogen signaling": ["ESR1", "PGR", "FOXA1", "GATA3", "XBP1"],
        "JAK-STAT signaling": ["JAK2", "STAT3", "IL6", "TNF", "SOCS1", "SOCS3"],
        "TGF-beta signaling": ["TGFB1", "SMAD4", "SMAD2", "SMAD3", "TGFBR1", "TGFBR2"],
        "Apoptosis": ["BAX", "BCL2", "CASP3", "CASP8", "CASP9", "BIRC5", "XIAP", "CYCS"],
        "Notch signaling": ["NOTCH1", "NOTCH2", "NOTCH3", "JAG1", "DLL1", "HES1"],
        "ErbB signaling": ["ERBB2", "ERBB3", "EGFR", "NRG1", "SHC1", "GRB2", "SOS1"],
        "Chromatin remodeling": ["ARID1A", "KMT2C", "NCOR1", "HDAC1", "HDAC2", "EP300"],
    }

    gene_set_upper = set(g.upper() for g in gene_list)
    pathways = {}
    for pathway, genes in known_pathways.items():
        matched = [g for g in genes if g.upper() in gene_set_upper]
        if matched:
            pathways[pathway] = matched

    logger.info(f"Using {len(pathways)} fallback pathways")
    return pathways


def _find_gmt(kg_dir: str, prefixes: tuple, fallback: str = None) -> str:
    """Return the first GMT in kg_dir whose filename starts with any of
    `prefixes`, otherwise the fallback filename if it exists, otherwise None.

    Why: we now support the MSigDB versioned filenames
    (e.g. h.all.v2024.1.Hs.symbols.gmt) directly without an intermediate
    rename step -- but we still fall back to the legacy `msigdb_hallmarks.gmt`
    name for backward-compat with older checkouts.
    """
    if not os.path.isdir(kg_dir):
        return None
    for name in sorted(os.listdir(kg_dir)):
        lower = name.lower()
        if not lower.endswith(".gmt"):
            continue
        if any(lower.startswith(p.lower()) for p in prefixes):
            return os.path.join(kg_dir, name)
    if fallback:
        candidate = os.path.join(kg_dir, fallback)
        if os.path.exists(candidate):
            return candidate
    return None


def load_gmt_pathways(gmt_path: str) -> dict:
    """Load an MSigDB-format GMT file into {pathway_name: [gene_symbols, ...]}.

    Each line: <name>\\t<description/url>\\t<gene1>\\t<gene2>...

    Guards for silent failures (the Phase 1 regression):
      * missing file -> warn and return {}
      * suspiciously small file (< 1 KB) -> warn: MSigDB Hallmarks GMT is
        ~48 KB and Reactome is ~900 KB, so anything smaller means a truncated
        download.
    """
    if not os.path.exists(gmt_path):
        logger.warning(f"GMT file not found: {gmt_path}")
        return {}

    size = os.path.getsize(gmt_path)
    if size < 1024:
        logger.warning(
            f"GMT file suspiciously small ({size} bytes) at {gmt_path}; "
            "likely truncated download. Returning empty pathway dict."
        )
        return {}

    pathways = {}
    with open(gmt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # Upper-case + strip immediately -- downstream matching is case-
            # insensitive against the LASSO gene universe, and some MSigDB
            # releases mix casing for old HGNC aliases.
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            if genes:
                pathways[name] = genes
    logger.info(f"Loaded {len(pathways)} pathways from {os.path.basename(gmt_path)} ({size} bytes)")
    return pathways


def load_msigdb_hallmarks(gmt_path: str, gene_list: list) -> dict:
    """Load MSigDB Hallmarks (50 hallmark gene sets). Keep only pathways that
    overlap the selected gene universe so we don't drag in noise.

    Per-hallmark match counts are logged so we can distinguish "file parsed
    fine but the LASSO gene list doesn't overlap with Hallmarks" (0 matches
    everywhere) from "the parser found genes but missed the universe" (some
    matches, but fewer than expected).
    """
    all_hallmarks = load_gmt_pathways(gmt_path)
    if not all_hallmarks:
        return {}

    gene_set_upper = {g.upper() for g in gene_list}
    filtered = {}
    total_matched = 0
    zero_match = 0
    for name, genes in all_hallmarks.items():
        matched = [g for g in genes if g in gene_set_upper]
        total_matched += len(matched)
        if matched:
            filtered[name] = matched
        else:
            zero_match += 1
    # One line per hallmark is chatty but invaluable when debugging a
    # symbol-mismatch regression.
    for name in sorted(filtered.keys()):
        logger.debug(
            f"  {name}: {len(all_hallmarks[name])} genes in set, "
            f"{len(filtered[name])} match our universe"
        )
    logger.info(
        f"MSigDB Hallmarks: {len(all_hallmarks)} loaded, "
        f"{len(filtered)} with gene overlap "
        f"({zero_match} hallmarks had 0 matches; "
        f"total edges will be ~{total_matched})"
    )
    return filtered


def load_reactome_pathways(
    gmt_path: str,
    gene_list: list,
    max_pathways: int = 200,
    cancer_keywords: tuple = (
        "cell_cycle", "apoptosis", "dna", "repair", "p53", "wnt", "pi3k",
        "mapk", "erbb", "notch", "tgf", "hedgehog", "estrogen", "jak",
        "signaling", "pathway", "cancer", "tumor", "metastasis", "egfr",
        "mtor", "kras", "receptor_tyrosine",
    ),
) -> dict:
    """Load Reactome pathways (via MSigDB C2 collection) and filter to the
    cancer/mechanistic subset. Limits to `max_pathways` by gene-overlap size
    so we don't explode the KG with 1700 largely-irrelevant pathway nodes.
    """
    all_pathways = load_gmt_pathways(gmt_path)
    if not all_pathways:
        return {}

    gene_set_upper = {g.upper() for g in gene_list}

    # Filter 1: keep only pathways with overlap
    overlap_map = {}
    for name, genes in all_pathways.items():
        matched = [g for g in genes if g.upper() in gene_set_upper]
        if matched:
            overlap_map[name] = matched

    # Filter 2: prioritize cancer-mechanistic pathways via keyword match
    name_lc = {n: n.lower() for n in overlap_map}
    prioritized = {
        n: g for n, g in overlap_map.items()
        if any(kw in name_lc[n] for kw in cancer_keywords)
    }
    # Fill remainder with highest-overlap Reactome pathways (non-prioritized)
    remainder = {n: g for n, g in overlap_map.items() if n not in prioritized}
    remainder_sorted = sorted(remainder.items(), key=lambda kv: -len(kv[1]))
    for n, g in remainder_sorted:
        if len(prioritized) >= max_pathways:
            break
        prioritized[n] = g

    # Hard cap
    if len(prioritized) > max_pathways:
        sorted_items = sorted(prioritized.items(), key=lambda kv: -len(kv[1]))[:max_pathways]
        prioritized = dict(sorted_items)

    logger.info(
        f"Reactome: {len(all_pathways)} loaded, {len(overlap_map)} with overlap, "
        f"{len(prioritized)} kept after cancer-keyword + max_pathways={max_pathways} filter"
    )
    return prioritized


def build_pathway_edges(
    pathways: dict,
    gene_list: list,
    gene_to_idx: dict,
) -> tuple:
    """Build gene-pathway edges from pathway membership data.

    Returns:
        (gene_pathway_edge_index, pathway_names, pathway_to_idx)
    """
    pathway_names = list(pathways.keys())
    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    gene_upper_to_original = {g.upper(): g for g in gene_list}

    src, dst = [], []
    for pathway, genes in pathways.items():
        pidx = pathway_to_idx[pathway]
        for gene in genes:
            gene_orig = gene_upper_to_original.get(gene.upper())
            if gene_orig and gene_orig in gene_to_idx:
                src.append(gene_to_idx[gene_orig])
                dst.append(pidx)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    logger.info(f"Gene-pathway edges: {edge_index.size(1)}, Pathways: {len(pathway_names)}")
    return edge_index, pathway_names, pathway_to_idx


def compute_graph_statistics(
    gene_gene_edges: torch.Tensor,
    gene_disease_edges: torch.Tensor,
    gene_pathway_edges: torch.Tensor,
    n_genes: int,
    gene_hallmark_edges: torch.Tensor = None,
    gene_reactome_edges: torch.Tensor = None,
) -> dict:
    """Compute and log knowledge graph statistics.

    The isolated-genes count counts any gene with zero edges across ALL edge
    types (gene-gene, gene-disease, gene-pathway, gene-hallmark, gene-reactome).
    This is the metric Phase 1.3 tracks: target < 10%.
    """
    stats = {
        "n_genes": n_genes,
        "n_gene_gene_edges": int(gene_gene_edges.size(1)),
        "n_gene_disease_edges": int(gene_disease_edges.size(1)),
        "n_gene_pathway_edges": int(gene_pathway_edges.size(1)),
        "n_gene_hallmark_edges": int(gene_hallmark_edges.size(1)) if gene_hallmark_edges is not None else 0,
        "n_gene_reactome_edges": int(gene_reactome_edges.size(1)) if gene_reactome_edges is not None else 0,
    }

    # Gene-gene graph density
    max_edges = n_genes * (n_genes - 1)
    stats["gene_gene_density"] = gene_gene_edges.size(1) / max_edges if max_edges > 0 else 0

    # Per-node degree across ALL edge types. The isolated metric used in
    # Phase 1.3 is across all graphs, since even a gene that has no PPI
    # neighbor can still receive message-passing through pathway/hallmark
    # hops in the heterogeneous setup.
    degrees = torch.zeros(n_genes)
    for tens in (gene_gene_edges, gene_disease_edges, gene_pathway_edges,
                 gene_hallmark_edges, gene_reactome_edges):
        if tens is None or tens.size(1) == 0:
            continue
        src = tens[0]
        for i in range(src.size(0)):
            node = int(src[i].item())
            if 0 <= node < n_genes:
                degrees[node] += 1

    stats["avg_degree"] = float(degrees.mean().item())
    stats["max_degree"] = float(degrees.max().item()) if n_genes > 0 else 0
    stats["isolated_genes"] = int((degrees == 0).sum().item())
    stats["isolated_pct"] = 100.0 * stats["isolated_genes"] / max(n_genes, 1)

    logger.info("=" * 60)
    logger.info("Knowledge Graph Statistics:")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    # Phase 1.3 targets, printed alongside actuals for easy pass/fail eyeballing
    logger.info("  -- Phase 1.3 targets --")
    logger.info(f"  n_gene_gene_edges target > 3000 (actual {stats['n_gene_gene_edges']})")
    logger.info(f"  n_gene_pathway_edges target > 5000 "
                f"(actual {stats['n_gene_pathway_edges'] + stats['n_gene_reactome_edges']})")
    logger.info(f"  n_gene_hallmark_edges target > 4000 (actual {stats['n_gene_hallmark_edges']})")
    logger.info(f"  isolated_pct target < 10% (actual {stats['isolated_pct']:.1f}%)")

    return stats


class KGBuildError(AssertionError):
    """Raised when a KG edge set falls below its configured minimum.

    Using a subclass (not plain AssertionError) so tests can target this
    specifically and upstream callers can catch "KG construction failed" as
    a semantic category rather than a generic assertion failure.
    """


def _strict_assert(condition: bool, message: str, strict: bool) -> None:
    """Fail loudly when strict mode is on, warn (but continue) otherwise.

    Gating behind config.kg.strict_asserts lets you flip asserts off for
    debugging the individual loaders without losing the safety net in
    the main training path.
    """
    if condition:
        return
    if strict:
        raise KGBuildError(message)
    logger.warning(f"[strict=off] {message}")


def _trim_isolated_genes(
    gene_gene_edges: torch.Tensor,
    gene_gene_weights: torch.Tensor,
    gene_disease_edges: torch.Tensor,
    gene_pathway_edges: torch.Tensor,
    gene_hallmark_edges: torch.Tensor,
    gene_reactome_edges: torch.Tensor,
    gene_list: list,
    gene_to_idx: dict,
    max_isolated_pct: float = 0.15,
):
    """Drop genes with zero incident edges across all relation types.

    Returns (possibly reindexed) edge tensors + weights, a possibly shorter
    gene_list, a fresh gene_to_idx, and a boolean `trimmed` flag.

    Only trims if isolated_pct > max_isolated_pct; otherwise returns inputs
    unchanged. When trimming fires, node indices are compacted so the new
    gene_list[i] corresponds to edge-tensor value `i`.

    Edge semantics:
      - gene_gene_edges is a [2, E] COO tensor with both rows = gene indices.
      - gene_disease / gene_pathway / gene_hallmark / gene_reactome are [2, E]
        with row 0 = gene, row 1 = non-gene (disease/pathway/hallmark/reactome).
      - Edges never reference an isolated gene (by definition), so trimming
        drops only nodes, not edges; all weights remain valid post-remap.
    """
    n_genes = len(gene_list)
    if n_genes == 0:
        return (
            gene_gene_edges, gene_gene_weights, gene_disease_edges,
            gene_pathway_edges, gene_hallmark_edges, gene_reactome_edges,
            gene_list, gene_to_idx, False,
        )

    touched = torch.zeros(n_genes, dtype=torch.bool)

    def _mark(edges: torch.Tensor, both_rows_genes: bool) -> None:
        if edges is None or edges.numel() == 0 or edges.size(1) == 0:
            return
        idx0 = edges[0].long()
        valid0 = (idx0 >= 0) & (idx0 < n_genes)
        touched[idx0[valid0]] = True
        if both_rows_genes:
            idx1 = edges[1].long()
            valid1 = (idx1 >= 0) & (idx1 < n_genes)
            touched[idx1[valid1]] = True

    _mark(gene_gene_edges, both_rows_genes=True)
    _mark(gene_disease_edges, both_rows_genes=False)
    _mark(gene_pathway_edges, both_rows_genes=False)
    _mark(gene_hallmark_edges, both_rows_genes=False)
    _mark(gene_reactome_edges, both_rows_genes=False)

    n_isolated = int((~touched).sum().item())
    isolated_pct = n_isolated / n_genes
    logger.info(
        f"Isolation audit: {n_isolated}/{n_genes} genes have zero incident "
        f"edges ({isolated_pct:.1%}); max_isolated_pct={max_isolated_pct:.1%}."
    )
    if isolated_pct <= max_isolated_pct:
        logger.info("  -> within tolerance; keeping universe as-is.")
        return (
            gene_gene_edges, gene_gene_weights, gene_disease_edges,
            gene_pathway_edges, gene_hallmark_edges, gene_reactome_edges,
            gene_list, gene_to_idx, False,
        )

    # Build the compaction map: old_idx -> new_idx (-1 if dropped).
    kept_old_idx = torch.nonzero(touched, as_tuple=False).squeeze(1)
    old_to_new = torch.full((n_genes,), -1, dtype=torch.long)
    old_to_new[kept_old_idx] = torch.arange(kept_old_idx.numel(), dtype=torch.long)

    new_gene_list = [gene_list[i] for i in kept_old_idx.tolist()]
    new_gene_to_idx = {g: i for i, g in enumerate(new_gene_list)}

    def _remap(edges: torch.Tensor, both_rows_genes: bool) -> torch.Tensor:
        if edges is None or edges.numel() == 0 or edges.size(1) == 0:
            return edges
        new_row0 = old_to_new[edges[0].long()]
        if both_rows_genes:
            new_row1 = old_to_new[edges[1].long()]
            return torch.stack([new_row0, new_row1], dim=0)
        return torch.stack([new_row0, edges[1].long()], dim=0)

    gene_gene = _remap(gene_gene_edges, both_rows_genes=True)
    gene_disease = _remap(gene_disease_edges, both_rows_genes=False)
    gene_pathway = _remap(gene_pathway_edges, both_rows_genes=False)
    gene_hallmark = _remap(gene_hallmark_edges, both_rows_genes=False)
    gene_reactome = _remap(gene_reactome_edges, both_rows_genes=False)

    # Safety check: after compaction, no -1 should survive in row 0 of any
    # edge tensor (isolated-by-definition genes cannot appear as edge sources).
    for name, e in [
        ("gene_gene", gene_gene),
        ("gene_disease", gene_disease),
        ("gene_pathway", gene_pathway),
        ("gene_hallmark", gene_hallmark),
        ("gene_reactome", gene_reactome),
    ]:
        if e is None or e.numel() == 0:
            continue
        if (e[0] < 0).any().item():
            raise KGBuildError(
                f"Trim sanity check failed: {name} has -1 in row 0 after remap. "
                "This means an edge referenced a gene we intended to drop."
            )

    logger.info(
        f"  -> trimmed {n_isolated} isolated genes "
        f"(universe {n_genes} -> {len(new_gene_list)})."
    )
    return (
        gene_gene, gene_gene_weights, gene_disease,
        gene_pathway, gene_hallmark, gene_reactome,
        new_gene_list, new_gene_to_idx, True,
    )


def build_knowledge_graph(config: dict, gene_list: list) -> dict:
    """Build the complete heterogeneous knowledge graph.

    Returns dictionary with all graph components needed for PyG HeteroData.
    """
    kg_dir = config["paths"]["knowledge_graph"]
    processed_dir = config["paths"]["processed_data"]
    os.makedirs(processed_dir, exist_ok=True)

    # kg.strict_asserts = true: any KG loader producing too few edges aborts
    # the whole run. Default on -- Phase 1 discovered four silent zero-edge
    # regressions that slipped through exactly because we swallowed the
    # warning. Turn off only for isolated loader debugging.
    kg_cfg = config.get("kg", {}) or {}
    strict = bool(kg_cfg.get("strict_asserts", True))
    # Per-loader minimums (override via config.kg.min_*). The numbers below
    # are the Phase 1.3 acceptance criteria minus a 10% safety margin so
    # small fluctuations in gene-universe composition don't trip the gate.
    min_gene_gene = int(kg_cfg.get("min_gene_gene_edges", 2000))
    min_hallmark = int(kg_cfg.get("min_gene_hallmark_edges", 3000))
    min_reactome = int(kg_cfg.get("min_gene_reactome_edges", 2000))
    min_pathway = int(kg_cfg.get("min_gene_pathway_edges", 2000))
    min_disease = int(kg_cfg.get("min_gene_disease_edges", 500))

    string_path = os.path.join(kg_dir, "string_ppi.tsv")
    mapping_path = os.path.join(kg_dir, "string_id_mapping.tsv")
    disgenet_path = os.path.join(kg_dir, "disgenet_gene_disease.tsv")

    # 1. STRING PPI edges (gene-gene)
    logger.info("=" * 60)
    logger.info("Building gene-gene edges from STRING PPI")
    logger.info("=" * 60)
    gene_gene_edges, gene_gene_weights, gene_to_idx = load_string_ppi(
        string_path,
        mapping_path,
        gene_list,
        confidence_threshold=config["data"]["string_confidence_threshold"],
    )

    # 1b. Co-expression edges derived from the normalized expression matrix.
    # Densifies the graph so the GAT has actual neighborhoods to attend over
    # (first run: ~81% of genes isolated at STRING threshold 700).
    coexp_thr = config["data"].get("coexpression_threshold")
    coexp_k = config["data"].get("coexpression_top_k", 8)
    if coexp_thr is not None:
        expr_path = os.path.join(processed_dir, "expression_selected.tsv")
        if os.path.exists(expr_path):
            try:
                expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
                coexp_edges, coexp_weights = build_coexpression_edges(
                    expr_df, gene_list, gene_to_idx,
                    threshold=float(coexp_thr), top_k=int(coexp_k),
                )
                gene_gene_edges, gene_gene_weights = merge_edge_sets(
                    gene_gene_edges, gene_gene_weights,
                    coexp_edges, coexp_weights,
                )
                logger.info(
                    f"Merged gene-gene edges (STRING + co-expression): "
                    f"{gene_gene_edges.size(1)}"
                )
            except Exception as e:
                logger.warning(f"Co-expression edges skipped: {e}")
        else:
            logger.warning(
                f"Expression file not found at {expr_path}; "
                "skipping co-expression edge construction."
            )

    # Phase 1 gate: STRING + co-expression should yield >2k edges on the 1000
    # LASSO-selected genes. If it doesn't, either STRING wasn't mapped right
    # or the co-expression step was skipped (check the warnings above).
    _strict_assert(
        gene_gene_edges.size(1) > min_gene_gene,
        f"STRING+coexpr produced only {gene_gene_edges.size(1)} gene-gene edges "
        f"(expected > {min_gene_gene}). Check STRING confidence threshold and "
        "whether expression_selected.tsv exists.",
        strict,
    )

    # 2. DisGeNET edges (gene-disease)
    logger.info("=" * 60)
    logger.info("Building gene-disease edges from DisGeNET")
    logger.info("=" * 60)
    gene_disease_edges, disease_names, disease_to_idx = load_disgenet_edges(
        disgenet_path,
        gene_list,
        gene_to_idx,
        config["data"]["disease_semantic_type"],
        disease_cuis=config.get("data", {}).get("disease_cuis"),
        min_score=float(config.get("data", {}).get("disgenet_min_score", 0.05)),
    )
    _strict_assert(
        gene_disease_edges.size(1) > min_disease,
        f"DisGeNET produced only {gene_disease_edges.size(1)} gene-disease edges "
        f"(expected > {min_disease}). Likely symbol/case mismatch or an over-"
        "aggressive score filter. See diagnostic log above.",
        strict,
    )

    # 3. KEGG pathway edges (gene-pathway)
    logger.info("=" * 60)
    logger.info("Building gene-pathway edges from KEGG")
    logger.info("=" * 60)
    pathways = fetch_kegg_pathways(gene_list, kg_dir=kg_dir)
    gene_pathway_edges, pathway_names, pathway_to_idx = build_pathway_edges(
        pathways, gene_list, gene_to_idx
    )
    _strict_assert(
        gene_pathway_edges.size(1) > min_pathway,
        f"KEGG produced only {gene_pathway_edges.size(1)} gene-pathway edges "
        f"(expected > {min_pathway}). Usually means the Entrez->symbol "
        "mapping didn't complete -- check data/knowledge_graph/kegg_entrez_to_symbol.json.",
        strict,
    )

    # 3b. MSigDB Hallmarks of Cancer -- 50 coarse-grained pathway nodes that
    # double as the clinical interpretability layer. Cites Liberzon et al., 2015.
    logger.info("=" * 60)
    logger.info("Building gene-hallmark edges from MSigDB Hallmarks")
    logger.info("=" * 60)
    hallmarks_path = _find_gmt(kg_dir, ("h.all",), fallback="msigdb_hallmarks.gmt")
    hallmarks = load_msigdb_hallmarks(hallmarks_path, gene_list) if hallmarks_path else {}
    gene_hallmark_edges, hallmark_names, hallmark_to_idx = build_pathway_edges(
        hallmarks, gene_list, gene_to_idx
    )
    _strict_assert(
        gene_hallmark_edges.size(1) > min_hallmark,
        f"MSigDB Hallmarks produced only {gene_hallmark_edges.size(1)} edges "
        f"(expected > {min_hallmark}). "
        + (
            "GMT file missing -- download h.all.v2024.1.Hs.symbols.gmt into "
            "data/knowledge_graph/."
            if gene_hallmark_edges.size(1) == 0
            else "Either the LASSO universe overlaps MSigDB Hallmarks less than "
            "expected, or `min_gene_hallmark_edges` in config is set too high "
            "for the current universe size. Expected ~N_hallmark_memberships * "
            "(universe_size / ~20k)."
        ),
        strict,
    )

    # 3c. Reactome -- finer mechanistic pathways (filtered to ~200 cancer-relevant).
    logger.info("=" * 60)
    logger.info("Building gene-reactome edges from MSigDB C2 Reactome")
    logger.info("=" * 60)
    reactome_path = _find_gmt(kg_dir, ("c2.cp.reactome",), fallback="msigdb_reactome.gmt")
    reactome_max = int(config.get("data", {}).get("reactome_max_pathways", 200))
    reactome = (
        load_reactome_pathways(reactome_path, gene_list, max_pathways=reactome_max)
        if reactome_path else {}
    )
    gene_reactome_edges, reactome_names, reactome_to_idx = build_pathway_edges(
        reactome, gene_list, gene_to_idx
    )
    _strict_assert(
        gene_reactome_edges.size(1) > min_reactome,
        f"Reactome produced only {gene_reactome_edges.size(1)} edges "
        f"(expected > {min_reactome}). "
        + (
            "GMT file missing -- check c2.cp.reactome.*.symbols.gmt in "
            "data/knowledge_graph/."
            if gene_reactome_edges.size(1) == 0
            else "Either the cancer-keyword filter is too tight, the LASSO "
            "universe overlaps Reactome less than expected, or "
            "`min_gene_reactome_edges` in config is set too high for the "
            "current universe size."
        ),
        strict,
    )

    # 4a. Trim isolated genes. Phase 1 found 48.7% of the 1500-gene universe
    # ended up with zero edges post-build: LASSO picks by survival variance,
    # DisGeNET by published association -- neither guarantees STRING/MSigDB
    # coverage. An isolated gene is a dead GAT node (no neighborhood to attend
    # to), so it contributes only its raw expression scalar and injects noise
    # into batch normalization statistics. Trim + rewrite the expression
    # matrix / selected_genes list so downstream (dataset.py, explain.py)
    # transparently picks up the compacted universe.
    max_iso = float(kg_cfg.get("max_isolated_pct", 0.15))
    (
        gene_gene_edges, gene_gene_weights, gene_disease_edges,
        gene_pathway_edges, gene_hallmark_edges, gene_reactome_edges,
        gene_list, gene_to_idx, trimmed,
    ) = _trim_isolated_genes(
        gene_gene_edges, gene_gene_weights, gene_disease_edges,
        gene_pathway_edges, gene_hallmark_edges, gene_reactome_edges,
        gene_list, gene_to_idx, max_isolated_pct=max_iso,
    )
    if trimmed:
        expr_path = os.path.join(processed_dir, "expression_selected.tsv")
        genes_path = os.path.join(processed_dir, "selected_genes.txt")
        if os.path.exists(expr_path):
            expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
            kept_in_expr = [g for g in gene_list if g in expr_df.index]
            if len(kept_in_expr) != len(gene_list):
                logger.warning(
                    f"Post-trim: {len(gene_list) - len(kept_in_expr)} genes in "
                    "the trimmed universe are not rows of expression_selected.tsv. "
                    "Rewriting the expression matrix with the intersection only."
                )
                # Reconcile: if expression lost any, shrink gene_list again.
                # This should be rare -- only happens if preprocessing's gene
                # filter already dropped some of the kept genes.
                gene_list = kept_in_expr
                gene_to_idx = {g: i for i, g in enumerate(gene_list)}
            expr_df.loc[gene_list].to_csv(expr_path, sep="\t")
            logger.info(
                f"  -> rewrote {expr_path} with {len(gene_list)} rows."
            )
        else:
            logger.warning(
                f"Trim fired but {expr_path} not found; skipping expression "
                "matrix rewrite (downstream may see a stale universe)."
            )
        with open(genes_path, "w") as f:
            f.write("\n".join(gene_list))
        logger.info(f"  -> rewrote {genes_path} with {len(gene_list)} genes.")

    # 4b. Compute statistics on the (possibly trimmed) graph.
    stats = compute_graph_statistics(
        gene_gene_edges, gene_disease_edges, gene_pathway_edges, len(gene_list),
        gene_hallmark_edges=gene_hallmark_edges,
        gene_reactome_edges=gene_reactome_edges,
    )
    stats["trimmed_isolated"] = bool(trimmed)

    # 5. Save knowledge graph
    kg_data = {
        "gene_gene_edges": gene_gene_edges,
        "gene_gene_weights": gene_gene_weights,
        "gene_to_idx": gene_to_idx,
        "gene_list": gene_list,
        "gene_disease_edges": gene_disease_edges,
        "disease_names": disease_names,
        "disease_to_idx": disease_to_idx,
        "gene_pathway_edges": gene_pathway_edges,
        "pathway_names": pathway_names,
        "pathway_to_idx": pathway_to_idx,
        "gene_hallmark_edges": gene_hallmark_edges,
        "hallmark_names": hallmark_names,
        "hallmark_to_idx": hallmark_to_idx,
        "gene_reactome_edges": gene_reactome_edges,
        "reactome_names": reactome_names,
        "reactome_to_idx": reactome_to_idx,
        "stats": stats,
    }

    # Save tensors
    torch.save(
        {
            "gene_gene_edges": gene_gene_edges,
            "gene_gene_weights": gene_gene_weights,
            "gene_disease_edges": gene_disease_edges,
            "gene_pathway_edges": gene_pathway_edges,
            "gene_hallmark_edges": gene_hallmark_edges,
            "gene_reactome_edges": gene_reactome_edges,
        },
        os.path.join(processed_dir, "kg_edges.pt"),
    )

    # Save metadata
    metadata = {
        "gene_to_idx": gene_to_idx,
        "disease_names": disease_names,
        "disease_to_idx": disease_to_idx,
        "pathway_names": pathway_names,
        "pathway_to_idx": pathway_to_idx,
        "hallmark_names": hallmark_names,
        "hallmark_to_idx": hallmark_to_idx,
        "reactome_names": reactome_names,
        "reactome_to_idx": reactome_to_idx,
        "stats": stats,
    }
    with open(os.path.join(processed_dir, "kg_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Knowledge graph saved successfully")
    return kg_data


if __name__ == "__main__":
    config = load_config()
    processed_dir = config["paths"]["processed_data"]

    # Load selected genes from preprocessing step
    genes_file = os.path.join(processed_dir, "selected_genes.txt")
    if not os.path.exists(genes_file):
        raise FileNotFoundError(f"Run preprocessing.py first. Missing: {genes_file}")

    with open(genes_file) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    kg_data = build_knowledge_graph(config, gene_list)
