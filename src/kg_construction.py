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
import numpy as np
import pandas as pd
import yaml
import torch
import requests
from collections import defaultdict
from pathlib import Path

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


def load_disgenet_edges(
    disgenet_path: str,
    gene_list: list,
    gene_to_idx: dict,
    disease_semantic_type: str = "Neoplastic Process",
) -> tuple:
    """Load DisGeNET gene-disease associations and build edges.

    Reference: Qumsiyeh et al., 2022

    Returns:
        (gene_disease_edge_index, disease_names, disease_to_idx)
    """
    logger.info("Loading DisGeNET gene-disease associations...")
    disgenet = pd.read_csv(disgenet_path, sep="\t")

    # Filter by disease semantic type
    if "diseaseSemanticType" in disgenet.columns:
        disgenet = disgenet[disgenet["diseaseSemanticType"].str.contains(disease_semantic_type, na=False)]
        logger.info(f"DisGeNET associations after semantic type filter: {len(disgenet)}")

    # Get gene-disease pairs for our selected genes
    gene_set_upper = set(g.upper() for g in gene_list)
    gene_upper_to_original = {g.upper(): g for g in gene_list}

    gene_col = "geneSymbol" if "geneSymbol" in disgenet.columns else "gene_symbol"
    disease_col = "diseaseName" if "diseaseName" in disgenet.columns else "disease_name"
    disease_id_col = "diseaseId" if "diseaseId" in disgenet.columns else "disease_id"

    disgenet["gene_upper"] = disgenet[gene_col].str.upper()
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


def fetch_kegg_pathways(gene_list: list) -> dict:
    """Fetch KEGG pathway memberships for selected genes.

    Returns:
        Dictionary mapping pathway_name -> list of gene symbols.
    """
    logger.info("Fetching KEGG pathway data...")
    pathways = defaultdict(list)

    # Use KEGG REST API to get human pathways
    try:
        # Get list of human pathways
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

        # For each gene, find which pathways it belongs to
        gene_set_upper = set(g.upper() for g in gene_list)
        batch_size = 10
        gene_batch = list(gene_list)[:100]  # Limit to avoid too many API calls

        for i in range(0, len(gene_batch), batch_size):
            batch = gene_batch[i : i + batch_size]
            for gene in batch:
                try:
                    resp = requests.get(f"https://rest.kegg.jp/find/genes/{gene}+hsa", timeout=10)
                    if resp.status_code == 200 and resp.text.strip():
                        for line in resp.text.strip().split("\n"):
                            kegg_id = line.split("\t")[0]
                            if kegg_id.startswith("hsa:"):
                                # Get pathways for this gene
                                presp = requests.get(
                                    f"https://rest.kegg.jp/link/pathway/{kegg_id}", timeout=10
                                )
                                if presp.status_code == 200 and presp.text.strip():
                                    for pline in presp.text.strip().split("\n"):
                                        pparts = pline.split("\t")
                                        if len(pparts) >= 2:
                                            pid = pparts[1].replace("path:", "")
                                            if pid in pathway_ids:
                                                pathways[pathway_ids[pid]].append(gene)
                                break  # Take first match
                except Exception:
                    continue

    except Exception as e:
        logger.warning(f"KEGG fetch failed: {e}. Using fallback pathways.")
        return _get_fallback_pathways(gene_list)

    logger.info(f"Found {len(pathways)} KEGG pathways for selected genes")
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
) -> dict:
    """Compute and log knowledge graph statistics."""
    stats = {
        "n_genes": n_genes,
        "n_gene_gene_edges": gene_gene_edges.size(1),
        "n_gene_disease_edges": gene_disease_edges.size(1),
        "n_gene_pathway_edges": gene_pathway_edges.size(1),
    }

    # Gene-gene graph density
    max_edges = n_genes * (n_genes - 1)
    stats["gene_gene_density"] = gene_gene_edges.size(1) / max_edges if max_edges > 0 else 0

    # Average node degree (gene-gene)
    if gene_gene_edges.size(1) > 0:
        degrees = torch.zeros(n_genes)
        for i in range(gene_gene_edges.size(1)):
            degrees[gene_gene_edges[0, i]] += 1
        stats["avg_degree"] = degrees.mean().item()
        stats["max_degree"] = degrees.max().item()
        stats["isolated_genes"] = (degrees == 0).sum().item()
    else:
        stats["avg_degree"] = 0
        stats["max_degree"] = 0
        stats["isolated_genes"] = n_genes

    logger.info("Knowledge Graph Statistics:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return stats


def build_knowledge_graph(config: dict, gene_list: list) -> dict:
    """Build the complete heterogeneous knowledge graph.

    Returns dictionary with all graph components needed for PyG HeteroData.
    """
    kg_dir = config["paths"]["knowledge_graph"]
    processed_dir = config["paths"]["processed_data"]
    os.makedirs(processed_dir, exist_ok=True)

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

    # 2. DisGeNET edges (gene-disease)
    logger.info("=" * 60)
    logger.info("Building gene-disease edges from DisGeNET")
    logger.info("=" * 60)
    gene_disease_edges, disease_names, disease_to_idx = load_disgenet_edges(
        disgenet_path,
        gene_list,
        gene_to_idx,
        config["data"]["disease_semantic_type"],
    )

    # 3. KEGG pathway edges (gene-pathway)
    logger.info("=" * 60)
    logger.info("Building gene-pathway edges from KEGG")
    logger.info("=" * 60)
    pathways = fetch_kegg_pathways(gene_list)
    gene_pathway_edges, pathway_names, pathway_to_idx = build_pathway_edges(
        pathways, gene_list, gene_to_idx
    )

    # 4. Compute statistics
    stats = compute_graph_statistics(
        gene_gene_edges, gene_disease_edges, gene_pathway_edges, len(gene_list)
    )

    # 5. Save knowledge graph
    kg_data = {
        "gene_gene_edges": gene_gene_edges,
        "gene_gene_weights": gene_gene_weights,
        "gene_to_idx": gene_to_idx,
        "gene_disease_edges": gene_disease_edges,
        "disease_names": disease_names,
        "disease_to_idx": disease_to_idx,
        "gene_pathway_edges": gene_pathway_edges,
        "pathway_names": pathway_names,
        "pathway_to_idx": pathway_to_idx,
        "stats": stats,
    }

    # Save tensors
    torch.save(
        {
            "gene_gene_edges": gene_gene_edges,
            "gene_gene_weights": gene_gene_weights,
            "gene_disease_edges": gene_disease_edges,
            "gene_pathway_edges": gene_pathway_edges,
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
