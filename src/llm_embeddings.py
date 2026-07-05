"""
Stage 2.3-2.4: LLM Gene Embeddings (GenePT method)

Generates 768-dimensional gene embeddings using BioBERT/BioLinkBERT
from NCBI gene summaries. Supports patient-specific weighted embeddings.

References:
    - GenePT method: Chen & Zou, 2023
    - BioBERT: Lee et al., 2020
    - BioLinkBERT: Yasunaga et al., 2022
    - Patient-specific weighting: GenePT-w style (Vavekanand, 2026)
"""

import os
import json
import logging
import numpy as np
import torch
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_gene_summaries(summaries_path: str) -> dict:
    """Load gene summary texts."""
    with open(summaries_path) as f:
        return json.load(f)


def generate_gene_embeddings(
    gene_list: list,
    summaries: dict,
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    batch_size: int = 32,
    max_length: int = 512,
    device: str = None,
) -> np.ndarray:
    """Generate [CLS] token embeddings for each gene using BioBERT/BioLinkBERT.

    Reference: Chen & Zou (GenePT), 2023

    Args:
        gene_list: List of gene symbols.
        summaries: Dict mapping gene symbol -> text summary.
        model_name: HuggingFace model name.
        batch_size: Batch size for inference.
        max_length: Maximum token length.
        device: Compute device.

    Returns:
        numpy array of shape [num_genes, 768]
    """
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    embeddings = np.zeros((len(gene_list), embedding_dim), dtype=np.float32)

    # Prepare texts
    texts = []
    for gene in gene_list:
        text = summaries.get(gene.upper(), summaries.get(gene, ""))
        if not text:
            text = f"{gene}: A human gene involved in cellular processes and regulation."
        # Truncate very long texts
        if len(text) > 2000:
            text = text[:2000]
        texts.append(text)

    logger.info(f"Generating embeddings for {len(gene_list)} genes...")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)

            # Extract [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[i : i + len(batch_texts)] = cls_embeddings

            if (i // batch_size) % 10 == 0:
                logger.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} genes")

    logger.info(f"Generated embeddings: shape {embeddings.shape}")
    return embeddings


def create_patient_weighted_embeddings(
    gene_embeddings: np.ndarray,
    expression_matrix: np.ndarray,
    gene_list: list,
) -> np.ndarray:
    """Create patient-specific node features by weighting gene embeddings with expression.

    Reference: GenePT-w style (Vavekanand, 2026)
    For Patient X, Gene G: node_feature[G] = LLM_embedding[G] * normalized_expression[X][G]

    Args:
        gene_embeddings: [num_genes, embedding_dim] base gene embeddings.
        expression_matrix: [num_genes, num_patients] normalized expression values.
        gene_list: List of gene symbols (for logging).

    Returns:
        [num_patients, num_genes, embedding_dim] patient-weighted embeddings.
    """
    n_genes, n_patients = expression_matrix.shape
    embedding_dim = gene_embeddings.shape[1]

    assert gene_embeddings.shape[0] == n_genes, (
        f"Mismatch: {gene_embeddings.shape[0]} embeddings vs {n_genes} genes in expression"
    )

    logger.info(
        f"Creating patient-weighted embeddings: {n_patients} patients x {n_genes} genes x {embedding_dim} dims"
    )

    # expression_matrix: [n_genes, n_patients] -> [n_patients, n_genes, 1]
    expr_weights = expression_matrix.T[:, :, np.newaxis]  # [n_patients, n_genes, 1]

    # gene_embeddings: [n_genes, embedding_dim] -> [1, n_genes, embedding_dim]
    gene_emb = gene_embeddings[np.newaxis, :, :]  # [1, n_genes, embedding_dim]

    # Element-wise multiplication broadcasts across patients
    patient_embeddings = gene_emb * expr_weights  # [n_patients, n_genes, embedding_dim]

    logger.info(f"Patient-weighted embeddings shape: {patient_embeddings.shape}")
    return patient_embeddings


def generate_pathway_embeddings(
    pathway_names: list,
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    device: str = None,
) -> np.ndarray:
    """Generate embeddings for pathway nodes using their names/descriptions.

    Reference: Vaida et al., 2025 — pathway nodes can use descriptions or learnable zeros.
    """
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Generating pathway embeddings for {len(pathway_names)} pathways...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    texts = [f"Biological pathway: {name}. A signaling pathway involved in cellular regulation." for name in pathway_names]

    embeddings = np.zeros((len(pathway_names), embedding_dim), dtype=np.float32)

    with torch.no_grad():
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    logger.info(f"Pathway embeddings shape: {embeddings.shape}")
    return embeddings


def generate_disease_embeddings(
    disease_names: list,
    model_name: str = "dmis-lab/biobert-base-cased-v1.2",
    device: str = None,
) -> np.ndarray:
    """Generate embeddings for disease nodes using their names."""
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Generating disease embeddings for {len(disease_names)} diseases...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    texts = [f"Disease: {name}. A medical condition affecting human health." for name in disease_names]

    embeddings = np.zeros((len(disease_names), embedding_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**encoded)
            embeddings[i : i + len(batch)] = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    logger.info(f"Disease embeddings shape: {embeddings.shape}")
    return embeddings


def run_embedding_generation(config: dict, gene_list: list, summaries_path: str) -> dict:
    """Run the complete embedding generation pipeline.

    Returns:
        Dictionary with gene, pathway, and disease embeddings.
    """
    model_name = config["model"]["llm_model"]
    emb_dir = config["paths"]["embeddings"]
    os.makedirs(emb_dir, exist_ok=True)

    # 1. Load gene summaries
    summaries = load_gene_summaries(summaries_path)

    # 2. Generate gene embeddings
    logger.info("=" * 60)
    logger.info("Generating gene embeddings with BioBERT")
    logger.info("=" * 60)

    gene_emb_path = os.path.join(emb_dir, "gene_embeddings.npy")
    if os.path.exists(gene_emb_path):
        logger.info(f"Loading cached gene embeddings from {gene_emb_path}")
        gene_embeddings = np.load(gene_emb_path)
        if gene_embeddings.shape[0] != len(gene_list):
            logger.warning("Cached embeddings size mismatch. Regenerating...")
            gene_embeddings = generate_gene_embeddings(gene_list, summaries, model_name)
            np.save(gene_emb_path, gene_embeddings)
    else:
        gene_embeddings = generate_gene_embeddings(gene_list, summaries, model_name)
        np.save(gene_emb_path, gene_embeddings)

    # 3. Generate pathway and disease embeddings
    processed_dir = config["paths"]["processed_data"]
    metadata_path = os.path.join(processed_dir, "kg_metadata.json")

    pathway_embeddings = None
    disease_embeddings = None

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

        pathway_names = metadata.get("pathway_names", [])
        disease_names = metadata.get("disease_names", [])

        if pathway_names:
            logger.info("=" * 60)
            logger.info("Generating pathway embeddings")
            logger.info("=" * 60)
            pathway_emb_path = os.path.join(emb_dir, "pathway_embeddings.npy")
            if os.path.exists(pathway_emb_path):
                pathway_embeddings = np.load(pathway_emb_path)
            else:
                pathway_embeddings = generate_pathway_embeddings(pathway_names, model_name)
                np.save(pathway_emb_path, pathway_embeddings)

        if disease_names:
            logger.info("=" * 60)
            logger.info("Generating disease embeddings")
            logger.info("=" * 60)
            disease_emb_path = os.path.join(emb_dir, "disease_embeddings.npy")
            if os.path.exists(disease_emb_path):
                disease_embeddings = np.load(disease_emb_path)
            else:
                disease_embeddings = generate_disease_embeddings(disease_names, model_name)
                np.save(disease_emb_path, disease_embeddings)

    results = {
        "gene_embeddings": gene_embeddings,
        "pathway_embeddings": pathway_embeddings,
        "disease_embeddings": disease_embeddings,
        "gene_list": gene_list,
    }

    logger.info("Embedding generation complete")
    return results


if __name__ == "__main__":
    config = load_config()
    processed_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]

    genes_file = os.path.join(processed_dir, "selected_genes.txt")
    summaries_file = os.path.join(emb_dir, "gene_summaries.json")

    if not os.path.exists(genes_file):
        raise FileNotFoundError(f"Run preprocessing.py first. Missing: {genes_file}")
    if not os.path.exists(summaries_file):
        raise FileNotFoundError(
            f"Run data_download.py with gene summaries first. Missing: {summaries_file}"
        )

    with open(genes_file) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    results = run_embedding_generation(config, gene_list, summaries_file)
