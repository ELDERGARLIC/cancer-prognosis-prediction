"""
Stage 2.3 (continued): Vector Database for Gene Embeddings

Stores and retrieves LLM gene embeddings using FAISS for efficient
similarity search and retrieval.

References:
    - FAISS: Johnson et al., 2019
    - GenePT embedding storage: Chen & Zou, 2023
"""

import os
import json
import logging
import numpy as np
import faiss
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class GeneEmbeddingStore:
    """FAISS-based vector store for gene embeddings.

    Supports:
        - Storing gene embeddings with metadata
        - k-NN similarity search
        - Batch retrieval by gene name
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = None
        self.gene_names = []
        self.metadata = {}

    def build_index(
        self,
        embeddings: np.ndarray,
        gene_names: list,
        metadata: dict = None,
        use_ivf: bool = False,
        nlist: int = 10,
    ):
        """Build FAISS index from gene embeddings.

        Args:
            embeddings: [num_genes, embedding_dim] array.
            gene_names: List of gene names.
            metadata: Optional dict of gene metadata.
            use_ivf: Use IVF index for large datasets.
            nlist: Number of Voronoi cells for IVF.
        """
        assert embeddings.shape[0] == len(gene_names), "Mismatch between embeddings and gene names"
        assert embeddings.shape[1] == self.embedding_dim, (
            f"Embedding dim mismatch: {embeddings.shape[1]} vs {self.embedding_dim}"
        )

        embeddings = embeddings.astype(np.float32)
        self.gene_names = list(gene_names)
        self.metadata = metadata or {}

        if use_ivf and len(gene_names) > 1000:
            # IVF index for large gene sets
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
            self.index.add(embeddings)
            logger.info(f"Built IVF index with {nlist} cells, {self.index.ntotal} vectors")
        else:
            # Flat L2 index for exact search
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings)
            logger.info(f"Built flat L2 index with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list:
        """Find k most similar genes to query embedding.

        Args:
            query_embedding: [embedding_dim] or [1, embedding_dim] query vector.
            k: Number of nearest neighbors.

        Returns:
            List of (gene_name, distance) tuples.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.gene_names):
                results.append((self.gene_names[idx], float(dist)))

        return results

    def search_by_gene(self, gene_name: str, k: int = 10) -> list:
        """Find k most similar genes to a given gene.

        Args:
            gene_name: Query gene symbol.
            k: Number of neighbors (including the query gene itself).

        Returns:
            List of (gene_name, distance) tuples.
        """
        if gene_name not in self.gene_names:
            logger.warning(f"Gene {gene_name} not found in store")
            return []

        idx = self.gene_names.index(gene_name)
        query = self.get_embedding(gene_name)
        return self.search(query, k=k)

    def get_embedding(self, gene_name: str) -> np.ndarray:
        """Retrieve embedding for a specific gene.

        Returns:
            [embedding_dim] numpy array.
        """
        if gene_name not in self.gene_names:
            raise KeyError(f"Gene {gene_name} not found in store")

        idx = self.gene_names.index(gene_name)
        return self.index.reconstruct(idx)

    def get_embeddings_batch(self, gene_names: list) -> np.ndarray:
        """Retrieve embeddings for multiple genes.

        Returns:
            [len(gene_names), embedding_dim] numpy array.
        """
        indices = []
        for gene in gene_names:
            if gene in self.gene_names:
                indices.append(self.gene_names.index(gene))
            else:
                logger.warning(f"Gene {gene} not found, using zero vector")
                indices.append(-1)

        embeddings = np.zeros((len(gene_names), self.embedding_dim), dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx >= 0:
                embeddings[i] = self.index.reconstruct(idx)

        return embeddings

    def save(self, output_dir: str):
        """Save FAISS index and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_dir, "gene_embeddings.faiss"))
        with open(os.path.join(output_dir, "gene_names.json"), "w") as f:
            json.dump(self.gene_names, f)
        if self.metadata:
            with open(os.path.join(output_dir, "gene_metadata.json"), "w") as f:
                json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved vector store to {output_dir}")

    def load(self, input_dir: str):
        """Load FAISS index and metadata."""
        self.index = faiss.read_index(os.path.join(input_dir, "gene_embeddings.faiss"))
        with open(os.path.join(input_dir, "gene_names.json")) as f:
            self.gene_names = json.load(f)
        metadata_path = os.path.join(input_dir, "gene_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        self.embedding_dim = self.index.d
        logger.info(f"Loaded vector store: {self.index.ntotal} vectors, dim={self.embedding_dim}")


def build_gene_vector_store(
    gene_embeddings: np.ndarray,
    gene_list: list,
    output_dir: str,
) -> GeneEmbeddingStore:
    """Build and save the gene embedding vector store.

    Args:
        gene_embeddings: [num_genes, embedding_dim] embeddings.
        gene_list: List of gene symbols.
        output_dir: Directory to save the FAISS index.

    Returns:
        Initialized GeneEmbeddingStore.
    """
    store = GeneEmbeddingStore(embedding_dim=gene_embeddings.shape[1])
    store.build_index(gene_embeddings, gene_list)
    store.save(output_dir)

    # Quick sanity check
    if len(gene_list) > 0:
        test_gene = gene_list[0]
        neighbors = store.search_by_gene(test_gene, k=5)
        logger.info(f"Sanity check - neighbors of {test_gene}:")
        for name, dist in neighbors:
            logger.info(f"  {name}: distance={dist:.4f}")

    return store


if __name__ == "__main__":
    config = load_config()
    emb_dir = config["paths"]["embeddings"]

    gene_emb_path = os.path.join(emb_dir, "gene_embeddings.npy")
    genes_file = os.path.join(config["paths"]["processed_data"], "selected_genes.txt")

    if not os.path.exists(gene_emb_path):
        raise FileNotFoundError(f"Run llm_embeddings.py first. Missing: {gene_emb_path}")

    gene_embeddings = np.load(gene_emb_path)
    with open(genes_file) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    store = build_gene_vector_store(gene_embeddings, gene_list, emb_dir)

    # Demo: find similar genes
    print("\nSimilarity search demo:")
    for gene in ["BRCA1", "TP53", "ERBB2"]:
        if gene in gene_list:
            results = store.search_by_gene(gene, k=5)
            print(f"\n{gene} neighbors:")
            for name, dist in results:
                print(f"  {name}: {dist:.4f}")
