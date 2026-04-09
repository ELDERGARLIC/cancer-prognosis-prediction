"""
Breast Cancer Prognosis Prediction — Main Pipeline

End-to-end pipeline for breast cancer survival prediction using:
    - TCGA-BRCA gene expression + clinical data
    - BioKG/DisGeNET biological knowledge graphs
    - BioBERT/BioLinkBERT LLM gene embeddings (GenePT-style)
    - FAISS vector database for gene embedding storage
    - Graph Attention Network (GAT) + Calibrated Random Forest
    - GNNExplainer / SHAP explainability

Usage:
    python main.py                    # Run full pipeline
    python main.py --stage 1          # Run only Stage 1 (data download)
    python main.py --stage 2          # Run only Stage 2 (KG + embeddings)
    python main.py --stage 3          # Run only Stage 3 (dataset)
    python main.py --stage 4          # Run only Stage 4 (training)
    python main.py --stage 5          # Run only Stage 5 (evaluation + viz)
"""

import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_stage1(config: dict) -> dict:
    """Stage 1: Data Acquisition & Preprocessing"""
    logger.info("=" * 70)
    logger.info("STAGE 1: DATA ACQUISITION & PREPROCESSING")
    logger.info("=" * 70)

    from src.data_download import run_data_download
    file_paths = run_data_download(config)

    from src.preprocessing import run_preprocessing
    results = run_preprocessing(
        config,
        expr_path=file_paths["expression"],
        clinical_path=file_paths["clinical"],
        disgenet_path=file_paths.get("disgenet"),
    )

    # Download gene summaries for LLM embeddings
    from src.data_download import download_ncbi_gene_summaries
    emb_dir = config["paths"]["embeddings"]
    download_ncbi_gene_summaries(results["selected_genes"], emb_dir)

    return {"file_paths": file_paths, "preprocessing": results}


def run_stage2(config: dict) -> dict:
    """Stage 2: Knowledge Graph Construction & LLM Embeddings"""
    logger.info("=" * 70)
    logger.info("STAGE 2: KNOWLEDGE GRAPH & LLM EMBEDDINGS")
    logger.info("=" * 70)

    processed_dir = config["paths"]["processed_data"]
    emb_dir = config["paths"]["embeddings"]

    # Load selected genes
    with open(os.path.join(processed_dir, "selected_genes.txt")) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    # Build knowledge graph
    from src.kg_construction import build_knowledge_graph
    kg_data = build_knowledge_graph(config, gene_list)

    # Generate LLM embeddings
    summaries_path = os.path.join(emb_dir, "gene_summaries.json")
    from src.llm_embeddings import run_embedding_generation
    emb_data = run_embedding_generation(config, gene_list, summaries_path)

    # Build FAISS vector store
    from src.vector_store import build_gene_vector_store
    store = build_gene_vector_store(emb_data["gene_embeddings"], gene_list, emb_dir)

    return {"kg_data": kg_data, "emb_data": emb_data, "vector_store": store}


def run_stage3(config: dict) -> dict:
    """Stage 3: Dataset Construction"""
    logger.info("=" * 70)
    logger.info("STAGE 3: DATASET CONSTRUCTION")
    logger.info("=" * 70)

    from src.dataset import build_dataset
    data = build_dataset(config)
    return data


def run_stage4(config: dict) -> dict:
    """Stage 4: Model Training"""
    logger.info("=" * 70)
    logger.info("STAGE 4: MODEL TRAINING")
    logger.info("=" * 70)

    from src.train import run_training
    results = run_training(config)
    return results


def run_stage5(config: dict, training_results: dict = None) -> dict:
    """Stage 5: Evaluation & Explainability"""
    logger.info("=" * 70)
    logger.info("STAGE 5: EVALUATION & EXPLAINABILITY")
    logger.info("=" * 70)

    # Evaluation
    from src.evaluate import run_evaluation, run_baseline_comparison, run_ablation_study
    eval_results = run_evaluation(config, training_results)

    # Run baselines and ablation if training results available
    if training_results and "dataset" in training_results:
        dataset_info = training_results["dataset"]
        dataset = dataset_info["dataset"]
        splits = dataset_info.get("splits", training_results.get("splits", []))

        if splits:
            train_idx, val_idx = splits[0]
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )

            baselines = run_baseline_comparison(dataset, train_idx, val_idx, config, device)
            eval_results["baselines"] = baselines

            ablation = run_ablation_study(dataset, train_idx, val_idx, config, device)
            eval_results["ablation"] = ablation

    # Explainability
    from src.explain import run_explainability
    if training_results:
        explain_results = run_explainability(config, training_results)
        eval_results["explainability"] = explain_results

    # Visualizations
    from src.visualize import generate_all_visualizations
    if training_results:
        viz_files = generate_all_visualizations(config, training_results)
        eval_results["visualizations"] = viz_files

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Prognosis Prediction Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--stage", type=int, default=None, help="Run specific stage (1-5)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    logger.info("Breast Cancer Prognosis Prediction Pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")

    if args.stage is None or args.stage == 1:
        run_stage1(config)

    if args.stage is None or args.stage == 2:
        run_stage2(config)

    if args.stage is None or args.stage == 3:
        run_stage3(config)

    training_results = None
    if args.stage is None or args.stage == 4:
        training_results = run_stage4(config)

    if args.stage is None or args.stage == 5:
        run_stage5(config, training_results)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
