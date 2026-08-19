import argparse
import json
import os
from pathlib import Path

import torch

from llm_wsgtm.experiment import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = {
    "dataset_name": "NeurIPS",
    "cache_path": str(PROJECT_ROOT / "datasets"),
    "runs_dir": str(PROJECT_ROOT / "runs"),
    "experiment_prefix": "llm-wsgtm",
    "num_topics": 60,
    "num_top_words": 20,
    "top_n_for_metrics": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "normalize_embeddings": True,
    "document_embedding_model": str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"),
    "epochs": 300,
    "learning_rate": 6e-4,
    "verbose": True,
    "log_interval": 1,
    "document_topic_alpha": 0.8,
    "topic_word_alpha": 0.6,
    "theta_temperature": 0.4,
    "low_memory": False,
    "low_memory_batch_size": None,
    "use_gensim_cv": True,
    "semantic_labeling_enable": True,
    "coverage_optimizer_enable": True,
    "anchor_prior_enable": True,
    "coverage_iterations": 1,
    "coverage_swaps_per_topic": 3,
    "coverage_candidates_per_topic": 400,
    "coverage_min_df": 5,
    "coverage_max_df_ratio": 0.20,
    "coverage_gamma": 0.10,
    "coverage_tau": 0.10,
    "coverage_global_exclusive": False,
    "anchor_prior_after_epochs": 5,
    "anchor_prior_weight": 0.05,
    "anchor_prior_top_k": 1000,
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    "ollama_model": "llama3:8b",
    "label_documents_per_topic": 3,
    "label_max_snippet_chars": 400,
    "seed": 2024,
}


def load_json(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Run LLM-WSGTM experiments")
    parser.add_argument("--config")
    parser.add_argument("--overrides")
    args = parser.parse_args()
    config = dict(BASE_CONFIG)
    config.update(load_json(args.config))
    if args.overrides:
        config.update(json.loads(args.overrides))
    run_dir, evaluation = run_experiment(config)
    print(run_dir)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
