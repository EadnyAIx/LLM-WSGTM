import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from topmost import Preprocess

from .coverage_optimizer import optimize_coverage
from .data import TopicDataset
from .metrics import coherence_cv, document_coverage, document_npmi, purity_and_nmi, topic_diversity, topic_overlap_jaccard
from .model import LLMWSGTM
from .semantic_labeler import label_topics


LOGGER = logging.getLogger("LLM-WSGTM.Experiment")


def load_dataset(dataset_name, cache_path):
    os.makedirs(cache_path, exist_ok=True)
    dataset_dir = os.path.join(cache_path, dataset_name)
    if os.path.isdir(dataset_dir) and any(os.scandir(dataset_dir)):
        return dataset_dir
    from topmost.data import download_dataset
    download_dataset(dataset_name, cache_path=cache_path)
    return dataset_dir


def read_dataset(dataset_dir):
    text_path = os.path.join(dataset_dir, "train_texts.txt")
    with open(text_path, "r", encoding="utf-8") as handle:
        documents = [line.strip() for line in handle if line.strip()]
    labels = None
    label_path = os.path.join(dataset_dir, "train_labels.txt")
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as handle:
            raw = [line.strip() for line in handle if line.strip()]
        try:
            labels = np.asarray([int(value) for value in raw], dtype=np.int64)
        except Exception:
            mapping = {value: index for index, value in enumerate(sorted(set(raw)))}
            labels = np.asarray([mapping[value] for value in raw], dtype=np.int64)
    return documents, labels


def make_run_directory(root, prefix):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_experiment(config):
    np.random.seed(int(config.get("seed", 2024)))
    torch.manual_seed(int(config.get("seed", 2024)))
    run_dir = make_run_directory(config["runs_dir"], config.get("experiment_prefix", "llm-wsgtm"))
    save_json(os.path.join(run_dir, "config.json"), config)
    dataset_dir = load_dataset(config["dataset_name"], config["cache_path"])
    documents, labels = read_dataset(dataset_dir)
    preprocess = Preprocess(verbose=False)

    class EvaluationEncoder:
        def encode(self, *args, **kwargs):
            raise RuntimeError("preset embeddings are required")

    evaluation_dataset = TopicDataset(
        documents,
        EvaluationEncoder(),
        preprocess,
        batch_size=len(documents),
        device="cpu",
        low_memory=False,
        preset_document_embeddings=np.zeros((len(documents), 1), dtype=np.float32),
    )
    bow = evaluation_dataset.train_bow.toarray().astype("float32")
    vocab = evaluation_dataset.vocab

    model = LLMWSGTM(
        num_topics=int(config["num_topics"]),
        preprocess=Preprocess(verbose=False),
        num_top_words=int(config.get("num_top_words", 20)),
        device=config.get("device"),
        normalize_embeddings=bool(config.get("normalize_embeddings", True)),
        document_embedding_model=str(config["document_embedding_model"]),
        document_topic_alpha=float(config.get("document_topic_alpha", 0.8)),
        topic_word_alpha=float(config.get("topic_word_alpha", 0.6)),
        theta_temperature=float(config.get("theta_temperature", 0.4)),
        low_memory=bool(config.get("low_memory", False)),
        low_memory_batch_size=config.get("low_memory_batch_size"),
        verbose=bool(config.get("verbose", True)),
        log_interval=int(config.get("log_interval", 1)),
        topic_orthogonality_weight=float(config.get("topic_orthogonality_weight", 0.12)),
        word_coherence_weight=float(config.get("word_coherence_weight", 0.30)),
        document_entropy_weight=float(config.get("document_entropy_weight", 0.03)),
        coherence_top_k=int(config.get("coherence_top_k", 20)),
        beta_uniform_mix=float(config.get("beta_uniform_mix", 0.0)),
        gradient_clip=float(config.get("gradient_clip", 5.0)),
        anchor_prior_enable=bool(config.get("anchor_prior_enable", True)),
        anchor_prior_after_epochs=int(config.get("anchor_prior_after_epochs", 5)),
        anchor_prior_weight=float(config.get("anchor_prior_weight", 0.05)),
        anchor_prior_top_k=int(config.get("anchor_prior_top_k", 1000)),
        anchor_llm_model=str(config.get("ollama_model", "llama3:8b")),
        anchor_llm_host=config.get("ollama_host"),
    )
    _, theta = model.fit_transform(documents, epochs=int(config.get("epochs", 300)), learning_rate=float(config.get("learning_rate", 6e-4)))
    beta = model.get_beta()
    if beta.shape[1] != len(vocab):
        raise ValueError("evaluation vocabulary does not match model vocabulary")
    metric_top_n = int(config.get("top_n_for_metrics", 10))
    initial_indices = []
    for topic_id in range(beta.shape[0]):
        indices = np.argsort(-beta[topic_id])[:metric_top_n]
        initial_indices.append(indices.tolist())
    coverage_report = None
    top_indices = initial_indices
    if bool(config.get("coverage_optimizer_enable", True)):
        top_indices, coverage_report = optimize_coverage(
            initial_indices,
            csr_matrix(bow),
            vocab,
            metric_top_n,
            gamma=float(config.get("coverage_gamma", 0.10)),
            tau=float(config.get("coverage_tau", 0.10)),
            min_df=int(config.get("coverage_min_df", 5)),
            max_df_ratio=float(config.get("coverage_max_df_ratio", 0.20)),
            candidates_per_topic=int(config.get("coverage_candidates_per_topic", 400)),
            swaps_per_topic=int(config.get("coverage_swaps_per_topic", 3)),
            iterations=int(config.get("coverage_iterations", 1)),
            global_exclusive=bool(config.get("coverage_global_exclusive", False)),
            seed=int(config.get("seed", 2024)),
        )
    output_top_n = int(config.get("num_top_words", 20))
    export_indices = []
    for topic_id in range(beta.shape[0]):
        selected = list(top_indices[topic_id])
        for index in np.argsort(-beta[topic_id]):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= output_top_n:
                break
        export_indices.append(selected[:output_top_n])
    with open(os.path.join(run_dir, "topic_top_words.txt"), "w", encoding="utf-8") as handle:
        for topic_id, indices in enumerate(export_indices):
            handle.write(f"{topic_id}\t{' '.join(vocab[index] for index in indices)}\n")
    torch.save(torch.as_tensor(beta), os.path.join(run_dir, "beta.pt"))
    torch.save(torch.as_tensor(theta), os.path.join(run_dir, "theta.pt"))
    binary_bow = (bow > 0).astype(np.int32)
    tokenized_documents = [[vocab[index] for index in np.where(binary_bow[row] == 1)[0]] for row in range(binary_bow.shape[0])]
    topic_words = [[vocab[index] for index in indices[:metric_top_n]] for indices in top_indices]
    coherence = coherence_cv(topic_words, tokenized_documents) if bool(config.get("use_gensim_cv", True)) else None
    if coherence is None:
        coherence = document_npmi([indices[:metric_top_n] for indices in top_indices], binary_bow.astype(np.float32))
    purity, nmi = purity_and_nmi(theta, labels)
    theta_coverage = document_coverage(theta, float(config.get("coverage_tau", 0.10)), float(config.get("coverage_gamma", 0.10)))
    if coverage_report is not None:
        reported_coverage = coverage_report["after"]
    else:
        reported_coverage = theta_coverage
    evaluation = {
        f"TopicDiversity@{metric_top_n}": topic_diversity(top_indices, metric_top_n),
        "Coherence": coherence,
        "Purity": purity,
        "NMI": nmi,
        "DocumentCoverage": reported_coverage["relaxed_soft"],
        "DocumentCoverageRelaxedHard": reported_coverage["relaxed_hard"],
        "DocumentCoverageStrictSoft": reported_coverage["strict_soft"],
        "DocumentCoverageStrictHard": reported_coverage["strict_hard"],
        "Diagnostics": {
            "model": "LLM-WSGTM",
            "topics": beta.shape[0],
            "vocabulary": beta.shape[1],
            "documents": len(documents),
            "average_topic_jaccard": topic_overlap_jaccard(top_indices, metric_top_n),
            "coverage_optimizer_enabled": bool(config.get("coverage_optimizer_enable", True)),
            "anchor_prior_enabled": bool(config.get("anchor_prior_enable", True)),
        },
    }
    save_json(os.path.join(run_dir, "evaluation.json"), evaluation)
    if coverage_report is not None:
        save_json(os.path.join(run_dir, "coverage_report.json"), coverage_report)
    if bool(config.get("semantic_labeling_enable", True)):
        label_input = [[vocab[index] for index in indices] for indices in export_indices]
        labels_output = label_topics(
            label_input,
            documents,
            theta,
            documents_per_topic=int(config.get("label_documents_per_topic", 3)),
            host=str(config.get("ollama_host", "http://127.0.0.1:11434")),
            model=str(config.get("ollama_model", "llama3:8b")),
            max_snippet_chars=int(config.get("label_max_snippet_chars", 400)),
            output_json=os.path.join(run_dir, "topic_labels.json"),
        )
        with open(os.path.join(run_dir, "topic_labels.txt"), "w", encoding="utf-8") as handle:
            for item in labels_output.get("results", []):
                handle.write(f"[{item['topic_id']:02d}] {item['label']}\n{item['description']}\n\n")
    model.save(os.path.join(run_dir, "llm_wsgtm.pt"))
    return run_dir, evaluation
