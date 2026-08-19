import logging
from typing import Callable, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_top_words(beta, vocab, num_top_words, verbose=False):
    topics = []
    vocab_array = np.array(vocab)
    for topic_id, distribution in enumerate(beta):
        words = vocab_array[np.argsort(distribution)][:-(num_top_words + 1):-1]
        text = " ".join(words)
        topics.append(text)
        if verbose:
            print(f"Topic {topic_id}: {text}")
    return topics


class DocumentEncoder:
    def __init__(self, model="all-MiniLM-L6-v2", device="cpu", normalize_embeddings=False, verbose=False):
        self.verbose = verbose
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model, device=device) if isinstance(model, str) else model

    def encode(self, documents: List[str]):
        return self.model.encode(documents, show_progress_bar=self.verbose, normalize_embeddings=self.normalize_embeddings)


class BatchIterator:
    def __init__(self, bag_of_words, document_embeddings, batch_size, device, low_memory=False, shuffle=True):
        self.document_embeddings = torch.tensor(document_embeddings, dtype=torch.float32)
        self.low_memory = low_memory
        self.shuffle = shuffle
        self.batch_size = batch_size
        if low_memory:
            self.bag_of_words = bag_of_words
        else:
            self.bag_of_words = torch.tensor(bag_of_words.toarray(), dtype=torch.float32).to(device)
            self.document_embeddings = self.document_embeddings.to(device)

    def __iter__(self):
        sample_count = self.bag_of_words.shape[0]
        indices = np.arange(sample_count)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, sample_count, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.low_memory:
                bag = torch.tensor(self.bag_of_words[batch_indices].toarray(), dtype=torch.float32)
            else:
                bag = self.bag_of_words[batch_indices]
            yield bag, self.document_embeddings[batch_indices]


class TopicDataset:
    def __init__(self, documents: List[str], document_encoder: Callable, preprocess: Callable, batch_size=200, device="cpu", low_memory=False, preset_document_embeddings=None):
        result = preprocess.preprocess(documents)
        self.train_bow = result["train_bow"]
        self.vocab = result["vocab"]
        self.vocab_size = len(self.vocab)
        self.document_embeddings = document_encoder.encode(documents) if preset_document_embeddings is None else preset_document_embeddings
        self.document_embedding_size = self.document_embeddings.shape[1]
        self.dataloader = BatchIterator(self.train_bow, self.document_embeddings, batch_size, device, low_memory)


def is_fitted(model):
    return model.beta is not None


def require_fitted(model):
    if not is_fitted(model):
        raise ValueError(f"This {type(model).__name__} instance is not fitted yet.")


class ProjectLogger:
    def __init__(self, level):
        self.logger = logging.getLogger("LLM-WSGTM")
        self.set_level(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
            self.logger.addHandler(handler)
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            self.logger.setLevel(level)
