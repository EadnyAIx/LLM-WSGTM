from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import visualization
from .anchor_prior import AnchorPriorBuilder
from .core import TopicTransportCore
from .data import DocumentEncoder, ProjectLogger, TopicDataset, get_top_words, is_fitted, require_fitted
from .llm_client import OllamaClient


LOGGER = ProjectLogger("WARNING")


class LLMWSGTM:
    model_name = "LLM-WSGTM"

    def __init__(self, num_topics, preprocess=None, num_top_words=15, device=None, normalize_embeddings=False, document_embedding_model="all-MiniLM-L6-v2", document_topic_alpha=3.0, topic_word_alpha=2.0, theta_temperature=1.0, low_memory=False, low_memory_batch_size=None, verbose=True, log_interval=10, topic_orthogonality_weight=0.12, word_coherence_weight=0.30, document_entropy_weight=0.03, coherence_top_k=20, beta_uniform_mix=0.0, gradient_clip=5.0, anchor_prior_enable=False, anchor_prior_after_epochs=5, anchor_prior_weight=0.05, anchor_prior_top_k=1000, anchor_llm_model="llama3.1:8b", anchor_llm_host=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_topics = int(num_topics)
        self.num_top_words = int(num_top_words)
        self.document_embedding_model = document_embedding_model
        self.normalize_embeddings = normalize_embeddings
        self.low_memory = low_memory
        self.low_memory_batch_size = low_memory_batch_size
        self.beta = None
        self.train_theta = None
        self.document_encoder = None
        self.gradient_clip = float(gradient_clip)
        self.anchor_prior_enable = bool(anchor_prior_enable)
        self.anchor_prior_after_epochs = int(anchor_prior_after_epochs)
        self.anchor_prior_weight = float(anchor_prior_weight)
        self.anchor_prior_top_k = int(anchor_prior_top_k)
        self.anchor_llm_model = str(anchor_llm_model)
        self.anchor_llm_host = anchor_llm_host
        self._anchor_prior_installed = False
        self.core = TopicTransportCore(
            num_topics=self.num_topics,
            theta_temperature=theta_temperature,
            document_topic_alpha=document_topic_alpha,
            topic_word_alpha=topic_word_alpha,
            topic_orthogonality_weight=topic_orthogonality_weight,
            word_coherence_weight=word_coherence_weight,
            document_entropy_weight=document_entropy_weight,
            coherence_top_k=coherence_top_k,
            beta_uniform_mix=beta_uniform_mix,
            anchor_prior_weight=0.0,
            anchor_prior_top_k=self.anchor_prior_top_k,
        )
        if preprocess is None:
            from topmost import Preprocess
            self.preprocess = Preprocess(verbose=verbose)
        else:
            self.preprocess = preprocess
        self.log_interval = int(log_interval)
        self.verbose = bool(verbose)
        LOGGER.set_level("DEBUG" if verbose else "WARNING")
        LOGGER.info(f"use device: {self.device}")

    def set_anchor_prior(self, beta_prior, weight=None, top_k=None, epsilon=None):
        self.core.set_beta_prior(beta_prior)
        if weight is not None:
            self.core.anchor_prior_weight = float(weight)
        if top_k is not None:
            self.core.anchor_prior_top_k = int(top_k)
        if epsilon is not None:
            self.core.epsilon = float(epsilon)
        self._anchor_prior_installed = True

    def make_optimizer(self, learning_rate):
        return torch.optim.Adam(self.core.parameters(), lr=learning_rate)

    def fit(self, documents, epochs=200, learning_rate=0.002, preset_document_embeddings=None):
        self.fit_transform(documents, epochs, learning_rate, preset_document_embeddings)
        return self

    def fit_transform(self, documents, epochs=200, learning_rate=0.002, preset_document_embeddings=None):
        data_size = len(documents)
        if self.low_memory:
            if self.low_memory_batch_size is None:
                raise ValueError("low_memory_batch_size is required when low_memory=True")
            batch_size = self.low_memory_batch_size
            dataset_device = "cpu"
        else:
            batch_size = data_size
            dataset_device = self.device
        fitted = bool(is_fitted(self))
        self.document_encoder = DocumentEncoder(self.document_embedding_model, self.device, self.normalize_embeddings, self.verbose)
        dataset = TopicDataset(documents, self.document_encoder, self.preprocess, batch_size, dataset_device, self.low_memory, preset_document_embeddings)
        self.train_document_embeddings = torch.as_tensor(dataset.document_embeddings)
        if not self.low_memory:
            self.train_document_embeddings = self.train_document_embeddings.to(self.device)
        if not fitted:
            self.core.initialize(dataset.vocab_size, dataset.document_embedding_size)
        else:
            previous_vocab = getattr(self, "vocab", None)
            self.core.initialize(dataset.vocab_size, dataset.document_embedding_size, fitted=True, previous_vocab=previous_vocab, vocab=dataset.vocab)
        self.vocab = dataset.vocab
        self.core = self.core.to(self.device)
        optimizer = self.make_optimizer(learning_rate)
        coherence_target = float(self.core.word_coherence_weight)
        orthogonality_start = float(self.core.topic_orthogonality_weight)
        mix_target = float(self.core.beta_uniform_mix) if self.core.beta_uniform_mix > 0 else 0.04
        self.core.train()
        for epoch in tqdm(range(1, epochs + 1), desc="Training LLM-WSGTM"):
            self._maybe_install_anchor_prior(epoch)
            progress = epoch / max(1, epochs)
            warm = min(progress / 0.2, 1.0)
            cool = max(1.0 - max(progress - 0.5, 0.0) / 0.5, 0.0)
            mix = min(progress / 0.8, 1.0) * mix_target
            self.core.word_coherence_weight = coherence_target * warm
            self.core.topic_orthogonality_weight = orthogonality_start * (0.6 + 0.4 * cool)
            self.core.beta_uniform_mix = mix
            accumulated = defaultdict(float)
            for bag, embeddings in dataset.dataloader:
                if self.low_memory:
                    bag = bag.to(self.device)
                    embeddings = embeddings.to(self.device)
                result = self.core(bag, embeddings)
                optimizer.zero_grad()
                result["loss"].backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.core.parameters(), self.gradient_clip)
                optimizer.step()
                for key, value in result.items():
                    accumulated[key] += float(value) * bag.shape[0]
            if epoch % self.log_interval == 0:
                message = f"Epoch: {epoch:03d}"
                for key in sorted(accumulated):
                    message += f" {key}: {accumulated[key] / data_size:.4f}"
                LOGGER.info(message)
        self.beta = self.get_beta()
        self.top_words = self.get_top_words(self.num_top_words)
        self.train_theta = self.transform(document_embeddings=self.train_document_embeddings)
        return self.top_words, self.train_theta

    def _maybe_install_anchor_prior(self, epoch):
        if not self.anchor_prior_enable or self._anchor_prior_installed or epoch < self.anchor_prior_after_epochs:
            return
        try:
            with torch.no_grad():
                current_beta = self.core.get_beta().detach().cpu().numpy()
            seed_topics = [text.split() for text in get_top_words(current_beta, self.vocab, 20, False)]
            client = OllamaClient(model=self.anchor_llm_model, base_url=self.anchor_llm_host)
            client.ensure_model()
            prior = AnchorPriorBuilder(self.vocab).build(seed_topics, client)
            self.set_anchor_prior(prior, weight=self.anchor_prior_weight, top_k=self.anchor_prior_top_k)
        except Exception as error:
            LOGGER.warning(f"anchor prior skipped: {error}")
            self._anchor_prior_installed = True

    def transform(self, documents=None, document_embeddings=None):
        if documents is None and document_embeddings is None:
            raise ValueError("documents or document_embeddings is required")
        if document_embeddings is None:
            if self.document_encoder is None:
                raise ValueError("document encoder is unavailable")
            document_embeddings = torch.as_tensor(self.document_encoder.encode(documents))
            if not self.low_memory:
                document_embeddings = document_embeddings.to(self.device)
        elif not isinstance(document_embeddings, torch.Tensor):
            document_embeddings = torch.as_tensor(document_embeddings, dtype=torch.float32)
        if document_embeddings.device != self.core.topic_embeddings.device:
            document_embeddings = document_embeddings.to(self.core.topic_embeddings.device)
        train_embeddings = self.train_document_embeddings.to(document_embeddings.device)
        with torch.no_grad():
            self.core.eval()
            return self.core.get_theta(document_embeddings, train_embeddings).detach().cpu().numpy()

    def get_beta(self):
        return self.core.get_beta().detach().cpu().numpy()

    def get_top_words(self, num_top_words=15, verbose=None):
        return get_top_words(self.get_beta(), self.vocab, num_top_words, self.verbose if verbose is None else verbose)

    @property
    def topic_embeddings(self):
        return self.core.topic_embeddings.detach().cpu().numpy()

    @property
    def word_embeddings(self):
        return self.core.word_embeddings.detach().cpu().numpy()

    @property
    def document_topic_transport(self):
        return self.core.get_document_topic_transport(self.train_document_embeddings)

    def save(self, path):
        require_fitted(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value for key, value in self.__dict__.items() if key != "document_encoder"}
        torch.save({"instance_dict": state}, path)

    @classmethod
    def from_pretrained(cls, path, preprocess=None, low_memory=None, low_memory_batch_size=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device, weights_only=False)
        instance_dict = state["instance_dict"]
        instance_dict["device"] = device
        if preprocess is not None:
            instance_dict["preprocess"] = preprocess
        if low_memory is not None:
            instance_dict["low_memory"] = low_memory
            instance_dict["low_memory_batch_size"] = low_memory_batch_size
        instance = cls.__new__(cls)
        instance.__dict__.update(instance_dict)
        instance.document_encoder = DocumentEncoder(instance_dict["document_embedding_model"], device=device, normalize_embeddings=instance_dict["normalize_embeddings"])
        LOGGER.set_level("DEBUG" if instance.verbose else "WARNING")
        return instance

    def get_topic(self, topic_index, num_top_words=5):
        words = self.top_words[topic_index].split()[:num_top_words]
        scores = np.sort(self.beta[topic_index])[:-(num_top_words + 1):-1]
        return tuple(zip(words, scores))

    def get_topic_weights(self):
        return self.document_topic_transport.sum(0)

    def topic_activity_over_time(self, time_slices):
        activity = self.document_topic_transport * self.document_topic_transport.shape[0]
        if len(time_slices) != activity.shape[0]:
            raise ValueError("time_slices length mismatch")
        frame = pd.DataFrame(activity)
        frame["time_slices"] = time_slices
        return frame.groupby("time_slices").mean().to_numpy().transpose()

    def visualize_topics(self, **kwargs):
        return visualization.visualize_topics(self, **kwargs)

    def visualize_topic_hierarchy(self, **kwargs):
        return visualization.visualize_hierarchy(self, **kwargs)

    def visualize_topic_activity(self, **kwargs):
        return visualization.visualize_activity(self, **kwargs)

    def visualize_topic_weights(self, **kwargs):
        return visualization.visualize_topic_weights(self, **kwargs)
