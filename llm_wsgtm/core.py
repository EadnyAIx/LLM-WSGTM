import torch
from torch import nn
import torch.nn.functional as F

from .math_utils import pairwise_squared_distance
from .transport import SinkhornTransport


def normalize_rows(x, eps=1e-12):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    denom = x.sum(dim=-1, keepdim=True)
    denom = torch.where(denom <= eps, torch.ones_like(denom), denom)
    x = torch.clamp(x / denom, min=eps)
    return x / x.sum(dim=-1, keepdim=True)


class TopicTransportCore(nn.Module):
    def __init__(self, num_topics, theta_temperature=1.0, document_topic_alpha=3.0, topic_word_alpha=2.0, topic_orthogonality_weight=0.12, word_coherence_weight=0.30, document_entropy_weight=0.03, coherence_top_k=20, beta_uniform_mix=0.0, anchor_prior_weight=0.0, anchor_prior_top_k=1000):
        super().__init__()
        self.num_topics = num_topics
        self.theta_temperature = theta_temperature
        self.document_topic_alpha = document_topic_alpha
        self.topic_word_alpha = topic_word_alpha
        self.topic_orthogonality_weight = float(topic_orthogonality_weight)
        self.word_coherence_weight = float(word_coherence_weight)
        self.document_entropy_weight = float(document_entropy_weight)
        self.coherence_top_k = int(coherence_top_k)
        self.beta_uniform_mix = float(beta_uniform_mix)
        self.anchor_prior_weight = float(anchor_prior_weight)
        self.anchor_prior_top_k = int(anchor_prior_top_k)
        self.epsilon = 1e-12
        self.beta_prior = None

    def initialize(self, vocab_size, embedding_size, fitted=False, previous_vocab=None, vocab=None):
        if fitted:
            topic_embeddings = self.topic_embeddings.data
            topic_weights = self.topic_weights.data
            del self.topic_weights
        else:
            topic_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty((self.num_topics, embedding_size))), dim=1)
            topic_weights = (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)
        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)
        word_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty(vocab_size, embedding_size)), dim=1)
        if fitted:
            previous_word_embeddings = self.word_embeddings.data
            previous_word_weights = F.softmax(self.word_weights.data, dim=0)
            del self.word_embeddings
            del self.word_weights
            word_weights = torch.zeros(vocab_size, 1)
            hits = 0
            previous_index = {word: i for i, word in enumerate(previous_vocab or [])}
            for i, word in enumerate(vocab or []):
                if word in previous_index:
                    j = previous_index[word]
                    word_embeddings[i] = previous_word_embeddings[j]
                    word_weights[i] = previous_word_weights[j]
                    hits += 1
            if hits > 0 and hits < vocab_size:
                remainder = (1.0 - word_weights.sum()).clamp_min(0.0) / (vocab_size - hits)
                word_weights[word_weights.squeeze(-1) == 0] = remainder
                word_weights = torch.log(word_weights.clamp_min(1e-8))
                word_weights = word_weights - word_weights.mean()
            elif hits == 0:
                word_weights = (torch.ones(vocab_size) / vocab_size).unsqueeze(1)
        else:
            word_weights = (torch.ones(vocab_size) / vocab_size).unsqueeze(1)
        self.word_embeddings = nn.Parameter(word_embeddings)
        self.word_weights = nn.Parameter(word_weights)
        self.document_topic_transport = SinkhornTransport(self.document_topic_alpha, init_target=self.topic_weights)
        self.topic_word_transport = SinkhornTransport(self.topic_word_alpha, init_target=self.word_weights)

    def set_beta_prior(self, beta_prior):
        prior = torch.as_tensor(beta_prior, dtype=torch.float32)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
        self.beta_prior = prior.detach()

    def get_document_topic_transport(self, document_embeddings):
        topics = self.topic_embeddings.detach().to(document_embeddings.device)
        _, transport = self.document_topic_transport(document_embeddings, topics)
        return transport.detach().cpu().numpy()

    @torch.no_grad()
    def get_beta(self):
        _, transport = self.topic_word_transport(self.topic_embeddings, self.word_embeddings)
        beta = normalize_rows(transport * transport.shape[0], self.epsilon)
        if self.beta_uniform_mix > 0:
            vocab_size = beta.shape[1]
            beta = normalize_rows((1.0 - self.beta_uniform_mix) * beta + self.beta_uniform_mix * (torch.ones_like(beta) / vocab_size), self.epsilon)
        return beta

    @torch.no_grad()
    def get_theta(self, document_embeddings, train_document_embeddings):
        topics = self.topic_embeddings.detach().to(document_embeddings.device)
        distance = pairwise_squared_distance(document_embeddings, topics)
        train_distance = pairwise_squared_distance(train_document_embeddings, topics)
        score = torch.exp(-distance / self.theta_temperature)
        train_score = torch.exp(-train_distance / self.theta_temperature)
        theta = score / (train_score.sum(0) + self.epsilon)
        return normalize_rows(theta, self.epsilon)

    def _orthogonality_loss(self):
        topics = F.normalize(self.topic_embeddings, dim=1)
        gram = topics @ topics.t()
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        off_diagonal = gram - identity
        off_diagonal = off_diagonal - torch.diag(torch.diag(off_diagonal))
        return (off_diagonal ** 2).mean()

    def _word_coherence_loss(self, beta):
        topic_count, vocab_size = beta.shape
        top_k = min(self.coherence_top_k, vocab_size)
        words = F.normalize(self.word_embeddings, dim=1)
        total = beta.new_tensor(0.0)
        for topic_id in range(topic_count):
            indices = torch.topk(beta[topic_id], k=top_k).indices
            selected = words.index_select(0, indices)
            similarity = (selected @ selected.t()).clamp(-1.0, 1.0)
            total = total + (similarity.sum() - torch.diagonal(similarity).sum()) / (top_k * (top_k - 1) + 1e-8)
        return -(total / max(topic_count, 1))

    def _entropy_loss(self, theta):
        theta = torch.clamp(theta, min=self.epsilon)
        entropy = -(theta * theta.log()).sum(dim=1).mean()
        return -entropy

    def _anchor_prior_loss(self, beta):
        if self.beta_prior is None or self.anchor_prior_weight <= 0:
            return beta.new_tensor(0.0)
        prior = self.beta_prior.to(beta.device)
        top_k = max(1, min(self.anchor_prior_top_k, prior.size(1)))
        indices = torch.topk(prior, k=top_k, dim=1).indices
        prior_values = prior.gather(1, indices).clamp_min(self.epsilon)
        beta_log_values = beta.gather(1, indices).clamp_min(self.epsilon).log()
        return -(prior_values * beta_log_values).sum(dim=1).mean()

    def forward(self, bag_of_words, document_embeddings):
        loss_document_topic, document_topic = self.document_topic_transport(document_embeddings, self.topic_embeddings)
        loss_topic_word, topic_word = self.topic_word_transport(self.topic_embeddings, self.word_embeddings)
        theta = normalize_rows(document_topic * document_topic.shape[0], self.epsilon)
        beta = normalize_rows(topic_word * topic_word.shape[0], self.epsilon)
        if self.beta_uniform_mix > 0:
            vocab_size = beta.shape[1]
            beta = normalize_rows((1.0 - self.beta_uniform_mix) * beta + self.beta_uniform_mix * (torch.ones_like(beta) / vocab_size), self.epsilon)
        reconstruction = torch.matmul(theta, beta).clamp_min(self.epsilon)
        reconstruction_loss = -(bag_of_words * reconstruction.log()).sum(dim=1).mean()
        transport_loss = loss_document_topic + loss_topic_word
        orthogonality_loss = self._orthogonality_loss()
        coherence_loss = self._word_coherence_loss(beta)
        entropy_loss = self._entropy_loss(theta)
        anchor_prior_loss = self._anchor_prior_loss(beta)
        loss = reconstruction_loss + transport_loss + self.topic_orthogonality_weight * orthogonality_loss + self.word_coherence_weight * coherence_loss + self.document_entropy_weight * entropy_loss + self.anchor_prior_weight * anchor_prior_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "transport_loss": transport_loss.detach(),
            "orthogonality_loss": orthogonality_loss.detach(),
            "coherence_loss": coherence_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "anchor_prior_loss": anchor_prior_loss.detach(),
        }
