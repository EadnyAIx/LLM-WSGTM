from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


class DenseRAGIndex:
    def __init__(self, encoder, normalize=True):
        self.encoder = encoder
        self.normalize = normalize
        self.texts: List[str] = []
        self.metadata: List[dict] = []
        self.embeddings = None

    def add(self, texts: Iterable[str], metadata: Optional[Iterable[dict]]=None):
        texts = list(texts)
        metadata = list(metadata) if metadata is not None else [{} for _ in texts]
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata lengths differ")
        embeddings = np.asarray(self.encoder.encode(texts), dtype=np.float32)
        if self.normalize:
            embeddings = self._normalize(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self.embeddings = embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])

    def search(self, query, top_k=5):
        if self.embeddings is None or not self.texts:
            return []
        query_embedding = np.asarray(self.encoder.encode([query]), dtype=np.float32)
        if self.normalize:
            query_embedding = self._normalize(query_embedding)
        scores = self.embeddings @ query_embedding[0]
        order = np.argsort(-scores)[:max(1, int(top_k))]
        return [RetrievedChunk(self.texts[index], float(scores[index]), self.metadata[index]) for index in order]

    def build_context(self, query, top_k=5, separator="\n\n"):
        return separator.join(chunk.text for chunk in self.search(query, top_k))

    @staticmethod
    def _normalize(values):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.clip(norms, 1e-12, None)
