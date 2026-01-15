import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class StoredVector:
    doc_id: str
    vector: np.ndarray
    metadata: Dict[str, Any]


class SimpleTextEmbedder:
    """
    Lightweight, deterministic text -> vector mapping.

    This is not a real ML model, but:
    - It avoids heavy dependencies (Torch, TensorRT, etc.).
    - It exposes an "embed(text) -> vector" interface that can easily be
      swapped for a call to a Triton Inference Server.
    """

    def __init__(self, dim: int = 128, seed: int = 42) -> None:
        self.dim = dim
        rng = np.random.default_rng(seed)
        # Random projection basis for ASCII characters.
        self._char_basis = rng.normal(size=(128, dim)).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        for ch in text.lower():
            idx = ord(ch)
            if 0 <= idx < 128:
                vec += self._char_basis[idx]

        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm


class InMemoryVectorStore:
    """
    In-memory vector store using cosine similarity.

    - Suitable as a prototype.
    - In production it can be replaced by FAISS, HNSW, Pinecone, etc.
    """

    def __init__(self, embedder: SimpleTextEmbedder) -> None:
        self._embedder = embedder
        self._vectors: List[StoredVector] = []
        self._lock = threading.Lock()

    def index(self, doc_id: str, text: str, metadata: Dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        vec = self._embedder.embed(text)
        stored = StoredVector(doc_id=doc_id, vector=vec, metadata=metadata)
        with self._lock:
            self._vectors.append(stored)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[StoredVector, float]]:
        if top_k <= 0:
            return []

        q_vec = self._embedder.embed(query)
        if not self._vectors:
            return []

        with self._lock:
            matrix = np.stack([v.vector for v in self._vectors], axis=0)
            dots = matrix @ q_vec
            matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
            q_norm = np.linalg.norm(q_vec) + 1e-8
            scores = dots / (matrix_norm * q_norm)

            top_k = min(top_k, len(self._vectors))
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            sorted_idx = idx[np.argsort(-scores[idx])]

            results: List[Tuple[StoredVector, float]] = []
            for i in sorted_idx:
                results.append((self._vectors[int(i)], float(scores[int(i)])))
            return results

