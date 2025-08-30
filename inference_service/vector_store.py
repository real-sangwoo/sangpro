import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class VectorEntry:
    doc_id: str
    embedding: np.ndarray
    attrs: Dict[str, Any]


class CharProjectionEmbedder:
    """
    Lightweight, deterministic text -> vector mapping.

    This is not a real ML model, but:
    - It avoids heavy dependencies (Torch, TensorRT, etc.).
    - It exposes an "embed(text) -> vector" interface that can easily be
      swapped for a call to a Triton Inference Server.
    """

    def __init__(self, output_dim: int = 128, rand_seed: int = 42) -> None:
        self.output_dim = output_dim
        rng = np.random.default_rng(rand_seed)
        # Random projection basis for ASCII characters.
        self._projection = rng.normal(size=(128, output_dim)).astype(np.float32)

    def embed(self, raw_text: str) -> np.ndarray:
        if not raw_text:
            return np.zeros(self.output_dim, dtype=np.float32)

        accumulator = np.zeros(self.output_dim, dtype=np.float32)
        for char in raw_text.lower():
            char_code = ord(char)
            if 0 <= char_code < 128:
                accumulator += self._projection[char_code]

        magnitude = np.linalg.norm(accumulator) + 1e-8
        return accumulator / magnitude


class VectorIndex:
    """
    In-memory vector store using cosine similarity.

    - Suitable as a prototype.
    - In production it can be replaced by FAISS, HNSW, Pinecone, etc.
    """

    def __init__(self, embedder: CharProjectionEmbedder) -> None:
        self._embedder = embedder
        self._entries: List[VectorEntry] = []
        self._mu = threading.Lock()

    def add(self, doc_id: str, text: str, attrs: Dict[str, Any] | None = None) -> None:
        attrs = attrs or {}
        vec = self._embedder.embed(text)
        entry = VectorEntry(doc_id=doc_id, embedding=vec, attrs=attrs)
        with self._mu:
            self._entries.append(entry)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[VectorEntry, float]]:
        if top_k <= 0:
            return []

        q_vec = self._embedder.embed(text)
        if not self._entries:
            return []

        with self._mu:
            mat = np.stack([e.embedding for e in self._entries], axis=0)
            dot_products = mat @ q_vec
            row_norms = np.linalg.norm(mat, axis=1) + 1e-8
            q_norm = np.linalg.norm(q_vec) + 1e-8
            cosine_scores = dot_products / (row_norms * q_norm)

            k = min(top_k, len(self._entries))
            top_indices = np.argpartition(-cosine_scores, k - 1)[:k]
            ranked = top_indices[np.argsort(-cosine_scores[top_indices])]

            output: List[Tuple[VectorEntry, float]] = []
            for idx in ranked:
                output.append((self._entries[int(idx)], float(cosine_scores[int(idx)])))
            return output


# Aliases kept for backward compatibility
SimpleTextEmbedder = CharProjectionEmbedder
InMemoryVectorStore = VectorIndex
