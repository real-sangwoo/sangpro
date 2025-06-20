from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .vector_store import InMemoryVectorStore, SimpleTextEmbedder


class IndexRequest(BaseModel):
    id: str = Field(..., description="Unique document ID")
    text: str = Field(..., description="Raw text to index")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional metadata")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")


class SearchHit(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    hits: List[SearchHit]


app = FastAPI(title="denifiles Inference + Vector Search Service")


_embedder = SimpleTextEmbedder(dim=128)
_store = InMemoryVectorStore(embedder=_embedder)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    """
    Simple healthcheck endpoint.

    Di environment production, health ini biasanya dipantau oleh
    orchestration layer (Kubernetes, ECS, dsb).
    """
    return {"status": "ok"}


@app.post("/index")
def index_document(body: IndexRequest) -> Dict[str, Any]:
    """
    Index satu dokumen ke vector store.

    Di setup yang terhubung Triton:
    - Di sini text dikirim ke Triton model (TensorRT) untuk dapat embedding.
    - Hasil embedding dimasukkan ke vector DB (FAISS/HNSW, dll).
    """
    _store.index(doc_id=body.id, text=body.text, metadata=body.metadata or {})
    return {"status": "indexed", "id": body.id}


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    """
    Lakukan semantic-ish search dengan cosine similarity.
    """
    results = _store.search(query=body.query, top_k=body.top_k)
    hits: List[SearchHit] = []
    for stored, score in results:
        hits.append(
            SearchHit(
                id=stored.doc_id,
                score=score,
                metadata=stored.metadata,
            )
        )
    return SearchResponse(hits=hits)

