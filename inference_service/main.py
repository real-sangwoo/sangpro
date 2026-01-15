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
    Simple health check endpoint.

    In production this is typically monitored by an orchestration layer
    such as Kubernetes or ECS.
    """
    return {"status": "ok"}


@app.post("/index")
def index_document(body: IndexRequest) -> Dict[str, Any]:
    """
    Index a single document into the vector store.

    In a Triton‑backed setup:
    - The text would be sent to a Triton (TensorRT) model to obtain an embedding.
    - The resulting embedding would be stored in a vector database (FAISS/HNSW, etc.).
    """
    _store.index(doc_id=body.id, text=body.text, metadata=body.metadata or {})
    return {"status": "indexed", "id": body.id}


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    """
    Perform a simple semantic search using cosine similarity.
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

