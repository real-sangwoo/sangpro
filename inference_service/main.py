from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .vector_store import VectorIndex, CharProjectionEmbedder


class AddDocRequest(BaseModel):
    id: str = Field(..., description="Unique document ID")
    text: str = Field(..., description="Raw text to index")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional metadata")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")


class ResultHit(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    hits: List[ResultHit]


app = FastAPI(title="denifiles Inference + Vector Search Service")


_enc = CharProjectionEmbedder(output_dim=128)
_idx = VectorIndex(embedder=_enc)


@app.get("/healthz")
def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.

    In production this is typically monitored by an orchestration layer
    such as Kubernetes or ECS.
    """
    return {"status": "ok"}


@app.post("/index")
def add_document(req: AddDocRequest) -> Dict[str, Any]:
    """
    Index a single document into the vector store.

    In a Triton-backed setup:
    - The text would be sent to a Triton (TensorRT) model to obtain an embedding.
    - The resulting embedding would be stored in a vector database (FAISS/HNSW, etc.).
    """
    _idx.add(doc_id=req.id, text=req.text, attrs=req.metadata or {})
    return {"status": "indexed", "id": req.id}


@app.post("/search", response_model=QueryResponse)
def run_search(req: QueryRequest) -> QueryResponse:
    """
    Perform a simple semantic search using cosine similarity.
    """
    matches = _idx.query(text=req.query, top_k=req.top_k)
    hits: List[ResultHit] = []
    for entry, relevance in matches:
        hits.append(
            ResultHit(
                id=entry.doc_id,
                score=relevance,
                metadata=entry.attrs,
            )
        )
    return QueryResponse(hits=hits)
