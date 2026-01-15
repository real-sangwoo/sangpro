denifiles – Vector Search Inference Service
===========================================

This repository is a compact demo project that is relevant to a **ML Engineer @ TwelveLabs**–style role:

- **AI inference infrastructure mindset**: a small microservice for inference + vector search.
- **Vector Search**: in‑memory vector index with cosine similarity.
- **Python**: backend using `FastAPI`.
- **Go**: small CLI client to query the service.
- **Triton/TensorRT oriented**: the embedding layer is designed so it can be swapped with a real NVIDIA Triton / TensorRT model server.

### High‑level architecture

- **`inference_service/` (Python, FastAPI)**  
  - `POST /index` to add text documents into the vector store.  
  - `POST /search` for simple semantic search.  
  - `GET /healthz` for a production‑style health check endpoint.
- **`go-client/` (Go)**  
  - CLI that sends search queries to `/search` and prints results.
- **`triton_model_repo/`**  
  - Skeleton + notes for how this service could be wired into **NVIDIA Triton Inference Server** and TensorRT models.

### Quickstart (Python service)

1. **Install dependencies** (ideally in a virtualenv):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the inference service**:

   ```bash
   uvicorn inference_service.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Index a sample document**:

   ```bash
   curl -X POST "http://localhost:8000/index" ^
     -H "Content-Type: application/json" ^
     -d "{\"id\":\"doc1\",\"text\":\"twelve labs video retrieval infra\",\"metadata\":{\"source\":\"demo\"}}"
   ```

4. **Run a search query**:

   ```bash
   curl -X POST "http://localhost:8000/search" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\":\"video understanding infrastructure\",\"top_k\":5}"
   ```

### Go client

- See `go-client/main.go` for an example:
  - Build: `go build -o search-client ./go-client`
  - Run: `./search-client "video understanding infrastructure"`

### Relationship to Triton & TensorRT

- In the Python service, the embedding layer is currently a **simple hash‑based vectorizer** (lightweight and portable).
- In a production setup, this part can be replaced by:
  - Calling **Triton Inference Server** (gRPC/HTTP) that serves a **TensorRT** model (e.g. text/video encoder).  
  - The Python endpoint becomes a thin wrapper: request → Triton → vector store (HNSW/FAISS/ScaNN).
- The `triton_model_repo/` folder documents a compatible Triton model repository layout (`config.pbtxt`, etc.).

Overall, this is a **small, self‑contained project** that demonstrates:

- Understanding of **inference serving patterns**, health checks, and service boundaries.
- Understanding of **vector search** concepts.
- Use of both **Python** (service) and **Go** (client).
