denifiles – Vector Search Inference Service
===========================================

**Bahasa campur Indo/English biar enak dibaca.**

Project ini contoh kecil yang relevan dengan role **ML Engineer @ TwelveLabs**:

- **AI inference infra mindset**: simple microservice untuk inference + vector search.
- **Vector Search**: in‑memory vector index dengan cosine similarity.
- **Python**: backend pakai `FastAPI`.
- **Go**: small CLI client buat query service.
- **Triton/TensorRT oriented**: desain embedding layer seakan‑akan di‑serve via NVIDIA Triton (di sini masih stub, gampang diganti ke Triton/TensorRT beneran).

### Arsitektur Singkat

- **`inference_service/` (Python, FastAPI)**  
  - Endpoint `POST /index` untuk menambahkan dokumen (text) ke vector store.  
  - Endpoint `POST /search` untuk semantic search sederhana.  
  - Endpoint `GET /healthz` untuk health check (tipikal di infra production).
- **`go-client/` (Go)**  
  - CLI buat kirim query ke `/search` dan print hasil.
- **`triton_model_repo/`**  
  - Skeleton + catatan gimana service ini bisa diintegrasikan dengan **NVIDIA Triton Inference Server** dan model TensorRT beneran.

### Quickstart (Python service)

1. **Install dependency** (idealnya di virtualenv):

   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan inference service**:

   ```bash
   uvicorn inference_service.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Index contoh dokumen**:

   ```bash
   curl -X POST "http://localhost:8000/index" ^
     -H "Content-Type: application/json" ^
     -d "{\"id\":\"doc1\",\"text\":\"twelve labs video retrieval infra\",\"metadata\":{\"source\":\"demo\"}}"
   ```

4. **Search**:

   ```bash
   curl -X POST "http://localhost:8000/search" ^
     -H "Content-Type: application/json" ^
     -d "{\"query\":\"video understanding infrastructure\",\"top_k\":5}"
   ```

### Go client

- Lihat `go-client/main.go` untuk contoh:
  - Build: `go build -o search-client ./go-client`
  - Pakai: `./search-client "video understanding infrastructure"`

### Hubungan dengan Triton & TensorRT

- Di file Python, embedding sekarang **masih simple hash-based vector** (biar ringan dan portable).
- Di production, bagian embedding ini bisa diganti:
  - Panggil **Triton Inference Server** (gRPC/HTTP) yang serve model **TensorRT** (mis. text/video encoder).  
  - Endpoint Python hanya jadi thin wrapper: request → Triton → vector store (HNSW/FAISS/ScaNN).
- Folder `triton_model_repo/` berisi catatan layout repo model yang cocok buat Triton (`config.pbtxt`, dll).

Dengan setup ini, kamu punya **satu project kecil** yang nunjukin:

- Paham **inference serving pattern**, health check, dan service boundary.
- Paham konsep **vector search**.
- Pake **Python** di sisi service dan **Go** di sisi client.
