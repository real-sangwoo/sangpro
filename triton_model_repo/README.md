Triton Model Repo Skeleton
==========================

Ini bukan full Triton setup, tapi skeleton yang nunjukin gimana project ini
bisa dihubungkan ke **NVIDIA Triton Inference Server** di environment beneran.

### Layout yang umum

Contoh layout minimal:

```text
triton_model_repo/
  text_embedder/
    1/
      model.onnx        # atau model.plan (TensorRT)
    config.pbtxt
```

`config.pbtxt` kurang-lebih berisi:

```text
name: "text_embedder"
platform: "tensorrt_plan"  # atau "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "TEXT"
    data_type: TYPE_BYTES
    dims: [ -1 ]
  }
]
output [
  {
    name: "EMBEDDING"
    data_type: TYPE_FP32
    dims: [ 768 ]  # contoh dim
  }
]
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

### Menghubungkan ke service Python di project ini

Di `inference_service/main.py`, saat ini kita pakai `SimpleTextEmbedder`
yang hashing‑based dan ringan. Di production, embedder itu bisa diganti jadi:

- Client HTTP/gRPC ke Triton:
  - Kirim input text (tokenized / serialized) ke model `text_embedder`.
  - Ambil output `EMBEDDING` (vector float32).
- Vector yang keluar kemudian disimpan ke vector store (FAISS / HNSW, dll),
  sementara demo ini masih pakai `InMemoryVectorStore`.

Dengan pendekatan ini, kamu bisa tunjukkan:

- Paham struktur **Triton model repository**.
- Paham pemisahan concern:
  - Triton: heavy model serving (TensorRT, GPU).
  - Python service: routing, healthcheck, vector DB orchestration.

