Triton Model Repo Skeleton
==========================

This is not a full Triton setup, but a skeleton that shows how this project
could be connected to **NVIDIA Triton Inference Server** in a real environment.

### Typical layout

A minimal example layout:

```text
triton_model_repo/
  text_embedder/
    1/
      model.onnx        # or model.plan (TensorRT)
    config.pbtxt
```

`config.pbtxt` would roughly contain:

```text
name: "text_embedder"
platform: "tensorrt_plan"  # or "onnxruntime_onnx"
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
    dims: [ 768 ]  # example dimension
  }
]
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

### Connecting to the Python service in this project

In `inference_service/main.py` the code currently uses `SimpleTextEmbedder`,
which is hash‑based and lightweight. In production this embedder can be replaced with:

- An HTTP/gRPC client to Triton:
  - Send input text (tokenized / serialized) to the `text_embedder` model.
  - Read the `EMBEDDING` output (float32 vector).
- The resulting vector is then stored in a vector store (FAISS / HNSW, etc.),
  while this demo still uses `InMemoryVectorStore`.

With this approach you can demonstrate:

- Understanding of **Triton model repository** structure.
- Understanding of clear separation of concerns:
  - Triton: heavy model serving (TensorRT, GPU).
  - Python service: routing, health checks, vector DB orchestration.

