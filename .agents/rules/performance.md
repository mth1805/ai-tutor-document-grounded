# Performance & Model Lifecycle Rules

## 1. Model Lifecycle & Initialization
- **Zero Per-Request Model Loading:**
  - NEVER load embedding models (`BAAI/bge-m3`) or reranker models (`Cross-Encoder`) inside HTTP request handlers.
  - Models must be loaded once per process during application lifespan startup (`lifespan` handler in FastAPI) or loaded lazily into a thread-safe singleton.
- **Local Model Caching:**
  - Explicitly configure Hugging Face / PyTorch cache directories (e.g., `HF_HOME` / `TRANSFORMERS_CACHE`) to point to a persistent local cache volume (e.g., `./cache/models` or system cache).
  - Check for local weights before attempting network downloads. Server restarts must reuse cached weights without re-downloading gigabytes over the network.
- **Process Memory Footprint:**
  - Use CPU/GPU evaluation modes (`model.eval()`) with `torch.no_grad()` to prevent memory leaks during embedding and reranking passes.
  - Set explicit batch sizes for chunk embedding (e.g., `batch_size=16` or `32`) to avoid OOM crashes on large documents.

## 2. Decoupled UI & Non-Blocking Frontend
- **Independence from Model Startup:**
  - The Next.js frontend must render shells, sidebars, and document views immediately. It must never freeze or wait for backend models to load into memory.
  - Implement a lightweight health check endpoint (`/api/v1/health`) that reports API availability independent of ML model warm-up status (`/api/v1/health/models`).

## 3. Asynchronous & Background Ingestion
- **Non-Blocking File Ingestion:**
  - Document parsing, text cleaning, chunking, and embedding generation are computationally intensive and must NEVER block the file upload HTTP request.
  - Upload endpoints must save the file to Supabase Storage, insert a `document` row with status `queued`, and immediately return `202 Accepted` with a `document_id`.
  - Execute ingestion tasks asynchronously via FastAPI `BackgroundTasks` or a background task worker.
  - The frontend tracks progress via status polling or webhook/SSE updates.

## 4. Streaming & Latency Budgets
- **Time-to-First-Token (TTFT):**
  - Generation endpoints must begin streaming tokens within 1.5 seconds of receiving the query (inclusive of retrieval and reranking).
  - Use token streaming over Server-Sent Events (SSE) so users see immediate output instead of waiting for full response synthesis.
- **Retrieval Latency Targets:**
  - Vector search (pgvector) + BM25: < 150ms.
  - Cross-Encoder reranking (top 20-30 chunks down to 3-5): < 350ms.
  - Keep total retrieval overhead under 500ms before LLM generation begins.

## 5. Database & Vector Query Optimization
- **pgvector Indexing:**
  - Use `HNSW` (Hierarchical Navigable Small World) index for embedding vectors (`halfvec` or `vector(1024)` with `vector_cosine_ops` or `vector_l2_ops`).
  - Set appropriate HNSW parameters (e.g., `m = 16`, `ef_construction = 64`) to balance index build time and query latency.
  - Always scope vector searches by `workspace_id` to limit vector scan partitions.
- **Connection Pooling:**
  - Use Supabase connection pooling (Supavisor / pgbouncer) with appropriate pool limits to avoid exhausting PostgreSQL connections under concurrent user requests.
