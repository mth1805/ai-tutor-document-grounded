---
name: performance-audit
description: Audits UI loading latency vs. AI processing latency, model caching, database query execution, streaming TTFT, and background ingestion throughput.
---

# Performance Audit Skill

## 1. Purpose
The `performance-audit` skill equips coding agents to profile, diagnose, and optimize end-to-end performance across the **AI Tutor Assistant** stack. It strictly enforces the distinction between **UI loading latency** and **AI processing latency**, guarantees model caching, detects query bottlenecks in pgvector, and ensures streaming responsiveness.

## 2. When to Use
Use this skill when:
- Investigating slow page loads or unresponsive UI transitions.
- Profiling chat completion latency and Time-to-First-Token (TTFT).
- Auditing memory consumption or CPU spikes on the backend.
- Verifying that BGE-M3 and Cross-Encoder weights are cached locally and not reloaded.
- Investigating slow document ingestion or database timeouts.
- Preparing the application for Phase 11 (Background Processing + Performance) and Phase 13 (Deployment).

## 3. What to Inspect First
Before profiling or modifying code, inspect:
1. `backend/app/core/config.py` — Cache directories (`HF_HOME`, `TRANSFORMERS_CACHE`), batch sizes, connection pool sizes.
2. `backend/app/ml/` — Lifespan loader functions to verify models are singletons.
3. Network profiles:
   - Next.js bundle sizes (`npm run build`).
   - FastAPI middleware / timing headers (`X-Process-Time`).
4. Database indexes via Supabase SQL editor or migration files:
   - Verify HNSW index on `document_chunks.embedding`.
   - Verify GIN index on `document_chunks.tsv`.
   - Verify B-tree indexes on foreign keys (`workspace_id`, `document_id`, `conversation_id`).

## 4. Performance Audit Workflow
Follow this structured 6-step audit process:

```
[Step 1: Metric Classification]
      │ Distinguish UI Loading Latency from AI Processing Latency
      ▼
[Step 2: Model Lifecycle & Caching Audit]
      │ Confirm zero per-request model instantiation & local weight caching
      ▼
[Step 3: Database & Vector Index Profiling]
      │ Run EXPLAIN ANALYZE on vector similarity and relational queries
      ▼
[Step 4: Ingestion Pipeline Profiling]
      │ Measure parsing, chunking, and embedding batch throughput
      ▼
[Step 5: Streaming & TTFT Measurement]
      │ Measure time from query dispatch to first streamed token
      ▼
[Step 6: Remediation & Benchmark Report]
      │ Deliver actionable optimization recommendations with before/after targets
```

### Step 1: Metric Classification: UI vs. AI Latency
Agents must treat UI latency and AI latency as fundamentally separate domains:

| Metric Category | Target SLA | Scope & Measurement Point |
|---|---|---|
| **UI First Contentful Paint (FCP)** | < 0.6s | Next.js server-rendered HTML delivery |
| **Workspace / Conversation Switch** | < 0.3s | Client navigation without model dependencies |
| **Document Upload Acknowledgment** | < 0.3s | Upload API returns `202 Accepted` |
| **Hybrid Retrieval Latency** | < 0.15s | Dense + BM25 queries in PostgreSQL |
| **Cross-Encoder Rerank Latency** | < 0.35s | Reranking top 25 chunks down to top 4 |
| **LLM Time-to-First-Token (TTFT)** | < 1.5s | Time from query post to first SSE token chunk |
| **Total Ingestion Throughput** | > 15 pgs/s | Background worker parsing + embedding |

### Step 2: Model Lifecycle & Caching Audit
- **Check 1: Single Initialization:** Verify model instantiation occurs inside FastAPI `lifespan(app)` or via thread-safe lazy singletons.
  ```python
  # BAD: Loading model inside endpoint
  @router.post("/chat")
  async def chat_handler(payload: ChatRequest):
      model = SentenceTransformer("BAAI/bge-m3") # CRITICAL VIOLATION!
  ```
- **Check 2: Local Directory Cache:** Ensure environment variables `HF_HOME` or `TORCH_HOME` point to a persistent directory and do not trigger HTTP downloads on restart:
  ```python
  # GOOD: Model path points to local cache directory
  model = SentenceTransformer("BAAI/bge-m3", cache_folder=settings.MODEL_CACHE_DIR)
  ```

### Step 3: Database & pgvector Query Profiling
- Run `EXPLAIN ANALYZE` on hybrid retrieval queries:
  ```sql
  EXPLAIN ANALYZE
  SELECT id, 1 - (embedding <=> :vector) as score
  FROM document_chunks
  WHERE workspace_id = :ws_id
  ORDER BY embedding <=> :vector
  LIMIT 25;
  ```
- **Verify:**
  - Query uses `Index Scan using idx_chunks_embedding on document_chunks` (HNSW).
  - Avoid `Seq Scan on document_chunks` (Sequential scans mean index is missing or disabled).

### Step 4: Ingestion Pipeline Profiling
- Break down ingestion into isolated timings:
  - $T_{parse}$: Time to extract text and page numbers from PDF.
  - $T_{chunk}$: Time to partition into structured chunks.
  - $T_{embed}$: Time to generate BGE-M3 embeddings.
  - $T_{db}$: Time to insert chunk records into PostgreSQL.
- Optimize $T_{embed}$ by verifying:
  - GPU acceleration (CUDA) is used if available; if CPU, thread pool concurrency is optimal (`torch.set_num_threads`).
  - Batching is enabled (e.g., `batch_size=32`).

### Step 5: Streaming & TTFT Measurement
- Measure TTFT using curl or client instrumentation:
  ```bash
  curl -w "TTFB: %{time_starttransfer}s | Total: %{time_total}s\n" \
       -N -X POST http://localhost:8000/api/v1/chat/stream \
       -H "Content-Type: application/json" \
       -d '{"workspace_id":"...", "query":"Explain photosynthesis"}'
  ```

## 5. Engineering Rules
1. **Decouple UI from Model Status:** A user visiting the website must never wait for PyTorch or Hugging Face to load into RAM. The UI loads instantly.
2. **Never Block Event Loop:** Any synchronous ML model call must run in `fastapi.concurrency.run_in_threadpool`.
3. **No Unindexed Foreign Keys:** Every foreign key in Supabase must have an associated B-tree index.
4. **Mandatory Vector Indexing:** Never run vector similarity searches without an HNSW index on `document_chunks.embedding`.
5. **Always Measure Before Optimizing:** Record baseline latency numbers before modifying code.

## 6. Validation Checklist
- [ ] Next.js UI loads without waiting for ML models.
- [ ] BGE-M3 and Cross-Encoder load once on application startup.
- [ ] Model weights are stored in local persistent cache (`HF_HOME`).
- [ ] pgvector queries utilize HNSW index (verified with `EXPLAIN ANALYZE`).
- [ ] Upload endpoint returns `202 Accepted` within 300ms.
- [ ] Streaming TTFT remains under 1.5s for typical queries.

## 7. Common Failure Modes & Mitigations
- **Failure:** Server runs out of memory (OOM) upon starting 4 uvicorn workers.
  - *Root Cause:* Each worker loaded a separate copy of BGE-M3 and Cross-Encoder in RAM.
  - *Mitigation:* Share model instances across workers or use a dedicated model microservice/singleton worker process.
- **Failure:** UI freezes for 30 seconds when an uploaded PDF is submitted.
  - *Root Cause:* File ingestion executed synchronously inside the POST `/upload` route.
  - *Mitigation:* Move ingestion to `BackgroundTasks` and return an immediate acceptance response.
