# AI Tutor Assistant

For frontend deployment preparation and manual Vercel settings, see [Vercel deployment guide](docs/vercel-deployment.md). FastAPI remains on Modal; use `frontend/.env.example` for browser-safe configuration.

AI Tutor Assistant is an enterprise-grade, document-grounded AI learning assistant designed to help users learn directly from their uploaded documents (PDF/DOCX) organized into private workspaces.

---

## 1. Project Purpose & Architecture

The core tenet of AI Tutor Assistant is **Document-First Grounding**:
- Uploaded learning documents are the primary knowledge source.
- Automated web search acts purely as a fallback when document retrieval yields insufficient relevance.
- Citations must strictly anchor back to exact page numbers and chunk IDs.
- State (workspaces, conversations, documents, messages) persists reliably in Supabase PostgreSQL across page reloads.
- The UI renders instantly with zero blocking on backend machine learning models.

```
       ┌────────────────────────┐
       │   Next.js (Frontend)   │  (Port 3000)
       └───────────┬────────────┘
                   │ HTTP / SSE
                   ▼
       ┌────────────────────────┐
       │   FastAPI (Backend)    │  (Port 8000)
       └───────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│  Supabase PG     │  │ AI & Embeddings  │
│  (pgvector + RLS)│  │ (BGE-M3, Gemini) │
└──────────────────┘  └──────────────────┘
```

---

## 2. Technology Stack

- **Frontend:** Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, Lucide Icons.
- **Backend:** FastAPI, Python 3.11+, Pydantic v2, SQLAlchemy (asyncpg).
- **Database & Storage:** Supabase PostgreSQL with `pgvector`, Supabase Storage (configured in upcoming phases).
- **AI & Embedding Models (Planned for Future Phases):**
  - Dense Embeddings: `BAAI/bge-m3` (1024-dimensional)
  - Reranker: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  - LLM: Google Gemini API

---

## 3. Repository Folder Structure

```
c:/2_AITutor/
├── AGENTS.md                   # Root source of truth for coding agents
├── README.md                   # Project overview & operational runbook
├── .env.example                # Canonical environment variable template
├── .gitignore                  # Git exclusions for dependencies and secrets
├── .agents/
│   ├── rules/                  # Domain-specific constraints (frontend, backend, rag, etc.)
│   └── skills/                 # Operational engineering workflows (rag, ingestion, audit)
├── backend/
│   ├── requirements.txt        # Python backend dependencies
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint, lifespan, CORS, /health
│   │   ├── api/v1/health.py    # Versioned health probe
│   │   ├── core/               # Configuration (pydantic-settings) & CORS
│   │   ├── db/session.py       # Async SQLAlchemy session management for Supabase
│   │   ├── schemas/health.py   # Pydantic schema validation
│   │   └── services/           # Decoupled business logic layer
│   └── tests/
│       └── test_health.py      # Pytest verification for /health
└── frontend/
    ├── package.json            # Node.js dependencies and scripts
    ├── tsconfig.json           # TypeScript configuration
    ├── next.config.mjs         # Next.js configuration
    ├── tailwind.config.ts      # Tailwind CSS theme configuration
    └── src/
        ├── app/                # App Router: layout, globals.css, page
        ├── components/         # Header, Sidebar, ChatArea
        └── lib/api.ts          # Centralized API client abstraction
```

---

## 4. Environment Variables

Copy `.env.example` to configure your environment:
```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

| Variable | Description | Target |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI origin; development default `http://localhost:8000`, required HTTPS origin on Vercel | Frontend Browser |
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project API URL | Frontend Browser |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Public anonymous Supabase key | Frontend Browser |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase privileged service role key (NEVER expose to frontend) | Backend Server |
| `DATABASE_URL` | Supabase PostgreSQL direct async connection string | Backend Server |
| `BACKEND_CORS_ORIGINS` | JSON list of allowed frontend origins (default: `["http://localhost:3000"]`) | Backend Server |

The FastAPI database dependency binds each SQLAlchemy transaction to the verified request user with transaction-local PostgreSQL claims and the Supabase `authenticated` role. This keeps RLS active even when service queries omit an ownership predicate. The role and claims reset automatically at transaction end. To run the RLS regression against a dedicated Supabase test database, set `RLS_TEST_DATABASE_URL` and run `python -m pytest backend/tests/test_rls_context.py` from the repository root.

---

## 5. Local Setup & Running Instructions

### One-command Docker demo

Copy `.env.example` to `.env`, then start both local services from the repository root:

```bash
docker compose up --build
```

Open `http://localhost:3000`; the browser calls `http://localhost:8000`. Stop the full stack with `docker compose down`. Compose starts frontend, backend, and the CPU ingestion worker automatically; no extra worker terminal is needed. Supabase remains managed/cloud. Root `.env` holds backend/worker Dev runtime settings, while frontend build arguments contain only browser-safe API/Supabase values. Models persist in a named Docker volume, so first model load may take longer but restarts reuse downloads. Placeholder auth values cannot provide a usable authenticated demo. Complete the one-time manual Dev migration/configuration in [local Docker runtime guide](docs/local-docker-runtime.md) before the first upload; never use a service-role key in the frontend.

### Phase 5 ingestion runtime dependencies

PDF/DOCX/TXT extraction runs in Python. OCR for image files and scanned PDF pages uses the `pytesseract` Python package plus a separately installed Tesseract executable. Install Tesseract and its `eng` and `vie` traineddata files; by default the backend requests `TESSERACT_LANGUAGES=vie+eng`. Set `TESSERACT_CMD` to an executable path or a command available on `PATH` when needed. The backend checks the executable and requested language data once per provider and reports a processing error if either is unavailable; OCR-required content is never treated as successfully extracted when OCR cannot run.

Legacy binary `.doc` files are converted in an isolated temporary directory with LibreOffice headless, then parsed by the existing DOCX parser. Install LibreOffice on any backend runtime that accepts `.doc` ingestion and set `LIBREOFFICE_CMD` if its executable is not `soffice` on `PATH`. `LIBREOFFICE_TIMEOUT_SECONDS` defaults to 60. If LibreOffice is unavailable, conversion fails, or times out, the document is marked failed with a diagnostic `processing_error`; temporary conversion files are removed and the uploaded original remains unchanged. `.docx` processing does not require LibreOffice.

For local development, install the system executables and OCR language data before starting the backend, or configure the paths in the root `.env` (copied from `.env.example`). These are backend runtime dependencies and must also be installed in the backend deployment image/host; installing Python requirements alone does not install system binaries or Tesseract traineddata. The optional smoke test for a real legacy `.doc` conversion can be enabled with `AI_TUTOR_LEGACY_DOC_FIXTURE` pointing to a valid local `.doc` fixture and LibreOffice installed.

### Backend (FastAPI)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
5. Verify health:
   - Root Health: `http://127.0.0.1:8000/health` $\rightarrow$ `{"status": "ok"}`
   - Interactive Docs: `http://127.0.0.1:8000/api/v1/docs`

### Frontend (Next.js)
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
4. Open the web interface:
   - Web application: `http://localhost:3000`

---

## 6. Current Scope & Phase Roadmap

> [!NOTE]
> **Phase 6 Status:** BGE-M3 Dense Embedding, pgvector HNSW indexing, embedding lifecycle tracking, and non-blocking batch execution are fully implemented. Chunks generated in Phase 5 are embedded into 1024-dimensional normalized vectors and persisted atomically in PostgreSQL. Model loading is centralized as a thread-safe singleton, with local weight caching and safe CPU fallback.

### Development Phases
- [x] **Phase 1: Foundation** — Scaffolding, agent governance, clean Next.js UI, FastAPI health probes, decoupled DB layer.
- [x] **Phase 2: Authentication + Workspace** — Supabase auth, user workspaces, workspace switcher, CRUD APIs with RLS.
- [x] **Phase 3: Conversation Persistence** — Multi-conversation threads per workspace, message history schema & persistence.
- [x] **Phase 4: Document Upload + Storage** — PDF/DOCX/TXT/Image file uploads, Supabase private storage bucket, file validation, document metadata persistence, and workspace isolation.
- [x] **Phase 5: Document Ingestion** — Text extraction, cleaning, structure-aware chunking, page-number tracking.
- [x] **Phase 6: Embedding + Vector Database** — BGE-M3 integration, local weight caching, batch embedding, pgvector storage with HNSW index.
- [ ] **Phase 7: Hybrid Retrieval + Reranking** — Dense vector search + BM25 lexical search, RRF fusion, Cross-Encoder reranking.
- [ ] **Phase 8: RAG Generation** — Prompt assembly, grounded LLM generation via Gemini API, prompt injection guards.
- [ ] **Phase 9: Document-First + Web Fallback** — Relevance gating, automated web search fallback when document evidence is insufficient.
- [ ] **Phase 10: Citation + Streaming** — Token streaming over SSE, inline document citations (`[Doc: file.pdf, p. 3]`), web source cards.
- [ ] **Phase 11: Background Processing + Performance** — Asynchronous task queues for ingestion, query optimization.
- [ ] **Phase 12: Evaluation** — RAG metrics benchmark (Recall@K, MRR, Faithfulness, Answer Relevance).
- [ ] **Phase 13: UI Polish + Deployment** — Production build audits, responsive UI refinement, Dockerization.

---

## 7. Phase 4: Document Upload & Persistent Storage

### System Architecture & Upload Flow

```
User (Browser)
   │
   │ 1. Multipart Form POST (file + JWT)
   ▼
FastAPI Backend (/api/v1/workspaces/{workspace_id}/documents)
   │
   ├─► Verify JWT & Authenticate User
   ├─► Verify Workspace Ownership (IDOR Defense)
   ├─► Content Validation (Magic bytes, extension, max 25MB, non-empty)
   ├─► Filename Sanitization (Strip directory traversal / null bytes)
   │
   ├──► Private Supabase Storage (Bucket: `documents`)
   │       Path: `{workspace_id}/{document_id}/{sanitized_filename}`
   │
   └──► Supabase PostgreSQL (`documents` Table)
           [Rollback storage file if DB metadata insertion fails]
```

### Storage Model & Isolation
- **Bucket:** Private `documents` bucket (`public = false`). Never accessible to anonymous users.
- **Path Structure:** `{workspace_id}/{document_id}/{sanitized_filename}`. Completely deterministic, unique per document, and scoped to the parent workspace.
- **Tenant Isolation:**
  - Row Level Security (RLS) is enabled on `public.documents` with policies enforcing `auth.uid() = user_id` and workspace ownership.
  - Storage Object RLS policies on `storage.objects` verify that the folder prefix matches a workspace owned by the user (`auth.uid()`).
  - Secret service role key (`SUPABASE_SERVICE_ROLE_KEY`) is stored strictly on the backend and never exposed to the frontend.

### Supported File Types & Size Limits
- **Allowed Extensions:** `.pdf`, `.doc`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.webp`.
- **Allowed MIME Types:** `application/pdf`, `application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/plain`, `image/png`, `image/jpeg`, `image/webp`.
- **Magic Bytes Validation:**
  - PDF: `%PDF-` header
  - DOCX: `PK\x03\x04` zip header
  - DOC: `\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1` OLE compound header
  - PNG: `\x89PNG\r\n\x1a\n` header
  - JPEG: `\xFF\xD8\xFF` header
  - WEBP: `RIFF` ... `WEBP` header
  - TXT: UTF-8/ASCII text decoding without null bytes
- **File Size Limits:** Minimum > 0 bytes (empty files rejected with 400), Maximum 25 MB (`26,214,400` bytes).

### Failure Consistency & Cleanup
- **Transactional Rollback:** If the Supabase Storage upload succeeds but the database insert fails, the backend immediately deletes the newly created storage file, preventing orphaned storage objects.
- **Cascade Deletion:** Deleting a document removes both its database metadata and its physical storage file. Deleting a workspace removes all its child documents and cleans up all associated storage files.

### API Endpoints

| Method | Endpoint | Description | Status |
|---|---|---|---|
| `POST` | `/api/v1/workspaces/{workspace_id}/documents` | Multipart file upload; verifies ownership, validates file, stores in private bucket, and records metadata. | `201 Created` |
| `GET` | `/api/v1/workspaces/{workspace_id}/documents` | Lists all documents belonging to a workspace, ordered newest first. | `200 OK` |
| `GET` | `/api/v1/documents/{document_id}` | Retrieves document metadata after verifying user ownership. | `200 OK` |
| `DELETE` | `/api/v1/documents/{document_id}` | Deletes document metadata and removes the file from storage. | `204 No Content` |
| `GET` | `/api/v1/documents/{document_id}/download` | Streams original file contents with `attachment` Content-Disposition. | `200 OK` |
| `GET` | `/api/v1/documents/{document_id}/url` | Generates a secure, time-limited signed URL for private access. | `200 OK` |

### Test Commands
```bash
# Run full backend test suite (88 tests covering auth, workspaces, conversations, messages, documents, ingestion, embeddings, pgvector, and RLS)
python -m pytest backend/tests

# Run frontend production build & TypeScript validation
npm --prefix frontend run build
```

---

## 8. Phase 6: BGE-M3 Embeddings & pgvector HNSW Vector Persistence

Phase 6 implements dense semantic vector embeddings for chunked learning materials using the state-of-the-art multilingual model **`BAAI/bge-m3`** and stores them in Supabase PostgreSQL using the **`pgvector`** extension with **HNSW** indexing.

### Architecture & Pipeline Flow

```
[Uploaded Document]
        │
        ▼ (Phase 5)
[Document Parsing & Chunking]
        │
        ▼ (Atomic commit)
[document_chunks persisted] (status = 'processed')
        │
        ├──► Automatic pipeline trigger (AUTO_EMBED_AFTER_INGESTION=True)
        │    OR Manual API trigger (POST /api/v1/documents/{id}/embed)
        │
        ▼
[BGE-M3 Batch Encoding] (offloaded via run_in_threadpool)
  - Processed in micro-batches (EMBEDDING_BATCH_SIZE=16)
  - Generates 1024-dimensional dense vectors
  - L2 normalized for cosine distance
        │
        ▼ (Atomic in-place update)
[pgvector Persistence] (document_chunks.embedding)
  - Canonical chunks updated in place (idempotent, no duplicates)
  - Metadata: embedding_model, embedding_version, embedded_at
  - HNSW index (vector_cosine_ops, m=16, ef_construction=64)
        │
        ▼
[Embedding Status Finalization] (embedding_status = 'completed')
```

### BGE-M3 Model Lifecycle & Centralized Singleton

- **Zero In-Request Loading:** Embedding models are never loaded inside HTTP request handlers. The model is managed as a process-level thread-safe singleton (`get_embedding_provider()`).
- **Startup Warmup:** Pre-warming is optionally executed during FastAPI application startup via the `@asynccontextmanager` lifespan handler (`PREWARM_MODELS=True`).
- **Local Weight Caching:** Model weights are cached locally using Hugging Face cache (`HF_HOME` or configured `EMBEDDING_MODEL_CACHE_DIR`). Restarts reuse cached weights without re-downloading gigabytes over the network.
- **Evaluation Mode & Memory Protection:** All inferences run in `.eval()` mode wrapped inside `torch.no_grad()`. Micro-batches prevent out-of-memory (OOM) errors on CPU or low-VRAM GPUs.

### Device Resolution & Hardware Flexibility

- The device is resolved automatically via `EMBEDDING_DEVICE`:
  - `auto`: Uses CUDA if `torch.cuda.is_available()` is True; otherwise falls back gracefully to CPU.
  - `cpu`: Forces CPU execution.
  - `cuda`: Uses CUDA if available; if requested but unavailable, logs a warning and falls back to CPU without breaking the server or tests.

### Database Schema & pgvector HNSW Index

Migration file: `supabase/migrations/20261001060000_bge_m3_pgvector_embeddings.sql`

- **Extension:** `CREATE EXTENSION IF NOT EXISTS vector;`
- **Columns on `public.document_chunks`:**
  - `embedding vector(1024)`: Dense vector produced by BGE-M3.
  - `embedding_model TEXT`: Model identifier (e.g., `BAAI/bge-m3`).
  - `embedding_version TEXT`: Provider version identifier for reproducibility (e.g., `bge-m3-dense-v1`).
  - `embedded_at TIMESTAMPTZ`: Exact timestamp when embeddings were persisted.
- **Vector Index:**
  ```sql
  CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON public.document_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  ```
- **Embedding Lifecycle on `public.documents`:**
  - `embedding_status TEXT`: Status lifecycle constraint: `CHECK (embedding_status IN ('pending', 'processing', 'completed', 'failed'))`.
  - `embedding_started_at TIMESTAMPTZ`: Timestamp when embedding inference started.
  - `embedded_at TIMESTAMPTZ`: Timestamp when vector persistence completed.
  - `embedding_error TEXT`: Detailed, sanitized error message if embedding fails.

### Idempotency & Re-Embedding Flow

- Re-running embedding on a document updates existing canonical `document_chunks` records in place.
- It does **not** create duplicate chunk rows, orphan existing records, or alter `chunk_index` / `page_number` citations.
- If an embedding pass fails, the database transaction rolls back so partial vectors are never presented as completed.

### API Endpoints (Phase 6)

| Method | Endpoint | Description | Status |
|---|---|---|---|
| `POST` | `/api/v1/documents/{document_id}/embed` | Schedules asynchronous BGE-M3 embedding generation for an authorized processed document. | `202 Accepted` |
| `GET` | `/api/v1/documents/{document_id}/embedding-status` | Retrieves real-time embedding progress, total chunks, embedded chunk count, model info, and timestamps. | `200 OK` |
| `GET` | `/api/v1/documents/{document_id}/chunks` | Lists parsed chunks with `has_embedding: bool`, `embedding_model`, and `embedded_at`. | `200 OK` |

### Environment Variables Added

| Variable | Type | Default | Description |
|---|---|---|---|
| `EMBEDDING_MODEL_NAME` | String | `BAAI/bge-m3` | Hugging Face model identifier for dense embeddings. |
| `EMBEDDING_DIMENSION` | Integer | `1024` | Vector dimensionality matching the embedding model. |
| `EMBEDDING_BATCH_SIZE` | Integer | `16` | Micro-batch size for chunk encoding to prevent OOM. |
| `EMBEDDING_DEVICE` | String | `auto` | Target compute device (`auto`, `cpu`, `cuda`). |
| `EMBEDDING_MODEL_CACHE_DIR` | Path | `None` | Optional custom persistent local directory for cached weights. |
| `EMBEDDING_NORMALIZE` | Boolean | `True` | L2-normalizes vectors for cosine distance indexing. |
| `AUTO_EMBED_AFTER_INGESTION` | Boolean | `True` | Automatically triggers embedding after successful Phase 5 chunking. |
| `PREWARM_MODELS` | Boolean | `False` | Pre-warms model singleton on startup in production. |
| `USE_MOCK_EMBEDDING` | Boolean | `False` | Forces deterministic MockEmbeddingProvider for testing without downloads. |

### Development & Test Setup

- **Fast Testing Without Downloads:** Test suites automatically use `MockEmbeddingProvider`, producing deterministic 1024-dimensional normalized vectors seeded from text hashes.
- **Optional Real Model Smoke Test:** To test actual model weight loading with Hugging Face, run:
  ```bash
  RUN_REAL_BGE_M3_TEST=1 python -m pytest backend/tests/test_embedding_ml.py -k test_real_bge_m3_smoke
  ```

---

## 8. Phase 7: Hybrid Retrieval + Cross-Encoder Reranking

Phase 7 implements hybrid semantic and lexical retrieval combined via Reciprocal Rank Fusion (RRF) and scored with a multilingual Cross-Encoder reranker. The output is a ranked list of authorized, relevance-gated document chunks with verified page provenance.

### End-to-End Retrieval Pipeline

```
User Query
    │
    ▼
[BGE-M3 Query Embedding] (1024-dim, L2-normalized)
    │
    ├─────────────────────────────────────────────┐
    ▼                                             ▼
[Dense Retrieval (pgvector)]            [BM25 Lexical Retrieval (PostgreSQL FTS)]
  - Cosine distance: embedding <=> query  - Cover density ranking: ts_rank_cd(tsv, query)
  - HNSW index scan (vector_cosine_ops)   - GIN index scan on document_chunks.tsv
  - Top-K dense candidates (DENSE_TOP_K)  - Top-K lexical candidates (LEXICAL_TOP_K)
  - Scoped to workspace & user            - Scoped to workspace & user
    │                                             │
    └──────────────────────┬──────────────────────┘
                           │
                           ▼
             [Reciprocal Rank Fusion (RRF)]
               - Merges dense + lexical candidates by chunk_id
               - Score formula: RRF(d) = Σ 1 / (k + rank_m(d)) (default k=60)
               - Deduplicates chunks and preserves source channels
               - Retains candidate pool (default pool size: 30)
                           │
                           ▼
             [Cross-Encoder Reranker]
               - Model: BAAI/bge-reranker-base (multilingual: English + Vietnamese)
               - Evaluates (query, chunk_text) pairs in micro-batches
               - Offloaded to threadpool (asyncio event loop never blocked)
               - Sigmoid normalization maps logits to [0.0, 1.0]
               - Sorts candidates descending by rerank score; selects Top-N
                           │
                           ▼
             [Relevance Gate]
               - Deterministic threshold check: score >= RELEVANCE_THRESHOLD (0.35)
               - Flags context sufficiency: has_sufficient_evidence
               - Returns ranked chunks with exact page provenance
```

### Cross-Encoder Model Lifecycle & Singleton

- **Zero In-Request Instantiation:** Cross-Encoder models are loaded once per backend process as a thread-safe singleton (`get_reranker_provider()`).
- **Persistent Local Cache:** Weights are stored in `RERANKER_MODEL_CACHE_DIR` or `EMBEDDING_MODEL_CACHE_DIR`. Server restarts reuse cached weights without re-downloading.
- **Multilingual Support:** Defaults to `BAAI/bge-reranker-base`, providing high-accuracy ranking across English and Vietnamese educational materials.
- **Device Fallback:** Automatically selects CUDA if available, falling back cleanly to CPU with memory-safe `torch.no_grad()` and `.eval()` mode.
- **Non-Blocking Inference:** CPU/GPU-bound tokenization and model inference are offloaded via `run_in_threadpool`.

### Database Schema & Migration (BM25 FTS)

Migration file: `supabase/migrations/20261001070000_document_chunks_fts_bm25.sql`

- **Generated tsvector Column:**
  ```sql
  ALTER TABLE public.document_chunks
      ADD COLUMN IF NOT EXISTS tsv tsvector
      GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
  ```
- **GIN Lexical Index:**
  ```sql
  CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON public.document_chunks USING gin(tsv);
  ```

### Configuration Parameters & Defaults

| Setting | Type | Default | Description |
|---|---|---|---|
| `DENSE_TOP_K` | Integer | `25` | Candidates retrieved via dense pgvector cosine search. |
| `LEXICAL_TOP_K` | Integer | `25` | Candidates retrieved via PostgreSQL FTS lexical search. |
| `RRF_K` | Integer | `60` | Reciprocal Rank Fusion smoothing constant. |
| `CANDIDATE_POOL_SIZE` | Integer | `30` | Number of candidate chunks preserved after RRF before reranking. |
| `RERANKER_MODEL_NAME` | String | `BAAI/bge-reranker-base` | Multilingual Cross-Encoder model. |
| `RERANKER_BATCH_SIZE` | Integer | `16` | Micro-batch size for Cross-Encoder inference. |
| `RERANKER_DEVICE` | String | `auto` | Compute device (`auto`, `cpu`, `cuda`). |
| `RERANK_TOP_K` | Integer | `5` | Final top-K ranked evidence chunks returned. |
| `RELEVANCE_THRESHOLD` | Float | `0.35` | Minimum reranker score required to pass relevance gate. |
| `USE_MOCK_RERANKER` | Boolean | `False` | Forces deterministic MockRerankerProvider in tests. |

### Retrieval API Endpoint

| Method | Endpoint | Description | Status |
|---|---|---|---|
| `POST` | `/api/v1/workspaces/{workspace_id}/retrieval/search` | Performs hybrid retrieval and reranking for authorized workspace owner. | `200 OK` |

#### Request Payload
```json
{
  "query": "What is Newton's second law of motion?",
  "dense_top_k": 25,
  "lexical_top_k": 25,
  "rrf_k": 60,
  "candidate_pool_size": 30,
  "rerank_top_k": 5,
  "relevance_threshold": 0.35
}
```

#### Response Structure
```json
{
  "workspace_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "query": "What is Newton's second law of motion?",
  "results": [
    {
      "chunk_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
      "document_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
      "content": "Newton's second law states that F = ma...",
      "page_number_start": 3,
      "page_number_end": 4,
      "chunk_index": 2,
      "dense_score": 0.9124,
      "lexical_score": 0.7651,
      "rrf_score": 0.032787,
      "rerank_score": 0.8845,
      "final_rank": 1,
      "passed_relevance_gate": true,
      "retrieval_sources": ["dense", "lexical"]
    }
  ],
  "total_results": 1,
  "has_sufficient_evidence": true,
  "relevance_threshold": 0.35,
  "timings": {
    "query_embedding_ms": 12.4,
    "dense_retrieval_ms": 18.2,
    "lexical_retrieval_ms": 8.1,
    "rrf_ms": 0.3,
    "rerank_ms": 42.6,
    "total_retrieval_ms": 81.6
  }
}
```

### Developer Inspection UI

The frontend includes a developer-focused **Retrieval Inspector** modal within `DocumentManager` allowing developers to test queries, adjust hyperparameters, inspect timing breakdowns, and verify page provenance in real time. Raw vectors and backend secrets are never exposed.
