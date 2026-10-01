# AGENTS.md — Root Engineering Instructions for AI Tutor Assistant

## 1. Project Purpose & System Overview
**AI Tutor Assistant** is an enterprise-grade, document-grounded AI learning assistant designed to help users learn from their uploaded documents (PDF/DOCX) within organized workspaces.

### Core Product Tenets
1. **Document-First Grounding:** Uploaded documents are the primary source of truth. All answers must prioritize document evidence.
2. **Web Search as Pure Fallback:** Web search is strictly used when document retrieval yields insufficient relevance. Web evidence must never supersede document evidence.
3. **Explicit Source Attribution:** Final responses must cleanly separate and clearly identify uploaded-document citations (with page numbers) from web citations.
4. **Persistent Multi-Tenant State:** Users, workspaces, documents, conversations, and messages must persist in the database across page reloads.
5. **Decoupled Performance & Streaming:** Fast initial page loads; zero model loading in the request path; responsive token streaming for generation.

---

## 2. Planned Technology Stack & Boundaries
- **Frontend (`/frontend`):** Next.js (App Router), React, TypeScript, Tailwind CSS. Strictly responsible for presentation, UI state, auth session consumption, and stream rendering.
- **Backend (`/backend`):** FastAPI (Python 3.11+), Pydantic v2. Responsible for business logic, document processing pipelines, hybrid search, ML model coordination, and streaming endpoints.
- **Database & Storage:** Supabase PostgreSQL with `pgvector` for vector embeddings; Supabase Storage for raw document storage; Supabase Auth for identity.
- **AI & Embedding Models:**
  - LLM: Google Gemini API (gemini-1.5-pro / gemini-1.5-flash)
  - Dense Embeddings: `BAAI/bge-m3` (1024-dim, dense + sparse/lexical capabilities)
  - Reranker: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-large`)

---

## 3. Autonomous Agent Operating Principles
Every coding agent modifying this repository must abide by these rules:
1. **Inspect Before Action:** Read relevant files, schemas, and tests before writing any code. Never assume existing implementations.
2. **Never Blindly Rewrite:** Make surgical, targeted edits. Preserve existing working features, interfaces, and docstrings.
3. **No Phantom Code:** Never hallucinate database columns, API routes, library exports, or config values. Verify against schemas and lockfiles.
4. **Plan Before Execution:** Explicitly outline architectural decisions and files to touch before writing code.
5. **Small, Testable Iterations:** Implement features incrementally with unit/integration validation at each step.
6. **Verify Behavior, Not Just Types:** Verify actual runtime execution (FastAPI responses, SSE streaming, Next.js page renders), not merely static type checks.
7. **Report Assumptions:** Document any external dependency or configuration assumptions explicitly.
8. **Phase-Aware Development Discipline:** Implement **only** the phase explicitly requested by the user. Do **not** anticipate or scaffold future phases prematurely.
9. **Zero Secret Leakage:** Never commit secrets, tokens, or credentials. Never expose backend secrets to client-facing code.
10. **Maintain Modular Boundaries:** Keep ingestion, retrieval, reranking, and generation strictly decoupled.
11. **Strict Stack & Architecture Discipline:** Do not introduce a technology, service, dependency, model, or architecture decision that is not explicitly approved in the current project plan. When multiple valid choices exist, present the options and rationale before making a major architectural change.


---

## 4. Phase-Aware Development Roadmap
Development is strictly partitioned into distinct phases. Agents must only work on the currently assigned phase:
- **Phase 1: Foundation** — Project scaffolding, linting, formatting, shared configs, base health checks.
- **Phase 2: Authentication + Workspace** — Supabase auth, user workspaces, workspace switcher, CRUD APIs.
- **Phase 3: Conversation Persistence** — Multi-conversation threads per workspace, message history schema & persistence.
- **Phase 4: Document Upload + Storage** — PDF/DOCX file uploads, Supabase storage bucket, file validation, doc metadata persistence.
- **Phase 5: Document Ingestion** — Text extraction, cleaning, structure-aware chunking, page-number tracking, chunk persistence.
- **Phase 6: Embedding + Vector Database** — BGE-M3 integration, local weight caching, batch embedding, pgvector storage with HNSW index.
- **Phase 7: Hybrid Retrieval + Reranking** — Dense vector search + BM25 lexical search, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking.
- **Phase 8: RAG Generation** — Prompt assembly, grounded LLM generation via Gemini API, prompt injection guards.
- **Phase 9: Document-First + Web Fallback** — Relevance gating, automated web search fallback when document evidence is insufficient.
- **Phase 10: Citation + Streaming** — Token streaming over SSE, inline document citations (`[Doc: file.pdf, p. 3]`), web source cards.
- **Phase 11: Background Processing + Performance** — Asynchronous task queues for ingestion, query optimization, connection pooling.
- **Phase 12: Evaluation** — RAG metrics benchmark (Recall@K, MRR, Faithfulness, Answer Relevance).
- **Phase 13: UI Polish + Deployment** — Production build audits, responsive UI refinement, Dockerization, production deployment.

---

## 5. Architectural & System Constraints

### 5.1 RAG Architecture & Logical Flow
Every RAG pipeline query must follow this exact sequence:
```
User Query
   │
   ▼
1. Uploaded Document Retrieval (Dense + BM25 in active workspace)
   │
   ▼
2. Reciprocal Rank Fusion (RRF) -> Candidate Pool
   │
   ▼
3. Cross-Encoder Reranking
   │
   ▼
4. Relevance Gating (Score threshold & context evaluation)
   ├─── Sufficient Evidence ───► Grounded Generation (Document Context)
   └─── Insufficient Evidence ─► Web Search Fallback ──► Grounded Generation (Web Context)
   │
   ▼
5. Token Streaming & Explicit Citation Attribution
```
- **Document Evidence Priority:** Web search must **never** run if document evidence meets the relevance threshold.
- **Citation Integrity:** Every citation must map to an existing `chunk_id` and document page. Fabricated citations are treated as critical bugs.

### 5.2 Model Caching & Lifecycle
- **Never Load Models in Request Handlers:** BGE-M3 and Cross-Encoder rerankers must load once during FastAPI lifespan startup or via process-level singletons.
- **Local Weight Caching:** Cache model checkpoints in a designated local directory (`HF_HOME` / persistent cache). Never re-download model weights per request or on normal server restarts.
- **Decoupled Frontend:** Frontend loading states must never block on backend ML model warmups.

### 5.3 Asynchronous Processing & Streaming
- **Heavy Ingestion is Non-Blocking:** Document parsing, chunking, and embedding must execute in background tasks or workers. Document upload endpoints must return immediately with a job ID and processing status.
- **Token Streaming:** LLM generation endpoints must stream tokens via Server-Sent Events (SSE) or chunked HTTP transfer to minimize Time-to-First-Token (TTFT).

### 5.4 Database & Persistence
- **Zero Browser-Memory Reliance:** Workspaces, conversations, messages, documents, and chunks must persist in PostgreSQL. Reloading any page must restore exact application state.
- **Migration Discipline:** All schema modifications must occur via explicit, versioned SQL migrations.
- **Referential Integrity:** Maintain foreign-key constraints with explicit cascade rules (e.g., deleting a workspace deletes its conversations and documents; deleting a document deletes its chunks).

### 5.5 Security & Multi-Tenancy
- **Strict Data Isolation:** Every database query and storage lookup must be scoped to the authenticated `user_id` and selected `workspace_id`.
- **Secret Isolation:** API keys (Gemini, Supabase service keys) live strictly in backend environment variables (`.env`). No server secrets may be prefixed with `NEXT_PUBLIC_`.
- **Upload Validation:** Strictly validate file extensions, MIME types, and magic bytes for PDF and DOCX before writing to storage.

---

## 6. Repository Rules & Skills Index

### Domain Rules (`/.agents/rules/`)
Consult domain-specific rule files before implementing or modifying code:
- [`frontend.md`](file:///.agents/rules/frontend.md) — Next.js, React, TypeScript, Tailwind, UI state, and streaming client rules.
- [`backend.md`](file:///.agents/rules/backend.md) — FastAPI, Pydantic schemas, dependency injection, async endpoints, and service layer.
- [`rag.md`](file:///.agents/rules/rag.md) — Retrieval, reranking, relevance gating, prompt formatting, citation rules, and web fallback.
- [`security.md`](file:///.agents/rules/security.md) — Authentication, data isolation, secret management, upload sanitization, injection defense.
- [`performance.md`](file:///.agents/rules/performance.md) — Model lifecycle, model caching, background workers, query optimization, and latency limits.
- [`database.md`](file:///.agents/rules/database.md) — Supabase PostgreSQL, pgvector indexing, migrations, schemas, and referential integrity.

### Specialized Agent Skills (`/.agents/skills/`)
Activate the relevant skill when undertaking complex workflows:
- [`project-architecture`](file:///.agents/skills/project-architecture/SKILL.md) — Structural guidance, dependency boundaries, feature planning.
- [`document-ingestion`](file:///.agents/skills/document-ingestion/SKILL.md) — Document parsing, cleaning, chunking, metadata extraction, embedding pipeline.
- [`rag-engineering`](file:///.agents/skills/rag-engineering/SKILL.md) — Hybrid retrieval (BM25 + Dense), RRF, reranking, relevance gate, generation.
- [`rag-evaluation`](file:///.agents/skills/rag-evaluation/SKILL.md) — Quantitative benchmarking (Recall@K, MRR, Faithfulness, Answer Relevance).
- [`performance-audit`](file:///.agents/skills/performance-audit/SKILL.md) — Latency profiling, TTFT analysis, memory auditing, model warmup optimization.
- [`production-review`](file:///.agents/skills/production-review/SKILL.md) — Senior engineer pre-merge audit covering reliability, security, and cleanliness.
