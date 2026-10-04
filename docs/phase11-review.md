# Phase 11 productionization review

## Verdict

**CHANGES REQUESTED before production launch.** The repository now contains a durable database-backed ingestion queue, CPU API plus separately scheduled ingestion workers, pinned backend dependencies, health/readiness checks, environment validation, request logging, smoke checks, and deployment guidance. Migration execution and deployment behavior have not been verified against a live environment.

**Phase naming note:** the repository `AGENTS.md` roadmap describes Phase 11 as background processing/performance and Phase 13 as UI polish/deployment. The attached task explicitly named its productionization/deployment scope “Phase 11”; this work follows that requested scope and preserves the existing Phase 10 evaluation artifacts.

## 1. Implemented

- Added a Modal `asgi_app` wrapper that returns the existing FastAPI application and preserves route paths.
- Added production configuration validation for Supabase Auth, database, Gemini, enabled Tavily fallback, and explicit CORS origins. Added `SUPABASE_DB_URL` support with `DATABASE_URL` compatibility.
- Added root `/ready` and `/api/v1/ready`; production readiness checks auth configuration and a lightweight PostgreSQL `SELECT 1`, without model inference.
- Added request IDs, route/status/latency logs, sanitized generic 500 responses, and a declared content-length request-size guard.
- Added bounded upload reads before full file buffering and a Gemini request/stream timeout.
- Added configurable SQLAlchemy pool bounds and limited Modal ASGI input concurrency to a configurable default of 10.
- Added a durable per-document ingestion job table and privileged queue RPCs with leases, bounded retries, and owner-scoped enqueue permissions. Database-backed uploads commit the document and job together; Modal's scheduled CPU poller dispatches individual jobs to a separately configurable CPU/GPU worker. In-memory development mode retains FastAPI background processing.
- Added exact-version Python 3.11/Linux x86_64 locks for CPU API and CUDA-capable ingestion worker dependencies; local test dependencies stay in `requirements-dev.txt`.
- Added structured `chat_stage_metrics` logs and SSE `done.stage_metrics` for retrieval stages, evidence decision, web fallback, provider/model, LLM TTFT, generation, and total request time.
- Added deployment docs, production checklist, and an environment-driven live smoke script.
- Kept provider singletons and retrieval configuration intact. Added a versioned migration, but did not apply it or change any production service.

## 2. Verified

- Backend safe suite: **236 passed, 4 skipped, 2 deselected**. The two deselected tests have existing benchmark output-directory/summary expectation issues. The RLS integration now opts in only through a process environment variable, not `.env`, and skipped without a dedicated disposable DB URL.
- Frontend tests: **23 passed**; Next.js production build passed.
- Python compile check passed for `backend/app`, `backend/deploy`, and `scripts`; environment `pip check` passed.
- `git diff --check`: passed.
- Repository search found no apparent key-shaped Gemini/OpenAI/service-role credential in the changed application, docs, or example configuration. The pre-existing `.gitignore` edit was left untouched.
- Inspected migration files for RLS, foreign keys/cascades, HNSW and GIN indexes. They remain enabled and no database changes were made.
- Inspected current SSE flow, authorization check, event headers, message persistence, model singleton loaders, upload validation, and in-memory/background ingestion behavior.

## 3. Unverified

- The workspace `.venv` launcher still targets a missing Windows Store Python. A task-local Python 3.11.15 interpreter was installed under the ignored workspace temp directory and used with the existing site-packages. `pip check` passed. The lock was resolved for Python 3.11/Linux x86_64, but a clean Linux/Modal image build was not run.
- No Modal deployment, Vercel deployment, production Supabase, live Gemini/Tavily calls, GPU allocation, or public endpoint was created or tested.
- End-to-end SSE disconnect cancellation, partial generation persistence, upstream timeout/error handling, and cross-tenant RLS through PostgreSQL remain unverified. Unit/context regressions ran without connecting to a database.
- Modal scheduling, RPC SQL, migration application, cache volume behavior, and worker execution have not been exercised against Modal or Supabase. The job table/RPC migration must be applied before database-backed uploads can enqueue successfully.
- BGE-M3 and Cross-Encoder query-time retrieval still execute inside the API process. The API is CPU-only by default and ingestion embedding can use an isolated GPU worker, but separate GPU inference serving for chat retrieval is not implemented.
- Target Modal cold start, GPU memory, warm inference, production retrieval latency, and generation TTFT remain unknown.

## 4. Findings

### [RESOLVED IN CODE, MIGRATION NOT APPLIED] Ingestion work is process-local

Database-backed uploads now persist one unique job per document in the same transaction as document metadata. Reprocessing re-arms terminal jobs; active jobs deduplicate. A scheduled Modal poller atomically claims jobs with `SKIP LOCKED` leases, and a separate worker marks completion or applies bounded exponential retries. This depends on applying the new migration and configuring the worker's service-role secret. SQL and lifecycle were inspected and unit-tested, but no PostgreSQL execution was performed. Local in-memory mode remains process-local by design.

### [PARTIALLY RESOLVED] Query-time retrieval still runs on the CPU API

The FastAPI Modal function is CPU-only. Document ingestion/embedding jobs are dispatched to a separate worker whose GPU is optional via `MODAL_INGESTION_GPU`. Chat query embedding and reranking remain in the API process, so the current split does not isolate all model inference. Independent GPU serving for retrieval remains needed if CPU API latency is not acceptable. No hardware benchmark was run.

### [RESOLVED IN REPOSITORY; IMAGE BUILD UNVERIFIED] Dependencies are not fully locked

Modal installs `backend/requirements.lock.txt` for the CPU API/poller and `backend/requirements-gpu.lock.txt` for ingestion workers. Both locks were resolver-checked for Python 3.11/Linux x86_64. The CPU lock selects `torch==2.14.1+cpu` and excludes NVIDIA/CUDA packages; the worker lock retains the CUDA-enabled `torch==2.14.1` dependency closure. Clean image installation and runtime imports still require Docker/Modal build validation.

### [MEDIUM] Request body guard depends on Content-Length

The API rejects declared oversized bodies, and document uploads read at most one byte beyond the file limit. Requests without a trustworthy `Content-Length` still need a hosting-proxy limit or streaming enforcement. Confirm limits at Modal/proxy and multipart layers.

### [RESOLVED IN CODE; SINK/PRODUCTION VALIDATION REMAIN]

Retrieval and chat now emit compact structured stage metrics without query text or document contents, including TTFT and full generation time. The HTTP middleware logs response start and completion after the SSE body finishes. Metrics are currently application logs; no external metrics sink, aggregation, sampling, or production log verification exists.

### [REMAINING] Backend verification has known failures and safety limits

The original `.venv` launcher is unavailable; a task-local interpreter ran the safe suite. Two excluded benchmark tests remain: answer-capture output assumes a results directory exists, and a benchmark-summary assertion expects different metric formatting. The Gemini provider test now verifies the authoritative model. The RLS integration reads only an explicit process environment URL and skips by default; run only against a dedicated disposable database in controlled CI. No production database was contacted successfully or modified.

## 5. Deployment prerequisites

- Resolve the two unrelated benchmark test failures; run the RLS integration only against a dedicated disposable database.
- Validate both CPU and GPU locks in clean Linux container/Modal image builds.
- Create the Modal secret and cache volume manually, then review costs and access controls before `modal deploy`.
- Configure exact Vercel origins, Supabase Auth settings, a persistent-backend-compatible database URL, Gemini, and Tavily secrets.
- Verify private Storage, production RLS, connection-pool capacity, migrations, and token authorization against production-equivalent services.
- Run `scripts/smoke_test_production.py` with a dedicated authorized user and conversation.

## 6. Performance

The Phase 10 retrieval artifact reports local mean query embedding **1,299.530 ms**, reranking **11,461.216 ms**, and total retrieval **12,791.938 ms**. The answer capture reports mean retrieval **839.422 ms** and generation **3,646.739 ms** over its successful answers. These artifacts cover different captures and neither predicts Modal production latency. GPU may reduce model inference time, but cold start, GPU memory, queueing, DB time, Gemini latency, and network overhead remain unmeasured.

## 7. Security findings

- No RLS policy was removed or weakened. Backend DB sessions continue to set transaction-local identity from the verified JWT.
- Chat checks conversation ownership before opening SSE; document routes check ownership before access, and storage uses the existing private bucket path.
- No production secrets were added to source. The Modal secret name is configurable; secret values are not embedded in the wrapper.
- Production CORS rejects wildcard, localhost, and loopback origins.
- Verify deployed secrets, Supabase bucket privacy, database role permissions, and proxy request limits in the target accounts before launch.

## 8. Configuration discrepancy

Current code defaults are authoritative for behavior when environment overrides are absent: dense/lexical top-k **25/25**, candidate pool **30**, rerank top-k **5**, routing **adaptive**, RRF k **60**, relevance threshold **0.35**, minimum answerable score **0.55**, BGE-M3, and `BAAI/bge-reranker-v2-m3`. The Phase 10 captured retrieval artifact records **20/20**, pool **20**, rerank top-k **10**, and `always_quality`, with the same listed score thresholds and model family. `.env.example` reflects the current code defaults. These configurations remain separate; this phase did not tune production retrieval to match benchmark capture metadata.

## 9. Recommended next steps

1. Resolve the two existing benchmark failures and run the complete non-database backend suite.
2. Build the pinned Modal image and verify SSE behavior in a controlled deployment.
3. Apply and validate the durable-job migration on a disposable Supabase-compatible database; verify retries, lease recovery, RLS, and cancellation.
4. Decide whether CPU query-time retrieval meets latency needs; isolate it behind a model-serving worker if not.
5. Connect structured logs to the production metrics platform and measure cold/warm deployment performance.
