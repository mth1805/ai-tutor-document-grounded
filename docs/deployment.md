# Deployment guide

This guide describes the current application deployment path. No resources are provisioned by the repository itself.

## A. Architecture

- Next.js 14 frontend on Vercel.
- Existing FastAPI application exposed as an ASGI app on Modal (`backend/deploy/modal_app.py`). The wrapper returns `app.main:app`; REST and SSE route paths stay the same.
- Supabase provides Auth, PostgreSQL/pgvector, and private Storage. Gemini generates responses and Tavily is used only by the existing web fallback.
- The API is CPU-only and currently invokes query-time embedding/reranking inside the API worker. Database-backed ingestion is queued durably; a scheduled CPU poller claims jobs and dispatches one job per separate ingestion worker. Set `MODAL_INGESTION_GPU` only when the ingestion worker should use a GPU. Query-time retrieval inference has not been isolated and may remain slower on CPU.

## B. Environment variables

### Local development

Copy `.env.example` to `.env`; use local frontend URL `http://localhost:3000`, local API URL `http://localhost:8000`, and development-only credentials. The frontend reads its API endpoint from `NEXT_PUBLIC_API_BASE_URL` (with legacy `NEXT_PUBLIC_API_URL` compatibility) and has no baked-in API host. `frontend/.env.local` is ignored and must not contain server secrets.

### Production backend

Configure these as Modal Secret values (never in source control):

| Variable | Purpose |
| --- | --- |
| `ENVIRONMENT` | Set to `production`; enables required configuration validation and DB readiness checks. |
| `SUPABASE_URL`, `SUPABASE_ANON_KEY` | Supabase Auth verification configuration. |
| `SUPABASE_DB_URL` | `asyncpg` PostgreSQL URL. A Supavisor session-mode or direct connection is appropriate for this persistent ASGI process. Legacy `DATABASE_URL` is accepted if this is absent. |
| `SUPABASE_SERVICE_ROLE_KEY` | Backend-only Storage access where required; never send to Vercel. |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini generation. |
| `TAVILY_API_KEY`, `TAVILY_MAX_RESULTS`, `WEB_SEARCH_TIMEOUT_SECONDS` | Bounded Tavily fallback. |
| `BACKEND_CORS_ORIGINS` | JSON array of exact Vercel origins, e.g. `["https://app.example.com"]`; no wildcard. |
| `LLM_PROVIDER`, `LLM_MAX_OUTPUT_TOKENS`, `LLM_STREAMING_TIMEOUT_SECONDS` | Existing LLM selection and generation limits. |
| `LOG_LEVEL`, `PREWARM_MODELS` | Logging and optional startup model warmup. |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT_SECONDS` | Per-container SQLAlchemy pool budget; coordinate with Modal container concurrency and the Supabase connection limit. |

`SUPABASE_DB_URL` and all database/service keys are backend-only. Use a database URL appropriate to a persistent backend and its connection budget; do not use a transaction-pool URL unless its prepared-statement and session behavior has been verified with this SQLAlchemy/asyncpg configuration.

## C. Supabase configuration

Apply the committed migrations through the normal Supabase migration workflow before deploying. Preserve RLS policies. Backend queries use the verified user's `authenticated` role and transaction-local JWT claims. Keep the `documents` bucket private. Configure Auth's allowed redirect URLs for the production Vercel origin. Do not apply ad-hoc production DDL.

## D. Gemini configuration

The authoritative project default is `GEMINI_MODEL=gemini-3.5-flash-lite`; set that value and the API key in the backend secret store. Provider calls are asynchronous; the configured streaming timeout is `LLM_STREAMING_TIMEOUT_SECONDS`. Check current provider quotas and model availability in your account before launch.

## E. Tavily configuration

Set `TAVILY_API_KEY`; `TAVILY_MAX_RESULTS` is bounded by the application's source limit and `WEB_SEARCH_TIMEOUT_SECONDS` bounds waiting. If fallback is intentionally disabled, set `WEB_SEARCH_FALLBACK_ENABLED=false` and do not require a Tavily key.

## F. Modal backend setup

1. Install and authenticate the Modal CLI locally (`pip install modal`, then `modal setup`).
2. Create a Modal Secret named `ai-tutor-production` containing the production backend variables above. Or set `MODAL_SECRET_NAME` to a secret you created.
3. From the repository root, deploy manually with `modal deploy backend/deploy/modal_app.py`.
4. Modal uses `backend/requirements.lock.txt` for the CPU API and poller, and `backend/requirements-gpu.lock.txt` only for ingestion workers. Both are exact Python 3.11/Linux x86_64 locks. Their source inputs are `requirements-cpu.in` (PyTorch CPU index and `torch==2.14.1+cpu`) and `requirements-gpu.in` (`torch==2.14.1` from PyPI); both keep the same Sentence Transformers, BGE-M3, and CrossEncoder dependencies. The API lock excludes CUDA/NVIDIA runtime packages, while the worker lock retains them. Both images install Tesseract with English/Vietnamese language data and LibreOffice Writer for scanned-PDF/image OCR and legacy `.doc` uploads. The root Dockerfile installs the CPU lock. Validate clean Linux image builds before rollout.
5. Apply the durable ingestion migration using the normal reviewed Supabase migration workflow before deploying this application version. Do not apply ad-hoc production DDL. The worker queue requires `SUPABASE_URL` and backend-only `SUPABASE_SERVICE_ROLE_KEY`; only the narrowly scoped queue RPCs use that key. Document content processing continues through the authenticated per-user database session.
6. The wrapper uses a persistent Modal Volume named `ai-tutor-model-cache` (override with `MODAL_MODEL_CACHE_VOLUME`) mounted at `/models`; Hugging Face uses `/models/huggingface`.
7. API and worker CPU/memory are configurable with `MODAL_API_CPU`, `MODAL_API_MEMORY_MB`, `MODAL_INGESTION_CPU`, and `MODAL_INGESTION_MEMORY_MB`. `MODAL_MAX_CONCURRENT_INPUTS` bounds ASGI concurrency per container. Set this with SQLAlchemy pool settings and Supabase connection limits in mind. `MODAL_INGESTION_GPU` optionally assigns a GPU only to per-job ingestion workers. The scheduled poller runs every 15 seconds, claims up to 10 due jobs with a 60-minute lease, and workers retry failures up to three attempts with backoff. The worker timeout is 30 minutes; expired leases are reclaimed by the database RPC.

`modal deploy` provisions public API and scheduled worker functions, and may incur charges. Do not run it until you intend to create those resources. The repository does not deploy automatically.

## G. Model configuration and queue operation

`EMBEDDING_DEVICE=auto` and `RERANKER_DEVICE=auto` select CUDA when available and otherwise CPU. The API image contains CPU-only PyTorch and serves query-time embedding/reranking on CPU. The separate ingestion-worker image contains CUDA-enabled PyTorch; set `MODAL_INGESTION_GPU` to attach a GPU to those workers. The model providers are process singletons and receive configured Hugging Face cache paths; models are not downloaded on every request. `PREWARM_MODELS=true` loads both models at startup, increasing cold start and startup memory. With prewarming off, first retrieval incurs lazy model initialization. Database access, auth, API orchestration, Gemini, and Tavily do not require a GPU.

Ingestion retries are persisted in `document_ingestion_jobs`; a document can show queued, processing, processed, or failed. Terminal jobs are re-armed when a user requests reprocessing. Job error records contain a bounded generic failure category, not document content. Queue RPC and Modal execution have not been run against a live service in this review. Verify model cache volume concurrency and worker memory on controlled infrastructure before production. Query-time retrieval remains CPU-bound in the API unless a later model-serving boundary is added.

Model memory needs, cold-start duration, and warm inference latency are unknown for target Modal hardware. The Phase 10 CPU artifact measured mean query embedding 1,299.530 ms, reranking 11,461.216 ms, and total retrieval 12,791.938 ms. These are local benchmark measurements, not production latency estimates. The Phase 10 answer capture reports mean retrieval 839.422 ms and mean generation 3,646.739 ms for its successful subset, a different capture. GPU behavior remains unmeasured.

## H. Vercel frontend setup

Set `NEXT_PUBLIC_API_BASE_URL` to the deployed Modal URL (no trailing slash). Set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` only when needed by the current browser auth flow. Never set service-role, DB, Gemini, or Tavily credentials in Vercel browser variables. Preview deployments need their own exact origins added to backend CORS if they must call the API.

## I. CORS configuration

Set `BACKEND_CORS_ORIGINS` to a JSON array of exact trusted origins. Production rejects wildcard configuration. Development defaults to localhost origins. CORS is not an authentication mechanism; protected endpoints still require bearer authentication.

## J. Health and readiness

- `GET /health` is a process liveness check and does not load models.
- `GET /ready` is a readiness check. In production it checks auth configuration and executes `SELECT 1` against PostgreSQL; it does not run model inference.
- `GET /api/v1/health` remains available for compatibility.

## K. SSE verification

Use the script below with a valid bearer token and conversation owned by that user. It checks the response content type and receives at least one SSE frame. Streaming depends on the hosting proxy forwarding `text/event-stream` without buffering; verify this against the deployed endpoint with `curl -N` as well.

## L. Smoke test checklist

```powershell
$env:PRODUCTION_API_BASE_URL = "https://<modal-url>"
$env:PRODUCTION_BEARER_TOKEN = "<user-access-token>"
$env:PRODUCTION_CONVERSATION_ID = "<owned-conversation-uuid>"
$env:PRODUCTION_CHAT_MESSAGE = "Summarize the uploaded notes"
python scripts/smoke_test_production.py
```

Run with actual deployment credentials only when you choose to perform a live check. For no-secret local checks, omit the token and conversation ID; the script checks health/readiness and verifies protected access rejects unauthenticated requests.

## M. Rollback and troubleshooting

- Redeploy the previously known-good Modal code revision; use Modal deployment history/rollback controls.
- Revert Vercel to its previous deployment and restore the matching API base URL.
- Do not roll back by dropping tables or disabling RLS. Database migrations are additive/currently deployed outside this app wrapper; use a reviewed forward migration for schema repair.
- `503 /ready`: check `SUPABASE_DB_URL`, network access, credentials, and pooler mode.
- `401` on protected routes: check Supabase URL/anon key, issuer/audience configuration, and token expiry.
- SSE appears buffered: inspect the proxy path and response headers; use `curl -N`.
- Slow first retrieval: distinguish model cold start from warm inference. Check Modal cache volume mount and model device logs before comparing with Phase 10.
