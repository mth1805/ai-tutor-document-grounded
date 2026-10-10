# Deployment guide

This guide describes the current application deployment path. No resources are provisioned by the repository itself.

For the CPU-only three-service local Compose demo and manual Dev queue migration, see [local Docker runtime](local-docker-runtime.md). Modal production boundaries stay unchanged.

## A. Architecture

- Next.js 14 frontend on Vercel.
- Existing FastAPI application exposed as an ASGI app on Modal (`backend/deploy/modal_app.py`). The wrapper returns `app.main:app`; REST and SSE route paths stay the same.
- Supabase provides Auth, PostgreSQL/pgvector, and private Storage. Gemini generates responses and Tavily is used only by the existing web fallback.
- The API is CPU-only and currently invokes query-time embedding/reranking inside the API worker. Database-backed ingestion is queued durably; a scheduled CPU poller claims jobs and dispatches one job per separate ingestion worker. Set `MODAL_INGESTION_GPU` only when the ingestion worker should use a GPU. Query-time retrieval inference has not been isolated and may remain slower on CPU.

## B. Environment variables

### Local development

Copy the root `.env.example` to `.env` for backend/Compose, and `frontend/.env.example` to `frontend/.env.local` for browser-safe frontend values. Local frontend/API origins are `http://localhost:3000` and `http://localhost:8000`. The centralized frontend configuration reads `NEXT_PUBLIC_API_BASE_URL` (legacy `NEXT_PUBLIC_API_URL` compatibility), defaults to localhost only in development, and requires an explicit origin for production builds. Private env files are ignored and frontend env must never contain backend secrets.

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
4. Modal uses `backend/requirements.lock.txt` for the CPU API, poller, and default CPU ingestion worker. Only explicitly GPU-configured ingestion workers use `backend/requirements-gpu.lock.txt`. Both are exact Python 3.11/Linux x86_64 locks. Their source inputs are `requirements-cpu.in` (PyTorch CPU index and `torch==2.14.1+cpu`) and `requirements-gpu.in` (`torch==2.14.1` from PyPI); both keep the same Sentence Transformers, BGE-M3, and CrossEncoder dependencies. The API lock excludes CUDA/NVIDIA runtime packages, while the GPU worker lock retains them. Both images install Tesseract with English/Vietnamese language data and LibreOffice Writer for scanned-PDF/image OCR and legacy `.doc` uploads. The root Dockerfile installs the CPU lock. Validate clean Linux image builds before rollout.
5. Apply the durable ingestion migration using the normal reviewed Supabase migration workflow before deploying this application version. Do not apply ad-hoc production DDL. The worker queue requires `SUPABASE_URL` and backend-only `SUPABASE_SERVICE_ROLE_KEY`; only the narrowly scoped queue RPCs use that key. Document content processing continues through the authenticated per-user database session.
6. The wrapper uses a persistent Modal Volume named `ai-tutor-model-cache` (override with `MODAL_MODEL_CACHE_VOLUME`) mounted at `/models`; Hugging Face uses `/models/huggingface`.
7. API and worker CPU/memory are configurable with `MODAL_API_CPU`, `MODAL_API_MEMORY_MB`, `MODAL_INGESTION_CPU`, and `MODAL_INGESTION_MEMORY_MB`. `MODAL_MAX_CONCURRENT_INPUTS` bounds ASGI concurrency per container. Set this with SQLAlchemy pool settings and Supabase connection limits in mind. `MODAL_INGESTION_GPU` defaults to no GPU: an absent, empty, or whitespace-only value selects CPU. An explicit GPU name allocates that GPU only to per-job ingestion workers. API and poller stay CPU-only. The scheduled poller runs every 15 seconds, claims up to 10 due jobs with a 60-minute lease, and workers retry failures up to three attempts with backoff. The worker timeout is 30 minutes; expired leases are reclaimed by the database RPC.

`modal deploy` provisions public API and scheduled worker functions, and may incur charges. Do not run it until you intend to create those resources. The repository does not deploy automatically.

## G. Model configuration and queue operation

`EMBEDDING_DEVICE` and `RERANKER_DEVICE` default to `cpu`. Explicit `auto` settings still select CUDA when available and otherwise CPU; explicit `cuda` retains the existing CPU fallback when CUDA is unavailable. The API image contains CPU-only PyTorch and serves query-time embedding/reranking on CPU. The ingestion worker selects the CPU image by default, or the CUDA-enabled image when a GPU is explicitly configured. It sets its own embedding device to match that selection before importing worker Settings, keeps the cache at `/models/huggingface/hub` (the normal HF_HOME hub directory), and enables automatic embedding. The model providers remain process singletons and receive configured Hugging Face cache paths; models are not downloaded on every request. `PREWARM_MODELS=true` loads both models at startup, increasing cold start and startup memory. With prewarming off, first retrieval incurs lazy model initialization. Database access, auth, API orchestration, Gemini, Tavily, and the SymPy math solver do not require a GPU.

### Compute-device modes

- **Local CPU:** `EMBEDDING_DEVICE=cpu` and `RERANKER_DEVICE=cpu`. These are the
  shared defaults, and Compose explicitly overrides both devices to CPU for the
  backend and local worker. Local Docker installs the CPU dependency lock and
  requires no NVIDIA runtime. A Modal GPU setting never changes this profile.
- **Modal CPU:** Leave `MODAL_INGESTION_GPU` unset or use `MODAL_INGESTION_GPU=`.
  The worker uses `image`, requests `gpu=None`, and sets `EMBEDDING_DEVICE=cpu`.
- **Modal GPU:** Use `MODAL_INGESTION_GPU=T4`, `MODAL_INGESTION_GPU=L4`, or another
  GPU supported by your Modal account. The worker uses `gpu_image`, requests that
  exact trimmed GPU name, and sets `EMBEDDING_DEVICE=cuda`. FastAPI and the poller
  continue to use the CPU image with no GPU allocation in every mode.

`MODAL_INGESTION_GPU` is read from the deployment launcher's environment when the
wrapper is imported and resources are registered. Set it in that environment
before a future manual deployment; putting it only in runtime Modal Secret values
does not select deployment resources. The wrapper does not load the local `.env`
to allocate GPUs. Actual GPU provisioning still requires account access to the
requested hardware; missing or blank configuration selects CPU, while unavailable
explicit allocations must be resolved in the deployment environment. No remote
GPU build or execution is implied by local configuration tests.

See [ingestion and document preview changes](production-ingestion-preview.md) for storage compensation, preview lifecycle, validation limits, and manual rollout commands.

Ingestion retries are persisted in `document_ingestion_jobs`; a document can show queued, processing, processed, or failed. Terminal jobs are re-armed when a user requests reprocessing. Job error records contain a bounded generic failure category, not document content. Queue RPC and Modal execution have not been run against a live service in this review. Verify model cache volume concurrency and worker memory on controlled infrastructure before production. Query-time retrieval remains CPU-bound in the API unless a later model-serving boundary is added.

Model memory needs, cold-start duration, and warm inference latency are unknown for target Modal hardware. The Phase 10 CPU artifact measured mean query embedding 1,299.530 ms, reranking 11,461.216 ms, and total retrieval 12,791.938 ms. These are local benchmark measurements, not production latency estimates. The Phase 10 answer capture reports mean retrieval 839.422 ms and mean generation 3,646.739 ms for its successful subset, a different capture. GPU behavior remains unmeasured.

## H. Vercel frontend setup

Follow the dedicated [Vercel deployment guide](vercel-deployment.md). All three public variables (`NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`) are required for Vercel builds. Never set service-role, DB, Gemini, or Tavily credentials in Vercel. Preview deployments need exact trusted CORS origins and their own reviewed auth configuration.

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
