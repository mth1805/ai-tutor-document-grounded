# Deployment guide

This guide describes the current application deployment path. No resources are provisioned by the repository itself.

For the CPU-only three-service local Compose demo and manual Dev queue migration, see [local Docker runtime](local-docker-runtime.md). Modal production boundaries stay unchanged.

## A. Architecture

- Next.js 14 frontend on Vercel.
- Existing FastAPI application exposed as an ASGI app on Modal (`backend/deploy/modal_app.py`). The wrapper returns `app.main:app`; REST and SSE route paths stay the same.
- Supabase provides Auth, PostgreSQL/pgvector, and private Storage. Gemini generates responses and Tavily is used only by the existing web fallback.
- The API and durable ingestion worker are CPU-only. Database-backed ingestion is queued durably; a scheduled CPU poller claims jobs and dispatches one job per worker. A separate authenticated GPU inference endpoint handles embeddings and reranking when configured; bounded requests fall back to the process-local CPU model.

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
2. Create a Modal Secret named `ai-tutor-production` containing the production backend variables above and a random server-only `MODAL_INFERENCE_TOKEN`. Or set `MODAL_SECRET_NAME` to a secret you created.
3. From the repository root, set `MODAL_INFERENCE_GPU=T4` (or another supported Modal GPU) in the deployment shell, then run `modal deploy backend/deploy/modal_app.py`. Copy the `gpu_inference` endpoint URL from deployment output into `MODAL_INFERENCE_URL` in the backend Modal Secret, then redeploy the callers with the same command. The first deployment without the URL safely uses CPU fallback.
4. Modal uses `backend/requirements.lock.txt` for the CPU API, poller, and durable ingestion worker. The dedicated inference endpoint uses `backend/requirements-gpu.lock.txt`. Both are exact Python 3.11/Linux x86_64 locks. Their source inputs are `requirements-cpu.in` (PyTorch CPU index and `torch==2.14.1+cpu`) and `requirements-gpu.in` (`torch==2.14.1` from PyPI); both keep the same Sentence Transformers, BGE-M3, and CrossEncoder dependencies. The API lock excludes CUDA/NVIDIA runtime packages, while the GPU endpoint lock retains them. The CPU image includes Tesseract with English/Vietnamese language data and LibreOffice Writer for scanned-PDF/image OCR and legacy `.doc` uploads; the smaller inference image does not include those ingestion tools. The root Dockerfile installs the CPU lock. Validate clean Linux image builds before rollout.
5. Apply the durable ingestion migration using the normal reviewed Supabase migration workflow before deploying this application version. Do not apply ad-hoc production DDL. The worker queue requires `SUPABASE_URL` and backend-only `SUPABASE_SERVICE_ROLE_KEY`; only the narrowly scoped queue RPCs use that key. Document content processing continues through the authenticated per-user database session.
6. The wrapper uses a persistent Modal Volume named `ai-tutor-model-cache` (override with `MODAL_MODEL_CACHE_VOLUME`) mounted at `/models`. After deploying the wrapper but before routing traffic, run `modal run backend/deploy/modal_app.py::prepare_model_cache`; it downloads both model checkpoints and commits the cache. Runtime API, ingestion, and inference containers mount that volume read-only and run Hugging Face in offline mode. Repeat cache preparation before deploying changes to either model checkpoint. Model-file cache compatibility must be validated in a staged deployment.
7. API and worker CPU/memory are configurable with `MODAL_API_CPU`, `MODAL_API_MEMORY_MB`, `MODAL_INGESTION_CPU`, and `MODAL_INGESTION_MEMORY_MB`. `MODAL_MAX_CONCURRENT_INPUTS` bounds ASGI concurrency per container. The API, poller, and durable ingestion job runner remain CPU-only. Only `gpu_inference` owns CUDA; it performs embedding/reranking inference and never accesses the ingestion queue or database. The scheduled poller runs every 15 seconds, claims up to 10 due jobs with a 60-minute lease, and workers retry failures up to three attempts with backoff. The worker timeout is 30 minutes; expired leases are reclaimed by the database RPC.

`modal deploy` provisions public API and scheduled worker functions, and may incur charges. Do not run it until you intend to create those resources. The repository does not deploy automatically.

## G. Model configuration and queue operation

`DEVICE_MODE=auto|cuda|cpu` controls model inference, and `GPU_FALLBACK_ENABLED=true` enables CPU fallback. Configure `MODAL_INFERENCE_URL` and `MODAL_INFERENCE_TOKEN` in the server-side Modal Secret. The endpoint URL is printed by `modal deploy` or shown in the Modal dashboard; set it in the secret before deploying the API/worker callers. The inference worker validates CUDA with an actual tensor operation, caches separate BGE and Cross-Encoder singletons, retries an OOM once with half the batch size, and validates numeric outputs. A bounded `GPU_WORKER_TIMEOUT_SECONDS` covers endpoint scheduling and execution; timeouts, scheduling errors, other failures, or invalid output use the existing CPU model when fallback is enabled. CPU fallback has a configurable `CPU_FALLBACK_TIMEOUT_SECONDS`. The API and ingestion job runner use the CPU-only PyTorch lock and initialize their CPU model only if needed for fallback. Local CPU development requires no Modal GPU or credential.

### Compute-device modes

- **Local CPU:** `DEVICE_MODE=cpu` (the `.env.example` setting). Compose explicitly overrides both devices to CPU for the
  backend and local worker. Local Docker installs the CPU dependency lock and
  requires no NVIDIA runtime. Start the stack from the repository root with
  `docker compose up --build`; a Modal GPU setting never changes this profile.
- **Modal CPU fallback:** Set `DEVICE_MODE=cpu`. No inference URL/token is required.
- **Modal GPU-first:** Set `DEVICE_MODE=auto`, `GPU_FALLBACK_ENABLED=true`,
  `MODAL_INFERENCE_GPU=T4` or another
  GPU supported by your account. Only `gpu_inference` receives CUDA. API and
  ingestion callers wait up to `GPU_WORKER_TIMEOUT_SECONDS`, then use CPU.

`MODAL_INFERENCE_GPU` is read by the deploy launcher when resources are
registered. Set it there before deployment; putting it only in runtime Secret
values does not select hardware. The old `MODAL_INGESTION_GPU` variable is
temporarily accepted as a deprecated alias only when `MODAL_INFERENCE_GPU` is
unset; it never allocates a GPU to ingestion. `MODAL_INFERENCE_URL` and
`MODAL_INFERENCE_TOKEN` are runtime server-side secret values. The GPU endpoint
uses bearer-token authentication and does not expose Modal credentials to the
frontend. No live GPU build or execution is implied by local tests.

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

## M. Optional daily development Supabase activity check

Supabase may pause Free projects with low activity over seven days to conserve
resources ([Supabase availability guidance](https://supabase.com/docs/guides/deployment/going-into-prod)).
The optional `Supabase development keep-alive` GitHub Actions workflow performs a
real database read once daily at **03:17 UTC (10:17 Asia/Saigon)** using cron
`17 3 * * *`. This may help maintain activity; it is **not a guarantee against
inactivity pausing**, availability issues, or future platform policy changes.

The workflow talks directly to Supabase PostgREST. It never contacts FastAPI,
Gemini, Tavily, Modal, Storage, Auth login, or any model loader. It needs no laptop,
application dependencies, Docker build, GPU, or always-running process.

### Setup for `ai-tutor-dev` only

1. In the Supabase dashboard, positively identify **ai-tutor-dev** and copy its
   project reference. Confirm that it is not your production project. Restore the
   development project manually first if it is already paused.
2. Review and apply
   `supabase/migrations/20261018140000_keepalive_heartbeat.sql` through the normal
   migration process against that development project. This additive, idempotent
   migration creates a two-column singleton and seeds `(1, 'ai-tutor-dev')` once.
   It enables RLS, removes default role grants, and grants `anon` only
   SELECT with a policy limited to that row. No application/auth policies change.
   The daily checker never runs migrations or writes rows. The static label is
   not proof of project identity; confirming the development project reference is
   an operator setup requirement. Do not point the workflow at production.
3. In the GitHub repository, open **Settings → Secrets and variables → Actions →
   New repository secret** and add these three values from **ai-tutor-dev**:
   - `SUPABASE_DEV_URL`: its HTTPS project origin, without an API path.
   - `SUPABASE_DEV_PROJECT_REF`: independently confirmed project reference. The
     script requires the URL hostname to match this reference exactly.
   - `SUPABASE_DEV_ANON_KEY`: its browser-safe **publishable key**
     (`sb_publishable_...`) or legacy **anon JWT**. Never supply a secret key,
     service-role key, personal access token, or user access token. Elevated keys
     are rejected before making a request. Supabase verifies the key; the script's
     legacy JWT parsing is only a configuration guard, not signature verification.
4. After reviewing the changes, commit/push them yourself and make the workflow
   available on the repository's default branch. This implementation does not push
   or configure GitHub/Supabase remotely. Scheduled runs require the default branch;
   manual runs on other branches are intentionally skipped.

The query is one GET to
`/rest/v1/keepalive_heartbeat?select=id,project_label&id=eq.1&limit=1`, equivalent to:

```sql
SELECT id, project_label FROM public.keepalive_heartbeat WHERE id = 1 LIMIT 1;
```

Only the harmless heartbeat is readable by this policy. An API key is required;
there is no additional unprotected HTTP endpoint. The public key is not a user
identity, and the singleton contains no user/workspace/document/chat information.
Existing RLS continues to protect private application data. The script uses only
the dedicated `SUPABASE_DEV_*` secrets, never `.env` or generic production settings,
never follows redirects, and makes no retries. The request timeout is 15 seconds,
the response is bounded to 1 KiB, and the job timeout is two minutes.

### Testing, monitoring, disabling, and resource use

- In **Actions → Supabase development keep-alive → Run workflow**, select the
  default branch. A successful job logs `PASS`, HTTP status, row count, and duration.
  Missing configuration, HTTP/network errors, missing heartbeat/RLS visibility,
  malformed responses, or the wrong label fail the job. Logs omit credentials,
  URLs, response bodies, and raw exception messages.
- Inspect previous runs and failed jobs in that same Actions page. GitHub's
  existing workflow notifications can be used; no notification service is added.
- Disable it using the workflow page's **… → Disable workflow**, or remove the
  workflow. Disable it before retiring/replacing the development project. Do not
  redirect its secrets to production. The singleton can remain without any process
  running or new rows accumulating.
- Expect **28–31 database GETs per month**, plus intentional manual tests, each
  returning one tiny row (typically tens of bytes). There are no recurring writes,
  LLM tokens, Gemini/Modal credits, uploaded artifacts, or paid external services.
  It still uses a small amount of Supabase compute/egress and one short Ubuntu
  GitHub Actions job per day; runner startup and your account's billing rules affect
  charged minutes. Check your own usage limits rather than assuming zero cost.
- GitHub schedules may be delayed or dropped under load; scheduling at minute 17
  avoids the top-of-hour peak. Public repositories may have scheduled workflows
  disabled after 60 days without repository activity. Check execution history and
  re-enable as needed; no artificial commits are generated. See
  [GitHub schedule documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
  A successful check confirms this read, not Supabase's future pause decision.

Offline validation from the repository root (existing development environment):

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_supabase_keepalive.py -v
.venv/Scripts/python.exe -m py_compile scripts/supabase_keepalive.py tests/test_supabase_keepalive.py
git diff --check
```

These checks mock network access, validate the workflow YAML/permissions and
success/failure/secret-handling behavior, and inspect the migration. They do not
prove live Supabase RLS or run GitHub Actions; the manual run above verifies the
configured development service after migration/setup. No live request is needed
for offline validation.

## N. Rollback and troubleshooting

- Redeploy the previously known-good Modal code revision; use Modal deployment history/rollback controls.
- Revert Vercel to its previous deployment and restore the matching API base URL.
- Do not roll back by dropping tables or disabling RLS. Database migrations are additive/currently deployed outside this app wrapper; use a reviewed forward migration for schema repair.
- `503 /ready`: check `SUPABASE_DB_URL`, network access, credentials, and pooler mode.
- `401` on protected routes: check Supabase URL/anon key, issuer/audience configuration, and token expiry.
- SSE appears buffered: inspect the proxy path and response headers; use `curl -N`.
- Slow first retrieval: distinguish model cold start from warm inference. Check Modal cache volume mount and model device logs before comparing with Phase 10.
