# Local Docker runtime

## One-command demo

After the one-time Dev setup below:

```powershell
docker compose up --build
```

Open http://localhost:3000. The browser calls http://localhost:8000. No extra worker terminal is needed. Stop the entire stack with `docker compose down`; do not use `down -v` if you want to keep downloaded models.

## One-time Supabase Dev configuration

Supabase stays managed/cloud; Compose adds no local database. The root `.env` must reference the **dedicated Dev** project throughout: `SUPABASE_URL`, browser `NEXT_PUBLIC_SUPABASE_URL`, anon/publishable keys, backend-only service role key, and `SUPABASE_DB_URL` (or `DATABASE_URL`). Auth, private documents Storage, reviewed schema/RLS, Gemini, and optionally Tavily must be configured there. These provider calls may use account quotas; do not point this demo at production. Root `.env` is passed only to backend/worker; frontend receives only three browser-safe build arguments.

Keep `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` for Compose. Container overrides force `ENVIRONMENT=development`, CPU embedding/reranking, automatic embeddings, batch size one, two inference threads, and the shared cache. Backend cold startup has a 120-second health-check grace period. `/health` is liveness; `/ready` in development is lightweight and does **not** prove DB/Auth/worker readiness. Confirm managed dependencies with the authenticated smoke flow below.

### Manual migration (never automatically pushed)

The new file is `supabase/migrations/20261005130000_grant_authenticated_ingestion_job_access.sql`. It reasserts exactly SELECT, INSERT, DELETE for `authenticated`; UPDATE remains revoked. SELECT checks existing jobs, INSERT enqueues, DELETE replaces the owner's terminal job on manual retry. Existing RLS and service-only claim/complete/fail/load RPC permissions are unchanged.

The old migration already declares these three grants; an applied database may differ. The application also incorrectly locked queue rows with FOR UPDATE, which requires UPDATE privileges. Enqueue now locks the authorized parent document and uses plain SELECT on jobs, preserving serialization without granting queue UPDATE access.

From the repository root, manually set a **standard PostgreSQL Dev migration URL** (not the SQLAlchemy `postgresql+asyncpg://` URL). Use the reviewed direct/session pooler target; confirm its project is Dev before running:

```powershell
$env:SUPABASE_DEV_MIGRATION_DB_URL = "postgresql://<dev-user>:<encoded-password>@<dev-host>:5432/postgres"
npx supabase db push --db-url "$env:SUPABASE_DEV_MIGRATION_DB_URL" --dry-run
npx supabase db push --db-url "$env:SUPABASE_DEV_MIGRATION_DB_URL"
```

Review the dry-run: it must target Dev and include only the intended pending migrations. Run `supabase/verification/verify_ingestion_queue_access.sql` manually in the Dev SQL editor afterward. It checks table grants, RLS/policy presence, and service-only RPC access without changing data. The user confirmed manual Dev application on 2026-10-05; the agent subsequently ran the read-only catalog assertions successfully. The agent did not push migrations.

## Process boundaries and model cache

- `frontend`: Next.js standalone UI; browser-safe Supabase auth and direct REST/SSE.
- `backend`: FastAPI CPU image; uploads file to private Storage, atomically commits document + durable queued job, serves retrieval/chat/viewer APIs. Reprocessing locks the owner-scoped document before rearming a terminal job.
- `ingestion-worker`: same CPU image, same Dev env/storage/DB/parser stack. Entrypoint `python -m app.workers.ingestion_queue`. It polls every five seconds, claims one due job through the service-role RPC, and reuses the same processing function as Modal. No separate local poller, GPU, or Modal invocation.

The named `model-cache` volume is mounted at `/models` on both backend and worker. `HF_HOME`, `EMBEDDING_MODEL_CACHE_DIR`, and `RERANKER_MODEL_CACHE_DIR` point to `/models/huggingface`, which the installed model providers use. `TIKTOKEN_CACHE_DIR=/models/tiktoken` persists tokenizer data too. First downloads/CPU initialization take longer; restarting containers reuses the volume. Weights are not committed or baked into source. Compose now enables `LOCAL_SHARED_MODELS=true`: only the backend owns BGE-M3 and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; the worker parses/OCRs/chunks and hands embedding to a protected internal API. Worker automatic embedding is disabled; job completion waits for committed backend vectors. Background warmup leaves API health and the frontend responsive; the worker waits for model readiness before claiming. Local inference is serialized in micro-batches with 512-token windows preserving all text. Production and Modal defaults are unchanged; the shared profile is rejected outside development. The validation PC has only 8 GB physical RAM and exposes about 3.7 GiB to Docker. It cannot allocate the previously suggested 12 GiB. Batch/thread limits do not remove model-weight memory requirements. The completed live workload peaked at 1.851 GiB for the backend cgroup, with no OOM event. See [the memory-saving design](local-docker-memory-design.md) for the measured results.

RPC/network polling failures are logged by exception category and retried after the polling interval. SQLAlchemy SQL echo is off so document text/vectors/claims are not logged. Stage logs include job/document IDs and counts, not content or credentials. SIGTERM/SIGINT stops new claims and drains an active job; Compose gives it two minutes. If Docker must kill a longer job, the persisted one-hour lease allows recovery; it is not silently marked complete. Queue backoff is 30/60 seconds across up to three attempts, as defined by the existing RPC.

## Actual status flow

Documents: `queued` -> `processing` -> `processed`; embeddings: `pending` -> `processing` -> `completed`. The UI displays processed + completed as Embedded. In-memory tests may initially use `uploaded`. A failed durable attempt remains processing until the queue RPC chooses `queued` (retry) or `failed` (terminal); no new ready/indexed database status is invented. Successful chunks/vectors are persisted and indexed by existing DB indexes. Polling covers queued/processing/uploaded and processed embeddings pending/processing, and stops on terminal completion/failure. Failed processing/embedding details remain visible.

## Error diagnosis

Queue permission errors are database failures, not CORS/network errors; they are not swallowed as successful uploads. In current code, initial document + job insertion is atomic; failures roll back metadata and clean up the stored original. Reprocessing previously requested an unauthorized queue row lock. Missing worker leaves jobs queued; missing queued polling makes that stale state persist in the UI.

The observed live 503 was traced to a Supabase Auth `ReadTimeout`. Auth verification now retries a transport failure once, with no retry or bypass for invalid credentials; Compose uses a 15-second auth timeout instead of the five-second default. The original browser Failed to fetch still lacks enough captured context to prove its cause. Actual 503 paths are unavailable Supabase Auth, production readiness dependencies, and an unconfigured production DB; enqueue exceptions produce a 500. A browser Failed to fetch indicates transport/CORS/connection interruption rather than a received 503. Compose keeps explicit localhost API/CORS, separates parsing/embedding from the API process, and retains error responses. If errors remain, inspect browser Network, API request IDs, `docker compose ps`, and sanitized backend/worker logs; do not call a permission fix proof that auth/network problems are resolved.

## Dev end-to-end checklist (after manual migration)

```powershell
docker compose down
docker compose up --build
# In a separate verification shell only (not required to operate the demo):
docker compose ps
```

- [x] All three services running; backend health healthy.
- [x] Frontend GET `/` returns 200; API `/health` and `/ready` return 200; localhost CORS preflight passes.
- [x] Dev login and workspace creation succeed through real authenticated API calls.
- [x] Small PDF, scanned PDF, and DOCX uploads queue and complete through the worker on one attempt.
- [x] Worker claims; parse/OCR completes; chunks and 1024-dimensional embeddings exist.
- [x] Documents are processed with completed embeddings; original-file retrieval works.
- [x] Grounded question retrieves document evidence; Gemini emits cited answers.
- [x] SSE arrives incrementally and stream cancellation works.
- [x] Logout/login preserves workspace, document, conversation, and message state.
- [x] No unexpected 503 occurred in the completed Dev smoke. The historic browser `Failed to fetch` remains un-reproduced.

Dev migration/configuration are confirmed. Backend tests: **272 passed, four skipped**, one warning; focused local-runtime tests: **22 passed**. Frontend tests: **28 passed**; type checking and production Docker builds passed. A completed disposable Dev smoke removed its test user, workspace, documents, conversation and storage objects. It verified queue completion, OCR, DOCX, 1024-dimensional vectors, original-file retrieval, English/Vietnamese retrieval quality, grounded Gemini SSE/citations, cancellation, and persisted state after logout/login. During the profile workload the backend cgroup peak was **1.851 GiB** under Docker's **3.717 GiB** allocation; no `oom` or `oom_kill` events occurred. Cold CPU model work remains slow and used up to about 0.638 GiB of Docker swap. See [the memory design review](local-docker-memory-design.md) for details.
