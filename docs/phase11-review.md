# Phase 11 final review

**Review date:** 2026-10-05

**Verdict:** **CHANGES REQUESTED before manual production deployment.** The local CPU Docker build and backend suite pass, and Supabase Dev verification is reported complete. Production release still requires environment-specific setup and smoke verification. The Modal remote image build was intentionally not run because it may incur cloud compute cost.

The repository roadmap names deployment as Phase 13; this review follows the project’s requested Phase 11 productionization scope.

## Verified evidence

- **CPU dependency lock:** `backend/requirements.lock.txt` contains no CUDA, NVIDIA, or Triton package pins and pins `torch==2.14.1+cpu`. The separate `backend/requirements-gpu.lock.txt` pins CUDA/NVIDIA/Triton dependencies and `torch==2.14.1` for the ingestion worker.
- **Backend tests:** `252 passed, 4 skipped` using `python -m pytest backend/tests -q`. The project `.venv` launcher points to an unavailable Windows Store Python, so the existing Python 3.11.15 task-local interpreter was run with `.venv\Lib\site-packages` on `PYTHONPATH`. `ENVIRONMENT=development` was set and RLS opt-in variables were cleared; no database was contacted.
- **Docker:** A clean Linux image build completed with `docker build --pull --no-cache --file Dockerfile --tag ai-tutor-backend:local .`. Container checks confirmed `/app/app/models`, successful `app.models` and FastAPI imports, and `python -m pip check` reported no broken requirements. Tesseract, English and Vietnamese language data, and LibreOffice Writer were present. Docker copies the CPU lock and backend application only; `.dockerignore` excludes `.env*` and local caches.
- **Supabase Dev:** User-reported verification is PASS for migration push, schema checks, and real RLS integration. Production Supabase was not contacted or modified in this review.
- **Modal local validation:** User-reported local wrapper import and ASGI resolution are PASS. **Modal remote cloud image build: NOT VERIFIED**; it was intentionally not attempted because it may incur cloud compute cost.
- **Whitespace:** `git diff --check` passed.
- No deployment, GPU execution, or production traffic was performed.

## Review findings

### Dependency and image boundaries

The Modal API/poller image installs the CPU lock. The isolated per-job ingestion worker image installs the GPU lock, and only that Modal function accepts the optional `MODAL_INGESTION_GPU` setting. BGE-M3 and CrossEncoder dependencies remain available in both locks so ingestion and CPU query-time retrieval retain their required inference libraries. The root Dockerfile installs the CPU lock. The CPU Linux image was built and checked; the GPU worker image and Modal cloud image were not built in this review.

The Dockerfile copies `backend/requirements.lock.txt` and `backend/app` only, with `PYTHONPATH=/app`. The `.dockerignore` exception for `backend/app/models` ensures the Python source package is copied while model cache folders remain excluded. No `.env` or secret value is copied into the image. Modal code references named Modal Secrets; secret values are not in source or image build instructions.

### Security, RLS, and migration

The durable ingestion migration adds an RLS-protected job table and queue-control `SECURITY DEFINER` RPCs with explicit service-role-only execute grants. Authenticated users can enqueue only jobs tied to their own documents and remove only terminal jobs. Queue-control calls use the backend service key; document processing continues through the authenticated user database session. Existing ownership and RLS behavior was not weakened. Supabase Dev migration and real RLS PASS are user-reported; no production migration was run.

Production secret values, role grants, bucket privacy, CORS origins, proxy upload limits, and deployed access controls still require verification in the target accounts. The request-size middleware enforces the declared `Content-Length`; hosting or proxy limits must also cover requests without a trustworthy length.

### Runtime and operational limits

Chat-time embedding and reranking remain CPU-bound inside the API process; target production latency and concurrency are not measured. The historical Phase 10 local CPU capture recorded mean embedding of about 1.30 seconds, reranking of 11.46 seconds, and total retrieval of 12.79 seconds. A separate answer capture recorded different retrieval and generation means, so the captures are not directly comparable. Treat query-time CPU performance as a release risk and benchmark against the intended workload. The durable queue, retries, leases, Modal scheduling, shared model-cache volume behavior, and worker execution have not been exercised on Modal. Logs include request and ingestion lifecycle events, but no production metrics sink or latency verification is established.

## Remaining blockers before manual production deployment

1. The Modal remote build remains unverified, and the GPU worker image has not been independently built. Decide whether to accept this validation gap or run a cost-reviewed build-only validation before deployment.
2. Review the CPU query-time retrieval latency against the product target and representative corpus; the local benchmark is well above the documented TTFT target before generation begins.
3. Apply the reviewed migration set to the intended production Supabase project through the normal migration workflow, then verify schema, RLS, queue RPC grants, and private Storage there. Supabase Dev verification does not apply changes to production.
4. Create and review the Modal Secret and model-cache Volume manually. Confirm the secret contains the required backend configuration and that credentials remain server-side. Set exact production CORS and database pool/concurrency budgets.
5. Review expected Modal costs and decide whether ingestion workers need a GPU. Configure `MODAL_INGESTION_GPU` only if approved.
6. Perform the manual deployment and validate health/readiness, authenticated ownership, document ingestion/retries, and SSE through the deployed proxy.
7. Configure the Vercel production API URL and browser-safe Supabase values, deploy the frontend when approved, and run the documented smoke test with a dedicated authorized user.

## Manual deployment sequence

1. Prepare the production Supabase project, review and apply the committed migrations, and verify the production schema/RLS and private bucket.
2. Create the Modal Secret (default name `ai-tutor-production`, or set `MODAL_SECRET_NAME`) and the model-cache Volume. Set production origins, connection-pool limits, and resource settings.
3. From the repository root, after explicitly accepting the remote-build/cost risk, run `modal deploy backend/deploy/modal_app.py`. This creates the API and scheduled poller functions; it was not run in this review.
4. Check the deployed `/health` and `/ready` endpoints, then perform authenticated API, document ingestion, queue retry, RLS ownership, and SSE checks using a dedicated production test user and non-sensitive test document.
5. Configure and deploy Vercel with `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SUPABASE_URL`, and `NEXT_PUBLIC_SUPABASE_ANON_KEY` only. Run `scripts/smoke_test_production.py` with dedicated authorized test credentials, then monitor logs, latency, and costs.

Do not place service-role, database, Gemini, or Tavily secrets in Vercel browser variables. Do not use production data for smoke tests.
