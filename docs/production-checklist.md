# Production checklist

Items are checked only when verified from repository code or locally run validation. Deployment-specific items remain pending.

Local Compose now includes the existing CPU ingestion worker and persistent model cache. The manual Dev migration and end-to-end queue verification are tracked separately in [local Docker runtime](local-docker-runtime.md); local success must not be treated as production verification.

## Security

- [x] Secrets are read through backend environment settings; frontend client code does not reference server keys.
- [x] Generic unhandled errors return a sanitized 500 envelope with a request ID.
- [ ] Verify deployed secret values are present and absent from logs/build artifacts.
- [ ] Verify request body limits and provider timeout behavior under the production proxy.

## Authentication

- [x] Protected API routes depend on verified Supabase user context.
- [x] Chat verifies conversation ownership before starting the stream.
- [ ] Verify production JWT issuer/audience and token expiry behavior against Supabase.

## RLS

- [x] SQLAlchemy transactions apply the authenticated role and transaction-local verified-user claims.
- [x] Migrations include tenant policies; no Phase 11 change weakens them.
- [ ] Run `RLS_TEST_DATABASE_URL` integration regression against production-equivalent PostgreSQL.

## Storage

- [x] Existing document storage service uses the configured private bucket and authorized document paths.
- [x] Upload path validates supported file type, size, and file signature.
- [ ] Verify bucket privacy and access policies in the target Supabase project.

## Database

- [x] Migrations are versioned and include relational, vector, lexical, and RLS setup.
- [x] Backend supports `SUPABASE_DB_URL` and legacy `DATABASE_URL`; readiness checks connectivity.
- [x] Per-container SQLAlchemy pool size, overflow, and checkout timeout are configurable.
- [x] Database-backed document uploads persist a unique ingestion job with metadata; queue claim/complete/fail RPCs are service-role only and implement leases/retries.
- [ ] Apply and execute the durable-ingestion migration in a disposable Supabase-compatible environment; verify lease expiry, retry exhaustion, and RLS.
- [ ] Verify production pooler mode, pool capacity, indexes, and query plans with Supabase.

## RAG

- [x] Existing flow performs document retrieval, reranking, evidence gating, and fallback.
- [x] Production notes preserve the benchmark configuration as separate from current code configuration.
- [ ] Run production corpus quality and citation-integrity evaluation.

## Model serving

- [x] Embedding and reranker providers are process-level singletons with configurable device selection.
- [x] Optional startup warmup runs off the event loop; model inference does not occur in readiness checks.
- [ ] Measure Modal cold start, warm inference, GPU memory, and CPU fallback.
- [x] API Modal function remains CPU-only; only the independent ingestion worker can request `MODAL_INGESTION_GPU`.
- [ ] Isolate chat query embedding and reranking from the CPU API if production latency requires GPU inference.

## SSE

- [x] Chat returns `text/event-stream`, no-cache, keep-alive, and proxy no-buffering headers.
- [x] Frontend parses SSE frames and sends bearer authorization.
- [ ] Verify disconnect cancellation, upstream failure, timeout, and completed-message persistence end to end through Modal.

## Observability

- [x] HTTP request ID, route, status, and latency are logged; request ID is returned in response headers.
- [x] Error logs classify unhandled exceptions by type and avoid request body contents.
- [x] Structured `chat_stage_metrics` captures retrieval stages, gate/fallback choice, provider/model, TTFT, generation, and total request latency; full response-completion duration is logged for SSE.
- [ ] Connect logs to a metrics sink and verify stage events/stream completion in production.

## Performance

- [x] Phase 10 local latency findings are retained with their benchmark context.
- [x] Production docs distinguish model cold start, warm inference, retrieval, and generation.
- [ ] Measure production latency; GPU performance is currently unknown.

## Environment variables

- [x] Example documents frontend API base URL and backend-only credentials.
- [x] Production settings validate required Supabase, DB, LLM, CORS, and enabled fallback configuration.
- [ ] Populate and verify environment values in Modal and Vercel without exposing backend-only secrets.

## Deployment verification

- [x] Modal wrapper reuses the current FastAPI application through `modal.asgi_app`.
- [x] Smoke test script supports live auth, conversation, chat, and SSE checks.
- [x] Resolved exact backend lock for Python 3.11/Linux x86_64; `pip check` passed in task-local Python environment.
- [ ] Clean Linux/Modal image build and runtime import validation.
- [x] Safe backend subset: 236 passed, 4 skipped, 2 existing benchmark tests deselected; Python compile check passed.
- [ ] Resolve the two pre-existing benchmark test failures and validate full suite.
- [ ] Run the destructive RLS integration only with an explicit process `RLS_TEST_DATABASE_URL` for a disposable test database.
- [x] Run frontend tests locally (23 passed).
- [x] Run frontend production build locally.
- [ ] Deploy only after manual review; verify health, readiness, auth, uploads, and SSE.

## Rollback

- [x] Deployment documentation describes code and frontend rollback without destructive database actions.
- [ ] Record target Modal/Vercel revisions and operational owner before launch.

## Vercel frontend

See [Vercel deployment guide](vercel-deployment.md) for settings, environment contract, and manual order. Live checks remain pending until an authorized deployment.

- [ ] Modal backend deployed.
- [ ] Supported patched Next.js major approved and validated before production release.
- [ ] Modal `/health` PASS.
- [ ] Modal `/ready` PASS.
- [ ] Final HTTPS Modal API URL recorded.
- [ ] Production Supabase schema/RLS/private Storage/Auth configured.
- [ ] Frontend browser-safe Supabase URL and anon/publishable key configured.
- [ ] `NEXT_PUBLIC_API_BASE_URL` configured at build time.
- [ ] Production Vercel origin added to Modal `BACKEND_CORS_ORIGINS`; backend restarted.
- [ ] Supabase Site URL and trusted auth redirects configured.
- [ ] Frontend production build PASS for final production values.
- [ ] Vercel deployment PASS.
- [ ] Auth and session refresh PASS.
- [ ] Upload and document viewer PASS.
- [ ] RAG PASS.
- [ ] SSE, Stop Generation, and disconnect handling PASS through Modal.
- [ ] Citations PASS.
- [ ] Reload and logout/login persistence PASS.
- [ ] No backend secrets in final browser bundle.
- [ ] Browser console free of critical errors.
