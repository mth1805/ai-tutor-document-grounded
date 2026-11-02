# Ingestion and document preview changes

This change prepares application code only. No deployment, production data changes, environment-file edits, or Supabase migrations were performed.

## GPU worker and observability

`backend/deploy/modal_app.py` allocates `gpu=INGESTION_GPU`, where `MODAL_INGESTION_GPU` defaults to `T4` (including an empty override). Only `process_ingestion_job` receives this resource; `fastapi_app` and `poll_ingestion_queue` retain their CPU images and have no GPU assignment. The worker retains `gpu_image`, `requirements-gpu.lock.txt`, and the existing `ai-tutor-model-cache` Volume at `/models`. Its worker-only environment selects CUDA, `/models/huggingface/hub` (the normal HF_HOME hub subdirectory, shared with default CPU weight caching), and automatic embedding. A persistent event loop and the existing thread-safe process singleton reuse the loaded model and async database pool inside warm containers. The Volume is explicitly committed after jobs; commit failures are logged.

Production embedding remains batched at `EMBEDDING_BATCH_SIZE` (default 16), with evaluation mode and `torch.no_grad()`. GPU inference should reduce embedding latency substantially relative to CPU, but no T4 benchmark or cloud execution was performed. Parsing/OCR, network downloads, cold model loading, and vector writes can still dominate total time. GPU workers incur GPU charges while allocated, including warm idle time within the existing 300-second scaledown window; scheduled/API functions remain CPU resources. Confirm account pricing and observed utilization before increasing concurrency.

JSON `ingestion_stage` events correlate job ID, document ID, and attempt. Stages include queued, claimed, storage_download, parsing, OCR when used, chunking, model_load, embedding, vector_store, ready, and failed. Final attempt summaries include CUDA availability, GPU name, model device, load/download/parsing/OCR/chunking/embedding/vector-store milliseconds, chunk/vector counts, batch size, total ingestion time, and final attempt status. Parsing time includes nested OCR time; do not add those two timings when computing totals. A zero duration means a stage was unused or not reached. Failed attempts remain subject to the existing queue's retries and backoff. Logs contain error categories/codes, never raw storage credentials or document text. Production document/embedding error metadata is also sanitized before it can be displayed in the frontend; development retains its existing diagnostics.

Changed backend files: `deploy/modal_app.py`, `app/ml/bge_provider.py`, `app/workers/ingestion_queue.py`, `app/services/ingestion/telemetry.py`, `app/services/ingestion/pipeline.py`, `app/services/ingestion_service.py`, and `app/services/embedding_service.py`.

## Product text and authentication

Visible labels removed/replaced:

- Header: `Phase 5` → `Document learning`.
- Chat: `Phase 9: Document-Grounded AI Tutor + Web Fallback Active` → `Document-grounded AI Tutor with Web Fallback`.
- Chat: `Phase 8 Active: Grounded LLM generation with streaming tokens, citation attribution, and relevance gating.` → `Answers stream with source citations from relevant evidence.`
- Chat: `Phase 8 Active: Grounded RAG token streaming with verifiable document citations.` → `Answers stream with verifiable document citations.`
- Viewer: `Phase 4 Notice` and the `will activate in Phase 5` notice → preview availability text.
- Search inspector: `Phase 7 Retrieval Inspector` → `Document Search Inspector`.

Engineering comments, architecture documents, and migration names remain intact. Changed UI files: `Header.tsx`, `ChatArea.tsx`, `DocumentViewer.tsx`, and `RetrievalInspectorModal.tsx`.

`src/app/(auth)/signup/page.tsx` stays on the signup page after success and renders “Account created. Please check your email and confirm your account before signing in.” with a sign-in link. `src/lib/auth-context.tsx` maps unconfirmed-email errors to “Please confirm your email before signing in. Check your inbox for the confirmation link.” through `auth-messages.ts`. Signup does not create a client-authenticated state from a no-session response. Supabase confirmation settings and the auth listener are unchanged.

## Word preview and download

Previously the viewer downloaded Word bytes but had no Word renderer and displayed only metadata. It now fetches the authenticated `/api/v1/documents/{document_id}/preview` endpoint for DOC/DOCX, then renders PDF in the existing iframe or plain text in the existing text view. Existing PDF/image preview paths remain unchanged.

The route verifies the authenticated owner through `DocumentService` before retrieving the private original. `document_preview.py` converts to PDF with the existing LibreOffice binary, a timeout, fixed internal source names, no shell, and a separate profile for each conversion. On failure it uses owner-scoped persisted chunks, or the existing Word parser if no chunks exist. If neither works, it returns a friendly preview-unavailable error and retains the original download action.

Preview artifacts exist only under the host's temporary directory in an `ai-tutor-preview-*` directory. Internal names are `source.doc`/`source.docx`, `source.pdf`, and `profile/`. The entire directory is removed on normal completion or exceptions by `TemporaryDirectory`; there is no persistent preview cache and no Supabase duplicate. Abrupt process termination relies on ephemeral container cleanup. Reopening a Word document performs conversion again. Originals stay private, routes remain authenticated, and RLS/CORS are unchanged.

Existing Word uploads need no re-upload if the original object still exists. Legacy DOC text fallback needs persisted chunks or a working LibreOffice DOC-to-DOCX conversion. Extracted text may lose layout and may contain chunk overlaps. Its page references are not a new source of citation truth. Word-to-PDF pagination can differ from ingestion's logical DOCX page numbering.

Backend files: `app/api/v1/documents.py`, new `app/services/document_preview.py`. Frontend files: `src/components/DocumentViewer.tsx`, `src/components/DocumentManager.tsx`, and new `src/lib/document-download.ts`. Both download controls fetch the original through bearer auth and use the original filename on a blob download; they never download a converted preview under a Word extension.

## Unicode and storage reliability

The old display-name sanitizer used `[^a-zA-Z0-9._\- ]`, replacing Vietnamese letters with underscores. It now preserves Unicode and normalizes NFC while removing path components and control characters. Names over the existing 255-character column limit are rejected instead of truncated. The existing `documents.original_filename` and `documents.storage_path` columns already separate display names from internal keys; no migration is needed. New keys are `<workspace_uuid>/<document_uuid>/source.<extension>`. Old objects remain accessible using their stored path without rewriting records or moving objects. Storage paths are percent-encoded exactly once when constructing HTTP URLs; slash separators are preserved. Downloads provide an ASCII fallback plus RFC 5987 `filename*=UTF-8''...`.

Uploads still write Storage first, then commit the document and ingestion job in one database transaction. Failed Storage writes cannot create metadata or jobs. Production storage now fails closed if backend credentials are missing, instead of silently writing to process memory that another worker cannot read. Private downloads use the authenticated object endpoint; compensation uses the bucket-level DELETE API with an exact `prefixes` list. All service-role access stays backend-only.

A confirmed database failure removes the uploaded object only after checking that no committed document exists. If a commit acknowledgement is lost and the row exists, or the database cannot be checked, the object is retained for operator reconciliation. A refresh failure after a successful commit can no longer delete the file referenced by its queued job. Cleanup failures are logged. This is compensation across two services, not a distributed atomic transaction; ambiguous outcomes can leave an unreferenced object, which requires reviewed reconciliation.

The exact historical production mismatch cannot be proven from code alone. Two concrete risks were found: production memory fallback with incomplete Storage credentials, and deleting the object after a committed transaction when refresh fails. Old records with missing objects require restoration at the exact stored key or a reviewed cleanup/re-upload. Names already replaced by underscores cannot be reconstructed from the database or identical sanitized Storage key; recover from independent metadata/backups if available, otherwise rename or re-upload. No production records were inspected or repaired.

Changed files: `app/services/storage_service.py`, `app/services/document_service.py`, `app/api/v1/documents.py`, `src/lib/document-download.ts`, and both document UI components.

## Validation and manual rollout

Regression coverage lives in `backend/tests/test_production_improvements.py`, the updated `backend/tests/test_documents.py`, and `frontend/tests/production-improvements.test.mjs`. It covers auth UI submission/rendering, no-session signup, Unicode upload/download, tenant isolation, PDF/image preview regression, Word conversion and fallback, Storage HTTP/auth/path encoding, upload ordering and compensation, GPU resource assignment, CUDA selection/fallback logging, cache/singleton reuse, and correlated stages. Tests use in-memory metadata/storage and mocked model providers/HTTP for safety; they do not prove live Supabase or Modal connectivity.

Validation completed on 2026-10-07:

- Full backend suite: **299 passed, 4 skipped**. The skipped cases require real BGE weights, real reranker weights, a legacy DOC fixture, or a disposable RLS test database. Three dependency deprecation warnings remain.
- Frontend `npm test`: **34 passed**. `npm run lint`: passed without warnings/errors. `npm run build`: passed, including production page generation and TypeScript checks.
- Modal import and resource configuration: passed; T4 on the worker, no GPU on API/poller, GPU image and cache Volume retained. The focused Modal checks also passed after aligning the worker cache with HF_HOME's hub subdirectory.
- Real local PDF/DOCX smoke: passed for Vietnamese filenames, private authenticated preview/download, parsing/chunking, and mock embedding. LibreOffice produced an actual PDF; no live Storage/database or real GPU/model download was used.
- `git diff --check`: passed.

The local environment's `LIBREOFFICE_CMD` entry contains extra text and is not a valid executable name. It was left unchanged. For ordinary local operation, manually correct that single setting to `soffice` or a valid absolute executable path. The smoke test temporarily selected the installed executable without modifying the environment file.

No new Supabase migration is required; existing queue/document/chunk migrations must already be applied. Manual release commands, run only when intentionally deploying:

```powershell
# From C:\2_AITutor, using the existing authenticated Modal CLI:
.venv\Scripts\modal.exe deploy backend/deploy/modal_app.py

# From the frontend directory, with your existing Vercel project configured:
Set-Location C:\2_AITutor\frontend
npx vercel --prod
```

Modal and Vercel both require redeployment to activate this code. Do not change production secrets unless separately reviewed. `MODAL_INGESTION_GPU` is read by the deployment CLI; unset or empty selects T4. After manual deployment, the Modal dashboard should show GPU allocation only on `process_ingestion_job`. Verify runtime CUDA/device logs, upload/preview/download with a dedicated test user, and private bucket/RLS behavior. Dashboard state and real CUDA execution cannot be validated without deployment.

Local tools must have a valid `LIBREOFFICE_CMD` (normally `soffice`). The real conversion smoke test uses a discovered local executable without editing environment files. Backend-only provider keys remain absent from browser code. Supabase email confirmation must remain enabled in project settings.

Configuration references: [Modal GPU allocation](https://modal.com/docs/guide/gpu), [Modal Volumes](https://modal.com/docs/guide/volumes), [Supabase Storage access control](https://supabase.com/docs/guides/storage/security/access-control), and [Supabase Storage API](https://supabase.github.io/storage/).
