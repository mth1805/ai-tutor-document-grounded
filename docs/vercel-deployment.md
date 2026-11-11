# Vercel frontend deployment

Preparation only: no project, deployment, paid resource, or production request was created by this task.

## Architecture and project settings

Browser -> Vercel Next.js frontend -> HTTPS REST/SSE directly to Modal FastAPI -> Supabase / Gemini / Tavily. FastAPI, ingestion, model caches, database connections, and provider secrets stay on Modal. No Vercel API proxy or `vercel.json` is needed; Next.js detection and dashboard settings suffice. `output: "standalone"` remains for the local Docker runner.

- Framework preset: **Next.js** (locked version **14.2.35**).
- Root Directory: **frontend**; access to files outside this root is not required.
- Node.js: **22.x**, matching Docker and supporting the existing test runner.
- Install Command: **npm ci**, using `frontend/package-lock.json`.
- Build Command: **npm run build**.
- Output Directory: **Next.js default**; disable override.
- Keep **Automatically expose System Environment Variables** enabled (default); the hosted configuration guard uses `VERCEL=1`.
- Production branch: select the intended release branch; review changes before pushing.

References: [Vercel build settings](https://vercel.com/docs/project-configuration/general-settings), [Next.js public environment variables](https://nextjs.org/docs/pages/guides/environment-variables).

Release decision: Next.js 14.x is now [outside the supported LTS versions](https://nextjs.org/support-policy). This task preserves the existing major version; approval is required by AGENTS.md before a major migration. Select and validate a supported patched version before production release (15.x is the smaller migration; 16.x is Active LTS). The [August 2026 security release](https://nextjs.org/blog/august-2026-security-release) documents patches in the supported lines. A successful 14.2.35 build does not resolve this release decision.

## Public environment contract

| Variable | Required | Classification | Placeholder | Purpose |
| --- | --- | --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | Yes | Browser-safe | `https://your-modal-api.modal.run` | Backend origin, without `/api/v1` or trailing slash |
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Browser-safe | `https://your-project.supabase.co` | Same Supabase project as Modal |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | Browser-safe | `your-browser-safe-key` | Anon JWT or `sb_publishable_` key; RLS remains required |

The legacy `NEXT_PUBLIC_API_URL` alias remains supported; use only `NEXT_PUBLIC_API_BASE_URL` for new deployments. Conflicting aliases fail validation. Public values are inlined at build time; changes require rebuilding. Do not promote a Preview build containing different API/auth settings to Production without rebuilding.

Vercel builds (`VERCEL=1`, supplied by Vercel) reject missing settings, loopback API URLs, non-HTTPS origins, placeholder auth, and malformed/privileged Supabase keys. Validation is structural, not credential/connectivity verification. Example values are placeholders, never deployment credentials.

Never put these in the frontend directory or Vercel environment: `GEMINI_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`, `SUPABASE_DB_URL`, `RLS_TEST_DATABASE_URL`, database passwords, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET`, private signing keys, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, or any privileged credentials. Never expose them via a `NEXT_PUBLIC_` prefix or Next.js `env` configuration.

## Local development and Docker

Copy `frontend/.env.example` to `frontend/.env.local` with browser-safe Supabase Dev values. Keep backend settings in the root/backend `.env`; never copy the whole backend file to the frontend. `npm run dev` defaults to `http://localhost:8000` if the API variable is absent. Production builds require an explicit origin. Explicit HTTP loopback is allowed for local production Docker builds; hosted Vercel builds reject it.

Compose remains local: frontend `localhost:3000`, backend `localhost:8000`; frontend build arguments contain only public values. Keep `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` in the Compose/root environment. Docker service hostnames are not browser API URLs. Supabase placeholders allow a Docker build but cannot provide authenticated production-mode login: supply Supabase Dev browser credentials for an authenticated demo. Existing fake auth is development-only.

## Authentication and streaming

The browser uses `@supabase/ssr`'s singleton client, persisted cookies, `getSession`, and `onAuthStateChange`. Auth refresh events update the bearer token forwarded to FastAPI. Password login/signup/logout remain browser-side; FastAPI/RLS authorize business data. Set Supabase Auth Site URL to the final frontend origin and allow exact trusted email-confirmation redirect URLs. Signup uses the default Site URL and has no custom callback route; test confirmation followed by password login. Preview signup must use a trusted redirect; do not add broad preview wildcards.

REST, uploads, authenticated viewer downloads, and chat derive from `src/lib/config.ts`. The viewer uses a browser blob, not a local filesystem. There are no Next.js API/server route handlers or Python/model/database dependencies in the frontend.

Chat is a bearer-authenticated POST directly to Modal, consumed incrementally with `ReadableStream`, partial UTF-8/chunk buffering, and AbortController for Stop Generation. EOF before `done` reports an error. Vercel function duration/body limits do not apply to this direct browser-to-Modal stream/upload. Modal/provider timeout and cancellation behavior still need deployed verification; the wrapper API timeout is 150 seconds. Local tests do not prove live streaming.

## CORS and manual deployment order

Set **`BACKEND_CORS_ORIGINS`** in the Modal Secret (default `ai-tutor-production`, selectable by `MODAL_SECRET_NAME`) to a JSON list of exact trusted origins, without paths/trailing slashes:

```text
BACKEND_CORS_ORIGINS=["https://your-project.vercel.app"]
```

Add an exact custom domain if needed. Previews require individually trusted exact origins and matching Supabase Auth settings; never allow `*.vercel.app` or `*`. Existing middleware allows bearer headers and GET/POST/PATCH/DELETE/OPTIONS. Production backend validation rejects loopback origins: keep `["http://localhost:3000"]` only in development Compose. Restart/redeploy Modal after changing the Secret so settings reload.

Future manual steps, in order; none executed here:

1. Complete [backend prerequisites](deployment.md) and [Phase 11 blockers](phase11-review.md): production Supabase schema/RLS/private Storage/Auth, Modal secrets, model/cache/worker validation and cost review.
2. Deploy Modal first; record its final HTTPS origin. Verify `/health` and `/ready` with authorized production checks.
3. Vercel Dashboard -> Add New -> Project -> Import GitHub repository. Choose Next.js, Root Directory `frontend`, Node 22.x, `npm ci`, `npm run build`, default output. Determine the intended project production domain before clicking Deploy; a custom domain may be selected in advance.
4. Enter the three public values for Production. Configure Preview separately only if intended; validation requires all three there too. Never import the backend `.env`.
5. Configure Supabase Site URL/redirect allowlist and Modal CORS with the intended production domain; restart/redeploy Modal. If Vercel assigns a different domain, update both before testing.
6. Click **Deploy** manually. Check build/domain, then run smoke checks below. Rebuild whenever public settings change.

Vercel CLI is not part of the repository workflow and is not required. Use the dashboard; no CLI installation, linking, or execution is needed for preparation.

## Post-deploy smoke checklist

- [ ] `/`, `/login`, `/signup` render without critical browser errors.
- [ ] Signup/email confirmation, login, auth refresh, logout, and login again work against the intended Supabase project.
- [ ] Workspace CRUD/switching and `/workspace/{id}` work; `/workspace/{id}/c/{conversationId}` restores history directly.
- [ ] Upload PDF/DOCX, observe indexing, open authorized viewer/download.
- [ ] Ask a grounded RAG question; verify document citations and pages, with separate web sources.
- [ ] Tokens arrive incrementally; Stop Generation, disconnect/retry, and backend cancellation work through Modal.
- [ ] Reload and logout/login preserve authorized workspaces, documents, conversations, messages.
- [ ] Network requests use HTTPS Modal, current bearer tokens, and successful CORS preflight.
- [ ] No backend secrets in browser bundles; console has no critical errors.

The [production checklist](production-checklist.md) remains the release gate. Local build success is preparation evidence, not proof of production auth, queues, or provider availability.

## Local preparation evidence (2026-10-05)

- Clean `npm ci` succeeded with the updated lockfile; host Node was 24.18.0, while the recommended Vercel/Docker target remains Node 22.x.
- Production build using synthetic hosted HTTPS/API/auth settings passed without backend services running. All six generated pages and both dynamic workspace/conversation routes compiled.
- Final production build with the local Docker arguments (`http://localhost:8000` and placeholder Supabase settings) also passed after the clean install, confirming compatibility with the local build contract. This was a host Next.js build, not a Docker image/full Compose runtime test.
- `npm run lint` passed with no warnings/errors; TypeScript/build checks were not weakened.
- `npm test`: 27 passed, including authenticated multipart uploads, URL validation, split UTF-8 SSE delivery, premature disconnect, and abort.
- Standalone production server: `/`, `/login`, `/signup`, workspace and conversation routes returned HTTP 200 locally. This checks server rendering, not live browser auth.
- Credential signature scan found no tracked secret values; private env files are ignored. The private frontend Supabase key was verified as the anon category without printing its value. The hosted browser bundle contained no configured backend credential values or backend secret variable names.
- `docker compose config --quiet` and `git diff --check` passed. Full Compose startup was intentionally not performed, since it could contact configured managed services. Existing local URLs and standalone Docker output were preserved.
- Vercel CLI was absent; it was not installed or linked. Production/Modal remote checks and deployments remain unexecuted.
