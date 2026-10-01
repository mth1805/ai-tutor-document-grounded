---
name: project-architecture
description: Guides repository structure navigation, architecture boundaries, phase alignment, and pre-implementation planning.
---

# Project Architecture Skill

## 1. Purpose
The `project-architecture` skill equips coding agents to maintain architectural purity across the **AI Tutor Assistant** repository. It ensures strict separation of concerns between Next.js (frontend) and FastAPI (backend), preserves dependency boundaries, enforces phase-aware incremental development, and prevents architectural drift or uncoordinated rewrites.

## 2. When to Use
Use this skill when:
- Beginning work on any new development phase (Phases 1 through 13).
- Adding new routes, tables, services, or shared contract interfaces.
- Assessing where a new component, module, or utility belongs in the directory tree.
- Reviewing cross-cutting concerns (authentication flows, streaming protocols, error schemas).
- Refactoring existing modules to maintain clean architectural boundaries.

## 3. What to Inspect First
Before proposing or making changes, always inspect:
1. `AGENTS.md` (root directory) — Confirm current phase scope and operational constraints.
2. `/.agents/rules/` — Check domain rules relevant to the planned changes.
3. Package manifests and configuration:
   - `frontend/package.json` — Frontend dependencies and script definitions.
   - `backend/pyproject.toml` or `backend/requirements.txt` — Backend dependencies.
   - Root project files (`docker-compose.yml`, `.env.example`, `supabase/config.toml`).
4. Schema definitions:
   - `backend/app/schemas/` — Current Pydantic request/response models.
   - `supabase/migrations/` — Existing database tables, columns, and foreign keys.

## 4. Standard Workflow
Follow this 5-step workflow whenever planning architectural changes:

```
[Step 1: Scope & Phase Check]
      │ Determine current phase; reject features belonging to later phases
      ▼
[Step 2: Boundary Identification]
      │ Map work strictly to Frontend, Backend, or Database layer
      ▼
[Step 3: Contract-First Design]
      │ Define Pydantic schemas (backend) and TypeScript interfaces (frontend)
      ▼
[Step 4: Implementation Plan]
      │ Formulate minimal, targeted file changes and dependencies
      ▼
[Step 5: Boundary & Regression Verification]
      │ Verify decoupled execution and ensure no leaky abstractions
```

### Step 1: Scope & Phase Verification
- Review the user's explicit request against the 13 defined development phases.
- If requested to implement Phase 2 (Auth + Workspace), **do not** implement Phase 5 (Ingestion) or Phase 7 (Hybrid Retrieval) ahead of time.

### Step 2: Layer & Boundary Identification
- **Frontend Layer (`/frontend`):**
  - UI components, layout trees, client state hooks, route handlers for static assets.
  - Must never import backend Python code, execute database queries directly, or hold secret API keys.
- **Backend Layer (`/backend`):**
  - FastAPI routers (`app/api/v1/`), service layer (`app/services/`), Pydantic schemas (`app/schemas/`), model orchestrator (`app/ml/`).
  - Must never generate HTML markup or couple endpoints to specific UI components.
- **Database & Storage Layer (`/supabase`):**
  - SQL migrations, schema constraints, pgvector indexes, RLS security policies.

### Step 3: Contract-First Interface Design
- Define the data transfer contract first:
  1. Write the Pydantic schema in `backend/app/schemas/`.
  2. Define the matching TypeScript interface in `frontend/src/types/`.
  3. Ensure naming conventions are aligned (snake_case in Python, camelCase serialization if configured).

### Step 4: Incremental Implementation Planning
- List the minimal set of files to create or modify.
- Check for existing helper functions and services before writing duplicate logic.

### Step 5: Verification & Boundary Audit
- Test that frontend builds without backend running (`npm run build`).
- Test that backend boots independently without frontend running (`pytest` or FastAPI test client).

## 5. Engineering Rules
1. **Unidirectional Dependency Flow:** Frontend depends on Backend HTTP/SSE endpoints; Backend depends on Database and AI APIs. Neither layer reaches inside the other's internal modules.
2. **Never Blindly Rewrite:** Extend existing abstractions. If a service needs new behavior, add a method or compose a new service rather than replacing working files.
3. **No Phantom Architecture:** Do not add Redis, Celery, or complex message brokers until Phase 11 explicitly requires distributed background queues. Keep early phases lightweight (FastAPI BackgroundTasks).
4. **Explicit Configuration:** All configurable options (model paths, thresholds, timeouts) must reside in `app/core/config.py` using `pydantic-settings`.

## 6. Validation Checklist
- [ ] Changes match the currently requested development phase only.
- [ ] No server secrets are referenced in frontend client components or prefixed with `NEXT_PUBLIC_`.
- [ ] New endpoints have corresponding Pydantic schemas for request and response.
- [ ] Database alterations include a corresponding migration file in `supabase/migrations/`.
- [ ] Model inference code is separated from FastAPI request handling logic.
- [ ] No circular imports exist in backend Python modules.

## 7. Common Failure Modes & Mitigations
- **Failure:** Implementing RAG retrieval logic during Phase 2 (Workspace Setup).
  - *Mitigation:* Halt and enforce phase boundaries. Focus exclusively on the workspace entity and auth guard.
- **Failure:** Direct database calls from Next.js server components bypassing FastAPI.
  - *Mitigation:* Route all business logic and RAG queries through FastAPI endpoints to maintain central security and telemetry.
- **Failure:** Hardcoding Supabase service role keys in frontend `.env.local`.
  - *Mitigation:* Reject immediately. Service role keys must only be read by FastAPI.
