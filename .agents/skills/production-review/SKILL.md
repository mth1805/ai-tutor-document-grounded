---
name: production-review
description: Conducts senior-level architectural and production-readiness reviews covering security, API design, RAG correctness, database integrity, and maintainability.
---

# Production Review Skill

## 1. Purpose
The `production-review` skill embodies the persona and rigor of a Staff / Lead Software Architect reviewing the **AI Tutor Assistant** repository. It conducts exhaustive code and architecture audits before PR merges or production milestones, identifying concrete vulnerabilities, anti-patterns, data leaks, and performance bottlenecks, and providing prioritized remediation steps.

## 2. When to Use
Use this skill when:
- Completing a major feature phase before marking it done.
- Conducting pre-merge reviews for architectural consistency.
- Auditing repository security (secrets, RLS, prompt injection, IDOR).
- Evaluating code maintainability, test coverage, and documentation.
- Preparing the application for showcase, portfolio evaluation, or production deployment.

## 3. What to Inspect First
The review agent must inspect the entire system in this order:
1. `AGENTS.md` and `/.agents/rules/` — The project constitution and domain standards.
2. Git status / recent diff:
   - Identify all files created or modified.
3. Security boundaries:
   - Frontend environment files (`.env.local`) and client bundles.
   - Backend auth middleware and RLS policies in `supabase/migrations/`.
4. API contracts & schemas:
   - `backend/app/schemas/` vs. `frontend/src/types/`.
5. RAG execution path:
   - Verify retrieval order: Uploaded Docs → Reranker → Relevance Gate → Web Fallback.
6. Test suites:
   - `backend/tests/` and `frontend/__tests__/`.

## 4. Production Review Workflow
Follow this 6-dimension review methodology:

```
[Dimension 1: Architectural Integrity & Separation]
      │ Check frontend/backend boundaries & phase alignment
      ▼
[Dimension 2: Security & Multi-Tenancy (IDOR, Secrets)]
      │ Audit auth token verification, RLS policies, file validation
      ▼
[Dimension 3: RAG Correctness & Grounding Hierarchy]
      │ Audit document primacy, relevance gate, citation verification
      ▼
[Dimension 4: Database Integrity & Migrations]
      │ Audit migrations, foreign keys, cascades, pgvector HNSW indexes
      ▼
[Dimension 5: Performance & Non-Blocking Async]
      │ Check model caching, background ingestion, streaming TTFT
      ▼
[Dimension 6: Code Quality, Tests & Observability]
      │ Audit type safety, error boundaries, test coverage, logging
```

### Review Findings Classification
Every issue identified during review must be categorized with clear severity:
- **`[CRITICAL]`**: Severe vulnerability, data loss risk, secret exposure, broken RLS, or per-request model loading. Must block deployment/merge.
- **`[HIGH]`**: Incorrect RAG hierarchy (e.g., web search bypassing documents), missing indexes causing table scans, or unhandled exceptions crashing workers.
- **`[MEDIUM]`**: Missing optimistic UI, lack of chunk boundary overlap, untyped API responses, or missing error boundaries.
- **`[LOW]`**: Code style nitpicks, minor docstring omission, or redundant imports.

### Review Report Format
The agent must generate structured audit reports formatted as:

```markdown
# Production Review Audit Report

## Summary
- **Overall Verdict:** [APPROVED / CHANGES REQUESTED / BLOCKED]
- **Target Phase:** Phase X
- **Issues Found:** X Critical | Y High | Z Medium | W Low

## Detailed Findings

### [CRITICAL] 1. Gemini API Key Exposed in Client Component
- **Location:** `frontend/src/components/ChatBox.tsx:14`
- **Impact:** Compromises API credentials; allows unauthorized billing.
- **Remediation:** Remove client-side API call. Route prompt generation through FastAPI backend endpoint `/api/v1/chat/stream`.

### [HIGH] 2. Web Search Overrides Document Evidence
- **Location:** `backend/app/services/rag_service.py:82`
- **Impact:** Violates core product tenet: document-grounded AI learning.
- **Remediation:** Implement relevance gating threshold check before invoking web search service.

...
```

## 5. Engineering Rules
1. **Be Constructive and Actionable:** Never point out a defect without offering a concrete code solution or architectural alternative.
2. **Uphold the Constitution:** Reject any PR or code that violates rules established in `AGENTS.md` or `/.agents/rules/`.
3. **No Compromise on Security:** Any detected hardcoded secret or bypassed authentication guard immediately results in a `BLOCKED` verdict.
4. **Enforce Test Evidence:** Code without automated tests verifying the happy path and edge cases cannot be approved for production.
5. **No Premature Complexity:** Flag over-engineered solutions (e.g., adding Kafka when a simple queue suffices).

## 6. Validation Checklist
- [ ] No secrets or service role keys in Git history or client code.
- [ ] All database queries enforce user and workspace ownership.
- [ ] Document retrieval always takes precedence over web search.
- [ ] Every citation in generated output is verified against retrieved chunks.
- [ ] Heavy ingestion tasks run in background workers, returning `202 Accepted`.
- [ ] Machine learning models are loaded once per process and cached locally.
- [ ] All migrations are versioned, reversible, and testable.
- [ ] Unit and integration tests pass successfully without mocks hiding real failures.

## 7. Common Failure Modes & Mitigations
- **Failure:** Approving a feature that implements future phases prematurely.
  - *Mitigation:* Explicitly verify PR scope against the currently active phase in `AGENTS.md`.
- **Failure:** Overlooking silent data deletion bugs in database schema changes.
  - *Mitigation:* Audit foreign key `ON DELETE` rules in migrations; verify that deleting a workspace properly cleans up associated vectors and files.
- **Failure:** Approving synchronous ML calls inside FastAPI async endpoints.
  - *Mitigation:* Check all model invocations; ensure they are wrapped with `run_in_threadpool` or dispatched to background tasks.
