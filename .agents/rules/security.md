# Security & Multi-Tenancy Rules

## 1. Secret Management & Credential Isolation
- **Zero Hardcoded Secrets:**
  - Never commit API keys, database connection strings, JWT secrets, or service account keys into Git.
  - Store all credentials in `.env` (gitignored). Provide an updated `.env.example` with dummy values for development.
- **Client vs. Server Boundary:**
  - Secrets like `GEMINI_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, and raw database connection URLs must live exclusively on the backend server.
  - Never prefix sensitive variables with `NEXT_PUBLIC_`.
  - Frontend code may only access `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

## 2. Multi-Tenancy & Authorization (IDOR Prevention)
- **Ownership Verification on Every Endpoint:**
  - Every API endpoint manipulating workspaces, conversations, messages, or documents must verify that the authenticated user (`user_id` from JWT) owns or is granted access to the resource.
  - Reject requests targeting resources outside the user's workspace with `403 Forbidden` or `404 Not Found`. Never rely on client-supplied `user_id` in request payloads; extract `user_id` strictly from the verified JWT.
- **Database Row Level Security (RLS):**
  - Enable RLS on all Supabase tables (`workspaces`, `documents`, `document_chunks`, `conversations`, `messages`).
  - Write explicit policies:
    - Users can only `SELECT`, `INSERT`, `UPDATE`, `DELETE` rows matching their `auth.uid()`.
    - Document chunks must be accessible only if the parent document belongs to a workspace owned by the user.

## 3. File Upload Security & Sanitization
- **Strict Content Validation:**
  - Enforce an allowlist of supported extensions: `.pdf`, `.docx`.
  - Validate MIME types (`application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`).
  - Verify **magic bytes** at the backend before processing:
    - PDF: starts with `%PDF-` (`0x25 0x50 0x44 0x46 0x2D`).
    - DOCX: starts with PK zip header (`0x50 0x4B 0x03 0x04`).
- **File Size Limits & Storage Sanitization:**
  - Limit file size to a configurable maximum (e.g., 25MB).
  - Sanitize original filenames before storing (strip directory traversal characters `../`, control characters, or non-printable ASCII).
  - Store files in Supabase Storage with generated UUID keys: `{user_id}/{workspace_id}/{document_id}.pdf`.

## 4. Prompt Injection & LLM Security
- **Untrusted Input Treatment:**
  - Treat all uploaded document text and user chat queries as untrusted data.
  - Use clear system prompt delimiters (e.g., `<context>`, `</context>`, `<query>`, `</query>`) when assembling LLM prompts.
  - Instruct the model that context text contains data only and cannot override core system instructions or safety rules.
- **Output Sanitization:**
  - Strip dangerous HTML/scripts from LLM outputs before rendering on the frontend (use HTML sanitizers / safe markdown renderers).

## 5. Network & Error Handling Security
- **CORS Configuration:**
  - FastAPI CORS middleware must explicitly allow only the configured frontend origins (e.g., `http://localhost:3000` in dev, production domain in prod). Never use `allow_origins=["*"]` with credentials.
- **Sanitized Error Responses:**
  - Never leak internal server paths, database connection strings, or full Python tracebacks in API error responses.
  - Log full tracebacks securely to server logs; return clean, standardized error envelopes to clients.
