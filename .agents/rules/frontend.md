# Frontend Engineering Rules

## 1. Stack & Architecture
- **Framework:** Next.js (App Router), React 18+, TypeScript (strict mode enabled).
- **Styling:** Tailwind CSS with a consistent design system (avoid arbitrary ad-hoc inline styles).
- **Component Separation:**
  - Use **React Server Components (RSC)** by default for layout skeletons, static metadata, and server-side route guards.
  - Use `"use client"` only for components that require hooks (`useState`, `useEffect`, `useReducer`), browser APIs, or direct user interaction.
- **Routing & State Synchronization:**
  - Workspaces and conversations must be reflected in the URL path (e.g., `/workspace/[workspaceId]/c/[conversationId]`).
  - Active workspace and conversation state must be recoverable directly from URL params upon page refresh.
  - Never store critical application state exclusively in transient React state without syncing to the URL or database.

## 2. Fast Rendering & UI Responsiveness
- **Zero ML Blocking:** The UI must render immediately. Never block page delivery or component mounting on backend ML model initialization.
- **Loading & Skeleton States:**
  - Provide skeleton loaders for workspace sidebars, conversation lists, and document panels.
  - Implement optimistic UI updates when sending messages (show user message immediately with a "sending" state before streaming begins).
- **Graceful Error Boundaries:**
  - Wrap conversation panels, document previewers, and chat message lists in React Error Boundaries to prevent full-page crashes.
  - Provide actionable retry mechanisms for failed requests or aborted streams.

## 3. Streaming & Real-Time Consumption
- **Stream Consumption:**
  - Consume backend LLM responses using `ReadableStream` / `fetch` with `TextDecoderStream` or standard Server-Sent Events (SSE).
  - Buffer partial chunks cleanly to avoid JSON parse errors or broken unicode characters.
  - Provide an explicit "Stop Generating" button that aborts the client `AbortController` and signals cancellation to the backend.
- **Time-to-First-Token (TTFT) Focus:**
  - Display the streaming indicator immediately upon receiving the initial response header.
  - Append tokens to the active message smoothly without triggering heavy layout shifts.

## 4. Citation Rendering & Source Distinction
- **Distinct Source Visuals:**
  - **Document Citations:** Must render as distinct badges (e.g., `[Doc: physics_notes.pdf, p. 12]`) with distinct styling (e.g., primary accent color, document icon).
  - **Web Citations:** Must render with a distinct "Web" indicator (e.g., globe icon, domain pill `[Web: wikipedia.org]`).
  - Web and document citations must never look identical.
- **Interactive Citation Inspection:**
  - Clicking or hovering over a document citation badge must open a popover or context drawer showing the exact text snippet from the chunk and page reference.
  - Web citation links must open in a new tab with `rel="noopener noreferrer"`.

## 5. Document Upload & Management UI
- **Client-Side Validation:**
  - Check file extensions (`.pdf`, `.docx`) and enforce file size limits (e.g., 25MB default) before initiating upload.
  - Display file size and upload progress indicators.
- **Asynchronous Status Polling / Events:**
  - Once uploaded, display ingestion status pills: `Queued` → `Processing` → `Indexed` / `Failed`.
  - Do not lock the chat UI while a document is indexing; allow the user to continue chatting with already-indexed documents.

## 6. Security & Environment Variable Rules
- **Secret Protection:**
  - NEVER expose `GEMINI_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, or database credentials to client-side code.
  - Only variables intended for public browser consumption may carry the `NEXT_PUBLIC_` prefix (e.g., `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_API_URL`).
- **Input Sanitization:**
  - Sanitize user markdown rendering (e.g., using `rehype-sanitize`) to prevent XSS attacks when displaying AI-generated markdown or code blocks.
