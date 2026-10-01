# Backend Engineering Rules

## 1. Stack & Directory Organization
- **Framework:** FastAPI (Python 3.11+), Pydantic v2.
- **Layered Architecture:**
  - `app/api/v1/`: Endpoint route definitions grouped by domain (`workspaces.py`, `documents.py`, `conversations.py`, `chat.py`).
  - `app/schemas/`: Pydantic models for request bodies, query params, and serialized responses.
  - `app/services/`: Pure business logic decoupled from HTTP layers (e.g., `ingestion_service.py`, `retrieval_service.py`, `rag_service.py`, `web_search_service.py`).
  - `app/core/`: Configuration via `pydantic-settings`, global exceptions, security helpers, and application lifecycle.
  - `app/db/`: Database clients, pgvector queries, and repository abstractions.
  - `app/ml/`: Model loaders and inference wrappers for BGE-M3 and Cross-Encoder.

## 2. API Contract & Validation
- **Pydantic Discipline:**
  - All endpoint inputs (body, query, path) and outputs must use explicit Pydantic v2 schemas.
  - Disallow untyped dictionaries (`dict` or `Any`) in route signatures.
  - Use `field_validator` / `model_validator` to enforce semantic constraints (e.g., non-empty strings, valid UUIDs, supported MIME types).
- **HTTP Status Code Conventions:**
  - `200 OK`: Standard successful retrieval/mutation.
  - `201 Created`: Resource successfully created (workspace, document record, conversation).
  - `202 Accepted`: Long-running task accepted for asynchronous processing (e.g., document ingestion job).
  - `400 Bad Request`: Malformed payload or failed business precondition.
  - `401 Unauthorized`: Missing or invalid authentication token.
  - `403 Forbidden`: Authenticated user does not own or have access to the target workspace or resource.
  - `404 Not Found`: Resource does not exist within the authorized scope.
  - `422 Unprocessable Entity`: Schema validation errors.
- **Uniform Error Envelope:** Return structured error objects:
  ```json
  {
    "error": {
      "code": "RESOURCE_NOT_FOUND",
      "message": "Document not found in active workspace",
      "details": {}
    }
  }
  ```

## 3. Asynchronous Execution & Non-Blocking Design
- **Event Loop Safety:**
  - Route handlers must be `async def`.
  - Use asynchronous database clients (e.g., `asyncpg` or Supabase async client).
  - Heavy CPU-bound tasks (tokenization, local embedding calculation, Cross-Encoder inference) must NEVER run directly on the event loop. Wrap them using `fastapi.concurrency.run_in_threadpool` or delegate to background task queues.
- **Streaming Handlers:**
  - Chat completions must stream via `StreamingResponse` using `text/event-stream` (Server-Sent Events) or NDJSON.
  - Stream events must follow a typed structure (e.g., `event: token`, `event: citations`, `event: error`, `event: done`).

## 4. Dependency Injection & Service Boundaries
- **FastAPI `Depends`:**
  - Inject database connections, authenticated user context (`get_current_user`), and service classes via `Depends`.
  - Never instantiate database connections or service classes directly inside route functions.
- **Decoupled Business Logic:**
  - Keep route functions thin (parsing input, calling service, returning response).
  - Services must be testable independently of FastAPI HTTP request objects.

## 5. Lifespan & Model Lifecycle
- **FastAPI Lifespan Context:**
  - Initialize long-lived resources (connection pools, cached model singletons) within the `@asynccontextmanager` lifespan handler of `FastAPI(lifespan=...)`.
  - Gracefully close connection pools on application shutdown.
- **Zero In-Request Model Instantiation:**
  - Never initialize BGE-M3 or Cross-Encoder inside a request handler.
  - Retrieve the pre-warmed model instance from application state or a thread-safe singleton.
