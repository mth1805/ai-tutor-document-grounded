# Database & Persistence Rules

## 1. Schema Management & Migration Discipline
- **Versioned Migrations:**
  - All database schema changes (tables, columns, indexes, RLS policies, triggers) must be managed through versioned SQL migration files (e.g., `supabase/migrations/YYYYMMDDHHMMSS_description.sql`).
  - Never apply ad-hoc, untracked DDL commands in production.
  - Every migration must be idempotent and testable against a clean local database instance.

## 2. Core Entities & Referential Integrity
The system persists the following relational hierarchy in Supabase PostgreSQL:
- **`workspaces`:**
  - Belongs to `user_id` (`auth.users.id`).
  - Primary organizational container for documents and conversations.
- **`documents`:**
  - References `workspace_id` (`ON DELETE CASCADE`) and `user_id`.
  - Stores metadata: `file_name`, `file_type` (PDF/DOCX), `file_size`, `storage_path`, `status` (`queued`, `processing`, `indexed`, `failed`), `created_at`.
- **`document_chunks`:**
  - References `document_id` (`ON DELETE CASCADE`) and `workspace_id`.
  - Stores: `chunk_index`, `page_number`, `section_title`, `chunk_text`, `token_count`, `embedding` (`vector(1024)`), `tsv` (`tsvector` for lexical search).
- **`conversations`:**
  - References `workspace_id` (`ON DELETE CASCADE`) and `user_id`.
  - Stores conversation title, timestamps, and active metadata.
- **`messages`:**
  - References `conversation_id` (`ON DELETE CASCADE`).
  - Stores: `sender` (`user` / `assistant`), `content`, `tokens_used`, `citations` (JSONB array containing cited chunk IDs, document names, page numbers, or web URLs).

## 3. Deletion Semantics & Data Integrity
- **No Silent Deletions:**
  - Cascade rules must be explicit:
    - Deleting a `workspace` cascades to its `documents`, `document_chunks`, `conversations`, and `messages`.
    - Deleting a `document` cascades to its `document_chunks` and deletes the corresponding raw file from Supabase Storage.
    - Deleting a `conversation` cascades to its `messages`.
  - Deleting messages or documents must never leave orphaned vector chunks in `document_chunks`.
- **Atomic Operations:**
  - When updating or replacing document chunks during re-indexing, execute the operation within a database transaction: delete old chunks and insert new chunks atomically.

## 4. Indexing Strategy
- **Relational Lookups:**
  - Standard B-Tree indexes on all foreign keys:
    - `CREATE INDEX idx_documents_workspace ON documents(workspace_id);`
    - `CREATE INDEX idx_chunks_doc ON document_chunks(document_id);`
    - `CREATE INDEX idx_chunks_workspace ON document_chunks(workspace_id);`
    - `CREATE INDEX idx_conversations_workspace ON conversations(workspace_id);`
    - `CREATE INDEX idx_messages_conversation ON messages(conversation_id);`
- **Vector Search Index:**
  - Create HNSW index on `document_chunks.embedding` using cosine distance:
    ```sql
    CREATE INDEX idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    ```
- **Lexical / Full-Text Search Index:**
  - Create GIN index on generated `tsvector` column for fast BM25 / keyword retrieval:
    ```sql
    CREATE INDEX idx_chunks_tsv ON document_chunks USING gin(tsv);
    ```

## 5. Persistence Guarantee
- **Zero Browser-Memory Reliance:**
  - Workspaces, documents, conversations, and messages must be retrieved from PostgreSQL upon initial page visit or refresh.
  - Client state stores (React state, Zustand, or TanStack Query) act purely as a cache layer for database records.
