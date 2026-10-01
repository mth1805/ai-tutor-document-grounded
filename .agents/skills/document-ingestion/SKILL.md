---
name: document-ingestion
description: Guides the extraction, cleaning, structure-aware chunking, metadata extraction, embedding, and vector persistence for PDF and DOCX files.
---

# Document Ingestion Skill

## 1. Purpose
The `document-ingestion` skill guides the end-to-end processing of uploaded learning materials (PDF and DOCX). It governs the transformation of raw binary documents into clean, structured, page-anchored text chunks with vector embeddings (`BAAI/bge-m3`) and full-text search representations persisted in Supabase PostgreSQL (`document_chunks`).

## 2. When to Use
Use this skill when:
- Implementing or modifying document parser pipelines (PDF/DOCX).
- Adjusting text extraction, header/section detection, and cleaning logic.
- Refining chunking strategies (token sizes, chunk overlaps, structure boundaries).
- Implementing or tuning batch vector embedding routines.
- Debugging ingestion performance, OOM crashes, or chunk metadata tracking.

## 3. What to Inspect First
Before modifying ingestion code, inspect:
1. `backend/app/services/ingestion/` — Existing parser and chunking implementations.
2. `backend/app/schemas/document.py` — Schema definitions for document status and chunk metadata.
3. Database table definitions in `supabase/migrations/` for:
   - `documents` (status, storage_path, file_size, etc.)
   - `document_chunks` (document_id, chunk_index, page_number, section_title, embedding, tsv).
4. `backend/app/ml/embeddings.py` — Current embedding batching and model wrapper.

## 4. Ingestion Pipeline Workflow
The document ingestion pipeline follows a strict, idempotent linear flow:

```
[Uploaded File (PDF/DOCX)]
          │
          ▼
[Step 1: Storage & Registration]
          │ Save binary to Supabase Storage; insert row in `documents` with status="queued"
          ▼
[Step 2: Parsing & Page Extraction]
          │ Extract text per page while preserving page numbers and structural headers
          ▼
[Step 3: Text Cleaning & Normalization]
          │ Strip non-printable chars, normalize whitespace, preserve code blocks/tables
          ▼
[Step 4: Structure-Aware Chunking]
          │ Partition text into 400-600 token chunks with 10-15% overlap and metadata
          ▼
[Step 5: Batch Embedding Generation]
          │ Pass chunks in batches (e.g., 16/32) to BGE-M3 (eval mode, no_grad)
          ▼
[Step 6: Atomic Database Persistence]
          │ Transactional insert into `document_chunks` with `embedding` and `tsv`
          ▼
[Step 7: Status Finalization]
          │ Update `documents.status` = "indexed" (or "failed" with error log)
```

### Detailed Pipeline Steps

#### Step 1: Storage & Registration
- Store original file in Supabase Storage under `{user_id}/{workspace_id}/{document_id}.{ext}`.
- Record document status as `queued`.

#### Step 2: Parsing & Page Extraction
- **PDF Extraction:** Use robust parsers (e.g., `pypdf`, `pymupdf` / `fitz`, or `pdfplumber`).
- Extract text per page explicitly: capture zero-indexed or one-indexed page counter so every extracted snippet knows its source `page_number`.
- **DOCX Extraction:** Use `python-docx` to parse paragraphs, tables, and heading styles (`Heading 1`, `Heading 2`) to retain section context.

#### Step 3: Text Cleaning & Normalization
- Remove extraneous hyphens across line breaks (de-hyphenation).
- Normalize excessive whitespace while preserving paragraph breaks.
- Retain markdown-friendly formatting for bullet points and tables.

#### Step 4: Structure-Aware Chunking
- Chunk size target: ~512 tokens (or 1500-2000 characters).
- Chunk overlap: 50-75 tokens (10-15%) to avoid splitting concepts across boundaries.
- **Mandatory Chunk Metadata:**
  ```python
  {
      "document_id": "uuid-...",
      "workspace_id": "uuid-...",
      "chunk_index": 0,          # Sequential integer order
      "page_number": 3,          # Exact page for citation
      "section_title": "Chapter 2: Vector Calculus",
      "token_count": 480,
      "chunk_text": "...",
  }
  ```

#### Step 5: Batch Embedding Generation
- Run BGE-M3 on chunk texts in configured batches (`batch_size=16` or `32`).
- Execute in a thread pool (`run_in_threadpool`) to avoid blocking the asyncio event loop.
- Generate 1024-dimensional dense vectors.

#### Step 6: Atomic Persistence & pgvector Storage
- Begin a database transaction.
- If re-indexing an existing document, delete previous chunks:
  `DELETE FROM document_chunks WHERE document_id = :doc_id;`
- Insert newly generated chunks with their embeddings:
  ```sql
  INSERT INTO document_chunks (
      id, document_id, workspace_id, chunk_index, page_number,
      section_title, chunk_text, token_count, embedding, tsv
  ) VALUES (
      :id, :document_id, :workspace_id, :chunk_index, :page_number,
      :section_title, :chunk_text, :token_count, :embedding,
      to_tsvector('english', :chunk_text)
  );
  ```
- Commit transaction.

#### Step 7: Status Finalization
- If all chunks are successfully written, update `documents.status = 'indexed'`.
- On any exception, update `documents.status = 'failed'` and record the sanitized error message in `documents.error_message`.

## 5. Engineering Rules
1. **Idempotency Guarantee:** Re-ingesting a document must completely replace old chunks without creating duplicate embeddings or orphaned database records.
2. **Page Anchoring is Non-Negotiable:** Every chunk must retain its source `page_number`. Chunks spanning page boundaries must record the primary page or list both (`page_start`, `page_end`).
3. **No Unbounded Memory Allocations:** Never read multi-hundred-megabyte files entirely into a single memory string. Stream or process page by page.
4. **Asynchronous Execution:** Document ingestion must run as a background task. The initial upload request must return immediately with `202 Accepted`.
5. **Handling Corrupt Files:** Encrypted, password-protected, or corrupted files must fail gracefully with descriptive error status, never crashing the worker.

## 6. Validation Checklist
- [ ] Uploaded PDF text extraction captures exact `page_number` for each chunk.
- [ ] Chunks generated do not exceed the configured token limit (512 tokens ± 10%).
- [ ] Re-running ingestion on the same file does not double the chunk count.
- [ ] Chunks are searchable via both vector cosine similarity and full-text `tsvector`.
- [ ] BGE-M3 embedding inference uses `torch.no_grad()` and is batched appropriately.
- [ ] Document status transitions cleanly: `queued` → `processing` → `indexed` (or `failed`).

## 7. Common Failure Modes & Mitigations
- **Failure:** Missing page numbers in citations.
  - *Root Cause:* Parser combined all pages into one giant string before chunking.
  - *Mitigation:* Chunk within page boundaries or carry page metadata forward in character index maps.
- **Failure:** Out of Memory (OOM) during embedding generation.
  - *Root Cause:* Passing 500 chunks to `model.encode()` in a single tensor pass.
  - *Mitigation:* Chunk embedding must use explicit micro-batches (16 to 32 items).
- **Failure:** Duplicate search results from the same document.
  - *Root Cause:* Re-uploading a modified document without deleting previous chunk rows.
  - *Mitigation:* Always wrap chunk replacement in an atomic `DELETE + INSERT` transaction keyed by `document_id`.
