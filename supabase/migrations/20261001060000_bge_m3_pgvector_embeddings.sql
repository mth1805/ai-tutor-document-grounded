-- ==============================================================================
-- Migration: BGE-M3 Embedding Persistence and pgvector HNSW Indexing
-- Phase 6: Vector Storage, Embedding Reproducibility Metadata, and Lifecycle
-- ==============================================================================

-- 1. Safely enable the PostgreSQL pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Extend public.document_chunks with vector embeddings and reproducibility metadata
-- BAAI/bge-m3 dense vector dimension is exactly 1024.
ALTER TABLE public.document_chunks
    ADD COLUMN IF NOT EXISTS embedding vector(1024),
    ADD COLUMN IF NOT EXISTS embedding_model TEXT,
    ADD COLUMN IF NOT EXISTS embedding_version TEXT,
    ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;

-- 3. Create HNSW Vector Index on document_chunks.embedding using cosine distance
-- Configured with m=16, ef_construction=64 according to performance guidelines.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON public.document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4. Extend public.documents with embedding lifecycle status and error tracking
ALTER TABLE public.documents
    ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS embedding_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS embedding_error TEXT;

-- Enforce valid embedding status values
ALTER TABLE public.documents
    DROP CONSTRAINT IF EXISTS documents_embedding_status_check;

ALTER TABLE public.documents
    ADD CONSTRAINT documents_embedding_status_check
    CHECK (embedding_status IN ('pending', 'processing', 'completed', 'failed'));

-- Index for embedding status queries and background queue lookups
CREATE INDEX IF NOT EXISTS idx_documents_embedding_status ON public.documents(embedding_status);

-- 5. Reaffirm permissions for authenticated role
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.document_chunks TO authenticated;
