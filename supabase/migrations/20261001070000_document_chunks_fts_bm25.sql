-- ==============================================================================
-- Migration: Full-Text Search tsvector Column and GIN Index for BM25 Retrieval
-- Phase 7: Hybrid Retrieval + Cross-Encoder Reranking
-- ==============================================================================

-- 1. Add generated tsvector column for English lexical full-text search
-- Uses to_tsvector('english', coalesce(content, '')) to index chunk contents
ALTER TABLE public.document_chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;

-- 2. Create GIN index on generated tsvector column for fast BM25 / lexical search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON public.document_chunks USING gin(tsv);

-- 3. Reaffirm table permissions for authenticated role
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.document_chunks TO authenticated;
