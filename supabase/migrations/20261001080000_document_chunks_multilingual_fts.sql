-- ==============================================================================
-- Migration: Multilingual (Vietnamese + English) Full-Text Search tsvector
-- Phase 7 Review Technical Correction:
-- Replaces English-only stemming with a dual tsvector combining 'simple' (for
-- exact Vietnamese/multilingual tokens & stopword preservation) and 'english'
-- (for English Porter stemming).
-- Uses PostgreSQL FTS with cover-density ranking (ts_rank_cd).
-- ==============================================================================

-- 1. Drop existing index and column safely
DROP INDEX IF EXISTS public.idx_chunks_tsv;
ALTER TABLE public.document_chunks DROP COLUMN IF EXISTS tsv;

-- 2. Add multilingual tsvector generated column
-- 'simple' dictionary preserves accented Vietnamese tokens and prevents English stopwords
-- from removing valid Vietnamese words (e.g. 'in', 'an').
-- 'english' dictionary provides Porter stemming for English content.
ALTER TABLE public.document_chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(content, '')) || to_tsvector('english', coalesce(content, ''))
    ) STORED;

-- 3. Recreate GIN index on multilingual tsvector column for fast cover-density retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON public.document_chunks USING gin(tsv);

-- 4. Reaffirm table permissions for authenticated role
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.document_chunks TO authenticated;
