-- ==============================================================================
-- Migration: Add Citations Column to Messages Table
-- Phase 8: Grounded RAG Generation & Citation Persistence
-- ==============================================================================

-- Add JSONB citations column to store structured source chunk provenance
ALTER TABLE public.messages
ADD COLUMN IF NOT EXISTS citations JSONB DEFAULT '[]'::jsonb;

-- Comment describing the citations structure
COMMENT ON COLUMN public.messages.citations IS 'Array of structured document citations: [{document_id, document_name, chunk_id, page_start, page_end, snippet}]';
