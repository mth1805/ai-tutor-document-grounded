-- ==============================================================================
-- Migration: Document Processing Status and Document Chunks Table
-- Phase 5: Document Parsing, Selective OCR, Cleaning, and Chunking
-- ==============================================================================

-- 1. Extend Document Status and Add Processing Metadata Columns
ALTER TABLE public.documents
    DROP CONSTRAINT IF EXISTS documents_status_check;

ALTER TABLE public.documents
    ADD CONSTRAINT documents_status_check
    CHECK (status IN ('uploaded', 'processing', 'processed', 'failed'));

ALTER TABLE public.documents
    ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS processing_error TEXT;

-- Index for status lookups
CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents(status);

-- 2. Create Document Chunks Table (Page-anchored text chunks)
-- Note: Embeddings and vector columns are deferred to Phase 6.
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number_start INTEGER NOT NULL,
    page_number_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT document_chunks_doc_chunk_uniq UNIQUE (document_id, chunk_index)
);

-- 3. Indexes for fast relational and document-chunk lookups
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON public.document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id ON public.document_chunks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON public.document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_chunk_idx ON public.document_chunks(document_id, chunk_index);

-- 4. Automatic updated_at trigger for document_chunks
DROP TRIGGER IF EXISTS set_document_chunks_updated_at ON public.document_chunks;
CREATE TRIGGER set_document_chunks_updated_at
BEFORE UPDATE ON public.document_chunks
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

-- 5. Enable Row Level Security (RLS)
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;

-- 6. Strict RLS Policies (Users can only access chunks from workspaces they own)
DROP POLICY IF EXISTS "Users can select their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can select their own document chunks"
ON public.document_chunks FOR SELECT
TO authenticated
USING (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Users can insert their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can insert their own document chunks"
ON public.document_chunks FOR INSERT
TO authenticated
WITH CHECK (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Users can update their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can update their own document chunks"
ON public.document_chunks FOR UPDATE
TO authenticated
USING (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
)
WITH CHECK (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Users can delete their own document chunks" ON public.document_chunks;
CREATE POLICY "Users can delete their own document chunks"
ON public.document_chunks FOR DELETE
TO authenticated
USING (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

-- 7. Grant operations to authenticated role
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.document_chunks TO authenticated;
