-- ==============================================================================
-- Migration: Create Documents Table and Configure Private Storage Bucket
-- Phase 4: Document Upload + Persistent Storage
-- ==============================================================================

-- 1. Create documents table
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    original_filename VARCHAR(255) NOT NULL,
    storage_path TEXT NOT NULL UNIQUE,
    mime_type VARCHAR(128) NOT NULL,
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    status VARCHAR(32) NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    CONSTRAINT documents_workspace_owner_fkey
        FOREIGN KEY (workspace_id, user_id)
        REFERENCES public.workspaces (id, user_id)
        ON DELETE CASCADE
);

-- 2. Indexes for workspace lookups, user lookups, and fast sorting
CREATE INDEX IF NOT EXISTS idx_documents_workspace_id ON public.documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON public.documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_storage_path ON public.documents(storage_path);

-- 3. Automatic updated_at trigger for documents
DROP TRIGGER IF EXISTS set_documents_updated_at ON public.documents;
CREATE TRIGGER set_documents_updated_at
BEFORE UPDATE ON public.documents
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

-- 4. Enable Row Level Security (RLS)
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- 5. Strict RLS Policies (Users can only access documents in their owned workspaces)
DROP POLICY IF EXISTS "Users can select their own documents" ON public.documents;
CREATE POLICY "Users can select their own documents"
ON public.documents FOR SELECT
TO authenticated
USING (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Users can insert their own documents" ON public.documents;
CREATE POLICY "Users can insert their own documents"
ON public.documents FOR INSERT
TO authenticated
WITH CHECK (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Users can update their own documents" ON public.documents;
CREATE POLICY "Users can update their own documents"
ON public.documents FOR UPDATE
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

DROP POLICY IF EXISTS "Users can delete their own documents" ON public.documents;
CREATE POLICY "Users can delete their own documents"
ON public.documents FOR DELETE
TO authenticated
USING (
    auth.uid() = user_id AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);

-- 6. Grant operations to authenticated role
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.documents TO authenticated;

-- 7. Configure Private Storage Bucket for Documents
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'documents',
    'documents',
    false,
    26214400, -- 25MB limit
    ARRAY[
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'image/png',
        'image/jpeg',
        'image/webp'
    ]
)
ON CONFLICT (id) DO UPDATE SET
    public = false,
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

-- 8. Storage Object RLS Policies
DROP POLICY IF EXISTS "Authenticated users can select workspace documents" ON storage.objects;
CREATE POLICY "Authenticated users can select workspace documents"
ON storage.objects FOR SELECT
TO authenticated
USING (
    bucket_id = 'documents' AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id::text = (storage.foldername(name))[1]
          AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Authenticated users can insert workspace documents" ON storage.objects;
CREATE POLICY "Authenticated users can insert workspace documents"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'documents' AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id::text = (storage.foldername(name))[1]
          AND w.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS "Authenticated users can delete workspace documents" ON storage.objects;
CREATE POLICY "Authenticated users can delete workspace documents"
ON storage.objects FOR DELETE
TO authenticated
USING (
    bucket_id = 'documents' AND
    EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id::text = (storage.foldername(name))[1]
          AND w.user_id = auth.uid()
    )
);
