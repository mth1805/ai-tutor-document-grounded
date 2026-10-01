-- Keep conversation ownership consistent with its workspace at the database layer.
-- The composite FK prevents a conversation from naming one user while referencing
-- another user's workspace, including updates made outside the FastAPI service.
CREATE UNIQUE INDEX IF NOT EXISTS idx_workspaces_id_user_id
    ON public.workspaces (id, user_id);

ALTER TABLE public.conversations
    DROP CONSTRAINT IF EXISTS conversations_workspace_id_fkey;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'conversations_workspace_owner_fkey'
          AND conrelid = 'public.conversations'::regclass
    ) THEN
        ALTER TABLE public.conversations
            ADD CONSTRAINT conversations_workspace_owner_fkey
            FOREIGN KEY (workspace_id, user_id)
            REFERENCES public.workspaces (id, user_id)
            ON DELETE CASCADE;
    END IF;
END $$;

DROP POLICY IF EXISTS "Users can update their own conversations" ON public.conversations;
CREATE POLICY "Users can update their own conversations"
ON public.conversations FOR UPDATE
TO authenticated
USING (
    auth.uid() = user_id AND EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
)
WITH CHECK (
    auth.uid() = user_id AND EXISTS (
        SELECT 1 FROM public.workspaces w
        WHERE w.id = workspace_id AND w.user_id = auth.uid()
    )
);
