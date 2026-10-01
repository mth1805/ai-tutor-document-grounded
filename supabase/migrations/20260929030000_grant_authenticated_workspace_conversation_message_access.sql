-- Grant only the table operations used by the authenticated application.
-- Row access remains restricted by the existing RLS policies.
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE
    public.workspaces,
    public.conversations,
    public.messages
TO authenticated;
