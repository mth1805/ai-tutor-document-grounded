-- Enqueue reads jobs, inserts an initial queued job, and deletes a terminal job
-- before a manual retry. Reassert these grants for existing Dev installations.
-- Row locks belong on the owned documents row, not this service-controlled queue.
BEGIN;
GRANT SELECT, INSERT, DELETE ON TABLE public.document_ingestion_jobs TO authenticated;
REVOKE UPDATE, TRUNCATE, REFERENCES, TRIGGER ON TABLE public.document_ingestion_jobs FROM authenticated;
-- Existing owner-scoped RLS policies and service_role-only RPC grants stay intact.
COMMIT;
