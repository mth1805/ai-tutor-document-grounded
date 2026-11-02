-- Durable ingestion queue. Normal app sessions can enqueue/read/delete only
-- their own terminal jobs. Worker state transitions happen through narrowly
-- granted SECURITY DEFINER RPCs called with the backend-only service key.

ALTER TABLE public.documents DROP CONSTRAINT IF EXISTS documents_status_check;
ALTER TABLE public.documents ADD CONSTRAINT documents_status_check
    CHECK (status IN ('uploaded', 'queued', 'processing', 'processed', 'failed'));

CREATE TABLE IF NOT EXISTS public.document_ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL UNIQUE
        REFERENCES public.documents(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 10),
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    locked_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ingestion_job_document_owner_fkey
        FOREIGN KEY (workspace_id, user_id)
        REFERENCES public.workspaces(id, user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_ready
    ON public.document_ingestion_jobs (available_at, created_at)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_expired
    ON public.document_ingestion_jobs (locked_at)
    WHERE status = 'processing';
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_user
    ON public.document_ingestion_jobs (user_id, created_at DESC);

DROP TRIGGER IF EXISTS set_document_ingestion_jobs_updated_at
    ON public.document_ingestion_jobs;
CREATE TRIGGER set_document_ingestion_jobs_updated_at
BEFORE UPDATE ON public.document_ingestion_jobs
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

ALTER TABLE public.document_ingestion_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view their ingestion jobs"
    ON public.document_ingestion_jobs FOR SELECT TO authenticated
    USING (
        auth.uid() = user_id AND EXISTS (
            SELECT 1 FROM public.documents d
            WHERE d.id = document_id AND d.user_id = auth.uid()
              AND d.workspace_id = workspace_id
        )
    );
CREATE POLICY "Users can enqueue their documents"
    ON public.document_ingestion_jobs FOR INSERT TO authenticated
    WITH CHECK (
        auth.uid() = user_id AND status = 'queued'
        AND attempt_count = 0 AND max_attempts = 3
        AND EXISTS (
            SELECT 1 FROM public.documents d
            WHERE d.id = document_id AND d.user_id = auth.uid()
              AND d.workspace_id = workspace_id
        )
    );
CREATE POLICY "Users can remove terminal ingestion jobs"
    ON public.document_ingestion_jobs FOR DELETE TO authenticated
    USING (
        auth.uid() = user_id AND status IN ('completed', 'failed')
    );

GRANT SELECT, INSERT, DELETE ON public.document_ingestion_jobs TO authenticated;
REVOKE UPDATE ON public.document_ingestion_jobs FROM authenticated;

CREATE OR REPLACE FUNCTION public.claim_document_ingestion_jobs(
    p_limit INTEGER DEFAULT 5,
    p_lease_seconds INTEGER DEFAULT 900
)
RETURNS TABLE(job_id UUID, claimed_document_id UUID, claimed_user_id UUID, claimed_attempt INTEGER)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    -- Exhausted jobs whose worker disappeared are terminally failed.
    UPDATE public.document_ingestion_jobs j
       SET status = 'failed', locked_at = NULL,
           last_error = COALESCE(j.last_error, 'Worker lease expired after maximum attempts')
     WHERE j.status = 'processing'
       AND j.locked_at < now() - make_interval(secs => GREATEST(60, LEAST(p_lease_seconds, 3600)))
       AND j.attempt_count >= j.max_attempts;

    UPDATE public.documents d
       SET status = 'failed', processed_at = now(),
           processing_error = 'Ingestion worker lease expired after maximum attempts'
     WHERE EXISTS (
        SELECT 1 FROM public.document_ingestion_jobs j
        WHERE j.document_id = d.id AND j.user_id = d.user_id
          AND j.status = 'failed' AND j.last_error = 'Worker lease expired after maximum attempts'
     ) AND d.status IN ('queued', 'processing');

    RETURN QUERY
    WITH candidates AS (
        SELECT j.id
          FROM public.document_ingestion_jobs j
         WHERE (j.status = 'queued' AND j.available_at <= now())
            OR (j.status = 'processing'
                AND j.locked_at < now() - make_interval(secs => GREATEST(60, LEAST(p_lease_seconds, 3600)))
                AND j.attempt_count < j.max_attempts)
         ORDER BY j.available_at, j.created_at
         FOR UPDATE SKIP LOCKED
         LIMIT GREATEST(1, LEAST(p_limit, 10))
    ), claimed AS (
        UPDATE public.document_ingestion_jobs j
           SET status = 'processing', locked_at = now(), attempt_count = j.attempt_count + 1
          FROM candidates c
         WHERE j.id = c.id
        RETURNING j.id, j.document_id, j.user_id, j.attempt_count
    )
    SELECT c.id, c.document_id, c.user_id, c.attempt_count FROM claimed c;
END;
$$;

CREATE OR REPLACE FUNCTION public.complete_document_ingestion_job(p_job_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE updated_rows INTEGER;
BEGIN
    UPDATE public.document_ingestion_jobs
       SET status = 'completed', locked_at = NULL, last_error = NULL
     WHERE id = p_job_id AND status = 'processing';
    GET DIAGNOSTICS updated_rows = ROW_COUNT;
    RETURN updated_rows = 1;
END;
$$;

CREATE OR REPLACE FUNCTION public.fail_document_ingestion_job(
    p_job_id UUID,
    p_error TEXT
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE job_row public.document_ingestion_jobs%ROWTYPE;
DECLARE next_status TEXT;
DECLARE retry_seconds INTEGER;
BEGIN
    SELECT * INTO job_row FROM public.document_ingestion_jobs
     WHERE id = p_job_id AND status = 'processing' FOR UPDATE;
    IF NOT FOUND THEN RETURN 'missing'; END IF;

    IF job_row.attempt_count < job_row.max_attempts THEN
        next_status := 'queued';
        retry_seconds := LEAST(
            900,
            30 * power(2::NUMERIC, LEAST(job_row.attempt_count - 1, 5))::INTEGER
        );
        UPDATE public.document_ingestion_jobs
           SET status = next_status, locked_at = NULL,
               available_at = now() + make_interval(secs => retry_seconds),
               last_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = p_job_id;
        UPDATE public.documents
           SET status = 'queued', processing_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = job_row.document_id AND user_id = job_row.user_id;
    ELSE
        next_status := 'failed';
        UPDATE public.document_ingestion_jobs
           SET status = next_status, locked_at = NULL,
               last_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = p_job_id;
        UPDATE public.documents
           SET status = 'failed', processed_at = now(),
               processing_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = job_row.document_id AND user_id = job_row.user_id;
    END IF;
    RETURN next_status;
END;
$$;

CREATE OR REPLACE FUNCTION public.get_claimed_document_ingestion_job(p_job_id UUID)
RETURNS TABLE(job_id UUID, claimed_document_id UUID, claimed_user_id UUID, claimed_attempt INTEGER)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
    SELECT j.id, j.document_id, j.user_id, j.attempt_count
      FROM public.document_ingestion_jobs j
     WHERE j.id = p_job_id AND j.status = 'processing'
       AND j.locked_at > now() - interval '1 hour'
$$;

REVOKE ALL ON FUNCTION public.claim_document_ingestion_jobs(INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.complete_document_ingestion_job(UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.fail_document_ingestion_job(UUID, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_claimed_document_ingestion_job(UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_document_ingestion_jobs(INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.complete_document_ingestion_job(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.fail_document_ingestion_job(UUID, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.get_claimed_document_ingestion_job(UUID) TO service_role;
