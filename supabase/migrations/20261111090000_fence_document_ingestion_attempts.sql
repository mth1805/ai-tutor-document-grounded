-- Fence every queued ingestion write with the monotonically increasing claim
-- attempt. Apply this migration before deploying workers that pass the token.
ALTER TABLE public.document_ingestion_jobs
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;

UPDATE public.document_ingestion_jobs
   SET lease_expires_at = COALESCE(locked_at, now()) + interval '3600 seconds'
 WHERE status = 'processing' AND lease_expires_at IS NULL;

ALTER TABLE public.document_ingestion_jobs
    DROP CONSTRAINT IF EXISTS document_ingestion_jobs_lease_state_check;
ALTER TABLE public.document_ingestion_jobs
    ADD CONSTRAINT document_ingestion_jobs_lease_state_check
    CHECK ((status = 'processing') = (lease_expires_at IS NOT NULL));

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_lease_expiry
    ON public.document_ingestion_jobs (lease_expires_at)
    WHERE status = 'processing';

CREATE OR REPLACE FUNCTION public.claim_document_ingestion_jobs(
    p_limit INTEGER DEFAULT 5,
    p_lease_seconds INTEGER DEFAULT 900
)
RETURNS TABLE(job_id UUID, claimed_document_id UUID, claimed_user_id UUID, claimed_attempt INTEGER)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE lease_seconds INTEGER := GREATEST(60, LEAST(p_lease_seconds, 3600));
BEGIN
    UPDATE public.document_ingestion_jobs j
       SET status = 'failed', locked_at = NULL, lease_expires_at = NULL,
           last_error = COALESCE(j.last_error, 'Worker lease expired after maximum attempts')
     WHERE j.status = 'processing'
       AND j.lease_expires_at <= clock_timestamp()
       AND j.attempt_count >= j.max_attempts;

    UPDATE public.documents d
       SET status = 'failed', processed_at = clock_timestamp(),
           processing_error = 'Ingestion worker lease expired after maximum attempts'
     WHERE EXISTS (
        SELECT 1 FROM public.document_ingestion_jobs j
        WHERE j.document_id = d.id AND j.user_id = d.user_id
          AND j.status = 'failed'
          AND j.last_error = 'Worker lease expired after maximum attempts'
     ) AND d.status IN ('queued', 'processing');

    RETURN QUERY
    WITH candidates AS (
        SELECT j.id
          FROM public.document_ingestion_jobs j
         WHERE (j.status = 'queued' AND j.available_at <= clock_timestamp())
            OR (j.status = 'processing' AND j.lease_expires_at <= clock_timestamp()
                AND j.attempt_count < j.max_attempts)
         ORDER BY j.available_at, j.created_at
         FOR UPDATE SKIP LOCKED
         LIMIT GREATEST(1, LEAST(p_limit, 10))
    ), claimed AS (
        UPDATE public.document_ingestion_jobs j
           SET status = 'processing', locked_at = clock_timestamp(),
               lease_expires_at = clock_timestamp() + make_interval(secs => lease_seconds),
               attempt_count = j.attempt_count + 1
          FROM candidates c
         WHERE j.id = c.id
        RETURNING j.id, j.document_id, j.user_id, j.attempt_count
    )
    SELECT c.id, c.document_id, c.user_id, c.attempt_count FROM claimed c;
END;
$$;

DROP FUNCTION IF EXISTS public.complete_document_ingestion_job(UUID);
CREATE OR REPLACE FUNCTION public.complete_document_ingestion_job(
    p_job_id UUID,
    p_attempt_count INTEGER
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE updated_rows INTEGER;
BEGIN
    UPDATE public.document_ingestion_jobs
       SET status = 'completed', locked_at = NULL, lease_expires_at = NULL,
           last_error = NULL
     WHERE id = p_job_id AND status = 'processing'
       AND attempt_count = p_attempt_count
       AND lease_expires_at > clock_timestamp();
    GET DIAGNOSTICS updated_rows = ROW_COUNT;
    RETURN updated_rows = 1;
END;
$$;

DROP FUNCTION IF EXISTS public.fail_document_ingestion_job(UUID, TEXT);
CREATE OR REPLACE FUNCTION public.fail_document_ingestion_job(
    p_job_id UUID,
    p_attempt_count INTEGER,
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
     WHERE id = p_job_id AND status = 'processing'
       AND attempt_count = p_attempt_count
       AND lease_expires_at > clock_timestamp()
     FOR UPDATE;
    IF NOT FOUND THEN RETURN 'missing'; END IF;

    IF job_row.attempt_count < job_row.max_attempts THEN
        next_status := 'queued';
        retry_seconds := LEAST(
            900,
            30 * power(2::NUMERIC, LEAST(job_row.attempt_count - 1, 5))::INTEGER
        );
        UPDATE public.document_ingestion_jobs
           SET status = next_status, locked_at = NULL, lease_expires_at = NULL,
               available_at = clock_timestamp() + make_interval(secs => retry_seconds),
               last_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = p_job_id;
        UPDATE public.documents
           SET status = 'queued', processing_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = job_row.document_id AND user_id = job_row.user_id;
    ELSE
        next_status := 'failed';
        UPDATE public.document_ingestion_jobs
           SET status = next_status, locked_at = NULL, lease_expires_at = NULL,
               last_error = LEFT(COALESCE(p_error, 'Ingestion failed'), 500)
         WHERE id = p_job_id;
        UPDATE public.documents
           SET status = 'failed', processed_at = clock_timestamp(),
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
       AND j.lease_expires_at > clock_timestamp()
$$;

CREATE OR REPLACE FUNCTION public.renew_document_ingestion_job_lease(
    p_job_id UUID,
    p_attempt_count INTEGER,
    p_lease_seconds INTEGER DEFAULT 1860
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE updated_rows INTEGER;
BEGIN
    UPDATE public.document_ingestion_jobs
       SET lease_expires_at = clock_timestamp()
             + make_interval(secs => GREATEST(60, LEAST(p_lease_seconds, 3600)))
     WHERE id = p_job_id AND status = 'processing'
       AND attempt_count = p_attempt_count
       AND lease_expires_at > clock_timestamp();
    GET DIAGNOSTICS updated_rows = ROW_COUNT;
    RETURN updated_rows = 1;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_document_ingestion_jobs(INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.complete_document_ingestion_job(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.fail_document_ingestion_job(UUID, INTEGER, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.get_claimed_document_ingestion_job(UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.renew_document_ingestion_job_lease(UUID, INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_document_ingestion_jobs(INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.complete_document_ingestion_job(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.fail_document_ingestion_job(UUID, INTEGER, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.get_claimed_document_ingestion_job(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.renew_document_ingestion_job_lease(UUID, INTEGER, INTEGER) TO service_role;

-- Triggers run for both authenticated and service-role writes. A valid token
-- locks the job row through transaction commit, so lease reclamation serializes
-- against the actual document/chunk/vector write rather than a Python precheck.
CREATE OR REPLACE FUNCTION public.assert_document_ingestion_attempt(p_document_id UUID)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE token_job_id TEXT := NULLIF(current_setting('app.ingestion_job_id', true), '');
DECLARE token_attempt TEXT := NULLIF(current_setting('app.ingestion_attempt', true), '');
DECLARE job_row public.document_ingestion_jobs%ROWTYPE;
BEGIN
    IF token_job_id IS NOT NULL OR token_attempt IS NOT NULL THEN
        IF token_job_id IS NULL OR token_attempt IS NULL THEN
            RAISE EXCEPTION 'Incomplete ingestion attempt token' USING ERRCODE = '40001';
        END IF;
        SELECT * INTO job_row
          FROM public.document_ingestion_jobs j
         WHERE j.id = token_job_id::UUID AND j.document_id = p_document_id
         FOR UPDATE;
        IF NOT FOUND OR job_row.status <> 'processing'
           OR job_row.attempt_count <> token_attempt::INTEGER
           OR job_row.lease_expires_at IS NULL
           OR job_row.lease_expires_at <= clock_timestamp() THEN
            RAISE EXCEPTION 'Stale or expired ingestion attempt' USING ERRCODE = '40001';
        END IF;
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1 FROM public.document_ingestion_jobs j
         WHERE j.document_id = p_document_id AND j.status = 'processing'
    ) THEN
        RAISE EXCEPTION 'Queued ingestion writes require an attempt token' USING ERRCODE = '40001';
    END IF;
END;
$$;

REVOKE ALL ON FUNCTION public.assert_document_ingestion_attempt(UUID) FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.fence_document_ingestion_update()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    IF NEW.status IS DISTINCT FROM OLD.status
       OR NEW.processed_at IS DISTINCT FROM OLD.processed_at
       OR NEW.processing_error IS DISTINCT FROM OLD.processing_error
       OR NEW.processing_started_at IS DISTINCT FROM OLD.processing_started_at
       OR NEW.embedding_status IS DISTINCT FROM OLD.embedding_status
       OR NEW.embedding_started_at IS DISTINCT FROM OLD.embedding_started_at
       OR NEW.embedded_at IS DISTINCT FROM OLD.embedded_at
       OR NEW.embedding_error IS DISTINCT FROM OLD.embedding_error THEN
        PERFORM public.assert_document_ingestion_attempt(NEW.id);
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS fence_document_ingestion_update ON public.documents;
CREATE TRIGGER fence_document_ingestion_update
BEFORE UPDATE ON public.documents
FOR EACH ROW EXECUTE FUNCTION public.fence_document_ingestion_update();

CREATE OR REPLACE FUNCTION public.fence_document_chunk_write()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    -- Parent-document cascade deletes are intentional user deletions.
    IF TG_OP = 'DELETE' AND pg_trigger_depth() > 1 THEN RETURN OLD; END IF;
    PERFORM public.assert_document_ingestion_attempt(
        CASE WHEN TG_OP = 'DELETE' THEN OLD.document_id ELSE NEW.document_id END
    );
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS fence_document_chunk_write ON public.document_chunks;
CREATE TRIGGER fence_document_chunk_write
BEFORE INSERT OR UPDATE OR DELETE ON public.document_chunks
FOR EACH ROW EXECUTE FUNCTION public.fence_document_chunk_write();
