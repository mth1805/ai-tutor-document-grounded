-- Run manually on Supabase Dev after applying the new migration.
-- Read-only catalog assertions: no application rows or schema are changed.
DO $$
DECLARE signature TEXT;
BEGIN
    IF NOT has_table_privilege('authenticated', 'public.document_ingestion_jobs', 'SELECT')
       OR NOT has_table_privilege('authenticated', 'public.document_ingestion_jobs', 'INSERT')
       OR NOT has_table_privilege('authenticated', 'public.document_ingestion_jobs', 'DELETE') THEN
        RAISE EXCEPTION 'Missing authenticated ingestion queue privileges';
    END IF;
    IF has_table_privilege('authenticated', 'public.document_ingestion_jobs', 'UPDATE') THEN
        RAISE EXCEPTION 'Authenticated users must not update worker queue state';
    END IF;
    IF NOT (SELECT relrowsecurity FROM pg_class WHERE oid = 'public.document_ingestion_jobs'::regclass) THEN
        RAISE EXCEPTION 'Ingestion queue RLS must be enabled';
    END IF;
    IF (SELECT count(*) FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'document_ingestion_jobs'
          AND policyname IN ('Users can view their ingestion jobs', 'Users can enqueue their documents', 'Users can remove terminal ingestion jobs')) <> 3 THEN
        RAISE EXCEPTION 'Expected owner-scoped queue policies are missing';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'document_ingestion_jobs'
          AND cmd IN ('UPDATE', 'ALL') AND 'authenticated' = ANY(roles)) THEN
        RAISE EXCEPTION 'Authenticated queue UPDATE policy must not exist';
    END IF;
    FOREACH signature IN ARRAY ARRAY[
        'public.claim_document_ingestion_jobs(integer,integer)',
        'public.get_claimed_document_ingestion_job(uuid)',
        'public.complete_document_ingestion_job(uuid)',
        'public.fail_document_ingestion_job(uuid,text)'
    ] LOOP
        IF has_function_privilege('authenticated', signature, 'EXECUTE')
           OR has_function_privilege('anon', signature, 'EXECUTE')
           OR NOT has_function_privilege('service_role', signature, 'EXECUTE') THEN
            RAISE EXCEPTION 'Queue control RPC access is incorrect: %', signature;
        END IF;
    END LOOP;
    RAISE NOTICE 'PASS: queue table privileges, RLS policies, and service-only RPC access';
END $$;

-- Inspect actual expressions as well: owner checks must still use auth.uid(),
-- INSERT requires queued/attempt_count=0/max_attempts=3, DELETE terminal states.
SELECT policyname, cmd, roles, qual, with_check
FROM pg_policies
WHERE schemaname = 'public' AND tablename = 'document_ingestion_jobs'
ORDER BY policyname;
