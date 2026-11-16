-- Real PostgreSQL integration test for the document-ingestion attempt fence.
-- The test data and helper function are rolled back when the script succeeds.
DO $$
BEGIN
    IF current_database() <> 'fencing_test' OR current_user <> 'fencing_test' THEN
        RAISE EXCEPTION 'Refusing fencing test outside fencing_test/fencing_test';
    END IF;
END;
$$;

BEGIN;

DO $$
DECLARE
    v_user UUID := gen_random_uuid();
    v_workspace UUID := gen_random_uuid();
    v_document UUID := gen_random_uuid();
    v_job UUID := gen_random_uuid();
    v_old_chunk UUID := gen_random_uuid();
    v_claim_a RECORD;
    v_claim_b RECORD;
    v_count INTEGER;
    v_value BOOLEAN;
    v_status TEXT;
    v_helper_oid OID := 'public.assert_document_ingestion_attempt(uuid)'::regprocedure;
    v_helper_owner OID;
    v_update_trigger_owner OID;
    v_chunk_trigger_owner OID;
BEGIN
    -- SECURITY DEFINER helper is internal to the trigger path. Confirm no
    -- directly callable EXECUTE grant remains for PUBLIC or client roles.
    SELECT proowner INTO STRICT v_helper_owner
      FROM pg_proc WHERE oid = v_helper_oid;
    IF EXISTS (
        SELECT 1
          FROM aclexplode(COALESCE(
              (SELECT proacl FROM pg_proc WHERE oid = v_helper_oid),
              acldefault('f', v_helper_owner)
          )) AS privilege
         WHERE privilege.grantee = 0 AND privilege.privilege_type = 'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'PUBLIC can execute assert_document_ingestion_attempt directly';
    END IF;
    IF has_function_privilege('anon', v_helper_oid, 'EXECUTE')
       OR has_function_privilege('authenticated', v_helper_oid, 'EXECUTE') THEN
        RAISE EXCEPTION 'anon or authenticated can execute assert_document_ingestion_attempt directly';
    END IF;

    -- Trigger functions are SECURITY DEFINER and retain the helper owner, so
    -- their internal PERFORM remains executable after client grants are revoked.
    SELECT p.proowner INTO STRICT v_update_trigger_owner
      FROM pg_proc p
     WHERE p.oid = 'public.fence_document_ingestion_update()'::regprocedure
       AND p.prosecdef;
    SELECT p.proowner INTO STRICT v_chunk_trigger_owner
      FROM pg_proc p
     WHERE p.oid = 'public.fence_document_chunk_write()'::regprocedure
       AND p.prosecdef;
    IF v_update_trigger_owner <> v_helper_owner
       OR v_chunk_trigger_owner <> v_helper_owner
       OR NOT has_function_privilege(v_helper_owner, v_helper_oid, 'EXECUTE') THEN
        RAISE EXCEPTION 'Trigger owner cannot execute assert_document_ingestion_attempt';
    END IF;

    INSERT INTO auth.users (id) VALUES (v_user);
    INSERT INTO public.workspaces (id, user_id, name)
    VALUES (v_workspace, v_user, 'fencing integration fixture');
    INSERT INTO public.documents (
        id, workspace_id, user_id, original_filename, storage_path,
        mime_type, file_size, status, embedding_status
    ) VALUES (
        v_document, v_workspace, v_user, 'fencing.txt',
        'fencing-test/' || v_document::TEXT, 'text/plain', 10,
        'processing', 'pending'
    );
    INSERT INTO public.document_chunks (
        id, document_id, workspace_id, user_id, chunk_index, content,
        page_number_start, page_number_end, token_count, embedding_model
    ) VALUES (
        v_old_chunk, v_document, v_workspace, v_user, 0, 'original chunk',
        1, 1, 2, 'original-model'
    );
    INSERT INTO public.document_ingestion_jobs (id, document_id, workspace_id, user_id)
    VALUES (v_job, v_document, v_workspace, v_user);

    -- Worker A claims the queued job at attempt 1.
    SELECT * INTO STRICT v_claim_a
      FROM public.claim_document_ingestion_jobs(1, 60);
    IF v_claim_a.job_id <> v_job OR v_claim_a.claimed_attempt <> 1 THEN
        RAISE EXCEPTION 'Worker A did not claim attempt 1: %', row_to_json(v_claim_a);
    END IF;

    -- Expire A's lease deterministically, then B reclaims attempt 2.
    UPDATE public.document_ingestion_jobs
       SET lease_expires_at = clock_timestamp() - interval '1 second'
     WHERE id = v_job;
    SELECT * INTO STRICT v_claim_b
      FROM public.claim_document_ingestion_jobs(1, 60);
    IF v_claim_b.job_id <> v_job OR v_claim_b.claimed_attempt <> 2 THEN
        RAISE EXCEPTION 'Worker B did not reclaim attempt 2: %', row_to_json(v_claim_b);
    END IF;

    -- A's lease renewal is rejected after B has reclaimed the job.
    SELECT public.renew_document_ingestion_job_lease(v_job, 1, 60) INTO v_value;
    IF v_value IS DISTINCT FROM FALSE THEN
        RAISE EXCEPTION 'Stale A lease renewal unexpectedly succeeded';
    END IF;

    PERFORM set_config('app.ingestion_job_id', v_job::TEXT, true);
    PERFORM set_config('app.ingestion_attempt', '1', true);

    -- A cannot change document processing state.
    BEGIN
        UPDATE public.documents SET status = 'processed' WHERE id = v_document;
        RAISE EXCEPTION 'Stale A document update unexpectedly succeeded';
    EXCEPTION WHEN SQLSTATE '40001' THEN
        NULL;
    END;

    -- A cannot change embedding lifecycle state either.
    BEGIN
        UPDATE public.documents SET embedding_status = 'completed' WHERE id = v_document;
        RAISE EXCEPTION 'Stale A embedding status update unexpectedly succeeded';
    EXCEPTION WHEN SQLSTATE '40001' THEN
        NULL;
    END;

    -- Every stale chunk operation is rejected independently so one error does
    -- not hide whether a later operation is also fenced.
    BEGIN
        DELETE FROM public.document_chunks WHERE document_id = v_document;
        RAISE EXCEPTION 'Stale A chunk delete unexpectedly succeeded';
    EXCEPTION WHEN SQLSTATE '40001' THEN
        NULL;
    END;
    BEGIN
        INSERT INTO public.document_chunks (
            document_id, workspace_id, user_id, chunk_index, content,
            page_number_start, page_number_end, token_count
        ) VALUES (v_document, v_workspace, v_user, 1, 'stale A insert', 2, 2, 3);
        RAISE EXCEPTION 'Stale A chunk insert unexpectedly succeeded';
    EXCEPTION WHEN SQLSTATE '40001' THEN
        NULL;
    END;
    BEGIN
        UPDATE public.document_chunks
           SET embedding = ARRAY[0.1, 0.2]::DOUBLE PRECISION[],
               embedding_model = 'stale-A'
         WHERE id = v_old_chunk;
        RAISE EXCEPTION 'Stale A embedding write unexpectedly succeeded';
    EXCEPTION WHEN SQLSTATE '40001' THEN
        NULL;
    END;

    SELECT public.complete_document_ingestion_job(v_job, 1) INTO v_value;
    IF v_value IS DISTINCT FROM FALSE THEN
        RAISE EXCEPTION 'Stale A completion unexpectedly succeeded';
    END IF;
    SELECT public.fail_document_ingestion_job(v_job, 1, 'stale A failure') INTO v_status;
    IF v_status <> 'missing' THEN
        RAISE EXCEPTION 'Stale A failure unexpectedly changed job state: %', v_status;
    END IF;

    SELECT count(*) INTO v_count
      FROM public.document_chunks
     WHERE document_id = v_document AND id = v_old_chunk AND content = 'original chunk';
    IF v_count <> 1 THEN
        RAISE EXCEPTION 'Stale A changed the existing chunk';
    END IF;
    SELECT count(*) INTO v_count
      FROM public.document_chunks
     WHERE document_id = v_document AND content = 'stale A insert';
    IF v_count <> 0 THEN
        RAISE EXCEPTION 'Stale A inserted a chunk';
    END IF;
    IF (SELECT status FROM public.documents WHERE id = v_document) <> 'processing'
       OR (SELECT embedding_status FROM public.documents WHERE id = v_document) <> 'pending' THEN
        RAISE EXCEPTION 'Stale A changed document state';
    END IF;

    -- Force an error after B deletes the old chunk and inserts one replacement.
    -- The PL/pgSQL exception block is a savepoint: rollback must restore the old
    -- chunk and remove the partially inserted replacement.
    PERFORM set_config('app.ingestion_attempt', '2', true);
    BEGIN
        DELETE FROM public.document_chunks WHERE document_id = v_document;
        INSERT INTO public.document_chunks (
            document_id, workspace_id, user_id, chunk_index, content,
            page_number_start, page_number_end, token_count
        ) VALUES (v_document, v_workspace, v_user, 0, 'partial replacement', 1, 1, 2);
        RAISE EXCEPTION 'forced replacement rollback';
    EXCEPTION WHEN OTHERS THEN
        IF SQLERRM <> 'forced replacement rollback' THEN RAISE; END IF;
    END;
    SELECT count(*) INTO v_count
      FROM public.document_chunks
     WHERE document_id = v_document AND id = v_old_chunk AND content = 'original chunk';
    IF v_count <> 1 THEN
        RAISE EXCEPTION 'Rollback did not restore original chunk';
    END IF;
    SELECT count(*) INTO v_count
      FROM public.document_chunks
     WHERE document_id = v_document AND content = 'partial replacement';
    IF v_count <> 0 THEN
        RAISE EXCEPTION 'Rollback left a partially inserted replacement';
    END IF;

    -- B writes replacement chunks and embeddings with attempt 2, then finalizes.
    UPDATE public.documents
       SET status = 'processed', processed_at = clock_timestamp(), processing_error = NULL,
           embedding_status = 'processing', embedding_started_at = clock_timestamp()
     WHERE id = v_document;
    DELETE FROM public.document_chunks WHERE document_id = v_document;
    INSERT INTO public.document_chunks (
        document_id, workspace_id, user_id, chunk_index, content,
        page_number_start, page_number_end, token_count
    ) VALUES
        (v_document, v_workspace, v_user, 0, 'worker B chunk 0', 1, 1, 3),
        (v_document, v_workspace, v_user, 1, 'worker B chunk 1', 2, 2, 3);
    UPDATE public.document_chunks
       SET embedding = ARRAY[0.3, 0.4]::DOUBLE PRECISION[],
           embedding_model = 'worker-B', embedding_version = 'integration-test',
           embedded_at = clock_timestamp()
     WHERE document_id = v_document;
    UPDATE public.documents
       SET embedding_status = 'completed', embedded_at = clock_timestamp(),
           embedding_error = NULL
     WHERE id = v_document;

    SELECT public.complete_document_ingestion_job(v_job, 2) INTO v_value;
    IF v_value IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'Worker B completion failed';
    END IF;

    IF (SELECT status FROM public.document_ingestion_jobs WHERE id = v_job) <> 'completed'
       OR (SELECT status FROM public.documents WHERE id = v_document) <> 'processed'
       OR (SELECT embedding_status FROM public.documents WHERE id = v_document) <> 'completed' THEN
        RAISE EXCEPTION 'Worker B did not persist final document/job states';
    END IF;
    SELECT count(*) INTO v_count
      FROM public.document_chunks
     WHERE document_id = v_document
       AND content LIKE 'worker B chunk %'
       AND embedding_model = 'worker-B'
       AND embedded_at IS NOT NULL
       AND cardinality(embedding) = 2;
    IF v_count <> 2 THEN
        RAISE EXCEPTION 'Worker B chunk/embedding writes are incomplete: %', v_count;
    END IF;
END;
$$;

ROLLBACK;
