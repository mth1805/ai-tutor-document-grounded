-- C2 Phase 1: immutable, private generation inputs and optimistic history fencing.

CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;

ALTER TABLE public.conversations
    ADD COLUMN conversation_version BIGINT NOT NULL DEFAULT 0,
    ADD CONSTRAINT conversations_context_version_nonnegative
        CHECK (conversation_version >= 0);

CREATE OR REPLACE FUNCTION public.assert_conversation_version_monotonic()
RETURNS TRIGGER LANGUAGE plpgsql SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
    IF NEW.conversation_version < OLD.conversation_version THEN
        RAISE EXCEPTION 'conversation_version cannot decrease' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER assert_conversation_version_monotonic_trg
BEFORE UPDATE OF conversation_version ON public.conversations
FOR EACH ROW EXECUTE FUNCTION public.assert_conversation_version_monotonic();

CREATE OR REPLACE FUNCTION public.bump_conversation_version_for_message()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE
    old_conversation_id UUID;
    new_conversation_id UUID;
BEGIN
    IF TG_OP <> 'INSERT' THEN old_conversation_id := OLD.conversation_id; END IF;
    IF TG_OP <> 'DELETE' THEN new_conversation_id := NEW.conversation_id; END IF;

    IF old_conversation_id IS NOT NULL THEN
        UPDATE public.conversations
           SET conversation_version = conversation_version + 1
         WHERE id = old_conversation_id;
    END IF;
    IF new_conversation_id IS NOT NULL
       AND new_conversation_id IS DISTINCT FROM old_conversation_id THEN
        UPDATE public.conversations
           SET conversation_version = conversation_version + 1
         WHERE id = new_conversation_id;
    ELSIF TG_OP = 'INSERT' THEN
        UPDATE public.conversations
           SET conversation_version = conversation_version + 1
         WHERE id = new_conversation_id;
    END IF;

    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER bump_conversation_version_for_message_trg
AFTER INSERT OR UPDATE OR DELETE ON public.messages
FOR EACH ROW EXECUTE FUNCTION public.bump_conversation_version_for_message();

CREATE TABLE public.chat_generation_context_snapshots (
    generation_id UUID PRIMARY KEY
        REFERENCES public.chat_generation_runs(generation_id) ON DELETE CASCADE,
    snapshot_version SMALLINT NOT NULL CHECK (snapshot_version = 1),
    snapshot_payload JSONB NOT NULL CHECK (jsonb_typeof(snapshot_payload) = 'object'),
    canonical_snapshot TEXT NOT NULL,
    context_hash CHAR(64) NOT NULL CHECK (context_hash ~ '^[0-9a-f]{64}$'),
    canonical_byte_count BIGINT NOT NULL,
    jsonb_rendered_byte_count BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT chat_generation_context_snapshot_shape_check CHECK (
        snapshot_payload ?& ARRAY[
            'original_request', 'normalized_mode', 'workspace_id',
            'selected_document_ids', 'history', 'resolved_document_references',
            'generation_path', 'system_instruction', 'prompt_plan', 'evidence',
            'routing_decisions', 'citation_sources', 'web_results', 'solver_output',
            'whole_document_source_chunks', 'whole_document_summary_plan'
        ]
    ),
    CONSTRAINT chat_generation_context_snapshot_size_check
        CHECK (canonical_byte_count <= 3500000 AND jsonb_rendered_byte_count <= 4194304),
    CONSTRAINT chat_generation_context_snapshot_canonical_json_check
        CHECK (canonical_snapshot::jsonb = snapshot_payload)
);

CREATE OR REPLACE FUNCTION public.set_chat_generation_context_snapshot_byte_counts()
RETURNS TRIGGER LANGUAGE plpgsql SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
    NEW.canonical_byte_count := octet_length(convert_to(NEW.canonical_snapshot, 'UTF8'));
    NEW.jsonb_rendered_byte_count := octet_length(NEW.snapshot_payload::text);
    RETURN NEW;
END;
$$;

CREATE TRIGGER chat_generation_context_snapshot_byte_counts_trg
BEFORE INSERT OR UPDATE ON public.chat_generation_context_snapshots
FOR EACH ROW EXECUTE FUNCTION public.set_chat_generation_context_snapshot_byte_counts();

ALTER TABLE public.chat_generation_context_snapshots ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.chat_generation_context_snapshots FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION public.reject_chat_generation_context_snapshot_update()
RETURNS TRIGGER LANGUAGE plpgsql SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
    RAISE EXCEPTION 'generation context snapshots are immutable' USING ERRCODE = '55000';
END;
$$;

CREATE TRIGGER chat_generation_context_snapshot_immutable_trg
BEFORE UPDATE ON public.chat_generation_context_snapshots
FOR EACH ROW EXECUTE FUNCTION public.reject_chat_generation_context_snapshot_update();

CREATE FUNCTION public.create_chat_generation_with_snapshot(
    p_generation_id UUID,
    p_conversation_id UUID,
    p_user_id UUID,
    p_expected_conversation_version BIGINT,
    p_request_fingerprint TEXT,
    p_content TEXT,
    p_chat_mode TEXT,
    p_workspace_id UUID,
    p_document_id UUID,
    p_active_document_id UUID,
    p_snapshot_version SMALLINT,
    p_snapshot JSONB,
    p_context_hash TEXT,
    p_canonical_snapshot TEXT,
    p_canonical_request TEXT
) RETURNS TABLE(generation_id UUID, state TEXT, user_message_id UUID, reused BOOLEAN)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, auth, pg_temp AS $$
DECLARE
    existing public.chat_generation_runs%ROWTYPE;
    v_user_message_id UUID;
    v_current_version BIGINT;
    v_conversation_workspace_id UUID;
    v_selected_document_ids JSONB;
    v_source JSONB;
    v_canonical_request JSONB;
BEGIN
    IF p_user_id IS NULL OR auth.uid() IS DISTINCT FROM p_user_id THEN
        RAISE EXCEPTION 'conversation not found or unauthorized' USING ERRCODE = '42501';
    END IF;
    IF p_generation_id IS NULL OR p_expected_conversation_version IS NULL
       OR p_request_fingerprint IS NULL OR p_request_fingerprint !~ '^[0-9a-f]{64}$'
       OR p_context_hash IS NULL OR p_context_hash !~ '^[0-9a-f]{64}$'
       OR p_canonical_snapshot IS NULL OR p_canonical_request IS NULL
       OR p_snapshot::jsonb IS DISTINCT FROM p_canonical_snapshot::jsonb
       OR encode(extensions.digest(convert_to(p_canonical_snapshot, 'UTF8'), 'sha256'), 'hex')
            IS DISTINCT FROM p_context_hash
       OR p_snapshot_version IS DISTINCT FROM 1 OR p_snapshot IS NULL
       OR p_content IS NULL OR length(btrim(p_content)) = 0
       OR p_chat_mode IS NULL
       OR p_chat_mode NOT IN ('Light Guidance', 'Detailed Guidance', 'Full Solution')
       OR jsonb_typeof(p_snapshot) <> 'object'
       OR NOT (p_snapshot ?& ARRAY[
            'original_request', 'normalized_mode', 'workspace_id',
            'selected_document_ids', 'history', 'resolved_document_references',
            'generation_path', 'system_instruction', 'prompt_plan', 'evidence',
            'routing_decisions', 'citation_sources', 'web_results', 'solver_output',
            'whole_document_source_chunks', 'whole_document_summary_plan'
       ])
       OR p_snapshot->>'original_request' IS DISTINCT FROM p_content
       OR p_snapshot->>'normalized_mode' IS DISTINCT FROM p_chat_mode
       OR p_snapshot->>'workspace_id' IS DISTINCT FROM p_workspace_id::text
       OR jsonb_typeof(p_snapshot->'selected_document_ids') IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_snapshot->'history') IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_snapshot->'resolved_document_references') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_snapshot->'prompt_plan') IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_snapshot->'evidence') IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_snapshot->'routing_decisions') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_snapshot->'citation_sources') IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_snapshot->'web_results') IS DISTINCT FROM 'array'
       OR jsonb_typeof(p_snapshot->'whole_document_source_chunks') IS DISTINCT FROM 'array'
       OR p_snapshot->>'generation_path' IS NULL
       OR p_snapshot->>'generation_path' NOT IN ('document', 'web_fallback', 'whole_document_summary') THEN
        RAISE EXCEPTION 'invalid generation context snapshot' USING ERRCODE = '22023';
    END IF;
    IF octet_length(p_snapshot::text) > 4194304 THEN
        RAISE EXCEPTION 'generation context snapshot exceeds the 4 MiB limit' USING ERRCODE = '22001';
    END IF;
    SELECT COALESCE(jsonb_agg(to_jsonb(selected.doc_id::text) ORDER BY selected.first_ordinal), '[]'::jsonb)
      INTO v_selected_document_ids
      FROM (
          SELECT doc_id, min(ordinality) AS first_ordinal
            FROM unnest(ARRAY[p_document_id, p_active_document_id]) WITH ORDINALITY AS item(doc_id, ordinality)
           WHERE doc_id IS NOT NULL
           GROUP BY doc_id
      ) AS selected;
    IF p_snapshot->'selected_document_ids' IS DISTINCT FROM v_selected_document_ids THEN
        RAISE EXCEPTION 'snapshot document selection does not match the request' USING ERRCODE = '22023';
    END IF;

    v_canonical_request := jsonb_build_object(
        'content', p_content,
        'mode', p_chat_mode,
        'workspace_id', p_workspace_id::text,
        'document_id', p_document_id::text,
        'active_document_id', p_active_document_id::text
    );
    IF p_canonical_request::jsonb IS DISTINCT FROM v_canonical_request
       OR encode(extensions.digest(convert_to(p_canonical_request, 'UTF8'), 'sha256'), 'hex')
            IS DISTINCT FROM p_request_fingerprint THEN
        RAISE EXCEPTION 'request fingerprint does not match the canonical request' USING ERRCODE = '22023';
    END IF;

    IF p_workspace_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM public.workspaces w
         WHERE w.id = p_workspace_id AND w.user_id = p_user_id
    ) THEN
        RAISE EXCEPTION 'workspace not found or unauthorized' USING ERRCODE = '42501';
    END IF;

    PERFORM pg_advisory_xact_lock(hashtextextended(p_generation_id::text, 0));
    SELECT * INTO existing FROM public.chat_generation_runs r
     WHERE r.generation_id = p_generation_id FOR UPDATE;
    IF FOUND THEN
        IF existing.user_id IS DISTINCT FROM p_user_id
           OR existing.conversation_id IS DISTINCT FROM p_conversation_id
           OR existing.request_fingerprint IS DISTINCT FROM p_request_fingerprint THEN
            RAISE EXCEPTION 'generation_id was already used for a different request' USING ERRCODE = '23505';
        END IF;
        IF existing.state IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified') THEN
            RETURN QUERY SELECT existing.generation_id, existing.state::text,
                                existing.user_message_id, true;
            RETURN;
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM public.chat_generation_context_snapshots s
             WHERE s.generation_id = p_generation_id AND s.context_hash::text = p_context_hash
               AND s.snapshot_version = p_snapshot_version
        ) THEN
            RAISE EXCEPTION 'generation retry has a different or missing context snapshot' USING ERRCODE = '23505';
        END IF;
        RETURN QUERY SELECT existing.generation_id, existing.state::text, existing.user_message_id, true;
        RETURN;
    END IF;

    SELECT c.conversation_version, c.workspace_id
      INTO v_current_version, v_conversation_workspace_id
      FROM public.conversations c
     WHERE c.id = p_conversation_id AND c.user_id = p_user_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conversation not found or unauthorized' USING ERRCODE = '42501';
    END IF;
    IF v_conversation_workspace_id IS DISTINCT FROM p_workspace_id THEN
        RAISE EXCEPTION 'workspace does not match conversation' USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM unnest(ARRAY[p_document_id, p_active_document_id]) AS selected(document_id)
         WHERE selected.document_id IS NOT NULL
           AND NOT EXISTS (
                SELECT 1 FROM public.documents d
                 WHERE d.id = selected.document_id
                   AND d.workspace_id = p_workspace_id AND d.user_id = p_user_id
           )
    ) THEN
        RAISE EXCEPTION 'selected document not found or unauthorized' USING ERRCODE = '42501';
    END IF;

    FOR v_source IN
        SELECT value FROM jsonb_array_elements(p_snapshot->'evidence')
        UNION ALL
        SELECT value FROM jsonb_array_elements(p_snapshot->'whole_document_source_chunks')
    LOOP
        IF v_source->>'document_id' IS NULL OR v_source->>'chunk_id' IS NULL
           OR NOT EXISTS (
                SELECT 1
                  FROM public.documents d
                  JOIN public.document_chunks c ON c.document_id = d.id
                 WHERE d.id = (v_source->>'document_id')::uuid
                   AND d.workspace_id = p_workspace_id AND d.user_id = p_user_id
                   AND c.id = (v_source->>'chunk_id')::uuid
                   AND c.workspace_id = p_workspace_id AND c.user_id = p_user_id
                   AND c.page_number_start = (v_source->>'page_start')::integer
                   AND c.page_number_end = (v_source->>'page_end')::integer
                   AND (v_source->>'token_count' IS NULL
                        OR c.token_count = (v_source->>'token_count')::integer)
                   AND d.original_filename = v_source->>'document_name'
                   AND (c.content = v_source->>'content' OR btrim(c.content) = v_source->>'content')
           ) THEN
            RAISE EXCEPTION 'snapshot evidence is not an authorized stored chunk' USING ERRCODE = '42501';
        END IF;
    END LOOP;

    FOR v_source IN SELECT value FROM jsonb_array_elements(p_snapshot->'citation_sources')
    LOOP
        IF v_source->>'document_id' IS NULL OR v_source->>'chunk_id' IS NULL
           OR v_source->>'document_name' IS NULL OR v_source->>'document_name' = ''
           OR jsonb_typeof(v_source->'source_index') IS DISTINCT FROM 'number'
           OR jsonb_typeof(v_source->'page_start') IS DISTINCT FROM 'number'
           OR jsonb_typeof(v_source->'page_end') IS DISTINCT FROM 'number'
           OR NOT EXISTS (
                SELECT 1
                  FROM jsonb_array_elements(p_snapshot->'evidence') AS evidence(source)
                 WHERE evidence.source->'source_index' IS NOT DISTINCT FROM v_source->'source_index'
                   AND evidence.source->'document_id' IS NOT DISTINCT FROM v_source->'document_id'
                   AND evidence.source->'document_name' IS NOT DISTINCT FROM v_source->'document_name'
                   AND evidence.source->'chunk_id' IS NOT DISTINCT FROM v_source->'chunk_id'
                   AND evidence.source->'page_start' IS NOT DISTINCT FROM v_source->'page_start'
                   AND evidence.source->'page_end' IS NOT DISTINCT FROM v_source->'page_end'
                UNION ALL
                SELECT 1
                  FROM jsonb_array_elements(p_snapshot->'whole_document_source_chunks') AS source(item)
                 WHERE source.item->'source_index' IS NOT DISTINCT FROM v_source->'source_index'
                   AND source.item->'document_id' IS NOT DISTINCT FROM v_source->'document_id'
                   AND source.item->'document_name' IS NOT DISTINCT FROM v_source->'document_name'
                   AND source.item->'chunk_id' IS NOT DISTINCT FROM v_source->'chunk_id'
                   AND source.item->'page_start' IS NOT DISTINCT FROM v_source->'page_start'
                   AND source.item->'page_end' IS NOT DISTINCT FROM v_source->'page_end'
           ) THEN
            RAISE EXCEPTION 'citation source is not present in snapshot evidence' USING ERRCODE = '22023';
        END IF;
    END LOOP;
    IF EXISTS (
        SELECT 1 FROM public.chat_generation_runs r
         WHERE r.conversation_id = p_conversation_id
           AND r.state IN ('queued', 'dispatching', 'running', 'cancel_requested')
    ) THEN
        RAISE EXCEPTION 'ACTIVE_GENERATION_EXISTS'
            USING ERRCODE = 'PZ001', DETAIL = 'conversation already has an active generation';
    END IF;
    IF v_current_version IS DISTINCT FROM p_expected_conversation_version THEN
        RAISE EXCEPTION 'conversation changed while generation context was prepared'
            USING ERRCODE = '40001', DETAIL = 'STALE_CONVERSATION_CONTEXT';
    END IF;

    v_user_message_id := gen_random_uuid();
    INSERT INTO public.messages(id, conversation_id, role, content)
    VALUES (v_user_message_id, p_conversation_id, 'user', p_content);
    INSERT INTO public.chat_generation_runs (
        generation_id, conversation_id, user_id, user_message_id,
        request_fingerprint, chat_mode, workspace_id, document_id, active_document_id
    ) VALUES (
        p_generation_id, p_conversation_id, p_user_id, v_user_message_id,
        p_request_fingerprint, p_chat_mode, p_workspace_id, p_document_id, p_active_document_id
    );
    INSERT INTO public.chat_generation_private_checkpoints(generation_id)
    VALUES (p_generation_id);
    INSERT INTO public.chat_generation_context_snapshots(
        generation_id, snapshot_version, snapshot_payload, canonical_snapshot, context_hash
    ) VALUES (p_generation_id, p_snapshot_version, p_snapshot, p_canonical_snapshot, p_context_hash);

    RETURN QUERY SELECT p_generation_id, 'queued'::text, v_user_message_id, false;
END;
$$;

-- Keep both claim signatures fail-closed/current-dispatch-only, and require an
-- immutable snapshot at the worker boundary as defense in depth.
CREATE OR REPLACE FUNCTION public.claim_chat_generation_worker(
    p_generation_id UUID,
    p_lease_seconds INTEGER DEFAULT 20
) RETURNS TABLE(worker_token UUID, worker_epoch BIGINT, state TEXT)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
    RETURN;
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_chat_generation_worker(
    p_generation_id UUID,
    p_dispatch_token UUID,
    p_lease_seconds INTEGER
) RETURNS TABLE(worker_token UUID, worker_epoch BIGINT, state TEXT)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_now TIMESTAMPTZ;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN; END IF;
    v_now := clock_timestamp();
    IF r.state <> 'dispatching'
       OR r.dispatch_token IS DISTINCT FROM p_dispatch_token OR p_dispatch_token IS NULL
       OR r.dispatch_lease_expires_at IS NULL OR r.dispatch_lease_expires_at <= v_now
       OR NOT EXISTS (SELECT 1 FROM public.chat_generation_context_snapshots s
                       WHERE s.generation_id = p_generation_id) THEN
        RETURN;
    END IF;
    RETURN QUERY UPDATE public.chat_generation_runs AS run
       SET state = 'running', worker_token = gen_random_uuid(),
           worker_epoch = run.worker_epoch + 1, worker_heartbeat_at = v_now,
           worker_lease_expires_at = v_now + make_interval(secs => GREATEST(10, LEAST(p_lease_seconds, 120))),
           dispatch_lease_expires_at = NULL
     WHERE run.generation_id = p_generation_id AND run.state = 'dispatching'
       AND run.dispatch_token = p_dispatch_token AND run.dispatch_lease_expires_at > v_now
       AND EXISTS (SELECT 1 FROM public.chat_generation_context_snapshots s
                   WHERE s.generation_id = p_generation_id)
     RETURNING run.worker_token, run.worker_epoch, run.state::text;
END;
$$;

-- The old creation signature cannot provide a complete context. Keep the symbol
-- for a deliberate fail-closed compatibility response; it cannot enqueue work.
CREATE OR REPLACE FUNCTION public.create_chat_generation(
    p_generation_id UUID, p_conversation_id UUID, p_user_id UUID,
    p_request_fingerprint TEXT, p_content TEXT, p_chat_mode TEXT,
    p_workspace_id UUID DEFAULT NULL, p_document_id UUID DEFAULT NULL,
    p_active_document_id UUID DEFAULT NULL
) RETURNS TABLE(generation_id UUID, state TEXT, user_message_id UUID, reused BOOLEAN)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, auth, pg_temp AS $$
DECLARE existing public.chat_generation_runs%ROWTYPE;
BEGIN
    IF p_user_id IS NULL OR auth.uid() IS DISTINCT FROM p_user_id THEN
        RAISE EXCEPTION 'conversation not found or unauthorized' USING ERRCODE = '42501';
    END IF;
    SELECT * INTO existing FROM public.chat_generation_runs r
     WHERE r.generation_id = p_generation_id;
    IF FOUND AND existing.user_id = p_user_id
       AND existing.conversation_id = p_conversation_id
       AND existing.request_fingerprint = p_request_fingerprint THEN
        RETURN QUERY SELECT existing.generation_id, existing.state::text,
                            existing.user_message_id, true;
        RETURN;
    END IF;
    RAISE EXCEPTION 'a complete generation context snapshot is required'
        USING ERRCODE = '0A000', DETAIL = 'USE_CREATE_CHAT_GENERATION_WITH_SNAPSHOT';
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_chat_generation_dispatch(
    p_generation_id UUID,
    p_lease_seconds INTEGER DEFAULT 30
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_token UUID;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND OR r.state <> 'queued' OR r.next_dispatch_at > now()
       OR NOT EXISTS (
            SELECT 1 FROM public.chat_generation_context_snapshots s
             WHERE s.generation_id = p_generation_id
       ) THEN
        RETURN NULL;
    END IF;
    IF r.dispatch_attempts >= 5 THEN
        PERFORM public.finalize_chat_generation_dispatcher(
            p_generation_id, 'queued', NULL, 'exhausted', r.last_dispatch_error
        );
        RETURN NULL;
    END IF;
    UPDATE public.chat_generation_runs
       SET state = 'dispatching', dispatch_attempts = r.dispatch_attempts + 1,
           dispatch_token = gen_random_uuid(),
           dispatch_lease_expires_at = now() + make_interval(secs => GREATEST(5, LEAST(p_lease_seconds, 120))),
           last_dispatch_error = NULL
     WHERE generation_id = p_generation_id AND state = 'queued'
       AND EXISTS (
            SELECT 1 FROM public.chat_generation_context_snapshots s
             WHERE s.generation_id = p_generation_id
       )
     RETURNING dispatch_token INTO v_token;
    RETURN v_token;
END;
$$;

CREATE OR REPLACE FUNCTION public.read_chat_generation_context_snapshot(
    p_generation_id UUID, p_worker_token UUID, p_worker_epoch BIGINT
) RETURNS TABLE(snapshot_version SMALLINT, snapshot_payload JSONB,
                 canonical_snapshot TEXT, context_hash TEXT,
                 canonical_byte_count BIGINT, jsonb_rendered_byte_count BIGINT)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, pg_temp AS $$
BEGIN
    RETURN QUERY
    SELECT s.snapshot_version, s.snapshot_payload, s.canonical_snapshot,
           s.context_hash::text, s.canonical_byte_count, s.jsonb_rendered_byte_count
      FROM public.chat_generation_runs r
      JOIN public.chat_generation_context_snapshots s USING (generation_id)
     WHERE r.generation_id = p_generation_id
       AND r.state = 'running'
       AND r.worker_token = p_worker_token AND p_worker_token IS NOT NULL
       AND r.worker_epoch IS NOT DISTINCT FROM p_worker_epoch AND p_worker_epoch IS NOT NULL
       AND r.worker_lease_expires_at > clock_timestamp();
END;
$$;

-- The existing C1 finalizer is the common terminal path for worker, dispatcher,
-- cancellation, and recovery. Replace it without changing its message/run effects,
-- and remove the private snapshot in the same transaction.
CREATE OR REPLACE FUNCTION public._finalize_chat_generation(
    p_generation_id UUID,
    p_terminal_state TEXT,
    p_termination_reason TEXT,
    p_content TEXT,
    p_citations JSONB,
    p_generation_metadata JSONB
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_message_id UUID;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN NULL; END IF;
    IF r.state IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified') THEN
        RETURN r.assistant_message_id;
    END IF;
    IF p_terminal_state NOT IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified') THEN
        RAISE EXCEPTION 'invalid terminal generation state' USING ERRCODE = '22023';
    END IF;

    v_message_id := gen_random_uuid();
    INSERT INTO public.messages (
        id, conversation_id, role, content, citations,
        generation_status, generation_metadata, generation_id
    ) VALUES (
        v_message_id, r.conversation_id, 'assistant', COALESCE(p_content, ''),
        COALESCE(p_citations, '[]'::jsonb), p_terminal_state,
        p_generation_metadata, p_generation_id
    );
    UPDATE public.chat_generation_runs
       SET state = p_terminal_state, termination_reason = p_termination_reason,
           assistant_message_id = v_message_id, generation_metadata = p_generation_metadata,
           finished_at = clock_timestamp(), worker_token = NULL,
           worker_epoch = worker_epoch + CASE WHEN worker_token IS NOT NULL THEN 1 ELSE 0 END,
           worker_lease_expires_at = NULL, dispatch_lease_expires_at = NULL,
           updated_at = clock_timestamp()
     WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_private_checkpoints WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_subscribers WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_context_snapshots WHERE generation_id = p_generation_id;
    RETURN v_message_id;
END;
$$;

REVOKE ALL ON FUNCTION public.create_chat_generation_with_snapshot(
    UUID, UUID, UUID, BIGINT, TEXT, TEXT, TEXT, UUID, UUID, UUID, SMALLINT, JSONB, TEXT, TEXT, TEXT
) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.assert_conversation_version_monotonic() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.bump_conversation_version_for_message() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.reject_chat_generation_context_snapshot_update() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.read_chat_generation_context_snapshot(UUID, UUID, BIGINT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.create_chat_generation_with_snapshot(
    UUID, UUID, UUID, BIGINT, TEXT, TEXT, TEXT, UUID, UUID, UUID, SMALLINT, JSONB, TEXT, TEXT, TEXT
) TO authenticated;
GRANT EXECUTE ON FUNCTION public.read_chat_generation_context_snapshot(UUID, UUID, BIGINT)
    TO service_role;
REVOKE ALL ON FUNCTION public.claim_chat_generation_worker(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_chat_generation_worker(UUID, UUID, INTEGER) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_chat_generation_worker(UUID, UUID, INTEGER) TO service_role;
