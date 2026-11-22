-- C1 follow-up: serialize worker authority checks with recovery, validate the
-- dispatch attempt before claiming, and retain identity only for successful
-- worker finalization retries.

CREATE OR REPLACE FUNCTION public.claim_chat_generation_worker(
    p_generation_id UUID,
    p_lease_seconds INTEGER DEFAULT 20
) RETURNS TABLE(worker_token UUID, worker_epoch BIGINT, state TEXT)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
BEGIN
    -- The legacy signature cannot prove which dispatch invocation is current.
    -- Keep it for compatibility, but fail closed.
    RETURN;
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_chat_generation_worker(
    p_generation_id UUID,
    p_dispatch_token UUID,
    p_lease_seconds INTEGER
) RETURNS TABLE(worker_token UUID, worker_epoch BIGINT, state TEXT)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    r public.chat_generation_runs%ROWTYPE;
    v_now TIMESTAMPTZ;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN; END IF;

    -- Evaluate time after acquiring the row lock. A waiting claim must not use
    -- an expiry timestamp captured before a competing requeue or reclaim.
    v_now := clock_timestamp();
    IF r.state <> 'dispatching'
       OR r.dispatch_token IS DISTINCT FROM p_dispatch_token
       OR p_dispatch_token IS NULL
       OR r.dispatch_lease_expires_at IS NULL
       OR r.dispatch_lease_expires_at <= v_now THEN
        RETURN;
    END IF;

    RETURN QUERY UPDATE public.chat_generation_runs AS run
       SET state = 'running', worker_token = gen_random_uuid(),
           worker_epoch = run.worker_epoch + 1,
           worker_heartbeat_at = v_now,
           worker_lease_expires_at = v_now + make_interval(secs => GREATEST(10, LEAST(p_lease_seconds, 120))),
           dispatch_lease_expires_at = NULL
     WHERE run.generation_id = p_generation_id AND run.state = 'dispatching'
       AND run.dispatch_token = p_dispatch_token
       AND run.dispatch_lease_expires_at > v_now
     RETURNING run.worker_token, run.worker_epoch, run.state::text;
END;
$$;

CREATE OR REPLACE FUNCTION public.checkpoint_chat_generation(
    p_generation_id UUID,
    p_worker_token UUID,
    p_worker_epoch BIGINT,
    p_checkpoint_seq BIGINT,
    p_checkpoint_delta TEXT,
    p_partial_content TEXT,
    p_safe_partial_content TEXT DEFAULT ''
) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    r public.chat_generation_runs%ROWTYPE;
    v_current_seq BIGINT;
    v_now TIMESTAMPTZ;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN false; END IF;
    v_now := clock_timestamp();
    IF r.worker_token IS DISTINCT FROM p_worker_token
       OR p_worker_token IS NULL
       OR r.worker_epoch IS DISTINCT FROM p_worker_epoch
       OR r.state <> 'running'
       OR r.worker_lease_expires_at IS NULL
       OR r.worker_lease_expires_at <= v_now THEN
        RETURN false;
    END IF;

    SELECT checkpoint_seq INTO v_current_seq
      FROM public.chat_generation_private_checkpoints
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF p_checkpoint_seq <> COALESCE(v_current_seq, 0) + 1 THEN RETURN false; END IF;
    INSERT INTO public.chat_generation_private_checkpoints AS cp(
        generation_id, checkpoint_delta, checkpoint_seq, partial_content,
        safe_partial_content, worker_epoch
    ) VALUES (
        p_generation_id, COALESCE(p_checkpoint_delta, ''), p_checkpoint_seq,
        COALESCE(p_partial_content, ''), COALESCE(p_safe_partial_content, ''), p_worker_epoch
    ) ON CONFLICT (generation_id) DO UPDATE SET
        checkpoint_delta = EXCLUDED.checkpoint_delta,
        checkpoint_seq = EXCLUDED.checkpoint_seq,
        partial_content = EXCLUDED.partial_content,
        safe_partial_content = EXCLUDED.safe_partial_content,
        worker_epoch = EXCLUDED.worker_epoch,
        updated_at = clock_timestamp()
    WHERE cp.worker_epoch <= EXCLUDED.worker_epoch;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.heartbeat_chat_generation(
    p_generation_id UUID, p_worker_token UUID, p_worker_epoch BIGINT,
    p_lease_seconds INTEGER DEFAULT 20
) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    r public.chat_generation_runs%ROWTYPE;
    v_now TIMESTAMPTZ;
    n INTEGER;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN false; END IF;
    v_now := clock_timestamp();
    IF r.worker_token IS DISTINCT FROM p_worker_token
       OR p_worker_token IS NULL
       OR r.worker_epoch IS DISTINCT FROM p_worker_epoch
       OR r.state NOT IN ('running', 'cancel_requested')
       OR r.worker_lease_expires_at IS NULL
       OR r.worker_lease_expires_at <= v_now THEN
        RETURN false;
    END IF;

    UPDATE public.chat_generation_runs
       SET worker_heartbeat_at = v_now,
           worker_lease_expires_at = v_now + make_interval(secs => GREATEST(10, LEAST(p_lease_seconds, 120)))
     WHERE generation_id = p_generation_id AND worker_token = p_worker_token
       AND worker_epoch = p_worker_epoch AND state IN ('running', 'cancel_requested')
       AND worker_lease_expires_at > v_now;
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n = 1;
END;
$$;

CREATE OR REPLACE FUNCTION public._finalize_chat_generation(
    p_generation_id UUID,
    p_terminal_state TEXT,
    p_termination_reason TEXT,
    p_content TEXT,
    p_citations JSONB,
    p_generation_metadata JSONB
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    r public.chat_generation_runs%ROWTYPE;
    v_message_id UUID;
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
       SET state = p_terminal_state,
           termination_reason = p_termination_reason,
           assistant_message_id = v_message_id,
           generation_metadata = p_generation_metadata,
           finished_at = clock_timestamp(),
           worker_token = NULL,
           worker_epoch = worker_epoch + CASE WHEN worker_token IS NOT NULL THEN 1 ELSE 0 END,
           worker_lease_expires_at = NULL,
           dispatch_lease_expires_at = NULL,
           updated_at = clock_timestamp()
     WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_private_checkpoints WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_subscribers WHERE generation_id = p_generation_id;
    RETURN v_message_id;
END;
$$;

CREATE OR REPLACE FUNCTION public._finalize_generation_as_worker(
    p_generation_id UUID, p_worker_token UUID, p_worker_epoch BIGINT,
    p_terminal_state TEXT, p_termination_reason TEXT, p_content TEXT,
    p_citations JSONB DEFAULT '[]'::jsonb, p_generation_metadata JSONB DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    r public.chat_generation_runs%ROWTYPE;
    v_state TEXT;
    v_reason TEXT;
    v_content TEXT;
    v_citations JSONB;
    v_now TIMESTAMPTZ;
    v_message_id UUID;
    v_restored INTEGER;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN NULL; END IF;

    -- Terminal retries disclose the ID only to the exact identity that
    -- successfully finalized this run. Dispatcher/recovery paths leave no
    -- retained token, and recovery advances the epoch before terminalizing.
    IF r.state IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified') THEN
        IF p_worker_token IS NOT NULL
           AND r.worker_token = p_worker_token
           AND r.worker_epoch IS NOT DISTINCT FROM p_worker_epoch THEN
            RETURN r.assistant_message_id;
        END IF;
        RETURN NULL;
    END IF;

    v_now := clock_timestamp();
    IF r.worker_token IS DISTINCT FROM p_worker_token
       OR p_worker_token IS NULL
       OR r.worker_epoch IS DISTINCT FROM p_worker_epoch
       OR r.state NOT IN ('running', 'cancel_requested')
       OR r.worker_lease_expires_at IS NULL
       OR r.worker_lease_expires_at <= v_now THEN
        RETURN NULL;
    END IF;

    v_state := p_terminal_state;
    v_reason := p_termination_reason;
    v_content := COALESCE(p_content, '');
    v_citations := COALESCE(p_citations, '[]'::jsonb);
    IF r.state = 'cancel_requested' THEN
        v_state := 'interrupted';
        v_reason := CASE r.cancel_reason
            WHEN 'user_requested' THEN 'user_cancelled'
            ELSE 'client_disconnected'
        END;
        SELECT COALESCE(safe_partial_content, '') INTO v_content
          FROM public.chat_generation_private_checkpoints
         WHERE generation_id = p_generation_id;
        v_content := COALESCE(v_content, '');
        v_citations := '[]'::jsonb;
    END IF;

    v_message_id := public._finalize_chat_generation(
        p_generation_id, v_state, v_reason, v_content,
        v_citations, p_generation_metadata
    );
    IF v_message_id IS NULL THEN RETURN NULL; END IF;

    -- The generic finalizer fences any identity it sees. Restore only the
    -- identity validated above, in this transaction and under the same row lock.
    UPDATE public.chat_generation_runs
       SET worker_token = p_worker_token, worker_epoch = p_worker_epoch
     WHERE generation_id = p_generation_id
       AND assistant_message_id = v_message_id
       AND state = v_state
       AND worker_token IS NULL
       AND worker_epoch = p_worker_epoch + 1;
    GET DIAGNOSTICS v_restored = ROW_COUNT;
    IF v_restored <> 1 THEN
        RAISE EXCEPTION 'worker finalization identity could not be retained'
            USING ERRCODE = '40001';
    END IF;
    RETURN v_message_id;
END;
$$;

CREATE OR REPLACE FUNCTION public.recover_stale_chat_generation(
    p_generation_id UUID, p_expected_worker_token UUID, p_expected_worker_epoch BIGINT
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE
    r public.chat_generation_runs%ROWTYPE;
    v_now TIMESTAMPTZ;
    v_safe_content TEXT;
    v_reason TEXT;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN NULL; END IF;
    v_now := clock_timestamp();
    IF r.state NOT IN ('running', 'cancel_requested')
       OR r.worker_token IS DISTINCT FROM p_expected_worker_token
       OR p_expected_worker_token IS NULL
       OR r.worker_epoch IS DISTINCT FROM p_expected_worker_epoch
       OR r.worker_lease_expires_at IS NULL
       OR r.worker_lease_expires_at > v_now THEN
        RETURN NULL;
    END IF;

    -- Fence before terminalization so a late worker cannot heartbeat, write,
    -- or retrieve the canonical assistant ID.
    UPDATE public.chat_generation_runs
       SET worker_epoch = worker_epoch + 1, worker_token = NULL
     WHERE generation_id = p_generation_id;

    SELECT COALESCE(NULLIF(safe_partial_content, ''),
                    'Generation stopped unexpectedly. Please try again.')
      INTO v_safe_content FROM public.chat_generation_private_checkpoints
     WHERE generation_id = p_generation_id;
    v_safe_content := COALESCE(v_safe_content, 'Generation stopped unexpectedly. Please try again.');
    v_reason := CASE r.cancel_reason
        WHEN 'user_requested' THEN 'user_cancelled'
        WHEN 'all_subscribers_disconnected' THEN 'client_disconnected'
        ELSE 'worker_lost'
    END;
    RETURN public._finalize_chat_generation(
        p_generation_id, 'interrupted', v_reason, v_safe_content,
        '[]'::jsonb, NULL
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.read_chat_generation_safe_checkpoint(
    p_generation_id UUID
) RETURNS TABLE(checkpoint_seq BIGINT, safe_partial_content TEXT)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, pg_temp AS $$
    SELECT cp.checkpoint_seq, cp.safe_partial_content
      FROM public.chat_generation_runs AS run
      JOIN public.chat_generation_private_checkpoints AS cp
        ON cp.generation_id = run.generation_id
     WHERE run.generation_id = p_generation_id
       AND auth.uid() IS NOT NULL
       AND run.user_id = auth.uid()
$$;

-- Keep both claim signatures explicit: the legacy signature is fail-closed;
-- only the token-aware signature can claim a current dispatch.
REVOKE ALL ON FUNCTION public.claim_chat_generation_worker(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_chat_generation_worker(UUID, UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.checkpoint_chat_generation(UUID, UUID, BIGINT, BIGINT, TEXT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.heartbeat_chat_generation(UUID, UUID, BIGINT, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public._finalize_chat_generation(UUID, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public._finalize_generation_as_worker(UUID, UUID, BIGINT, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.finalize_chat_generation_worker(UUID, UUID, BIGINT, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.recover_stale_chat_generation(UUID, UUID, BIGINT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.read_chat_generation_safe_checkpoint(UUID) FROM PUBLIC, anon;
REVOKE ALL ON FUNCTION public.read_chat_generation_safe_checkpoint(UUID) FROM authenticated;

GRANT EXECUTE ON FUNCTION public.claim_chat_generation_worker(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.claim_chat_generation_worker(UUID, UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.checkpoint_chat_generation(UUID, UUID, BIGINT, BIGINT, TEXT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.heartbeat_chat_generation(UUID, UUID, BIGINT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.finalize_chat_generation_worker(UUID, UUID, BIGINT, TEXT, TEXT, TEXT, JSONB, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.recover_stale_chat_generation(UUID, UUID, BIGINT) TO service_role;
GRANT EXECUTE ON FUNCTION public.read_chat_generation_safe_checkpoint(UUID) TO authenticated;
