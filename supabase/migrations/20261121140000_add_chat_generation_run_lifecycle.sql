-- C1: durable generation ownership, dispatch fencing, private checkpoints,
-- subscriber leases, and transactionally safe assistant finalization.

ALTER TABLE public.conversations
    ADD CONSTRAINT conversations_id_user_id_key UNIQUE (id, user_id);

ALTER TABLE public.messages
    ADD COLUMN generation_id UUID,
    ADD CONSTRAINT messages_id_conversation_role_key UNIQUE (id, conversation_id, role),
    ADD CONSTRAINT messages_id_conversation_generation_key UNIQUE (id, conversation_id, generation_id),
    ADD CONSTRAINT messages_generation_id_key UNIQUE (generation_id),
    ADD CONSTRAINT messages_generation_id_assistant_only_check
        CHECK (generation_id IS NULL OR role = 'assistant');

CREATE TABLE public.chat_generation_runs (
    generation_id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    user_id UUID NOT NULL,
    user_message_id UUID NOT NULL,
    user_message_role VARCHAR(32) NOT NULL DEFAULT 'user'
        CHECK (user_message_role = 'user'),
    request_fingerprint CHAR(64) NOT NULL
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    chat_mode VARCHAR(32) NOT NULL,
    workspace_id UUID,
    document_id UUID,
    active_document_id UUID,
    state VARCHAR(24) NOT NULL DEFAULT 'queued'
        CHECK (state IN (
            'queued', 'dispatching', 'running', 'cancel_requested',
            'completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified'
        )),
    termination_reason VARCHAR(64),
    cancel_reason VARCHAR(48)
        CHECK (cancel_reason IS NULL OR cancel_reason IN (
            'user_requested', 'all_subscribers_disconnected'
        )),
    assistant_message_id UUID UNIQUE,
    dispatch_attempts INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_attempts >= 0),
    dispatch_token UUID,
    dispatch_lease_expires_at TIMESTAMPTZ,
    next_dispatch_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_dispatch_error TEXT,
    modal_call_id TEXT,
    worker_token UUID,
    worker_epoch BIGINT NOT NULL DEFAULT 0 CHECK (worker_epoch >= 0),
    worker_heartbeat_at TIMESTAMPTZ,
    worker_lease_expires_at TIMESTAMPTZ,
    disconnect_grace_started_at TIMESTAMPTZ,
    generation_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT chat_generation_runs_generation_conversation_key
        UNIQUE (generation_id, conversation_id),
    CONSTRAINT chat_generation_runs_conversation_owner_fkey
        FOREIGN KEY (conversation_id, user_id)
        REFERENCES public.conversations(id, user_id) ON DELETE CASCADE,
    CONSTRAINT chat_generation_runs_user_message_fkey
        FOREIGN KEY (user_message_id, conversation_id, user_message_role)
        REFERENCES public.messages(id, conversation_id, role) ON DELETE CASCADE,
    CONSTRAINT chat_generation_runs_assistant_message_fkey
        FOREIGN KEY (assistant_message_id, conversation_id, generation_id)
        REFERENCES public.messages(id, conversation_id, generation_id)
        ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT chat_generation_runs_terminal_shape_check CHECK (
        (state IN ('queued', 'dispatching', 'running', 'cancel_requested')
            AND assistant_message_id IS NULL AND finished_at IS NULL
            AND termination_reason IS NULL)
        OR
        (state IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified')
            AND assistant_message_id IS NOT NULL AND finished_at IS NOT NULL
            AND termination_reason IS NOT NULL)
    )
);

ALTER TABLE public.messages
    ADD CONSTRAINT messages_generation_run_fkey
        FOREIGN KEY (generation_id, conversation_id)
        REFERENCES public.chat_generation_runs(generation_id, conversation_id)
        ON DELETE CASCADE;

CREATE UNIQUE INDEX chat_generation_one_active_per_conversation_idx
    ON public.chat_generation_runs(conversation_id)
    WHERE state IN ('queued', 'dispatching', 'running', 'cancel_requested');
CREATE INDEX chat_generation_dispatch_due_idx
    ON public.chat_generation_runs(next_dispatch_at, created_at)
    WHERE state = 'queued';
CREATE INDEX chat_generation_dispatch_lease_idx
    ON public.chat_generation_runs(dispatch_lease_expires_at)
    WHERE state = 'dispatching';
CREATE INDEX chat_generation_worker_lease_idx
    ON public.chat_generation_runs(worker_lease_expires_at)
    WHERE state IN ('running', 'cancel_requested');
CREATE INDEX chat_generation_user_history_idx
    ON public.chat_generation_runs(user_id, created_at DESC);

CREATE TABLE public.chat_generation_private_checkpoints (
    generation_id UUID PRIMARY KEY
        REFERENCES public.chat_generation_runs(generation_id) ON DELETE CASCADE,
    checkpoint_delta TEXT NOT NULL DEFAULT '',
    checkpoint_seq BIGINT NOT NULL DEFAULT 0 CHECK (checkpoint_seq >= 0),
    partial_content TEXT NOT NULL DEFAULT '',
    safe_partial_content TEXT NOT NULL DEFAULT '',
    worker_epoch BIGINT NOT NULL DEFAULT 0 CHECK (worker_epoch >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.chat_generation_subscribers (
    generation_id UUID NOT NULL
        REFERENCES public.chat_generation_runs(generation_id) ON DELETE CASCADE,
    subscriber_id UUID NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    last_checkpoint_seq BIGINT NOT NULL DEFAULT 0 CHECK (last_checkpoint_seq >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, subscriber_id)
);
CREATE INDEX chat_generation_subscriber_expiry_idx
    ON public.chat_generation_subscribers(generation_id, lease_expires_at);

CREATE OR REPLACE FUNCTION public.set_chat_generation_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql SET search_path = public, pg_temp AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;
CREATE TRIGGER set_chat_generation_runs_updated_at
BEFORE UPDATE ON public.chat_generation_runs
FOR EACH ROW EXECUTE FUNCTION public.set_chat_generation_updated_at();
CREATE TRIGGER set_chat_generation_checkpoints_updated_at
BEFORE UPDATE ON public.chat_generation_private_checkpoints
FOR EACH ROW EXECUTE FUNCTION public.set_chat_generation_updated_at();

ALTER TABLE public.chat_generation_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_generation_private_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_generation_subscribers ENABLE ROW LEVEL SECURITY;

CREATE POLICY chat_generation_runs_owner_select
    ON public.chat_generation_runs FOR SELECT TO authenticated
    USING (user_id = auth.uid());
CREATE POLICY chat_generation_subscriber_owner_select
    ON public.chat_generation_subscribers FOR SELECT TO authenticated
    USING (EXISTS (
        SELECT 1 FROM public.chat_generation_runs r
        WHERE r.generation_id = chat_generation_subscribers.generation_id
          AND r.user_id = auth.uid()
    ));
CREATE POLICY chat_generation_private_checkpoint_service_access
    ON public.chat_generation_private_checkpoints FOR ALL TO service_role
    USING (true) WITH CHECK (true);

REVOKE ALL ON public.chat_generation_runs FROM PUBLIC, anon, authenticated;
REVOKE ALL ON public.chat_generation_private_checkpoints FROM PUBLIC, anon, authenticated;
REVOKE ALL ON public.chat_generation_subscribers FROM PUBLIC, anon, authenticated;
GRANT SELECT ON public.chat_generation_runs TO authenticated;
GRANT SELECT ON public.chat_generation_subscribers TO authenticated;
GRANT ALL ON public.chat_generation_runs, public.chat_generation_private_checkpoints,
    public.chat_generation_subscribers TO service_role;

-- Generation creation is only through its atomic idempotency RPC. Ordinary
-- client inserts cannot attach a message to a generation. Existing Phase 20
-- internal assistant-message paths with generation_id NULL remain valid.
DROP POLICY IF EXISTS "Users can insert messages in their conversations" ON public.messages;
CREATE POLICY "Users can insert messages in their conversations"
ON public.messages FOR INSERT TO authenticated
WITH CHECK (
    generation_id IS NULL AND EXISTS (
        SELECT 1 FROM public.conversations c
        WHERE c.id = conversation_id AND c.user_id = auth.uid()
    )
);

CREATE OR REPLACE FUNCTION public.assert_chat_generation_integrity()
RETURNS TRIGGER LANGUAGE plpgsql SET search_path = public, pg_temp AS $$
DECLARE
    v_generation_id UUID;
    v_state TEXT;
    v_assistant_id UUID;
    v_count BIGINT;
    v_found_message_id UUID;
    v_found_message_status TEXT;
BEGIN
    v_generation_id := COALESCE(NEW.generation_id, OLD.generation_id);
    IF v_generation_id IS NULL THEN
        v_generation_id := COALESCE(NEW.id, OLD.id);
    END IF;
    SELECT r.state, r.assistant_message_id
      INTO v_state, v_assistant_id
      FROM public.chat_generation_runs r WHERE r.generation_id = v_generation_id;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    SELECT count(*) INTO v_count FROM public.messages m
     WHERE m.generation_id = v_generation_id AND m.role = 'assistant';
    SELECT m.id, m.generation_status INTO v_found_message_id, v_found_message_status
      FROM public.messages m WHERE m.generation_id = v_generation_id AND m.role = 'assistant';
    IF v_state IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified') THEN
        IF v_count <> 1 OR v_assistant_id IS NULL
           OR v_found_message_id <> v_assistant_id
           OR v_found_message_status <> v_state THEN
            RAISE EXCEPTION 'terminal generation must have exactly one matching assistant message'
                USING ERRCODE = '23514';
        END IF;
    ELSIF v_count <> 0 OR v_assistant_id IS NOT NULL THEN
        RAISE EXCEPTION 'active generation cannot have an assistant message'
            USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER chat_generation_runs_integrity_ct
AFTER INSERT OR UPDATE ON public.chat_generation_runs
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION public.assert_chat_generation_integrity();
CREATE CONSTRAINT TRIGGER chat_generation_messages_integrity_ct
AFTER INSERT OR UPDATE OR DELETE ON public.messages
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW
EXECUTE FUNCTION public.assert_chat_generation_integrity();

-- Private implementation: callers below provide distinct actor-specific fences.
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
           finished_at = now(),
           worker_lease_expires_at = NULL,
           dispatch_lease_expires_at = NULL,
           updated_at = now()
     WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_private_checkpoints WHERE generation_id = p_generation_id;
    DELETE FROM public.chat_generation_subscribers WHERE generation_id = p_generation_id;
    RETURN v_message_id;
END;
$$;

CREATE OR REPLACE FUNCTION public.create_chat_generation(
    p_generation_id UUID,
    p_conversation_id UUID,
    p_user_id UUID,
    p_request_fingerprint TEXT,
    p_content TEXT,
    p_chat_mode TEXT,
    p_workspace_id UUID DEFAULT NULL,
    p_document_id UUID DEFAULT NULL,
    p_active_document_id UUID DEFAULT NULL
) RETURNS TABLE(generation_id UUID, state TEXT, user_message_id UUID, reused BOOLEAN)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, auth, pg_temp AS $$
DECLARE
    existing public.chat_generation_runs%ROWTYPE;
    v_user_message_id UUID;
BEGIN
    IF p_user_id IS NULL OR auth.uid() IS DISTINCT FROM p_user_id THEN
        RAISE EXCEPTION 'conversation not found or unauthorized' USING ERRCODE = '42501';
    END IF;
    IF p_request_fingerprint IS NULL OR p_request_fingerprint !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'invalid request fingerprint' USING ERRCODE = '22023';
    END IF;

    -- Resolve idempotent retries before taking the conversation lock.
    PERFORM pg_advisory_xact_lock(hashtextextended(p_generation_id::text, 0));
    SELECT * INTO existing FROM public.chat_generation_runs r
     WHERE r.generation_id = p_generation_id FOR UPDATE;
    IF FOUND THEN
        IF existing.user_id IS DISTINCT FROM p_user_id
           OR existing.conversation_id IS DISTINCT FROM p_conversation_id
           OR existing.request_fingerprint IS DISTINCT FROM p_request_fingerprint THEN
            RAISE EXCEPTION 'generation_id was already used for a different request' USING ERRCODE = '23505';
        END IF;
        RETURN QUERY SELECT existing.generation_id, existing.state::text, existing.user_message_id, true;
        RETURN;
    END IF;
    -- Lock the owned conversation before checking active runs. Competing
    -- generation IDs for this conversation then observe a committed winner
    -- and receive a stable domain conflict rather than an index violation.
    PERFORM 1 FROM public.conversations c
     WHERE c.id = p_conversation_id AND c.user_id = p_user_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conversation not found or unauthorized' USING ERRCODE = '42501';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.chat_generation_runs r
         WHERE r.conversation_id = p_conversation_id
           AND r.state IN ('queued', 'dispatching', 'running', 'cancel_requested')
    ) THEN
        RAISE EXCEPTION 'ACTIVE_GENERATION_EXISTS'
            USING ERRCODE = 'PZ001', DETAIL = 'conversation already has an active generation';
    END IF;
    v_user_message_id := gen_random_uuid();
    INSERT INTO public.messages(id, conversation_id, role, content)
    VALUES (v_user_message_id, p_conversation_id, 'user', p_content);
    INSERT INTO public.chat_generation_runs AS run (
        generation_id, conversation_id, user_id, user_message_id,
        request_fingerprint, chat_mode, workspace_id, document_id, active_document_id
    ) VALUES (
        p_generation_id, p_conversation_id, p_user_id, v_user_message_id,
        p_request_fingerprint, p_chat_mode, p_workspace_id, p_document_id, p_active_document_id
    )
    ON CONFLICT (conversation_id)
        WHERE run.state IN ('queued', 'dispatching', 'running', 'cancel_requested')
        DO NOTHING;
    -- The index also covers writers outside this RPC. Raising here rolls back
    -- the user message and leaves violations of other constraints untouched.
    IF NOT FOUND THEN
        RAISE EXCEPTION 'ACTIVE_GENERATION_EXISTS'
            USING ERRCODE = 'PZ001', DETAIL = 'conversation already has an active generation';
    END IF;
    INSERT INTO public.chat_generation_private_checkpoints(generation_id)
    VALUES (p_generation_id);
    RETURN QUERY SELECT p_generation_id, 'queued'::text, v_user_message_id, false;
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_chat_generation_dispatch(
    p_generation_id UUID,
    p_lease_seconds INTEGER DEFAULT 30
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_token UUID;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND OR r.state <> 'queued' OR r.next_dispatch_at > now() THEN
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
     RETURNING dispatch_token INTO v_token;
    RETURN v_token;
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_chat_generation_worker(
    p_generation_id UUID,
    p_lease_seconds INTEGER DEFAULT 20
) RETURNS TABLE(worker_token UUID, worker_epoch BIGINT, state TEXT)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
BEGIN
    RETURN QUERY UPDATE public.chat_generation_runs r
       SET state = 'running', worker_token = gen_random_uuid(), worker_epoch = r.worker_epoch + 1,
           worker_heartbeat_at = now(),
           worker_lease_expires_at = now() + make_interval(secs => GREATEST(10, LEAST(p_lease_seconds, 120))),
           dispatch_lease_expires_at = NULL
     WHERE r.generation_id = p_generation_id AND r.state = 'dispatching'
       AND r.dispatch_lease_expires_at > now()
     RETURNING r.worker_token, r.worker_epoch, r.state::text;
END;
$$;

CREATE OR REPLACE FUNCTION public.record_chat_generation_modal_call(
    p_generation_id UUID, p_dispatch_token UUID, p_modal_call_id TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE n INTEGER;
BEGIN
    UPDATE public.chat_generation_runs SET modal_call_id = LEFT(p_modal_call_id, 255)
     WHERE generation_id = p_generation_id AND dispatch_token = p_dispatch_token
       AND state IN ('dispatching', 'running');
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n = 1;
END;
$$;

CREATE OR REPLACE FUNCTION public.requeue_expired_chat_generation_dispatches(p_limit INTEGER DEFAULT 100)
RETURNS INTEGER
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_count INTEGER := 0; v_delay INTEGER;
BEGIN
    FOR r IN
        SELECT * FROM public.chat_generation_runs
         WHERE state = 'dispatching' AND dispatch_lease_expires_at <= now()
         ORDER BY dispatch_lease_expires_at
         LIMIT GREATEST(1, LEAST(p_limit, 500))
         FOR UPDATE SKIP LOCKED
    LOOP
        IF r.dispatch_attempts >= 5 THEN
            PERFORM public.finalize_chat_generation_dispatcher(
                r.generation_id, 'dispatching', r.dispatch_token, 'exhausted',
                COALESCE(r.last_dispatch_error, 'dispatch lease expired')
            );
        ELSE
            v_delay := LEAST(120, 15 * power(2, GREATEST(0, r.dispatch_attempts - 1))::INTEGER);
            UPDATE public.chat_generation_runs
               SET state = 'queued', dispatch_token = NULL, dispatch_lease_expires_at = NULL,
                   next_dispatch_at = now() + make_interval(secs => v_delay),
                   last_dispatch_error = COALESCE(last_dispatch_error, 'dispatch lease expired')
             WHERE generation_id = r.generation_id AND state = 'dispatching'
               AND dispatch_token = r.dispatch_token;
        END IF;
        v_count := v_count + 1;
    END LOOP;
    RETURN v_count;
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
DECLARE v_current_seq BIGINT;
BEGIN
    PERFORM 1 FROM public.chat_generation_runs r
     WHERE r.generation_id = p_generation_id AND r.worker_token = p_worker_token
       AND r.worker_epoch = p_worker_epoch AND r.state = 'running'
     FOR UPDATE;
    IF NOT FOUND THEN RETURN false; END IF;
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
        updated_at = now()
    WHERE cp.worker_epoch <= EXCLUDED.worker_epoch;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.heartbeat_chat_generation(
    p_generation_id UUID, p_worker_token UUID, p_worker_epoch BIGINT,
    p_lease_seconds INTEGER DEFAULT 20
) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE n INTEGER;
BEGIN
    UPDATE public.chat_generation_runs
       SET worker_heartbeat_at = now(),
           worker_lease_expires_at = now() + make_interval(secs => GREATEST(10, LEAST(p_lease_seconds, 120)))
     WHERE generation_id = p_generation_id AND worker_token = p_worker_token
       AND worker_epoch = p_worker_epoch AND state IN ('running', 'cancel_requested');
    GET DIAGNOSTICS n = ROW_COUNT;
    RETURN n = 1;
END;
$$;

CREATE OR REPLACE FUNCTION public._finalize_generation_as_worker(
    p_generation_id UUID, p_worker_token UUID, p_worker_epoch BIGINT,
    p_terminal_state TEXT, p_termination_reason TEXT, p_content TEXT,
    p_citations JSONB DEFAULT '[]'::jsonb, p_generation_metadata JSONB DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_state TEXT; v_reason TEXT; v_content TEXT; v_citations JSONB;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND OR r.worker_token IS DISTINCT FROM p_worker_token
       OR r.worker_epoch <> p_worker_epoch
       OR r.state NOT IN ('running', 'cancel_requested') THEN
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
        SELECT COALESCE(NULLIF(safe_partial_content, ''), v_content, '') INTO v_content
          FROM public.chat_generation_private_checkpoints WHERE generation_id = p_generation_id;
        v_citations := '[]'::jsonb;
    END IF;
    RETURN public._finalize_chat_generation(
        p_generation_id, v_state, v_reason, v_content,
        v_citations, p_generation_metadata
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.finalize_chat_generation_worker(
    p_generation_id UUID, p_worker_token UUID, p_worker_epoch BIGINT,
    p_terminal_state TEXT, p_termination_reason TEXT, p_content TEXT,
    p_citations JSONB DEFAULT '[]'::jsonb, p_generation_metadata JSONB DEFAULT NULL
) RETURNS UUID
LANGUAGE sql SECURITY DEFINER SET search_path = public, pg_temp AS $$
    SELECT public._finalize_generation_as_worker(
        p_generation_id, p_worker_token, p_worker_epoch, p_terminal_state,
        p_termination_reason, p_content, p_citations, p_generation_metadata
    )
$$;

CREATE OR REPLACE FUNCTION public.finalize_chat_generation_dispatcher(
    p_generation_id UUID,
    p_expected_state TEXT,
    p_dispatch_token UUID DEFAULT NULL,
    p_action TEXT DEFAULT 'cancel',
    p_error TEXT DEFAULT NULL
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_state TEXT; v_reason TEXT; v_content TEXT;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND OR r.state <> p_expected_state THEN RETURN NULL; END IF;
    IF p_action = 'cancel' AND r.state = 'queued' THEN
        v_state := 'interrupted'; v_reason := 'user_cancelled';
        v_content := 'Generation was cancelled before it started.';
    ELSIF p_action = 'disconnect' AND r.state = 'queued' THEN
        v_state := 'interrupted'; v_reason := 'client_disconnected';
        v_content := 'Generation stopped because all viewers disconnected.';
        UPDATE public.chat_generation_runs
           SET cancel_reason = 'all_subscribers_disconnected'
         WHERE generation_id = p_generation_id;
    ELSIF p_action = 'exhausted' AND r.state = 'dispatching'
          AND r.dispatch_token = p_dispatch_token AND r.dispatch_attempts >= 5 THEN
        v_state := 'failed'; v_reason := 'worker_spawn_failed';
        v_content := 'The response could not be started. Please try again.';
    ELSIF p_action = 'exhausted' AND r.state = 'queued'
          AND r.dispatch_token IS NULL AND p_dispatch_token IS NULL
          AND r.dispatch_attempts >= 5 THEN
        v_state := 'failed'; v_reason := 'worker_spawn_failed';
        v_content := 'The response could not be started. Please try again.';
    ELSE
        RETURN NULL;
    END IF;
    UPDATE public.chat_generation_runs SET last_dispatch_error = LEFT(p_error, 500)
     WHERE generation_id = p_generation_id;
    RETURN public._finalize_chat_generation(p_generation_id, v_state, v_reason, v_content, '[]'::jsonb, NULL);
END;
$$;

CREATE OR REPLACE FUNCTION public.fail_chat_generation_dispatch(
    p_generation_id UUID, p_dispatch_token UUID, p_error TEXT, p_max_attempts INTEGER DEFAULT 5
) RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_delay INTEGER;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND OR r.state <> 'dispatching' OR r.dispatch_token IS DISTINCT FROM p_dispatch_token THEN
        RETURN 'stale';
    END IF;
    IF r.dispatch_attempts >= GREATEST(1, LEAST(p_max_attempts, 5)) THEN
        PERFORM public.finalize_chat_generation_dispatcher(
            p_generation_id, 'dispatching', p_dispatch_token, 'exhausted', p_error
        );
        RETURN 'failed';
    END IF;
    v_delay := LEAST(120, 15 * power(2, GREATEST(0, r.dispatch_attempts - 1))::INTEGER);
    UPDATE public.chat_generation_runs
       SET state = 'queued', dispatch_token = NULL, dispatch_lease_expires_at = NULL,
           next_dispatch_at = now() + make_interval(secs => v_delay),
           last_dispatch_error = LEFT(p_error, 500)
     WHERE generation_id = p_generation_id AND state = 'dispatching'
       AND dispatch_token = p_dispatch_token;
    RETURN 'queued';
END;
$$;

CREATE OR REPLACE FUNCTION public.request_chat_generation_cancel(p_generation_id UUID)
RETURNS TABLE(state TEXT, assistant_message_id UUID)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, auth, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_message UUID;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id AND user_id = auth.uid() FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'generation not found' USING ERRCODE = '42501'; END IF;
    IF r.state IN ('completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified') THEN
        RETURN QUERY SELECT r.state::text, r.assistant_message_id; RETURN;
    ELSIF r.state IN ('queued', 'dispatching') THEN
        UPDATE public.chat_generation_runs SET cancel_reason = 'user_requested'
         WHERE generation_id = p_generation_id;
        v_message := public._finalize_chat_generation(
            p_generation_id, 'interrupted', 'user_cancelled',
            'Generation was cancelled before it started.', '[]'::jsonb, NULL
        );
    ELSIF r.state IN ('running', 'cancel_requested') THEN
        UPDATE public.chat_generation_runs AS run
           SET state = 'cancel_requested', cancel_reason = 'user_requested'
         WHERE run.generation_id = p_generation_id AND run.state = 'running';
    END IF;
    RETURN QUERY SELECT run.state::text, run.assistant_message_id
      FROM public.chat_generation_runs AS run WHERE run.generation_id = p_generation_id;
END;
$$;

CREATE OR REPLACE FUNCTION public.recover_stale_chat_generation(
    p_generation_id UUID, p_expected_worker_token UUID, p_expected_worker_epoch BIGINT
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_safe_content TEXT;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND OR r.state NOT IN ('running', 'cancel_requested')
       OR r.worker_token IS DISTINCT FROM p_expected_worker_token
       OR r.worker_epoch <> p_expected_worker_epoch
       OR r.worker_lease_expires_at IS NULL OR r.worker_lease_expires_at > now() THEN
        RETURN NULL;
    END IF;
    UPDATE public.chat_generation_runs SET worker_epoch = worker_epoch + 1, worker_token = NULL
     WHERE generation_id = p_generation_id;
    SELECT COALESCE(NULLIF(safe_partial_content, ''), 'Generation stopped unexpectedly. Please try again.')
      INTO v_safe_content FROM public.chat_generation_private_checkpoints
     WHERE generation_id = p_generation_id;
    RETURN public._finalize_chat_generation(
        p_generation_id, 'interrupted', 'worker_lost', COALESCE(v_safe_content, 'Generation stopped unexpectedly. Please try again.'),
        '[]'::jsonb, NULL
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.upsert_chat_generation_subscriber(
    p_generation_id UUID, p_subscriber_id UUID, p_lease_seconds INTEGER DEFAULT 30
) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, auth, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id AND user_id = auth.uid() FOR UPDATE;
    IF NOT FOUND OR r.state NOT IN ('queued', 'dispatching', 'running', 'cancel_requested') THEN
        RETURN false;
    END IF;
    DELETE FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at <= now();
    INSERT INTO public.chat_generation_subscribers(generation_id, subscriber_id, lease_expires_at)
    VALUES (p_generation_id, p_subscriber_id, now() + make_interval(secs => GREATEST(10, LEAST(p_lease_seconds, 60))))
    ON CONFLICT (generation_id, subscriber_id) DO UPDATE
       SET lease_expires_at = EXCLUDED.lease_expires_at;
    UPDATE public.chat_generation_runs SET disconnect_grace_started_at = NULL
     WHERE generation_id = p_generation_id;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.release_chat_generation_subscriber(
    p_generation_id UUID, p_subscriber_id UUID
) RETURNS BOOLEAN
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, auth, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_count BIGINT;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs
     WHERE generation_id = p_generation_id AND user_id = auth.uid() FOR UPDATE;
    IF NOT FOUND THEN RETURN false; END IF;
    DELETE FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND subscriber_id = p_subscriber_id;
    DELETE FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at <= now();
    SELECT count(*) INTO v_count FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at > now();
    IF v_count = 0 AND r.state IN ('queued', 'dispatching', 'running') THEN
        UPDATE public.chat_generation_runs
           SET disconnect_grace_started_at = COALESCE(disconnect_grace_started_at, now())
         WHERE generation_id = p_generation_id;
    END IF;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.cleanup_chat_generation_subscribers(p_generation_id UUID)
RETURNS INTEGER
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_count BIGINT; v_deleted INTEGER;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN 0; END IF;
    DELETE FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at <= now();
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    SELECT count(*) INTO v_count FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at > now();
    IF v_count = 0 AND r.state IN ('queued', 'dispatching', 'running') THEN
        UPDATE public.chat_generation_runs
           SET disconnect_grace_started_at = COALESCE(disconnect_grace_started_at, now())
         WHERE generation_id = p_generation_id;
    END IF;
    RETURN v_deleted;
END;
$$;

CREATE OR REPLACE FUNCTION public.reconcile_chat_generation_disconnect(p_generation_id UUID)
RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public, pg_temp AS $$
DECLARE r public.chat_generation_runs%ROWTYPE; v_count BIGINT;
BEGIN
    SELECT * INTO r FROM public.chat_generation_runs WHERE generation_id = p_generation_id FOR UPDATE;
    IF NOT FOUND THEN RETURN 'missing'; END IF;
    DELETE FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at <= now();
    SELECT count(*) INTO v_count FROM public.chat_generation_subscribers
     WHERE generation_id = p_generation_id AND lease_expires_at > now();
    IF v_count > 0 THEN
        UPDATE public.chat_generation_runs SET disconnect_grace_started_at = NULL
         WHERE generation_id = p_generation_id;
        RETURN 'subscribed';
    END IF;
    IF r.state NOT IN ('queued', 'dispatching', 'running') THEN RETURN r.state; END IF;
    UPDATE public.chat_generation_runs
       SET disconnect_grace_started_at = COALESCE(disconnect_grace_started_at, now())
     WHERE generation_id = p_generation_id;
    SELECT * INTO r FROM public.chat_generation_runs WHERE generation_id = p_generation_id;
    IF r.disconnect_grace_started_at > now() - interval '15 seconds' THEN RETURN 'grace'; END IF;
    IF r.state IN ('queued', 'dispatching') THEN
        UPDATE public.chat_generation_runs SET cancel_reason = 'all_subscribers_disconnected'
         WHERE generation_id = p_generation_id;
        PERFORM public._finalize_chat_generation(
            p_generation_id, 'interrupted', 'client_disconnected',
            'Generation stopped because all viewers disconnected.', '[]'::jsonb, NULL
        );
        RETURN 'interrupted';
    END IF;
    UPDATE public.chat_generation_runs
       SET state = 'cancel_requested', cancel_reason = 'all_subscribers_disconnected'
     WHERE generation_id = p_generation_id AND state IN ('dispatching', 'running');
    RETURN 'cancel_requested';
END;
$$;

-- Terminal runs retain only safe message/status metadata for history and
-- idempotency. Private checkpoints and subscriber leases are deleted atomically
-- during finalization; active-run data cascades when its user message or
-- conversation is deleted. Expired subscriber leases are reaped by the
-- reconciler. Terminal runs are not time-pruned independently of conversation
-- history.
REVOKE ALL ON FUNCTION public._finalize_chat_generation(UUID, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public._finalize_generation_as_worker(UUID, UUID, BIGINT, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.assert_chat_generation_integrity() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.set_chat_generation_updated_at() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.create_chat_generation(UUID, UUID, UUID, TEXT, TEXT, TEXT, UUID, UUID, UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.request_chat_generation_cancel(UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.upsert_chat_generation_subscriber(UUID, UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.release_chat_generation_subscriber(UUID, UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_chat_generation_dispatch(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_chat_generation_worker(UUID, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.record_chat_generation_modal_call(UUID, UUID, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.requeue_expired_chat_generation_dispatches(INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.checkpoint_chat_generation(UUID, UUID, BIGINT, BIGINT, TEXT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.heartbeat_chat_generation(UUID, UUID, BIGINT, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.finalize_chat_generation_worker(UUID, UUID, BIGINT, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.finalize_chat_generation_dispatcher(UUID, TEXT, UUID, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.fail_chat_generation_dispatch(UUID, UUID, TEXT, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.recover_stale_chat_generation(UUID, UUID, BIGINT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.cleanup_chat_generation_subscribers(UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.reconcile_chat_generation_disconnect(UUID) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.create_chat_generation(UUID, UUID, UUID, TEXT, TEXT, TEXT, UUID, UUID, UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION public.request_chat_generation_cancel(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION public.upsert_chat_generation_subscriber(UUID, UUID, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION public.release_chat_generation_subscriber(UUID, UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION public.claim_chat_generation_dispatch(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.claim_chat_generation_worker(UUID, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.record_chat_generation_modal_call(UUID, UUID, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.requeue_expired_chat_generation_dispatches(INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.checkpoint_chat_generation(UUID, UUID, BIGINT, BIGINT, TEXT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.heartbeat_chat_generation(UUID, UUID, BIGINT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.finalize_chat_generation_worker(UUID, UUID, BIGINT, TEXT, TEXT, TEXT, JSONB, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.finalize_chat_generation_dispatcher(UUID, TEXT, UUID, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.fail_chat_generation_dispatch(UUID, UUID, TEXT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.recover_stale_chat_generation(UUID, UUID, BIGINT) TO service_role;
GRANT EXECUTE ON FUNCTION public.cleanup_chat_generation_subscribers(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.reconcile_chat_generation_disconnect(UUID) TO service_role;

COMMENT ON TABLE public.chat_generation_private_checkpoints IS
    'Backend-only raw/safe generation checkpoints. Never grant SELECT to authenticated or anon.';
COMMENT ON COLUMN public.chat_generation_runs.state IS
    'Single source of truth for active and terminal generation state; terminal states match messages.generation_status.';
