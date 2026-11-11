-- One non-sensitive heartbeat. No application tables or auth policies change.
-- Only this setup migration writes; the recurring workflow performs SELECT only.
BEGIN;

CREATE TABLE IF NOT EXISTS public.keepalive_heartbeat (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    project_label TEXT NOT NULL CHECK (project_label = 'ai-tutor-dev')
);

ALTER TABLE public.keepalive_heartbeat ENABLE ROW LEVEL SECURITY;

-- Remove default table grants; expose only this harmless singleton through anon.
REVOKE ALL ON TABLE public.keepalive_heartbeat FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.keepalive_heartbeat TO anon;

DROP POLICY IF EXISTS "Anon can read development heartbeat" ON public.keepalive_heartbeat;
CREATE POLICY "Anon can read development heartbeat"
ON public.keepalive_heartbeat FOR SELECT TO anon
USING (id = 1 AND project_label = 'ai-tutor-dev');

INSERT INTO public.keepalive_heartbeat (id, project_label)
VALUES (1, 'ai-tutor-dev') ON CONFLICT (id) DO NOTHING;

COMMENT ON TABLE public.keepalive_heartbeat IS
    'Non-sensitive singleton for the optional development database activity check; no user data.';

COMMIT;
