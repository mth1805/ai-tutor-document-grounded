-- Persist safe completion state for assistant generation outcomes.
ALTER TABLE public.messages
    ADD COLUMN IF NOT EXISTS generation_status VARCHAR(16),
    ADD COLUMN IF NOT EXISTS generation_metadata JSONB;

-- Backfill historical and malformed assistant outcomes without deleting messages.
UPDATE public.messages
   SET generation_status = 'completed'
 WHERE role = 'assistant'
   AND (
       generation_status IS NULL
       OR generation_status NOT IN (
           'completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified'
       )
   );

-- Status belongs only to assistant messages. Clear any legacy invalid values elsewhere.
UPDATE public.messages
   SET generation_status = NULL
 WHERE role <> 'assistant' AND generation_status IS NOT NULL;

ALTER TABLE public.messages
    DROP CONSTRAINT IF EXISTS messages_generation_status_check;
ALTER TABLE public.messages
    ADD CONSTRAINT messages_generation_status_check
    CHECK (
        (
            role = 'assistant'
            AND generation_status IS NOT NULL
            AND generation_status IN (
                'completed', 'incomplete', 'interrupted', 'blocked', 'failed', 'unverified'
            )
        )
        OR (role <> 'assistant' AND generation_status IS NULL)
    );

COMMENT ON COLUMN public.messages.generation_status IS
    'Assistant generation completion outcome; NULL for user/system messages.';
COMMENT ON COLUMN public.messages.generation_metadata IS
    'Safe operational generation metadata; excludes prompts, generated thought content, and provider payloads.';
