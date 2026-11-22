"""C1 durable generation schema and RPC integration tests.

The live-database test is opt-in and refuses to run without the same explicit
disposable-target confirmations as the existing RLS integration suite.
"""
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.test_rls_context import _validate_rls_test_target


MIGRATIONS = (
    "20261121140000_add_chat_generation_run_lifecycle.sql",
    "20261122150000_harden_chat_generation_worker_fencing.sql",
)


async def _as_role(connection, role: str, user_id: uuid.UUID | None = None) -> None:
    await connection.exec_driver_sql(f"SET LOCAL ROLE {role}")
    if user_id is not None:
        await connection.execute(
            text("SELECT set_config('request.jwt.claim.sub', :uid, true)"),
            {"uid": str(user_id)},
        )
        await connection.execute(
            text("SELECT set_config('request.jwt.claims', :claims, true)"),
            {"claims": json.dumps({"sub": str(user_id), "role": "authenticated", "aud": "authenticated"})},
        )


async def _create_generation(connection, schema: str, *, generation_id, conversation_id,
                             user_id, fingerprint: str | None = None, content="question"):
    return (await connection.execute(text(f"""
        SELECT * FROM "{schema}".create_chat_generation(
            :generation_id, :conversation_id, :user_id, :fingerprint, :content,
            'detailed', NULL, NULL, NULL
        )
    """), {
        "generation_id": generation_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "fingerprint": fingerprint or uuid.uuid4().hex * 2,
        "content": content,
    })).one()


async def _insert_conversation(connection, schema: str, conversation_id, user_id) -> None:
    await connection.execute(
        text(f'INSERT INTO "{schema}".conversations (id, user_id) VALUES (:id, :uid)'),
        {"id": conversation_id, "uid": user_id},
    )


def _assert_active_generation_conflict(error: DBAPIError) -> None:
    assert error.orig.sqlstate == "PZ001"
    assert error.orig.__cause__.message == "ACTIVE_GENERATION_EXISTS"


async def _wait_for_lock_waiter(engine, application_name: str, timeout: float = 5.0) -> None:
    """Wait until the named test connection is demonstrably blocked on a DB lock."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with engine.connect() as monitor:
            row = (await monitor.execute(text("""
                SELECT wait_event_type, cardinality(pg_blocking_pids(pid))
                  FROM pg_stat_activity WHERE application_name=:name
            """), {"name": application_name})).first()
        if row is not None and row[0] == "Lock" and row[1] > 0:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"database RPC did not block on the generation row lock: {application_name}")


async def _create_claimed_generation(engine, schema: str, owner: uuid.UUID):
    generation_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    async with engine.begin() as connection:
        await _insert_conversation(connection, schema, conversation_id, owner)
    async with engine.begin() as connection:
        await _as_role(connection, "authenticated", owner)
        await _create_generation(connection, schema, generation_id=generation_id,
            conversation_id=conversation_id, user_id=owner)
    async with engine.begin() as connection:
        await _as_role(connection, "service_role")
        dispatch_token = await connection.scalar(text(
            f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
        ), {"id": generation_id})
        claim = (await connection.execute(text(
            f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
        ), {"id": generation_id, "dispatch": dispatch_token})).one()
    return generation_id, claim


async def _start_competing_rpc(engine, application_name: str, sql: str, params: dict,
                               role: str = "service_role", user_id=None):
    started = asyncio.Event()

    async def run():
        async with engine.connect() as connection:
            transaction = await connection.begin()
            try:
                await _as_role(connection, role, user_id)
                await connection.execute(text("SELECT set_config('application_name', :name, true)"),
                                         {"name": application_name})
                started.set()
                row = (await connection.execute(text(sql), params)).first()
                if row is None:
                    result = None
                elif len(row) == 1:
                    result = row[0]
                else:
                    result = tuple(row)
                await transaction.commit()
                return result
            except BaseException:
                await transaction.rollback()
                raise

    task = asyncio.create_task(run())
    await started.wait()
    try:
        await _wait_for_lock_waiter(engine, application_name)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    return task


@pytest.mark.asyncio
async def test_chat_generation_migration_lifecycle_rls_fencing_and_cleanup():
    database_url = os.environ.get("RLS_TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("Set RLS_TEST_DATABASE_URL to run chat generation migration integration")
    _validate_rls_test_target(
        database_url,
        os.environ.get("RLS_TEST_DATABASE_CONFIRMATION", "").strip(),
        os.environ.get("RLS_TEST_PROJECT_CONFIRMATION", "").strip(),
        os.environ.get("RLS_TEST_DATABASE_ACKNOWLEDGE_DISPOSABLE", "").strip(),
    )

    schema = f"codex_chat_generation_{uuid.uuid4().hex}"
    migration_dir = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
    engine = create_async_engine(database_url, pool_size=5, max_overflow=2)
    owner = uuid.uuid4()
    other_owner = uuid.uuid4()
    conversation_id = uuid.uuid4()
    other_conversation_id = uuid.uuid4()

    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            await connection.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{schema}" TO authenticated, service_role')
            await connection.exec_driver_sql(f'''CREATE TABLE "{schema}".conversations (
                id uuid PRIMARY KEY, user_id uuid NOT NULL
            )''')
            await connection.exec_driver_sql(f'''CREATE TABLE "{schema}".messages (
                id uuid PRIMARY KEY, conversation_id uuid NOT NULL
                    REFERENCES "{schema}".conversations(id) ON DELETE CASCADE,
                role varchar(32) NOT NULL CHECK (role IN ('user','assistant','system')),
                content text NOT NULL, citations jsonb DEFAULT '[]'::jsonb,
                generation_status varchar(16), generation_metadata jsonb,
                CHECK ((role = 'assistant' AND generation_status IN
                    ('completed','incomplete','interrupted','blocked','failed','unverified'))
                    OR (role <> 'assistant' AND generation_status IS NULL))
            )''')
            await connection.exec_driver_sql(f'ALTER TABLE "{schema}".conversations ENABLE ROW LEVEL SECURITY')
            await connection.exec_driver_sql(f'ALTER TABLE "{schema}".messages ENABLE ROW LEVEL SECURITY')
            await connection.exec_driver_sql(f'''CREATE POLICY "Users can insert messages in their conversations"
                ON "{schema}".messages FOR INSERT TO authenticated WITH CHECK (
                    EXISTS (SELECT 1 FROM "{schema}".conversations c
                        WHERE c.id = conversation_id AND c.user_id = auth.uid()))''')
            await connection.exec_driver_sql(f'''CREATE POLICY c_select ON "{schema}".conversations
                FOR SELECT TO authenticated USING (user_id = auth.uid())''')
            await connection.exec_driver_sql(f'''CREATE POLICY c_delete ON "{schema}".conversations
                FOR DELETE TO authenticated USING (user_id = auth.uid())''')
            await connection.exec_driver_sql(f'''CREATE POLICY m_select ON "{schema}".messages
                FOR SELECT TO authenticated USING (EXISTS (SELECT 1 FROM "{schema}".conversations c
                    WHERE c.id = conversation_id AND c.user_id = auth.uid()))''')
            await connection.exec_driver_sql(f'''CREATE POLICY m_delete ON "{schema}".messages
                FOR DELETE TO authenticated USING (EXISTS (SELECT 1 FROM "{schema}".conversations c
                    WHERE c.id = conversation_id AND c.user_id = auth.uid()))''')
            await connection.exec_driver_sql(f'GRANT SELECT, DELETE ON "{schema}".conversations TO authenticated')
            await connection.exec_driver_sql(f'GRANT SELECT ON "{schema}".messages TO authenticated')
            await connection.exec_driver_sql(f'GRANT INSERT, DELETE ON "{schema}".messages TO authenticated')
            await connection.exec_driver_sql(f'GRANT SELECT, INSERT ON "{schema}".messages TO service_role')
            await connection.execute(text(f'''INSERT INTO "{schema}".conversations(id,user_id)
                VALUES (:c1,:u1),(:c2,:u2)'''), {"c1": conversation_id, "u1": owner,
                                                 "c2": other_conversation_id, "u2": other_owner})
            import sqlparse
            for migration in MIGRATIONS:
                sql = (migration_dir / migration).read_text(encoding="utf-8").replace(
                    "public.", f'"{schema}".'
                )
                for statement in sqlparse.split(sql):
                    if statement.strip():
                        await connection.exec_driver_sql(statement)

        # Run.state is the sole status source; no shadow terminal_status field.
        async with engine.begin() as connection:
            terminal_status_column = await connection.scalar(text('''SELECT count(*) FROM information_schema.columns
                WHERE table_schema=:schema AND table_name='chat_generation_runs' AND column_name='terminal_status' '''),
                {"schema": schema})
            assert terminal_status_column == 0

        # Same-ID retries return the existing durable run and user message;
        # different payload fingerprints are rejected without a duplicate row.
        generation_id = uuid.uuid4()
        fingerprint = uuid.uuid4().hex * 2
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            first = await _create_generation(connection, schema, generation_id=generation_id,
                conversation_id=conversation_id, user_id=owner, fingerprint=fingerprint)
            again = await _create_generation(connection, schema, generation_id=generation_id,
                conversation_id=conversation_id, user_id=owner, fingerprint=fingerprint)
            assert first.generation_id == again.generation_id == generation_id
            assert first.state == again.state == "queued"
            assert first.user_message_id == again.user_message_id
            assert first.reused is False and again.reused is True
            with pytest.raises(DBAPIError) as reused_id:
                async with connection.begin_nested():
                    await _create_generation(connection, schema, generation_id=generation_id,
                        conversation_id=conversation_id, user_id=owner,
                        fingerprint=uuid.uuid4().hex * 2, content="different request")
            assert reused_id.value.orig.sqlstate == "23505"
            assert reused_id.value.orig.__cause__.message == (
                "generation_id was already used for a different request"
            )
            with pytest.raises(DBAPIError) as active_conflict:
                async with connection.begin_nested():
                    await _create_generation(connection, schema, generation_id=uuid.uuid4(),
                        conversation_id=conversation_id, user_id=owner, content="second request")
            _assert_active_generation_conflict(active_conflict.value)
            existing_counts = (await connection.execute(text(f'''SELECT
                (SELECT count(*) FROM "{schema}".messages WHERE conversation_id=:cid AND role='user'),
                (SELECT count(*) FROM "{schema}".chat_generation_runs WHERE conversation_id=:cid)'''),
                {"cid": conversation_id})).one()
            assert existing_counts == (1, 1)

        # Concurrent same-ID starts serialize on the idempotency key.
        same_id = uuid.uuid4()
        same_fp = uuid.uuid4().hex * 2
        same_id_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, same_id_conversation, owner)

        async def same_id_start():
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                return await _create_generation(connection, schema, generation_id=same_id,
                    conversation_id=same_id_conversation, user_id=owner, fingerprint=same_fp)

        same_results = await asyncio.gather(same_id_start(), same_id_start())
        assert len({row.user_message_id for row in same_results}) == 1
        assert sorted(row.reused for row in same_results) == [False, True]

        # Distinct IDs racing for the same conversation serialize on the
        # conversation row and return a stable domain conflict before inserting
        # a second user message. The partial index remains the final DB guard.
        race_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, race_conversation, owner)

        async def competing_start():
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                await asyncio.wait_for(race_ready.wait(), timeout=10)
                return await _create_generation(connection, schema, generation_id=uuid.uuid4(),
                    conversation_id=race_conversation, user_id=owner)

        race_ready = asyncio.Barrier(2)
        competing_results = await asyncio.gather(
            competing_start(), competing_start(), return_exceptions=True
        )
        conflicts = [result for result in competing_results if isinstance(result, DBAPIError)]
        assert len(conflicts) == 1
        _assert_active_generation_conflict(conflicts[0])
        winner = next(result for result in competing_results if result is not conflicts[0])
        assert winner.state == "queued" and winner.reused is False
        async with engine.begin() as connection:
            counts = (await connection.execute(text(f'''SELECT
                (SELECT count(*) FROM "{schema}".messages WHERE conversation_id=:cid AND role='user'),
                (SELECT count(*) FROM "{schema}".chat_generation_runs WHERE conversation_id=:cid)'''),
                {"cid": race_conversation})).one()
            assert counts == (1, 1)

        # Different conversations lock independently and can be created in
        # parallel by the same owner.
        parallel_conversations = [uuid.uuid4(), uuid.uuid4()]
        async with engine.begin() as connection:
            await connection.execute(text(f'''INSERT INTO "{schema}".conversations(id,user_id)
                VALUES (:c1,:uid),(:c2,:uid)'''),
                {"c1": parallel_conversations[0], "c2": parallel_conversations[1], "uid": owner})

        both_created = asyncio.Barrier(2)

        async def start_in_conversation(conversation):
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                created = await _create_generation(connection, schema, generation_id=uuid.uuid4(),
                    conversation_id=conversation, user_id=owner)
                # Both inserts must finish while both transactions are open.
                await asyncio.wait_for(both_created.wait(), timeout=10)
                return created

        parallel_results = await asyncio.gather(*(start_in_conversation(cid) for cid in parallel_conversations))
        assert len({row.generation_id for row in parallel_results}) == 2

        # Bypass the RPC to prove the partial unique index still rejects an
        # invalid active duplicate as the final database-level invariant.
        async with engine.begin() as connection:
            with pytest.raises(IntegrityError) as duplicate_error:
                async with connection.begin_nested():
                    duplicate_user_message = uuid.uuid4()
                    duplicate_generation = uuid.uuid4()
                    await connection.execute(text(f'''INSERT INTO "{schema}".messages(
                        id,conversation_id,role,content) VALUES (:mid,:cid,'user','index guard probe')'''),
                        {"mid": duplicate_user_message, "cid": race_conversation})
                    await connection.execute(text(f'''INSERT INTO "{schema}".chat_generation_runs(
                        generation_id,conversation_id,user_id,user_message_id,request_fingerprint,chat_mode)
                        VALUES (:gid,:cid,:uid,:mid,:fingerprint,'detailed')'''),
                        {"gid": duplicate_generation, "cid": race_conversation, "uid": owner,
                         "mid": duplicate_user_message, "fingerprint": uuid.uuid4().hex * 2})
            assert duplicate_error.value.orig.sqlstate == "23505"
            assert duplicate_error.value.orig.__cause__.constraint_name == (
                "chat_generation_one_active_per_conversation_idx"
            )

        # Dispatch and worker claim, private checkpoints, terminal finalization.
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id)'
            ), {"id": generation_id})).first() is None
            legacy_state = (await connection.execute(text(f'''SELECT state,worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": generation_id})).one()
            assert legacy_state == ("queued", None, 0)
            dispatch_token = await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id, 30)'
            ), {"id": generation_id})
            assert dispatch_token is not None
            claim = (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id, :dispatch, 20)'
            ), {"id": generation_id, "dispatch": dispatch_token})).one()
            assert claim.state == "running"
            identity_before_null_calls = (await connection.execute(text(f'''SELECT state,worker_token,
                worker_epoch,worker_heartbeat_at,worker_lease_expires_at,assistant_message_id
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": generation_id})).one()
            assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,NULL::uuid,:epoch,1,'null token','null token','null token')'''),
                {"id": generation_id, "epoch": claim.worker_epoch}) is False
            assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,:token,NULL::bigint,1,'null epoch','null epoch','null epoch')'''),
                {"id": generation_id, "token": claim.worker_token}) is False
            assert await connection.scalar(text(f'''SELECT "{schema}".heartbeat_chat_generation(
                :id,NULL::uuid,:epoch,20)'''),
                {"id": generation_id, "epoch": claim.worker_epoch}) is False
            assert await connection.scalar(text(f'''SELECT "{schema}".heartbeat_chat_generation(
                :id,:token,NULL::bigint,20)'''),
                {"id": generation_id, "token": claim.worker_token}) is False
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,NULL::uuid,:epoch,'completed','null token','null token','[]'::jsonb,NULL)'''),
                {"id": generation_id, "epoch": claim.worker_epoch}) is None
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,NULL::bigint,'completed','null epoch','null epoch','[]'::jsonb,NULL)'''),
                {"id": generation_id, "token": claim.worker_token}) is None
            identity_after_null_calls = (await connection.execute(text(f'''SELECT state,worker_token,
                worker_epoch,worker_heartbeat_at,worker_lease_expires_at,assistant_message_id
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": generation_id})).one()
            assert identity_after_null_calls == identity_before_null_calls
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_private_checkpoints
                WHERE generation_id=:id AND checkpoint_seq<>0'''), {"id": generation_id}) == 0
            assert (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": generation_id, "dispatch": dispatch_token})).first() is None
            assert (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,20)'
            ), {"id": generation_id})).first() is None
            checkpointed = await connection.scalar(text(
                f'''SELECT "{schema}".checkpoint_chat_generation(
                    :id,:token,:epoch,1,'safe delta','private raw','approved partial')'''
            ), {"id": generation_id, "token": claim.worker_token, "epoch": claim.worker_epoch})
            assert checkpointed is True
            checkpoint = (await connection.execute(text(f'''SELECT checkpoint_delta,checkpoint_seq,
                partial_content,safe_partial_content,worker_epoch,updated_at
                FROM "{schema}".chat_generation_private_checkpoints WHERE generation_id=:id'''),
                {"id": generation_id})).one()
            assert checkpoint.checkpoint_delta == "safe delta"
            assert checkpoint.checkpoint_seq == 1
            assert checkpoint.partial_content == "private raw"
            assert checkpoint.safe_partial_content == "approved partial"
            assert checkpoint.worker_epoch == claim.worker_epoch and checkpoint.updated_at is not None
            message_id = await connection.scalar(text(
                f'''SELECT "{schema}".finalize_chat_generation_worker(
                    :id,:token,:epoch,'completed','normal_finish','canonical answer',
                    '[]'::jsonb,jsonb_build_object('ok', true))'''
            ), {"id": generation_id, "token": claim.worker_token, "epoch": claim.worker_epoch})
            assert message_id is not None
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','normal_finish','retry body','[]'::jsonb,NULL)'''),
                {"id": generation_id, "token": claim.worker_token,
                 "epoch": claim.worker_epoch}) == message_id
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','normal_finish','wrong token','[]'::jsonb,NULL)'''),
                {"id": generation_id, "token": uuid.uuid4(),
                 "epoch": claim.worker_epoch}) is None
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','normal_finish','old epoch','[]'::jsonb,NULL)'''),
                {"id": generation_id, "token": claim.worker_token,
                 "epoch": claim.worker_epoch - 1}) is None
            retained_identity = (await connection.execute(text(f'''SELECT worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": generation_id})).one()
            assert retained_identity == (claim.worker_token, claim.worker_epoch)
            state, linked_id, count = (await connection.execute(text(f'''
                SELECT r.state, r.assistant_message_id,
                       (SELECT count(*) FROM "{schema}".messages m WHERE m.generation_id=r.generation_id)
                  FROM "{schema}".chat_generation_runs r WHERE r.generation_id=:id
            '''), {"id": generation_id})).one()
            assert (state, linked_id, count) == ("completed", message_id, 1)
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_private_checkpoints
                WHERE generation_id=:id'''), {"id": generation_id}) == 0
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_subscribers
                WHERE generation_id=:id'''), {"id": generation_id}) == 0
            with pytest.raises(DBAPIError):
                async with connection.begin_nested():
                    await connection.execute(text(f'''INSERT INTO "{schema}".messages(
                        id,conversation_id,role,content,generation_status,generation_id)
                        VALUES (:id,:conversation,'assistant','mismatch','completed',:generation)'''),
                        {"id": uuid.uuid4(), "conversation": other_conversation_id, "generation": generation_id})
            with pytest.raises(DBAPIError):
                async with connection.begin_nested():
                    await connection.execute(text(f'''INSERT INTO "{schema}".messages(
                        id,conversation_id,role,content,generation_status,generation_id)
                        VALUES (:id,:conversation,'assistant','duplicate','completed',:generation)'''),
                        {"id": uuid.uuid4(), "conversation": conversation_id, "generation": generation_id})
            stale_checkpoint = await connection.scalar(text(
                f'''SELECT "{schema}".checkpoint_chat_generation(
                    :id,:token,:epoch,2,'late','late overwrite','late safe')'''
            ), {"id": generation_id, "token": claim.worker_token, "epoch": claim.worker_epoch})
            stale_final = await connection.scalar(text(
                f'''SELECT "{schema}".finalize_chat_generation_worker(
                    :id,:token,:epoch,'failed','late_failure','late overwrite','[]'::jsonb,NULL)'''
            ), {"id": generation_id, "token": claim.worker_token, "epoch": claim.worker_epoch})
            assert stale_checkpoint is False and stale_final == message_id

        # Owner RLS hides another user's run; private checkpoint permissions do
        # not expose even rejected raw content, and history contains no checkpoint.
        blocked_id = uuid.uuid4()
        blocked_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, blocked_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            blocked = await _create_generation(connection, schema, generation_id=blocked_id,
                conversation_id=blocked_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_private_checkpoints
                SET partial_content='blocked secret', checkpoint_delta='blocked secret',
                    safe_partial_content='approved safe view', checkpoint_seq=7 WHERE generation_id=:id'''),
                {"id": blocked_id})
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", other_owner)
            visible = await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_runs
                WHERE generation_id=:id'''), {"id": blocked_id})
            assert visible == 0
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            reader_acl = (await connection.execute(text(f'''SELECT p.prosecdef,p.proconfig,
                has_function_privilege('authenticated',CAST(:signature AS text),'EXECUTE'),
                has_function_privilege('anon',CAST(:signature AS text),'EXECUTE'),
                has_function_privilege('service_role',CAST(:signature AS text),'EXECUTE'),
                has_table_privilege('authenticated',CAST(:checkpoint_table AS text),'SELECT')
                FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                WHERE n.nspname=:schema AND p.proname='read_chat_generation_safe_checkpoint' '''),
                {"signature": f'"{schema}".read_chat_generation_safe_checkpoint(uuid)',
                 "checkpoint_table": f'"{schema}".chat_generation_private_checkpoints',
                 "schema": schema})).one()
            assert reader_acl[0] is True
            assert "search_path=pg_catalog, pg_temp" in reader_acl[1]
            assert reader_acl[2:] == (True, False, False, False)
            safe_result = await connection.execute(text(
                f'SELECT * FROM "{schema}".read_chat_generation_safe_checkpoint(:id)'
            ), {"id": blocked_id})
            safe_row = safe_result.one()
            assert safe_row == (7, "approved safe view")
            assert set(safe_result.keys()) == {"checkpoint_seq", "safe_partial_content"}
            with pytest.raises(DBAPIError):
                async with connection.begin_nested():
                    await connection.execute(text(f'SELECT * FROM "{schema}".chat_generation_private_checkpoints'))
            history = (await connection.execute(text(f'''SELECT content FROM "{schema}".messages
                WHERE conversation_id=:conversation_id AND content='blocked secret' '''),
                {"conversation_id": blocked_conversation})).all()
            assert history == []
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", other_owner)
            safe_result = await connection.execute(text(
                f'SELECT * FROM "{schema}".read_chat_generation_safe_checkpoint(:id)'
            ), {"id": blocked_id})
            assert safe_result.all() == []

        # Queued cancellation uses dispatcher finalization authority, not a
        # nonexistent worker token, and prevents dispatch.
        queued_id = uuid.uuid4()
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", other_owner)
            queued = await _create_generation(connection, schema, generation_id=queued_id,
                conversation_id=other_conversation_id, user_id=other_owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", other_owner)
            result = (await connection.execute(text(
                f'SELECT * FROM "{schema}".request_chat_generation_cancel(:id)'
            ), {"id": queued_id})).one()
            assert result.state == "interrupted" and result.assistant_message_id is not None
            queued_identity = (await connection.execute(text(f'''SELECT worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": queued_id})).one()
            assert queued_identity == (None, 0)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','late','late','[]'::jsonb,NULL)'''),
                {"id": queued_id, "token": uuid.uuid4(), "epoch": 1}) is None
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": queued_id}) is None

        # Dispatching cancellation terminalizes before a late worker claim.
        dispatch_cancel_id = uuid.uuid4()
        dispatch_cancel_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, dispatch_cancel_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=dispatch_cancel_id,
                conversation_id=dispatch_cancel_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": dispatch_cancel_id}) is not None
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            result = (await connection.execute(text(
                f'SELECT * FROM "{schema}".request_chat_generation_cancel(:id)'
            ), {"id": dispatch_cancel_id})).one()
            assert result.state == "interrupted" and result.assistant_message_id is not None
            dispatch_cancel_identity = (await connection.execute(text(f'''SELECT worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": dispatch_cancel_id})).one()
            assert dispatch_cancel_identity == (None, 0)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            late_claim = await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,20)'
            ), {"id": dispatch_cancel_id})
            assert late_claim.first() is None

        race_id = uuid.uuid4()
        race_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, race_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=race_id,
                conversation_id=race_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            race_dispatch_token = await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": race_id})

        async def claim_during_cancel():
            async with engine.begin() as connection:
                await _as_role(connection, "service_role")
                claim = (await connection.execute(text(
                    f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
                ), {"id": race_id, "dispatch": race_dispatch_token})).first()
                if claim is not None:
                    assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                        :id,:token,:epoch,1,'unchecked citations','unchecked citations','safe partial')'''),
                        {"id": race_id, "token": claim.worker_token, "epoch": claim.worker_epoch}) is True
                return claim

        async def cancel_during_claim():
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                return (await connection.execute(text(
                    f'SELECT * FROM "{schema}".request_chat_generation_cancel(:id)'
                ), {"id": race_id})).one()

        race_claim, _ = await asyncio.gather(claim_during_cancel(), cancel_during_claim())
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            race_state = await connection.scalar(text(
                f'SELECT state FROM "{schema}".chat_generation_runs WHERE generation_id=:id'
            ), {"id": race_id})
            if race_state == "cancel_requested":
                assert race_claim is not None
                await connection.execute(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                    :id,:token,:epoch,'completed','normal_finish','unchecked answer',
                    '[{{"url":"https://unvalidated.invalid"}}]'::jsonb,NULL)'''),
                    {"id": race_id, "token": race_claim.worker_token, "epoch": race_claim.worker_epoch})
            else:
                assert race_state == "interrupted" and race_claim is None
            race_row = (await connection.execute(text(f'''SELECT r.state,r.termination_reason,m.content,
                m.citations,m.generation_status,(SELECT count(*) FROM "{schema}".messages linked
                  WHERE linked.generation_id=r.generation_id)
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                  ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''), {"id": race_id})).one()
            assert race_row[0:2] == ("interrupted", "user_cancelled")
            assert race_row[2] == ("safe partial" if race_claim is not None else "Generation was cancelled before it started.")
            assert race_row[3:] == ([], "interrupted", 1)

        expired_dispatch_id = uuid.uuid4()
        expired_dispatch_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, expired_dispatch_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=expired_dispatch_id,
                conversation_id=expired_dispatch_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            old_dispatch = await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": expired_dispatch_id})
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET dispatch_lease_expires_at=now()-interval '1 second' WHERE generation_id=:id'''),
                {"id": expired_dispatch_id})
            expired_dispatch_before_claim = (await connection.execute(text(f'''SELECT state,dispatch_token,
                dispatch_lease_expires_at,worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": expired_dispatch_id})).one()
            assert (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": expired_dispatch_id, "dispatch": old_dispatch})).first() is None
            expired_dispatch_after_claim = (await connection.execute(text(f'''SELECT state,dispatch_token,
                dispatch_lease_expires_at,worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": expired_dispatch_id})).one()
            assert expired_dispatch_after_claim == expired_dispatch_before_claim
            assert await connection.scalar(text(f'SELECT "{schema}".requeue_expired_chat_generation_dispatches(10)')) == 1
            dispatch_row = (await connection.execute(text(f'''SELECT state,dispatch_token FROM "{schema}".chat_generation_runs
                WHERE generation_id=:id'''), {"id": expired_dispatch_id})).one()
            assert dispatch_row == ("queued", None)
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET next_dispatch_at=now() WHERE generation_id=:id'''),
                {"id": expired_dispatch_id})
            new_dispatch = await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": expired_dispatch_id})
            assert new_dispatch is not None and new_dispatch != old_dispatch
            assert (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": expired_dispatch_id, "dispatch": old_dispatch})).first() is None
            new_claim = (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": expired_dispatch_id, "dispatch": new_dispatch})).one()
            assert new_claim.state == "running"
            assert (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": expired_dispatch_id, "dispatch": new_dispatch})).first() is None

        expired_exhaustion_id = uuid.uuid4()
        expired_exhaustion_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, expired_exhaustion_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=expired_exhaustion_id,
                conversation_id=expired_exhaustion_conversation, user_id=owner)
        for attempt in range(1, 6):
            async with engine.begin() as connection:
                await _as_role(connection, "service_role")
                await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                    SET next_dispatch_at=now() WHERE generation_id=:id'''), {"id": expired_exhaustion_id})
                dispatch = await connection.scalar(text(
                    f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
                ), {"id": expired_exhaustion_id})
                assert dispatch is not None
                await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                    SET dispatch_lease_expires_at=now()-interval '1 second',
                        last_dispatch_error='worker lease expired'
                    WHERE generation_id=:id'''), {"id": expired_exhaustion_id})
                assert await connection.scalar(text(
                    f'SELECT "{schema}".requeue_expired_chat_generation_dispatches(10)'
                )) == 1
                dispatch_state = await connection.scalar(text(
                    f'SELECT state FROM "{schema}".chat_generation_runs WHERE generation_id=:id'
                ), {"id": expired_exhaustion_id})
                assert dispatch_state == ("failed" if attempt == 5 else "queued")
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": expired_exhaustion_id}) is None
            exhaustion = (await connection.execute(text(f'''SELECT r.state,r.termination_reason,
                m.generation_status,r.worker_token,r.worker_epoch,
                (SELECT count(*) FROM "{schema}".messages linked
                  WHERE linked.generation_id=r.generation_id)
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                  ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''),
                {"id": expired_exhaustion_id})).one()
            assert exhaustion == ("failed", "worker_spawn_failed", "failed", None, 0, 1)

        exhausted_id = uuid.uuid4()
        exhausted_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, exhausted_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=exhausted_id,
                conversation_id=exhausted_conversation, user_id=owner)
        for attempt in range(1, 6):
            async with engine.begin() as connection:
                await _as_role(connection, "service_role")
                await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                    SET next_dispatch_at=now() WHERE generation_id=:id'''), {"id": exhausted_id})
                dispatch = await connection.scalar(text(
                    f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
                ), {"id": exhausted_id})
                assert dispatch is not None
                result = await connection.scalar(text(
                    f'''SELECT "{schema}".fail_chat_generation_dispatch(:id,:token,'modal spawn unavailable',5)'''
                ), {"id": exhausted_id, "token": dispatch})
                assert result == ("failed" if attempt == 5 else "queued")
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            exhausted_row = (await connection.execute(text(f'''SELECT state,worker_token,worker_epoch,
                (SELECT count(*) FROM "{schema}".messages m WHERE m.generation_id=:id)
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": exhausted_id})).one()
            assert exhausted_row == ("failed", None, 0, 1)

        # Recovery fences a stale worker, finalizes only its last safe checkpoint,
        # and rejects all late worker writes.
        stale_id = uuid.uuid4()
        stale_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, stale_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=stale_id,
                conversation_id=stale_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            token = await connection.scalar(text(f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'), {"id": stale_id})
            claim = (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": stale_id, "dispatch": token})).one()
            await connection.execute(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,:token,:epoch,1,'raw rejected','raw rejected','approved partial')'''),
                {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch})
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET worker_lease_expires_at=now()-interval '1 second' WHERE generation_id=:id'''), {"id": stale_id})
            null_epoch_recovery_before = (await connection.execute(text(f'''SELECT r.state,r.worker_token,
                r.worker_epoch,r.assistant_message_id,r.worker_lease_expires_at,r.termination_reason,
                cp.checkpoint_seq,cp.safe_partial_content
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".chat_generation_private_checkpoints cp
                  USING(generation_id) WHERE r.generation_id=:id'''), {"id": stale_id})).one()
            assert await connection.scalar(text(f'''SELECT "{schema}".recover_stale_chat_generation(
                :id,:token,NULL::bigint)'''),
                {"id": stale_id, "token": claim.worker_token}) is None
            assert await connection.scalar(text(f'''SELECT "{schema}".recover_stale_chat_generation(
                :id,NULL::uuid,:epoch)'''),
                {"id": stale_id, "epoch": claim.worker_epoch}) is None
            null_epoch_recovery_after = (await connection.execute(text(f'''SELECT r.state,r.worker_token,
                r.worker_epoch,r.assistant_message_id,r.worker_lease_expires_at,r.termination_reason,
                cp.checkpoint_seq,cp.safe_partial_content
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".chat_generation_private_checkpoints cp
                  USING(generation_id) WHERE r.generation_id=:id'''), {"id": stale_id})).one()
            assert null_epoch_recovery_after == null_epoch_recovery_before
            heartbeat_before = (await connection.execute(text(f'''SELECT worker_heartbeat_at,
                worker_lease_expires_at FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": stale_id})).one()
            assert await connection.scalar(text(f'''SELECT "{schema}".heartbeat_chat_generation(
                :id,:token,:epoch,20)'''),
                {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch}) is False
            assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,:token,:epoch,2,'late delta','late raw','late safe')'''),
                {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch}) is False
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','late','late answer','[]'::jsonb,NULL)'''),
                {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch}) is None
            before_recovery = (await connection.execute(text(f'''SELECT cp.checkpoint_seq,
                cp.partial_content,cp.safe_partial_content,r.state,
                r.worker_heartbeat_at,r.worker_lease_expires_at,
                (SELECT count(*) FROM "{schema}".messages m WHERE m.generation_id=r.generation_id)
                FROM "{schema}".chat_generation_runs r
                JOIN "{schema}".chat_generation_private_checkpoints cp USING(generation_id)
                WHERE r.generation_id=:id'''), {"id": stale_id})).one()
            assert before_recovery == (1, "raw rejected", "approved partial", "running",
                                       heartbeat_before[0], heartbeat_before[1], 0)
            recovered = await connection.scalar(text(f'''SELECT "{schema}".recover_stale_chat_generation(
                :id,:token,:epoch)'''), {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch})
            assert recovered is not None
            assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,:token,:epoch,2,'late','late','late')'''),
                {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch}) is False
            final_row = (await connection.execute(text(f'''SELECT r.state,r.termination_reason,m.content,m.citations,
                r.worker_token,r.worker_epoch,m.id,(SELECT count(*) FROM "{schema}".messages linked
                    WHERE linked.generation_id=r.generation_id)
                FROM "{schema}".chat_generation_runs r
                JOIN "{schema}".messages m ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''),
                {"id": stale_id})).one()
            assert final_row == ("interrupted", "worker_lost", "approved partial", [], None,
                                 claim.worker_epoch + 1, recovered, 1)
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','late retry','late answer','[]'::jsonb,NULL)'''),
                {"id": stale_id, "token": claim.worker_token, "epoch": claim.worker_epoch}) is None

        for cancel_reason, expected_reason in (
            ("user_requested", "user_cancelled"),
            ("all_subscribers_disconnected", "client_disconnected"),
        ):
            recovery_id = uuid.uuid4()
            recovery_conversation = uuid.uuid4()
            async with engine.begin() as connection:
                await _insert_conversation(connection, schema, recovery_conversation, owner)
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                await _create_generation(connection, schema, generation_id=recovery_id,
                    conversation_id=recovery_conversation, user_id=owner)
            async with engine.begin() as connection:
                await _as_role(connection, "service_role")
                recovery_dispatch = await connection.scalar(text(
                    f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
                ), {"id": recovery_id})
                recovery_claim = (await connection.execute(text(
                    f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
                ), {"id": recovery_id, "dispatch": recovery_dispatch})).one()
                assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                    :id,:token,:epoch,1,'raw cancellation','raw cancellation','safe cancellation')'''),
                    {"id": recovery_id, "token": recovery_claim.worker_token,
                     "epoch": recovery_claim.worker_epoch}) is True
                await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs SET
                    state='cancel_requested',cancel_reason=:reason,
                    worker_lease_expires_at=clock_timestamp()-interval '1 second'
                    WHERE generation_id=:id'''), {"id": recovery_id, "reason": cancel_reason})
                recovery_message = await connection.scalar(text(f'''SELECT "{schema}".recover_stale_chat_generation(
                    :id,:token,:epoch)'''),
                    {"id": recovery_id, "token": recovery_claim.worker_token,
                     "epoch": recovery_claim.worker_epoch})
                recovery_result = (await connection.execute(text(f'''SELECT r.state,r.termination_reason,
                    m.content,m.citations,r.worker_token,r.worker_epoch,
                    (SELECT count(*) FROM "{schema}".messages linked WHERE linked.generation_id=r.generation_id)
                    FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                      ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''),
                    {"id": recovery_id})).one()
                assert recovery_result == ("interrupted", expected_reason, "safe cancellation", [],
                                           None, recovery_claim.worker_epoch + 1, 1)
                assert recovery_message is not None
                assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                    :id,:token,:epoch,'completed','late','late','[]'::jsonb,NULL)'''),
                    {"id": recovery_id, "token": recovery_claim.worker_token,
                     "epoch": recovery_claim.worker_epoch}) is None

        # Each worker write and recovery serialize on the generation row. The
        # competing call is verified in pg_stat_activity as lock-blocked before
        # the lock holder commits; this avoids relying on task scheduling.
        for operation in ("heartbeat", "checkpoint", "finalize"):
            generation_id, claim = await _create_claimed_generation(engine, schema, owner)
            async with engine.connect() as winner:
                winner_tx = await winner.begin()
                await _as_role(winner, "service_role")
                if operation == "heartbeat":
                    winner_result = await winner.scalar(text(f'''SELECT "{schema}".heartbeat_chat_generation(
                        :id,:token,:epoch,20)'''),
                        {"id": generation_id, "token": claim.worker_token,
                         "epoch": claim.worker_epoch})
                    assert winner_result is True
                elif operation == "checkpoint":
                    winner_result = await winner.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                        :id,:token,:epoch,1,'race delta','race raw','race safe')'''),
                        {"id": generation_id, "token": claim.worker_token,
                         "epoch": claim.worker_epoch})
                    assert winner_result is True
                else:
                    winner_result = await winner.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                        :id,:token,:epoch,'completed','normal_finish','winner answer','[]'::jsonb,NULL)'''),
                        {"id": generation_id, "token": claim.worker_token,
                         "epoch": claim.worker_epoch})
                    assert winner_result is not None
                blocked_recovery = await _start_competing_rpc(engine,
                    f"cg_{operation}_winner_{uuid.uuid4().hex[:8]}",
                    f'SELECT "{schema}".recover_stale_chat_generation(:id,:token,:epoch)',
                    {"id": generation_id, "token": claim.worker_token,
                     "epoch": claim.worker_epoch})
                await winner_tx.commit()
                assert await blocked_recovery is None
            if operation != "finalize":
                async with engine.begin() as connection:
                    await _as_role(connection, "service_role")
                    assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                        :id,:token,:epoch,'completed','normal_finish','worker answer','[]'::jsonb,NULL)'''),
                        {"id": generation_id, "token": claim.worker_token,
                         "epoch": claim.worker_epoch}) is not None

        for operation in ("heartbeat", "checkpoint", "finalize"):
            generation_id, claim = await _create_claimed_generation(engine, schema, owner)
            async with engine.begin() as connection:
                await _as_role(connection, "service_role")
                if operation != "heartbeat":
                    assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                        :id,:token,:epoch,1,'safe delta','raw','safe before recovery')'''),
                        {"id": generation_id, "token": claim.worker_token,
                         "epoch": claim.worker_epoch}) is True
                await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                    SET worker_lease_expires_at=clock_timestamp()-interval '1 second'
                    WHERE generation_id=:id'''), {"id": generation_id})
            async with engine.connect() as winner:
                winner_tx = await winner.begin()
                await _as_role(winner, "service_role")
                recovered_id = await winner.scalar(text(f'''SELECT "{schema}".recover_stale_chat_generation(
                    :id,:token,:epoch)'''),
                    {"id": generation_id, "token": claim.worker_token,
                     "epoch": claim.worker_epoch})
                assert recovered_id is not None
                if operation == "heartbeat":
                    competing_sql = f'SELECT "{schema}".heartbeat_chat_generation(:id,:token,:epoch,20)'
                elif operation == "checkpoint":
                    competing_sql = f'''SELECT "{schema}".checkpoint_chat_generation(
                        :id,:token,:epoch,2,'late','late','late')'''
                else:
                    competing_sql = f'''SELECT "{schema}".finalize_chat_generation_worker(
                        :id,:token,:epoch,'completed','late','late','[]'::jsonb,NULL)'''
                blocked_worker = await _start_competing_rpc(engine,
                    f"cg_recovery_winner_{operation}_{uuid.uuid4().hex[:8]}",
                    competing_sql,
                    {"id": generation_id, "token": claim.worker_token,
                     "epoch": claim.worker_epoch})
                await winner_tx.commit()
                rejected = await blocked_worker
                if operation == "finalize":
                    assert rejected is None
                else:
                    assert rejected is False
            async with engine.begin() as connection:
                await _as_role(connection, "service_role")
                final_content = (await connection.execute(text(f'''SELECT m.content,m.citations
                    FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                      ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''),
                    {"id": generation_id})).one()
                assert final_content[1] == []
                assert final_content[0] == (
                    "safe before recovery" if operation != "heartbeat"
                    else "Generation stopped unexpectedly. Please try again."
                )

        # Cancellation and worker finalization races use separate transactions
        # and are ordered by an observed row-lock wait in each direction.
        finalize_first_id, finalize_first_claim = await _create_claimed_generation(engine, schema, owner)
        async with engine.connect() as winner:
            winner_tx = await winner.begin()
            await _as_role(winner, "service_role")
            canonical_id = await winner.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','normal_finish','canonical winner','[]'::jsonb,NULL)'''),
                {"id": finalize_first_id, "token": finalize_first_claim.worker_token,
                 "epoch": finalize_first_claim.worker_epoch})
            assert canonical_id is not None
            blocked_cancel = await _start_competing_rpc(engine,
                f"cg_finalize_before_cancel_{uuid.uuid4().hex[:8]}",
                f'SELECT * FROM "{schema}".request_chat_generation_cancel(:id)',
                {"id": finalize_first_id}, role="authenticated", user_id=owner)
            await winner_tx.commit()
            cancellation_result = await blocked_cancel
            assert cancellation_result == ("completed", canonical_id)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            canonical = (await connection.execute(text(f'''SELECT r.state,m.content,m.generation_status,
                m.id,(SELECT count(*) FROM "{schema}".messages linked
                      WHERE linked.generation_id=r.generation_id)
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                  ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''),
                {"id": finalize_first_id})).one()
            assert canonical == ("completed", "canonical winner", "completed", canonical_id, 1)

        cancel_first_id, cancel_first_claim = await _create_claimed_generation(engine, schema, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,:token,:epoch,1,'unsafe raw','unsafe raw','safe cancellation checkpoint')'''),
                {"id": cancel_first_id, "token": cancel_first_claim.worker_token,
                 "epoch": cancel_first_claim.worker_epoch}) is True
        async with engine.connect() as winner:
            winner_tx = await winner.begin()
            await _as_role(winner, "authenticated", owner)
            cancellation_request = (await winner.execute(text(
                f'SELECT * FROM "{schema}".request_chat_generation_cancel(:id)'
            ), {"id": cancel_first_id})).one()
            assert cancellation_request.state == "cancel_requested"
            blocked_finalize = await _start_competing_rpc(engine,
                f"cg_cancel_before_finalize_{uuid.uuid4().hex[:8]}",
                f'''SELECT "{schema}".finalize_chat_generation_worker(
                    :id,:token,:epoch,'completed','normal_finish','unsafe full response','[]'::jsonb,NULL)''',
                {"id": cancel_first_id, "token": cancel_first_claim.worker_token,
                 "epoch": cancel_first_claim.worker_epoch})
            await winner_tx.commit()
            interrupted_id = await blocked_finalize
            assert interrupted_id is not None
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            interrupted = (await connection.execute(text(f'''SELECT r.state,r.termination_reason,
                m.content,m.citations,m.generation_status,m.id,
                (SELECT count(*) FROM "{schema}".messages linked
                  WHERE linked.generation_id=r.generation_id)
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                  ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''),
                {"id": cancel_first_id})).one()
            assert interrupted == ("interrupted", "user_cancelled", "safe cancellation checkpoint",
                                   [], "interrupted", interrupted_id, 1)

        # Dispatcher terminalizers and retry/requeue RPCs cannot terminalize a
        # claimed run. Disconnect may request cancellation, after which only
        # the worker or stale recovery may finish it.
        dispatcher_id, dispatcher_claim = await _create_claimed_generation(engine, schema, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            dispatch_identity = (await connection.execute(text(f'''SELECT dispatch_token
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": dispatcher_id})).one()
            dispatch_token_after_claim = dispatch_identity[0]
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_dispatcher(
                :id,'running',:dispatch,'cancel',NULL)'''),
                {"id": dispatcher_id, "dispatch": dispatch_token_after_claim}) is None
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_dispatcher(
                :id,'running',:dispatch,'exhausted','exhausted')'''),
                {"id": dispatcher_id, "dispatch": dispatch_token_after_claim}) is None
            assert await connection.scalar(text(f'''SELECT "{schema}".fail_chat_generation_dispatch(
                :id,:dispatch,'late dispatch failure',5)'''),
                {"id": dispatcher_id, "dispatch": dispatch_token_after_claim}) == "stale"
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET disconnect_grace_started_at=now()-interval '16 seconds'
                WHERE generation_id=:id'''), {"id": dispatcher_id})
            assert await connection.scalar(text(f'''SELECT "{schema}".reconcile_chat_generation_disconnect(:id)'''),
                {"id": dispatcher_id}) == "cancel_requested"
            assert await connection.scalar(text(f'''SELECT "{schema}".requeue_expired_chat_generation_dispatches(100)''')) == 0
            dispatcher_state = (await connection.execute(text(f'''SELECT state,worker_token,worker_epoch,
                assistant_message_id,cancel_reason FROM "{schema}".chat_generation_runs
                WHERE generation_id=:id'''), {"id": dispatcher_id})).one()
            assert dispatcher_state == ("cancel_requested", dispatcher_claim.worker_token,
                dispatcher_claim.worker_epoch, None, "all_subscribers_disconnected")
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET worker_lease_expires_at=clock_timestamp()-interval '1 second'
                WHERE generation_id=:id'''), {"id": dispatcher_id})
            dispatcher_message_id = await connection.scalar(text(f'''SELECT "{schema}".recover_stale_chat_generation(
                :id,:token,:epoch)'''),
                {"id": dispatcher_id, "token": dispatcher_claim.worker_token,
                 "epoch": dispatcher_claim.worker_epoch})
            assert dispatcher_message_id is not None
            assert await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','late','late','[]'::jsonb,NULL)'''),
                {"id": dispatcher_id, "token": dispatcher_claim.worker_token,
                 "epoch": dispatcher_claim.worker_epoch}) is None

        # Subscriber lease operations serialize on the run row. One release does
        # not start grace while another subscriber remains; final release does.
        sub_generation = uuid.uuid4()
        sub_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, sub_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=sub_generation,
                conversation_id=sub_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": sub_generation}) is not None

        subscriber_ids = [uuid.uuid4(), uuid.uuid4()]
        async def add_subscriber(subscriber_id):
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                return await connection.scalar(text(f'''SELECT "{schema}".upsert_chat_generation_subscriber(
                    :generation_id,:subscriber_id,30)'''),
                    {"generation_id": sub_generation, "subscriber_id": subscriber_id})
        assert await asyncio.gather(*(add_subscriber(sid) for sid in subscriber_ids)) == [True, True]
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await connection.execute(text(f'''SELECT "{schema}".release_chat_generation_subscriber(:gid,:sid)'''),
                                     {"gid": sub_generation, "sid": subscriber_ids[0]})
            grace = await connection.scalar(text(f'''SELECT disconnect_grace_started_at
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''), {"id": sub_generation})
            assert grace is None
            await connection.execute(text(f'''SELECT "{schema}".release_chat_generation_subscriber(:gid,:sid)'''),
                                     {"gid": sub_generation, "sid": subscriber_ids[1]})
            grace = await connection.scalar(text(f'''SELECT disconnect_grace_started_at
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''), {"id": sub_generation})
            assert grace is not None
            await connection.execute(text(f'''SELECT "{schema}".upsert_chat_generation_subscriber(:gid,:sid,30)'''),
                                     {"gid": sub_generation, "sid": uuid.uuid4()})
            assert await connection.scalar(text(f'''SELECT disconnect_grace_started_at
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''), {"id": sub_generation}) is None
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_subscribers
                SET lease_expires_at=now()-interval '1 second' WHERE generation_id=:id'''),
                {"id": sub_generation})
            assert await connection.scalar(text(f'''SELECT "{schema}".reconcile_chat_generation_disconnect(:id)'''),
                                           {"id": sub_generation}) == "grace"
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET disconnect_grace_started_at=now()-interval '16 seconds' WHERE generation_id=:id'''),
                {"id": sub_generation})
            assert await connection.scalar(text(f'''SELECT "{schema}".reconcile_chat_generation_disconnect(:id)'''),
                                           {"id": sub_generation}) == "interrupted"
            disconnect_row = (await connection.execute(text(f'''SELECT state,cancel_reason,termination_reason
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": sub_generation})).one()
            assert disconnect_row == ("interrupted", "all_subscribers_disconnected", "client_disconnected")

        # Explicit Stop wins while a subscriber remains; finalization is still
        # fenced by the worker token and records user cancellation.
        stop_id = uuid.uuid4()
        stop_conversation = uuid.uuid4()
        stop_subscriber = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, stop_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=stop_id,
                conversation_id=stop_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            stop_dispatch = await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": stop_id})
            stop_claim = (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": stop_id, "dispatch": stop_dispatch})).one()
            assert await connection.scalar(text(f'''SELECT "{schema}".checkpoint_chat_generation(
                :id,:token,:epoch,1,'unsafe interim','unsafe interim','approved partial')'''),
                {"id": stop_id, "token": stop_claim.worker_token, "epoch": stop_claim.worker_epoch}) is True
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            assert await connection.scalar(text(f'''SELECT "{schema}".upsert_chat_generation_subscriber(:gid,:sid,30)'''),
                                           {"gid": stop_id, "sid": stop_subscriber}) is True
            stopped = (await connection.execute(text(f'SELECT * FROM "{schema}".request_chat_generation_cancel(:id)'),
                                                {"id": stop_id})).one()
            assert stopped.state == "cancel_requested"
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_subscribers
                WHERE generation_id=:id AND lease_expires_at>now()'''), {"id": stop_id}) == 1
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            stopped_message_id = await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','normal_finish','unvalidated full response',
                '[{{"url":"https://unvalidated.invalid"}}]'::jsonb,NULL)'''),
                {"id": stop_id, "token": stop_claim.worker_token, "epoch": stop_claim.worker_epoch})
            stopped_row = (await connection.execute(text(f'''SELECT r.state,r.termination_reason,m.generation_status,
                m.content,m.citations
                FROM "{schema}".chat_generation_runs r JOIN "{schema}".messages m
                  ON m.id=r.assistant_message_id WHERE r.generation_id=:id'''), {"id": stop_id})).one()
            assert stopped_message_id is not None
            assert stopped_row == ("interrupted", "user_cancelled", "interrupted", "approved partial", [])

        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await connection.execute(text(f'DELETE FROM "{schema}".conversations WHERE id=:id'),
                                     {"id": stop_conversation})
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(f'SELECT count(*) FROM "{schema}".chat_generation_runs WHERE generation_id=:id'),
                                           {"id": stop_id}) == 0
            assert await connection.scalar(text(f'SELECT count(*) FROM "{schema}".messages WHERE generation_id=:id'),
                                           {"id": stop_id}) == 0

        # Individual assistant deletion fails due to the deferred pointer FK;
        # deleting the initiating user message cascades run and assistant.
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError):
                async with connection.begin_nested():
                    await connection.execute(text(f'DELETE FROM "{schema}".messages WHERE id=:id'), {"id": message_id})
                    await connection.exec_driver_sql("SET CONSTRAINTS ALL IMMEDIATE")

        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            user_message_id = await connection.scalar(text(f'SELECT user_message_id FROM "{schema}".chat_generation_runs WHERE generation_id=:id'), {"id": generation_id})
            await connection.execute(text(f'DELETE FROM "{schema}".messages WHERE id=:id'), {"id": user_message_id})
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(f'SELECT count(*) FROM "{schema}".chat_generation_runs WHERE generation_id=:id'), {"id": generation_id}) == 0
            assert await connection.scalar(text(f'SELECT count(*) FROM "{schema}".messages WHERE generation_id=:id'), {"id": generation_id}) == 0

    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            assert await connection.scalar(text("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=:schema)"),
                                           {"schema": schema}) is False
        await engine.dispose()
