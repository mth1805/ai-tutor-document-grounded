"""C1 durable generation schema and RPC integration tests.

The live-database test is opt-in and refuses to run without the same explicit
disposable-target confirmations as the existing RLS integration suite.
"""
import asyncio
import hashlib
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
    "20261122170000_add_immutable_chat_generation_context_snapshots.sql",
)


def _generation_snapshot(content: str, mode: str = "Detailed Guidance") -> dict:
    return {
        "original_request": content,
        "normalized_mode": mode,
        "workspace_id": None,
        "selected_document_ids": [],
        "history": [],
        "resolved_document_references": {},
        "generation_path": "document",
        "system_instruction": "Captured system instruction",
        "prompt_plan": [{"stage": "final", "prompt": "Captured prompt", "max_output_tokens": 8192}],
        "evidence": [],
        "routing_decisions": {},
        "citation_sources": [],
        "web_results": [],
        "solver_output": None,
        "whole_document_source_chunks": [],
        "whole_document_summary_plan": None,
    }


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
                             user_id, content="question"):
    version = await connection.scalar(text(
        f'SELECT conversation_version FROM "{schema}".conversations WHERE id=:id'
    ), {"id": conversation_id})
    snapshot = _generation_snapshot(content)
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    request = {
        "content": content, "mode": "Detailed Guidance", "workspace_id": None,
        "document_id": None, "active_document_id": None,
    }
    canonical_request = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (await connection.execute(text(f"""
        SELECT * FROM "{schema}".create_chat_generation_with_snapshot(
            :generation_id, :conversation_id, :user_id, :expected_version,
            :fingerprint, :content, :mode, NULL, NULL, NULL, 1::smallint,
            CAST(:snapshot AS jsonb), :context_hash, :canonical_snapshot, :canonical_request
        )
    """), {
        "generation_id": generation_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "expected_version": version,
        "fingerprint": hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        "content": content,
        "mode": "Detailed Guidance",
        "snapshot": canonical,
        "canonical_snapshot": canonical,
        "canonical_request": canonical_request,
        "context_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    })).one()


async def _insert_conversation(connection, schema: str, conversation_id, user_id, workspace_id=None) -> None:
    await connection.execute(
        text(f'INSERT INTO "{schema}".conversations (id, user_id, workspace_id) VALUES (:id, :uid, :wid)'),
        {"id": conversation_id, "uid": user_id, "wid": workspace_id},
    )


def _assert_active_generation_conflict(error: DBAPIError) -> None:
    assert error.orig.sqlstate == "PZ001"
    assert error.orig.__cause__.message == "ACTIVE_GENERATION_EXISTS"


async def _wait_for_lock_waiter(engine, application_name: str, timeout: float = 5.0,
                               task: asyncio.Task | None = None) -> None:
    """Wait until the named test connection is demonstrably blocked on a DB lock."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if task is not None and task.done():
            result = task.result()  # Propagate the RPC error instead of masking it as a lock timeout.
            raise AssertionError(f"database RPC completed before acquiring the expected lock: {result!r}")
        async with engine.connect() as monitor:
            row = (await monitor.execute(text("""
                SELECT wait_event_type, cardinality(pg_blocking_pids(pid))
                  FROM pg_stat_activity WHERE application_name=:name
            """), {"name": application_name})).first()
        if row is not None and row[0] == "Lock" and row[1] > 0:
            return
        await asyncio.sleep(0.02)
    if task is not None and task.done():
        result = task.result()
        raise AssertionError(f"database RPC completed before acquiring the expected lock: {result!r}")
    raise AssertionError(f"database RPC did not block on the generation row lock: {application_name}")


@pytest.mark.asyncio
async def test_lock_wait_helper_surfaces_competing_rpc_errors():
    async def fail_rpc():
        raise RuntimeError("underlying RPC failure")

    task = asyncio.create_task(fail_rpc())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="underlying RPC failure"):
        await _wait_for_lock_waiter(None, "test-rpc", task=task)


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
                               role: str = "service_role", user_id=None,
                               task_registry: set[asyncio.Task] | None = None):
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
    if task_registry is not None:
        task_registry.add(task)
    await started.wait()
    try:
        await _wait_for_lock_waiter(engine, application_name, task=task)
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
    writer = None
    writer_tx = None
    acceptance_task = None
    competing_tasks: set[asyncio.Task] = set()

    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            await connection.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{schema}" TO authenticated, service_role')
            await connection.exec_driver_sql(f'''CREATE TABLE "{schema}".conversations (
                id uuid PRIMARY KEY, user_id uuid NOT NULL, workspace_id uuid
            )''')
            await connection.exec_driver_sql(f'''CREATE TABLE "{schema}".workspaces (
                id uuid PRIMARY KEY, user_id uuid NOT NULL
            )''')
            await connection.exec_driver_sql(f'''CREATE TABLE "{schema}".documents (
                id uuid PRIMARY KEY, workspace_id uuid NOT NULL, user_id uuid NOT NULL,
                original_filename text NOT NULL
            )''')
            await connection.exec_driver_sql(f'''CREATE TABLE "{schema}".document_chunks (
                id uuid PRIMARY KEY, document_id uuid NOT NULL, workspace_id uuid NOT NULL,
                user_id uuid NOT NULL, content text NOT NULL,
                page_number_start integer NOT NULL, page_number_end integer NOT NULL,
                token_count integer NOT NULL DEFAULT 0
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
            conversation_read_boundary = (await connection.execute(text('''
                SELECT has_table_privilege('authenticated', :table_name, 'SELECT'),
                       has_table_privilege('service_role', :table_name, 'SELECT')
            '''), {"table_name": f'"{schema}".conversations'})).one()
            assert conversation_read_boundary == (True, False)
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
            acceptance_signatures = (await connection.execute(text('''
                SELECT p.pronargs, ARRAY(
                    SELECT format_type(arg_type, NULL)
                      FROM unnest(p.proargtypes::oid[]) WITH ORDINALITY AS args(arg_type, arg_position)
                     ORDER BY arg_position
                )
                  FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                 WHERE n.nspname=:schema AND p.proname='create_chat_generation_with_snapshot'
            '''), {"schema": schema})).all()
            assert acceptance_signatures == [(
                15,
                ["uuid", "uuid", "uuid", "bigint", "text", "text", "text",
                 "uuid", "uuid", "uuid", "smallint", "jsonb", "text", "text", "text"],
            )]

            terminal_status_column = await connection.scalar(text('''SELECT count(*) FROM information_schema.columns
                WHERE table_schema=:schema AND table_name='chat_generation_runs' AND column_name='terminal_status' '''),
                {"schema": schema})
            assert terminal_status_column == 0
            snapshot_access = (await connection.execute(text('''
                SELECT c.relrowsecurity,
                       has_table_privilege('authenticated', c.oid, 'SELECT'),
                       has_table_privilege('anon', c.oid, 'SELECT'),
                       has_table_privilege('service_role', c.oid, 'SELECT')
                  FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                 WHERE n.nspname=:schema AND c.relname='chat_generation_context_snapshots'
            '''), {"schema": schema})).one()
            assert snapshot_access == (True, False, False, False)
            snapshot_reader_acl = (await connection.execute(text('''
                SELECT has_function_privilege('service_role', p.oid, 'EXECUTE'),
                       has_function_privilege('authenticated', p.oid, 'EXECUTE'),
                       has_function_privilege('anon', p.oid, 'EXECUTE'),
                       p.prosecdef, p.proconfig
                  FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                 WHERE n.nspname=:schema AND p.proname='read_chat_generation_context_snapshot'
            '''), {"schema": schema})).one()
            assert snapshot_reader_acl[0:4] == (True, False, False, True)
            assert any("search_path=pg_catalog, public, pg_temp" in item
                       for item in snapshot_reader_acl[4])
            worker_claim_paths = (await connection.execute(text('''
                SELECT p.proconfig
                  FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
                 WHERE n.nspname=:schema AND p.proname='claim_chat_generation_worker'
                 ORDER BY p.pronargs
            '''), {"schema": schema})).scalars().all()
            assert len(worker_claim_paths) == 2
            assert all(any("search_path=pg_catalog, public, pg_temp" in setting
                           for setting in path) for path in worker_claim_paths)

        # Every message write path advances the monotonic conversation version.
        version_conversation = uuid.uuid4()
        version_message = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, version_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            assert await connection.scalar(text(
                f'SELECT conversation_version FROM "{schema}".conversations WHERE id=:id'
            ), {"id": version_conversation}) == 0
            await connection.execute(text(f'''INSERT INTO "{schema}".messages(
                id,conversation_id,role,content) VALUES (:mid,:cid,'user','version one')'''),
                {"mid": version_message, "cid": version_conversation})
        async with engine.begin() as connection:
            inspection_context = (await connection.execute(text('''
                SELECT current_user = session_user,
                       has_table_privilege(current_user, :table_name, 'SELECT')
            '''), {"table_name": f'"{schema}".conversations'})).one()
            assert inspection_context == (True, True)
            assert await connection.scalar(text(
                f'SELECT conversation_version FROM "{schema}".conversations WHERE id=:id'
            ), {"id": version_conversation}) == 1
        async with engine.begin() as connection:
            await connection.execute(text(f'''UPDATE "{schema}".messages
                SET content='version two' WHERE id=:id'''), {"id": version_message})
        async with engine.begin() as connection:
            assert await connection.scalar(text(
                f'SELECT conversation_version FROM "{schema}".conversations WHERE id=:id'
            ), {"id": version_conversation}) == 2
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await connection.execute(text(f'DELETE FROM "{schema}".messages WHERE id=:id'),
                                     {"id": version_message})
        async with engine.begin() as connection:
            assert await connection.scalar(text(
                f'SELECT conversation_version FROM "{schema}".conversations WHERE id=:id'
            ), {"id": version_conversation}) == 3

        # Moving a message bumps the old and new conversation versions.
        move_source, move_target, move_message = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, move_source, owner)
            await _insert_conversation(connection, schema, move_target, owner)
            await connection.execute(text(f'''INSERT INTO "{schema}".messages(
                id,conversation_id,role,content) VALUES (:mid,:cid,'user','move me')'''),
                {"mid": move_message, "cid": move_source})
        async with engine.begin() as connection:
            await connection.execute(text(f'''UPDATE "{schema}".messages
                SET conversation_id=:target WHERE id=:mid'''),
                {"target": move_target, "mid": move_message})
        async with engine.begin() as connection:
            versions = (await connection.execute(text(f'''SELECT id,conversation_version
                FROM "{schema}".conversations WHERE id IN (:source,:target) ORDER BY id'''),
                {"source": move_source, "target": move_target})).all()
            version_by_id = {row.id: row.conversation_version for row in versions}
            assert version_by_id == {move_source: 2, move_target: 1}

        # A message transaction that owns the conversation row first makes the
        # competing acceptance block, then fail stale after that writer commits.
        version_race_conversation, race_message = uuid.uuid4(), uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, version_race_conversation, owner)
        writer = await engine.connect()
        writer_tx = await writer.begin()
        await _as_role(writer, "authenticated", owner)
        await writer.execute(text(f'''INSERT INTO "{schema}".messages(
            id,conversation_id,role,content) VALUES (:mid,:cid,'user','racing write')'''),
            {"mid": race_message, "cid": version_race_conversation})
        accepted_started = asyncio.Event()
        application_name = f"snapshot_accept_race_{uuid.uuid4().hex}"

        async def accept_during_write():
            async with engine.connect() as connection:
                transaction = await connection.begin()
                try:
                    await _as_role(connection, "authenticated", owner)
                    await connection.execute(text("SELECT set_config('application_name', :name, true)"),
                                             {"name": application_name})
                    accepted_started.set()
                    result = await _create_generation(connection, schema,
                        generation_id=uuid.uuid4(), conversation_id=version_race_conversation,
                        user_id=owner, content="racing generation")
                    await transaction.commit()
                    return result
                except BaseException:
                    await transaction.rollback()
                    raise

        acceptance_task = asyncio.create_task(accept_during_write())
        await accepted_started.wait()
        await _wait_for_lock_waiter(engine, application_name, task=acceptance_task)
        await writer_tx.commit()
        await writer.close()
        with pytest.raises(DBAPIError) as stale_race:
            await acceptance_task
        assert stale_race.value.orig.sqlstate == "40001"
        async with engine.begin() as connection:
            assert await connection.scalar(text(f'''SELECT conversation_version
                FROM "{schema}".conversations WHERE id=:id'''),
                {"id": version_race_conversation}) == 1
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_runs
                WHERE conversation_id=:id'''), {"id": version_race_conversation}) == 0
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".messages
                WHERE conversation_id=:id'''), {"id": version_race_conversation}) == 1

        # Authenticated acceptance cannot substitute another workspace or
        # document, nor attach evidence whose chunk is not the stored source.
        secured_workspace, other_workspace = uuid.uuid4(), uuid.uuid4()
        owned_document, foreign_document = uuid.uuid4(), uuid.uuid4()
        evidence_chunk = uuid.uuid4()
        workspace_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await connection.execute(text(f'''INSERT INTO "{schema}".workspaces(id,user_id)
                VALUES (:w1,:owner),(:w2,:other)'''), {
                "w1": secured_workspace, "w2": other_workspace, "owner": owner, "other": other_owner,
            })
            await connection.execute(text(f'''INSERT INTO "{schema}".documents(
                id,workspace_id,user_id,original_filename) VALUES
                (:owned,:workspace,:owner,'owned.pdf'),(:foreign,:workspace,:other,'foreign.pdf')'''), {
                "owned": owned_document, "foreign": foreign_document,
                "workspace": secured_workspace, "owner": owner, "other": other_owner,
            })
            await connection.execute(text(f'''INSERT INTO "{schema}".document_chunks(
                id,document_id,workspace_id,user_id,content,page_number_start,page_number_end,token_count)
                VALUES (:id,:doc,:workspace,:owner,'Stored evidence',4,4,2)'''), {
                "id": evidence_chunk, "doc": owned_document,
                "workspace": secured_workspace, "owner": owner,
            })
            await _insert_conversation(connection, schema, workspace_conversation, owner, secured_workspace)

        def acceptance_payload(doc_id, *, chunk_id=None, evidence=False, workspace_id=secured_workspace,
                               content="secure snapshot request", system_instruction=None,
                               snapshot_overrides=None):
            snapshot = _generation_snapshot(content)
            snapshot["workspace_id"] = str(workspace_id) if workspace_id else None
            snapshot["selected_document_ids"] = [str(doc_id)] if doc_id else []
            if system_instruction is not None:
                snapshot["system_instruction"] = system_instruction
            if evidence:
                snapshot["evidence"] = [{
                    "source_index": 1, "document_id": str(doc_id), "document_name": "owned.pdf",
                    "chunk_id": str(chunk_id), "page_start": 4, "page_end": 4,
                    "content": "Stored evidence",
                }]
                snapshot["citation_sources"] = [{
                    "source_index": 1, "document_id": str(doc_id), "document_name": "owned.pdf",
                    "chunk_id": str(chunk_id), "page_start": 4, "page_end": 4,
                }]
            if snapshot_overrides:
                snapshot.update(snapshot_overrides)
            canonical_snapshot = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            identity = {
                "content": content, "mode": "Detailed Guidance",
                "workspace_id": str(workspace_id) if workspace_id else None,
                "document_id": str(doc_id) if doc_id else None, "active_document_id": None,
            }
            canonical_request = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return content, canonical_snapshot, canonical_request

        async def attempt_protected_accept(connection, *, conversation, workspace, doc_id, chunk_id=None,
                                           evidence=False, corrupt_hash=False, corrupt_fingerprint=False,
                                           content="secure snapshot request", system_instruction=None,
                                           snapshot_overrides=None, generation_id=None):
            content, canonical_snapshot, canonical_request = acceptance_payload(
                doc_id, chunk_id=chunk_id, evidence=evidence, workspace_id=workspace,
                content=content, system_instruction=system_instruction,
                snapshot_overrides=snapshot_overrides,
            )
            version = await connection.scalar(text(f'''SELECT conversation_version FROM
                "{schema}".conversations WHERE id=:id'''), {"id": conversation})
            fingerprint = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
            digest = hashlib.sha256(canonical_snapshot.encode("utf-8")).hexdigest()
            if corrupt_hash:
                digest = "0" * 64 if digest != "0" * 64 else "1" * 64
            if corrupt_fingerprint:
                fingerprint = "0" * 64 if fingerprint != "0" * 64 else "1" * 64
            return await connection.execute(text(f'''SELECT * FROM "{schema}".create_chat_generation_with_snapshot(
                :gid,:cid,:uid,:version,:fingerprint,:content,'Detailed Guidance',:workspace,:doc,NULL,1::smallint,
                CAST(:snapshot AS jsonb),:digest,:canonical_snapshot,:canonical_request)'''), {
                "gid": generation_id or uuid.uuid4(), "cid": conversation, "uid": owner, "version": version,
                "fingerprint": fingerprint, "content": content, "workspace": workspace, "doc": doc_id,
                "snapshot": canonical_snapshot, "digest": digest,
                "canonical_snapshot": canonical_snapshot, "canonical_request": canonical_request,
            })

        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError) as wrong_workspace:
                async with connection.begin_nested():
                    await attempt_protected_accept(connection, conversation=workspace_conversation,
                        workspace=other_workspace, doc_id=owned_document)
            assert wrong_workspace.value.orig.sqlstate == "42501"
            with pytest.raises(DBAPIError) as foreign_doc:
                async with connection.begin_nested():
                    await attempt_protected_accept(connection, conversation=workspace_conversation,
                        workspace=secured_workspace, doc_id=foreign_document)
            assert foreign_doc.value.orig.sqlstate == "42501"
            with pytest.raises(DBAPIError) as foreign_chunk:
                async with connection.begin_nested():
                    await attempt_protected_accept(connection, conversation=workspace_conversation,
                        workspace=secured_workspace, doc_id=owned_document,
                        chunk_id=uuid.uuid4(), evidence=True)
            assert foreign_chunk.value.orig.sqlstate == "42501"
            with pytest.raises(DBAPIError) as forged_citation_metadata:
                async with connection.begin_nested():
                    await attempt_protected_accept(connection, conversation=workspace_conversation,
                        workspace=secured_workspace, doc_id=owned_document,
                        chunk_id=evidence_chunk, evidence=True,
                        snapshot_overrides={"citation_sources": [{
                            "source_index": 1, "document_id": str(owned_document),
                            "document_name": "owned.pdf", "chunk_id": str(evidence_chunk),
                            "page_start": 999, "page_end": 4,
                        }]})
            assert forged_citation_metadata.value.orig.sqlstate == "22023"
            with pytest.raises(DBAPIError) as wrong_context_hash:
                async with connection.begin_nested():
                    await attempt_protected_accept(connection, conversation=workspace_conversation,
                        workspace=secured_workspace, doc_id=owned_document, corrupt_hash=True)
            assert wrong_context_hash.value.orig.sqlstate == "22023"
            with pytest.raises(DBAPIError) as wrong_request_hash:
                async with connection.begin_nested():
                    await attempt_protected_accept(connection, conversation=workspace_conversation,
                        workspace=secured_workspace, doc_id=owned_document, corrupt_fingerprint=True)
            assert wrong_request_hash.value.orig.sqlstate == "22023"

        # A stale context version rejects acceptance before any generation rows
        # or user message are written.
        stale_generation = uuid.uuid4()
        stale_content = "stale context request"
        stale_snapshot = _generation_snapshot(stale_content)
        stale_canonical = json.dumps(stale_snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stale_request = {
            "content": stale_content, "mode": "Detailed Guidance", "workspace_id": None,
            "document_id": None, "active_document_id": None,
        }
        stale_canonical_request = json.dumps(stale_request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stale_fingerprint = hashlib.sha256(stale_canonical_request.encode("utf-8")).hexdigest()
        stale_hash = hashlib.sha256(stale_canonical.encode("utf-8")).hexdigest()
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError) as stale_context:
                async with connection.begin_nested():
                    await connection.execute(text(f'''SELECT * FROM "{schema}".create_chat_generation_with_snapshot(
                        :gid,:cid,:uid,0,:fingerprint,:content,'Detailed Guidance',NULL,NULL,NULL,1::smallint,
                        CAST(:snapshot AS jsonb),:hash,:canonical_snapshot,:canonical_request)'''), {
                        "gid": stale_generation, "cid": version_conversation, "uid": owner,
                        "fingerprint": stale_fingerprint, "content": stale_content,
                        "snapshot": stale_canonical, "hash": stale_hash,
                        "canonical_snapshot": stale_canonical,
                        "canonical_request": stale_canonical_request,
                    })
            assert stale_context.value.orig.sqlstate == "40001"
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_runs
                WHERE generation_id=:id'''), {"id": stale_generation}) == 0
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".messages
                WHERE conversation_id=:cid AND content=:content'''),
                {"cid": version_conversation, "content": stale_content}) == 0

        # The canonical byte count is calculated by the table trigger before
        # CHECK constraints run. Oversized acceptance must roll back every
        # earlier write made by the atomic RPC.
        oversized_conversation = uuid.uuid4()
        oversized_generation = uuid.uuid4()
        oversized_content = "oversized canonical request"
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, oversized_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError) as oversized_snapshot:
                async with connection.begin_nested():
                    await attempt_protected_accept(
                        connection, conversation=oversized_conversation, workspace=None, doc_id=None,
                        content=oversized_content, system_instruction="x" * 3_500_000,
                        snapshot_overrides={"routing_decisions": {
                            "canonical_byte_count": 1, "jsonb_rendered_byte_count": 1,
                        }},
                        generation_id=oversized_generation,
                    )
            assert oversized_snapshot.value.orig.sqlstate == "23514"
        async with engine.begin() as connection:
            rolled_back = (await connection.execute(text(f'''SELECT
                (SELECT count(*) FROM "{schema}".chat_generation_runs WHERE generation_id=:id),
                (SELECT count(*) FROM "{schema}".messages WHERE conversation_id=:cid),
                (SELECT count(*) FROM "{schema}".chat_generation_private_checkpoints WHERE generation_id=:id),
                (SELECT count(*) FROM "{schema}".chat_generation_context_snapshots WHERE generation_id=:id),
                (SELECT conversation_version FROM "{schema}".conversations WHERE id=:cid)'''), {
                "id": oversized_generation, "cid": oversized_conversation,
            })).one()
            assert rolled_back == (0, 0, 0, 0, 0)

        # This payload is compact enough for the canonical cap but expands
        # beyond 4 MiB when PostgreSQL renders JSONB with separator spaces.
        # Forged count-like fields in the payload cannot bypass the RPC's
        # database-side JSONB size check.
        jsonb_conversation = uuid.uuid4()
        jsonb_generation = uuid.uuid4()
        jsonb_content = "oversized JSONB request"
        jsonb_size_probe = [""] * 1_100_000
        jsonb_snapshot_overrides = {
            "routing_decisions": {
                "jsonb_size_probe": jsonb_size_probe,
                "canonical_byte_count": 1,
                "jsonb_rendered_byte_count": 1,
            },
        }
        _, jsonb_canonical_snapshot, _ = acceptance_payload(
            None, workspace_id=None, content=jsonb_content,
            snapshot_overrides=jsonb_snapshot_overrides,
        )
        assert len(jsonb_canonical_snapshot.encode("utf-8")) <= 3_500_000
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, jsonb_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            rendered_jsonb_bytes = await connection.scalar(text(
                "SELECT octet_length(CAST(:payload AS jsonb)::text)"
            ), {"payload": jsonb_canonical_snapshot})
            assert rendered_jsonb_bytes > 4_194_304
            with pytest.raises(DBAPIError) as oversized_jsonb:
                async with connection.begin_nested():
                    await attempt_protected_accept(
                        connection, conversation=jsonb_conversation, workspace=None, doc_id=None,
                        content=jsonb_content, snapshot_overrides=jsonb_snapshot_overrides,
                        generation_id=jsonb_generation,
                    )
            assert oversized_jsonb.value.orig.sqlstate == "22001"
        async with engine.begin() as connection:
            rolled_back = (await connection.execute(text(f'''SELECT
                (SELECT count(*) FROM "{schema}".chat_generation_runs WHERE generation_id=:id),
                (SELECT count(*) FROM "{schema}".messages WHERE conversation_id=:cid),
                (SELECT count(*) FROM "{schema}".chat_generation_private_checkpoints WHERE generation_id=:id),
                (SELECT count(*) FROM "{schema}".chat_generation_context_snapshots WHERE generation_id=:id),
                (SELECT conversation_version FROM "{schema}".conversations WHERE id=:cid)'''), {
                "id": jsonb_generation, "cid": jsonb_conversation,
            })).one()
            assert rolled_back == (0, 0, 0, 0, 0)

        # Force a failure at the final snapshot insert and prove the RPC rolls
        # back its earlier user message, run, and checkpoint writes.
        rollback_conversation = uuid.uuid4()
        rollback_generation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, rollback_conversation, owner)
            await connection.exec_driver_sql(f'''CREATE FUNCTION "{schema}".reject_snapshot_insert()
                RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN
                    RAISE EXCEPTION 'forced snapshot insert failure' USING ERRCODE='P0001';
                END $$''')
            await connection.exec_driver_sql(f'''CREATE TRIGGER reject_snapshot_insert_trg
                BEFORE INSERT ON "{schema}".chat_generation_context_snapshots
                FOR EACH ROW EXECUTE FUNCTION "{schema}".reject_snapshot_insert()''')
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError) as forced_failure:
                async with connection.begin_nested():
                    await _create_generation(connection, schema, generation_id=rollback_generation,
                                             conversation_id=rollback_conversation, user_id=owner)
            assert forced_failure.value.orig.sqlstate == "P0001"
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f'''DROP TRIGGER reject_snapshot_insert_trg
                ON "{schema}".chat_generation_context_snapshots''')
            await connection.exec_driver_sql(f'DROP FUNCTION "{schema}".reject_snapshot_insert()')
            rolled_back = (await connection.execute(text(f'''SELECT
                (SELECT count(*) FROM "{schema}".chat_generation_runs WHERE generation_id=:id),
                (SELECT count(*) FROM "{schema}".messages WHERE conversation_id=:cid),
                (SELECT count(*) FROM "{schema}".chat_generation_private_checkpoints WHERE generation_id=:id),
                (SELECT count(*) FROM "{schema}".chat_generation_context_snapshots WHERE generation_id=:id),
                (SELECT conversation_version FROM "{schema}".conversations WHERE id=:cid)'''), {
                "id": rollback_generation, "cid": rollback_conversation,
            })).one()
            assert rolled_back == (0, 0, 0, 0, 0)

        # Legacy C1 creation remains present as a deliberate fail-closed API.
        legacy_generation = uuid.uuid4()
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError) as legacy_rejected:
                async with connection.begin_nested():
                    await connection.execute(text(f'''SELECT * FROM "{schema}".create_chat_generation(
                        :gid,:cid,:uid,:fingerprint,'legacy request','Detailed Guidance',NULL,NULL,NULL)'''), {
                        "gid": legacy_generation, "cid": version_conversation, "uid": owner,
                        "fingerprint": uuid.uuid4().hex * 2,
                    })
            assert legacy_rejected.value.orig.sqlstate == "0A000"
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(f'''SELECT count(*) FROM "{schema}".chat_generation_runs
                WHERE generation_id=:id'''), {"id": legacy_generation}) == 0

        # A run inserted through any legacy/direct path without a snapshot is
        # never eligible for dispatch.
        no_snapshot_conversation = uuid.uuid4()
        no_snapshot_generation = uuid.uuid4()
        no_snapshot_message = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, no_snapshot_conversation, owner)
            await connection.execute(text(f'''INSERT INTO "{schema}".messages(
                id,conversation_id,role,content) VALUES (:mid,:cid,'user','no snapshot')'''), {
                "mid": no_snapshot_message, "cid": no_snapshot_conversation,
            })
            await connection.execute(text(f'''INSERT INTO "{schema}".chat_generation_runs(
                generation_id,conversation_id,user_id,user_message_id,request_fingerprint,chat_mode)
                VALUES (:gid,:cid,:uid,:mid,:fingerprint,'Detailed Guidance')'''), {
                "gid": no_snapshot_generation, "cid": no_snapshot_conversation, "uid": owner,
                "mid": no_snapshot_message, "fingerprint": uuid.uuid4().hex * 2,
            })
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": no_snapshot_generation}) is None
        manual_dispatch_token = uuid.uuid4()
        async with engine.begin() as connection:
            await connection.execute(text(f'''UPDATE "{schema}".chat_generation_runs
                SET state='dispatching', dispatch_token=:token,
                    dispatch_lease_expires_at=clock_timestamp()+interval '60 seconds'
                WHERE generation_id=:id'''), {"token": manual_dispatch_token, "id": no_snapshot_generation})
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            assert (await connection.execute(text(f'''SELECT * FROM
                "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'''), {
                "id": no_snapshot_generation, "dispatch": manual_dispatch_token,
            })).first() is None
            no_snapshot_state = (await connection.execute(text(f'''SELECT state,worker_token,worker_epoch
                FROM "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": no_snapshot_generation})).one()
            assert no_snapshot_state == ("dispatching", None, 0)

        # Supplied values never control either size constraint: the BEFORE
        # trigger overwrites them with byte lengths derived from the payload.
        caller_count_snapshot = _generation_snapshot("caller supplied counts")
        caller_count_canonical = json.dumps(
            caller_count_snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        forged_oversized_snapshot = _generation_snapshot("forged oversized counts")
        forged_oversized_snapshot["routing_decisions"] = jsonb_snapshot_overrides["routing_decisions"]
        forged_oversized_canonical = json.dumps(
            forged_oversized_snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        assert len(forged_oversized_canonical.encode("utf-8")) <= 3_500_000
        async with engine.begin() as connection:
            rendered_jsonb_bytes = await connection.scalar(text(
                "SELECT octet_length(CAST(:payload AS jsonb)::text)"
            ), {"payload": forged_oversized_canonical})
            assert rendered_jsonb_bytes > 4_194_304
            with pytest.raises(DBAPIError) as forged_counts:
                async with connection.begin_nested():
                    await connection.execute(text(f'''INSERT INTO "{schema}".chat_generation_context_snapshots(
                        generation_id,snapshot_version,snapshot_payload,canonical_snapshot,context_hash,
                        canonical_byte_count,jsonb_rendered_byte_count)
                        VALUES (:id,1,CAST(:snapshot AS jsonb),:canonical,:hash,1,1)'''), {
                        "id": no_snapshot_generation,
                        "snapshot": forged_oversized_canonical,
                        "canonical": forged_oversized_canonical,
                        "hash": hashlib.sha256(forged_oversized_canonical.encode("utf-8")).hexdigest(),
                    })
            assert forged_counts.value.orig.sqlstate == "23514"
            assert await connection.scalar(text(f'''SELECT count(*)
                FROM "{schema}".chat_generation_context_snapshots WHERE generation_id=:id'''), {
                "id": no_snapshot_generation,
            }) == 0
            await connection.execute(text(f'''INSERT INTO "{schema}".chat_generation_context_snapshots(
                generation_id,snapshot_version,snapshot_payload,canonical_snapshot,context_hash,
                canonical_byte_count,jsonb_rendered_byte_count)
                VALUES (:id,1,CAST(:snapshot AS jsonb),:canonical,:hash,1,1)'''), {
                "id": no_snapshot_generation,
                "snapshot": caller_count_canonical,
                "canonical": caller_count_canonical,
                "hash": hashlib.sha256(caller_count_canonical.encode("utf-8")).hexdigest(),
            })
            generated_counts = (await connection.execute(text(f'''SELECT
                canonical_byte_count,
                octet_length(convert_to(canonical_snapshot,'UTF8')),
                jsonb_rendered_byte_count,
                octet_length(snapshot_payload::text)
                FROM "{schema}".chat_generation_context_snapshots WHERE generation_id=:id'''), {
                "id": no_snapshot_generation,
            })).one()
            assert generated_counts[0:2] == (
                len(caller_count_canonical.encode("utf-8")),
                len(caller_count_canonical.encode("utf-8")),
            )
            assert generated_counts[2] == generated_counts[3]
            assert generated_counts[0] != 1 and generated_counts[2] != 1

        # Owner access to the run does not imply snapshot table access. The
        # service-only reader requires a live worker identity and verifies an
        # unavailable/mismatched identity returns no data.
        read_generation = uuid.uuid4()
        read_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, read_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=read_generation,
                                     conversation_id=read_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            with pytest.raises(DBAPIError):
                async with connection.begin_nested():
                    await connection.execute(text(f'''SELECT snapshot_payload
                        FROM "{schema}".chat_generation_context_snapshots WHERE generation_id=:id'''),
                        {"id": read_generation})
            with pytest.raises(DBAPIError):
                async with connection.begin_nested():
                    await connection.execute(text(f'''SELECT * FROM
                        "{schema}".read_chat_generation_context_snapshot(:id,NULL,NULL)'''),
                        {"id": read_generation})
        async with engine.begin() as connection:
            with pytest.raises(DBAPIError) as immutable:
                async with connection.begin_nested():
                    await connection.execute(text(f'''UPDATE "{schema}".chat_generation_context_snapshots
                        SET context_hash=:hash WHERE generation_id=:id'''), {
                        "id": read_generation, "hash": "0" * 64,
                    })
            assert immutable.value.orig.sqlstate == "55000"
        async with engine.begin() as connection:
            await _as_role(connection, "service_role")
            dispatch = await connection.scalar(text(
                f'SELECT "{schema}".claim_chat_generation_dispatch(:id,30)'
            ), {"id": read_generation})
            claim = (await connection.execute(text(
                f'SELECT * FROM "{schema}".claim_chat_generation_worker(:id,:dispatch,20)'
            ), {"id": read_generation, "dispatch": dispatch})).one()
            snapshot_row = (await connection.execute(text(f'''SELECT * FROM
                "{schema}".read_chat_generation_context_snapshot(:id,:token,:epoch)'''), {
                "id": read_generation, "token": claim.worker_token, "epoch": claim.worker_epoch,
            })).one()
            assert snapshot_row.snapshot_payload["original_request"] == "question"
            assert snapshot_row.context_hash
            assert snapshot_row.canonical_byte_count == len(snapshot_row.canonical_snapshot.encode("utf-8"))
            rendered_snapshot_bytes = await connection.scalar(text(
                "SELECT octet_length(CAST(:canonical AS jsonb)::text)"
            ), {"canonical": snapshot_row.canonical_snapshot})
            assert snapshot_row.jsonb_rendered_byte_count == rendered_snapshot_bytes
            assert hashlib.sha256(snapshot_row.canonical_snapshot.encode("utf-8")).hexdigest() == snapshot_row.context_hash
            assert (await connection.execute(text(f'''SELECT * FROM
                "{schema}".read_chat_generation_context_snapshot(:id,:token,:epoch)'''), {
                "id": read_generation, "token": uuid.uuid4(), "epoch": claim.worker_epoch,
            })).first() is None
            assert (await connection.execute(text(f'''SELECT * FROM
                "{schema}".read_chat_generation_context_snapshot(:id,:token,:epoch)'''), {
                "id": read_generation, "token": claim.worker_token, "epoch": claim.worker_epoch + 1,
            })).first() is None
            read_message = await connection.scalar(text(f'''SELECT "{schema}".finalize_chat_generation_worker(
                :id,:token,:epoch,'completed','test','done','[]'::jsonb,NULL)'''), {
                "id": read_generation, "token": claim.worker_token, "epoch": claim.worker_epoch,
            })
            assert read_message is not None

        async with engine.begin() as connection:
            assert await connection.scalar(text(f'''SELECT count(*) FROM
                "{schema}".chat_generation_context_snapshots WHERE generation_id=:id'''),
                {"id": read_generation}) == 0

        # Deleting a conversation cascades through the run to its private
        # snapshot, including when the generation never reaches finalization.
        cascade_conversation = uuid.uuid4()
        cascade_generation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, cascade_conversation, owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await _create_generation(connection, schema, generation_id=cascade_generation,
                                     conversation_id=cascade_conversation, user_id=owner)
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            await connection.execute(text(f'DELETE FROM "{schema}".conversations WHERE id=:id'),
                                     {"id": cascade_conversation})
        async with engine.begin() as connection:
            assert await connection.scalar(text(f'''SELECT count(*) FROM
                "{schema}".chat_generation_context_snapshots WHERE generation_id=:id'''),
                {"id": cascade_generation}) == 0
            assert await connection.scalar(text(f'''SELECT count(*) FROM
                "{schema}".chat_generation_runs WHERE generation_id=:id'''),
                {"id": cascade_generation}) == 0
            assert await connection.scalar(text(f'''SELECT count(*) FROM
                "{schema}".messages WHERE conversation_id=:id'''),
                {"id": cascade_conversation}) == 0
            assert await connection.scalar(text(f'''SELECT count(*) FROM
                "{schema}".conversations WHERE id=:id'''),
                {"id": cascade_conversation}) == 0

        # Same-ID retries return the existing durable run and user message;
        # different payload fingerprints are rejected without a duplicate row.
        generation_id = uuid.uuid4()
        async with engine.begin() as connection:
            await _as_role(connection, "authenticated", owner)
            first = await _create_generation(connection, schema, generation_id=generation_id,
                conversation_id=conversation_id, user_id=owner)
            again = await _create_generation(connection, schema, generation_id=generation_id,
                conversation_id=conversation_id, user_id=owner)
            assert first.generation_id == again.generation_id == generation_id
            assert first.state == again.state == "queued"
            assert first.user_message_id == again.user_message_id
            assert first.reused is False and again.reused is True
            with pytest.raises(DBAPIError) as reused_id:
                async with connection.begin_nested():
                    await _create_generation(connection, schema, generation_id=generation_id,
                        conversation_id=conversation_id, user_id=owner, content="different request")
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
        same_id_conversation = uuid.uuid4()
        async with engine.begin() as connection:
            await _insert_conversation(connection, schema, same_id_conversation, owner)

        async def same_id_start():
            async with engine.begin() as connection:
                await _as_role(connection, "authenticated", owner)
                return await _create_generation(connection, schema, generation_id=same_id,
                    conversation_id=same_id_conversation, user_id=owner)

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
                     "epoch": claim.worker_epoch}, task_registry=competing_tasks)
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
                     "epoch": claim.worker_epoch}, task_registry=competing_tasks)
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
                {"id": finalize_first_id}, role="authenticated", user_id=owner,
                task_registry=competing_tasks)
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
                 "epoch": cancel_first_claim.worker_epoch}, task_registry=competing_tasks)
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
        try:
            pending_tasks = set(competing_tasks)
            if acceptance_task is not None:
                pending_tasks.add(acceptance_task)
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
        finally:
            try:
                if writer_tx is not None and writer_tx.is_active:
                    await writer_tx.rollback()
            finally:
                try:
                    if writer is not None:
                        await writer.close()
                finally:
                    try:
                        async with engine.begin() as connection:
                            await connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                            assert await connection.scalar(text(
                                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=:schema)"
                            ), {"schema": schema}) is False
                    finally:
                        await engine.dispose()
