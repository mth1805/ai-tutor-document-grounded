"""Generation metadata migration coverage on the opt-in disposable DB target."""
import os
import uuid
import sqlparse
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from tests.test_rls_context import _validate_rls_test_target


@pytest.mark.asyncio
async def test_generation_migration_enforces_assistant_status_and_cleans_up_schema():
    database_url = os.environ.get("RLS_TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("Set RLS_TEST_DATABASE_URL to run generation migration integration")
    _validate_rls_test_target(
        database_url,
        os.environ.get("RLS_TEST_DATABASE_CONFIRMATION", "").strip(),
        os.environ.get("RLS_TEST_PROJECT_CONFIRMATION", "").strip(),
        os.environ.get("RLS_TEST_DATABASE_ACKNOWLEDGE_DISPOSABLE", "").strip(),
    )

    migration_path = Path(__file__).resolve().parents[2] / "supabase" / "migrations" / "20261120120000_add_message_generation_metadata.sql"
    schema = f"codex_generation_{uuid.uuid4().hex}"
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            await connection.exec_driver_sql(
                f'''CREATE TABLE "{schema}".messages (
                    id integer primary key,
                    role varchar(32) not null,
                    content text not null,
                    generation_status varchar(16),
                    generation_metadata jsonb
                )'''
            )
            await connection.exec_driver_sql(
                f'''INSERT INTO "{schema}".messages
                    (id, role, content, generation_status)
                    VALUES
                      (1, 'user', 'question', NULL),
                      (2, 'assistant', 'old answer', NULL),
                      (3, 'assistant', 'legacy invalid status', 'unknown'),
                      (4, 'user', 'legacy invalid role status', 'completed')'''
            )

            migration_sql = migration_path.read_text(encoding="utf-8").replace(
                "public.messages", f'"{schema}".messages'
            )
            for statement in sqlparse.split(migration_sql):
                sql = statement.strip()
                if sql:
                    await connection.exec_driver_sql(sql)

            historical = (await connection.execute(text(
                f'SELECT id, generation_status FROM "{schema}".messages ORDER BY id'
            ))).all()
            assert [(row.id, row.generation_status) for row in historical] == [
                (1, None), (2, "completed"), (3, "completed"), (4, None)
            ]

            await connection.exec_driver_sql(
                f'''INSERT INTO "{schema}".messages
                    (id, role, content, generation_status, generation_metadata)
                    VALUES (5, 'assistant', 'completed answer', 'completed', NULL),
                        (6, 'assistant', 'partial answer', 'incomplete',
                            '{{"finish_reason":"MAX_TOKENS","generation_duration_ms":123.4}}'::jsonb)'''
            )

            partial = (await connection.execute(text(
                f'SELECT content, generation_status, generation_metadata FROM "{schema}".messages WHERE id = 6'
            ))).one()
            assert partial.content == "partial answer"
            assert partial.generation_status == "incomplete"
            assert partial.generation_metadata["finish_reason"] == "MAX_TOKENS"

            await connection.execute(text(
                f'''INSERT INTO "{schema}".messages (id, role, content, generation_status)
                    VALUES (7, 'user', 'new user message', NULL),
                           (8, 'system', 'system instruction', NULL)'''
            ))

            with pytest.raises(IntegrityError):
                async with connection.begin_nested():
                    await connection.execute(text(
                        f'''INSERT INTO "{schema}".messages
                            (id, role, content, generation_status)
                            VALUES (9, 'assistant', 'missing outcome', NULL)'''
                    ))

            with pytest.raises(IntegrityError):
                async with connection.begin_nested():
                    await connection.execute(text(
                        f'''INSERT INTO "{schema}".messages
                            (id, role, content, generation_status)
                            VALUES (10, 'user', 'must not be generated', 'completed')'''
                    ))
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            schema_exists = await connection.scalar(text(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = :schema)"
            ), {"schema": schema})
            assert schema_exists is False
        await engine.dispose()
