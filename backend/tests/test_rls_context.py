import json
import uuid
import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.session import _apply_rls_context


class _RecordingConnection:
    def __init__(self):
        self.statements = []

    def exec_driver_sql(self, statement):
        self.statements.append((statement, None))

    def execute(self, statement, params):
        self.statements.append((str(statement), params))


def test_transaction_hook_sets_authenticated_role_and_verified_claims():
    user_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    connection = _RecordingConnection()
    session = type("SessionStub", (), {"info": {"rls_user_id": user_id}})()

    _apply_rls_context(session, transaction=None, connection=connection)

    assert connection.statements[0] == ("SET LOCAL ROLE authenticated", None)
    claims = json.loads(connection.statements[1][1]["claims"])
    assert claims == {
        "sub": "11111111-1111-1111-1111-111111111111",
        "role": "authenticated",
        "aud": "authenticated",
    }
    assert connection.statements[2][1] == {
        "user_id": "11111111-1111-1111-1111-111111111111"
    }


def test_transaction_hook_without_identity_still_fails_closed_under_rls():
    connection = _RecordingConnection()
    session = type("SessionStub", (), {"info": {}})()

    _apply_rls_context(session, transaction=None, connection=connection)

    assert connection.statements[0] == ("SET LOCAL ROLE authenticated", None)
    assert connection.statements[1][1] == {
        "claims": '{"role": "authenticated", "aud": "authenticated"}'
    }
    assert connection.statements[2][1] == {"user_id": ""}


@pytest.mark.asyncio
async def test_postgres_rls_hides_other_users_rows_without_app_filter():
    """End-to-end regression; run against a dedicated Supabase/Postgres test DB."""
    # Do not inherit a database URL from the repository's .env. This test creates
    # and drops a table, so it must be opted into through the process environment
    # with a dedicated disposable database only.
    database_url = os.environ.get("RLS_TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("Set RLS_TEST_DATABASE_URL to run the database RLS integration test")

    engine = create_async_engine(database_url)
    table = f"codex_rls_probe_{uuid.uuid4().hex}"
    user_a = "11111111-1111-1111-1111-111111111111"
    user_b = "22222222-2222-2222-2222-222222222222"
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(
                f"CREATE TABLE public.{table} (user_id uuid NOT NULL, secret text NOT NULL)"
            )
            await connection.exec_driver_sql(
                f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"
            )
            await connection.exec_driver_sql(
                f"CREATE POLICY owner_only ON public.{table} "
                "FOR SELECT TO authenticated USING (auth.uid() = user_id)"
            )
            await connection.exec_driver_sql(
                f"GRANT SELECT ON public.{table} TO authenticated"
            )
            await connection.execute(
                text(f"INSERT INTO public.{table} (user_id, secret) VALUES (:a, 'A'), (:b, 'B')"),
                {"a": user_a, "b": user_b},
            )

        visible_by_user = {}
        for user_id in (user_a, user_b):
            async with AsyncSession(engine) as session:
                session.info["rls_user_id"] = uuid.UUID(user_id)
                rows = await session.execute(
                    # Deliberately omit a user_id predicate: RLS must do the filtering.
                    text(f"SELECT secret FROM public.{table}")
                )
                visible_by_user[user_id] = {row.secret for row in rows}

        assert visible_by_user[user_a] == {"A"}
        assert visible_by_user[user_b] == {"B"}
    finally:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(f"DROP TABLE IF EXISTS public.{table}")
        await engine.dispose()
