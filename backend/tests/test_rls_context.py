import json
import uuid
import os
import re
from sqlalchemy.engine import make_url

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.session import _apply_rls_context


def _validate_rls_test_target(
    database_url: str,
    confirmed_database: str,
    confirmed_project: str,
    acknowledge_disposable: str,
) -> None:
    """Require exact target confirmations and explicit disposable intent."""
    url = make_url(database_url)
    database_name = url.database or ""
    host = (url.host or "").lower().rstrip(".")
    host_parts = host.split(".")
    username = url.username or ""
    if len(host_parts) == 4 and host_parts[0] == "db" and host_parts[2:] == ["supabase", "co"]:
        # Direct: db.<project-ref>.supabase.co
        project_id = host_parts[1]
    elif host.endswith(".pooler.supabase.com"):
        # Pooler: the ref is encoded in the login as postgres.<project-ref>.
        username_parts = username.split(".")
        if len(username_parts) != 2 or username_parts[0] != "postgres" or not re.fullmatch(r"[a-z0-9-]+", username_parts[1]):
            pytest.fail("Supabase pooler username must be postgres.<project-ref>")
        project_id = username_parts[1]
    else:
        # Non-Supabase hosts must be confirmed by their complete hostname.
        project_id = host
    if not database_name:
        pytest.fail("RLS test URL must include an explicit database name")
    if any(label in f"{host} {project_id} {database_name}" for label in ("prod", "production", "primary")):
        pytest.fail("Refusing an RLS test target that looks like production")
    if confirmed_database != database_name:
        pytest.fail("RLS_TEST_DATABASE_CONFIRMATION must exactly match the database name")
    if ":" not in confirmed_project:
        pytest.fail("RLS_TEST_PROJECT_CONFIRMATION must be '<test|staging|disposable>:<host/project identifier>'")
    project_environment, confirmed_project_id = confirmed_project.split(":", 1)
    if project_environment.lower() not in {"test", "staging", "disposable"} or confirmed_project_id != project_id:
        pytest.fail("RLS_TEST_PROJECT_CONFIRMATION must explicitly classify and exactly match this host/project")

    target_hint = f"{project_id} {database_name}".lower()
    explicitly_disposable = any(label in target_hint for label in ("test", "staging", "disposable"))
    if database_name.lower() == "postgres":
        if project_environment.lower() not in {"test", "staging", "disposable"}:
            pytest.fail("A postgres database requires explicit test/staging/disposable project confirmation")
        if acknowledge_disposable != "YES":
            pytest.fail("Set RLS_TEST_DATABASE_ACKNOWLEDGE_DISPOSABLE=YES to confirm the postgres target is disposable")
    elif not explicitly_disposable:
        pytest.fail("Target must identify itself as test, staging, or disposable")


def test_rls_target_guard_accepts_named_disposable_database():
    _validate_rls_test_target(
        "postgresql+asyncpg://user:pass@test-db.example.com/app_test", "app_test", "test:test-db.example.com", ""
    )


def test_rls_target_guard_rejects_production_looking_target():
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            "postgresql+asyncpg://user:pass@prod-db.example.com/app_test", "app_test", "staging:prod-db.example.com", ""
        )


def test_rls_target_guard_rejects_postgres_without_staging_confirmation():
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            "postgresql+asyncpg://user:pass@db.example.com/postgres", "postgres", "db.example.com", "YES"
        )


def test_rls_target_guard_accepts_confirmed_staging_supabase_postgres():
    _validate_rls_test_target(
        "postgresql+asyncpg://user:pass@db.ai-tutor-staging.supabase.co/postgres",
        "postgres", "staging:ai-tutor-staging", "YES",
    )


def test_rls_target_guard_accepts_direct_supabase_project_ref():
    _validate_rls_test_target(
        "postgresql+asyncpg://postgres:pass@db.abc123staging.supabase.co/postgres",
        "postgres", "staging:abc123staging", "YES",
    )


def test_rls_target_guard_accepts_supabase_pooler_project_ref():
    _validate_rls_test_target(
        "postgresql+asyncpg://postgres.abc123staging:pass@aws-0-ap-northeast-1.pooler.supabase.com/postgres",
        "postgres", "staging:abc123staging", "YES",
    )


def test_rls_target_guard_rejects_pooler_project_ref_mismatch():
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            "postgresql+asyncpg://postgres.abc123staging:pass@aws-0-ap-northeast-1.pooler.supabase.com/postgres",
            "postgres", "staging:otherproject", "YES",
        )


@pytest.mark.parametrize("username", ["postgres", "other.abc123staging", "postgres.", "postgres.a.b"])
def test_rls_target_guard_rejects_malformed_pooler_username(username):
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            f"postgresql+asyncpg://{username}:pass@aws-0-ap-northeast-1.pooler.supabase.com/postgres",
            "postgres", "staging:abc123staging", "YES",
        )


def test_rls_target_guard_rejects_production_supabase_project_ref():
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            "postgresql+asyncpg://postgres.prodproject:pass@aws-0-ap-northeast-1.pooler.supabase.com/app_test",
            "app_test", "staging:prodproject", "",
        )


def test_rls_target_guard_rejects_mismatched_confirmation():
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            "postgresql+asyncpg://user:pass@test-db.example.com/app_test", "another_db", "test:test-db.example.com", ""
        )


def test_rls_target_guard_rejects_missing_postgres_acknowledgement():
    with pytest.raises(pytest.fail.Exception):
        _validate_rls_test_target(
            "postgresql+asyncpg://user:pass@db.ai-tutor-staging.supabase.co/postgres",
            "postgres", "staging:ai-tutor-staging", "",
        )


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
    _validate_rls_test_target(
        database_url,
        os.environ.get("RLS_TEST_DATABASE_CONFIRMATION", "").strip(),
        os.environ.get("RLS_TEST_PROJECT_CONFIRMATION", "").strip(),
        os.environ.get("RLS_TEST_DATABASE_ACKNOWLEDGE_DISPOSABLE", "").strip(),
    )

    engine = create_async_engine(database_url)
    table = f"codex_rls_probe_{uuid.uuid4().hex}"
    user_a = "11111111-1111-1111-1111-111111111111"
    user_b = "22222222-2222-2222-2222-222222222222"
    try:
        # Verify the connected target and the Supabase-compatible primitives before DDL.
        async with engine.connect() as connection:
            database_name, current_user = (
                await connection.execute(text("SELECT current_database(), current_user"))
            ).one()
            if database_name.lower() != (engine.url.database or "").lower():
                pytest.fail("Connected database identity does not match the configured URL")
            roles = (await connection.execute(text(
                "SELECT rolname, rolsuper, rolbypassrls FROM pg_roles "
                "WHERE rolname IN ('authenticated', 'anon')"
            ))).all()
            role_map = {row.rolname: row for row in roles}
            if "authenticated" not in role_map:
                pytest.fail("RLS test target must provide the Supabase authenticated role")
            if role_map["authenticated"].rolsuper or role_map["authenticated"].rolbypassrls:
                pytest.fail("authenticated role must not bypass row-level security")
            if await connection.scalar(text("SELECT to_regprocedure('auth.uid()') IS NOT NULL")) is not True:
                pytest.fail("RLS test target must provide auth.uid()")

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
