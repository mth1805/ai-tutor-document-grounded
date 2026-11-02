"""Focused tests for durable ingestion job handling without external services."""
import uuid
import asyncio
from types import SimpleNamespace
import pytest

from app.models.ingestion_job import DocumentIngestionJob
from app.models.document import Document
from app.services.ingestion_service import IngestionService
from app.workers import ingestion_queue


class FakeSession:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.deleted = []
        self.statements = []
        self.owner_exists = True

    async def execute(self, _statement):
        self.statements.append(_statement)
        if _statement.column_descriptions[0]["entity"] is Document:
            return SimpleNamespace(scalar_one_or_none=lambda: uuid.UUID(int=1) if self.owner_exists else None)
        return SimpleNamespace(scalar_one_or_none=lambda: self.existing)

    def add(self, row):
        self.added.append(row)

    async def delete(self, row):
        self.deleted.append(row)

    async def flush(self):
        return None


@pytest.mark.asyncio
async def test_enqueue_does_not_duplicate_queued_or_processing_jobs():
    for state in ("queued", "processing"):
        session = FakeSession(SimpleNamespace(status=state))
        accepted = await IngestionService.enqueue_document(
            session,
            SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4()),
            uuid.uuid4(),
        )
        assert accepted is False
        assert session.added == []
        assert session.deleted == []


@pytest.mark.asyncio
async def test_enqueue_rearms_terminal_job_for_owner():
    old = SimpleNamespace(status="failed")
    document_id, workspace_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    document = SimpleNamespace(
        id=document_id, workspace_id=workspace_id, status="failed",
        processing_error="previous failure", processed_at=object(),
    )
    session = FakeSession(old)

    accepted = await IngestionService.enqueue_document(session, document, user_id)

    assert accepted is True
    assert session.deleted == [old]
    assert len(session.added) == 1
    queued = session.added[0]
    assert isinstance(queued, DocumentIngestionJob)
    assert queued.document_id == document_id
    assert queued.workspace_id == workspace_id
    assert queued.user_id == user_id
    assert queued.status == "queued"
    assert queued.attempt_count == 0
    assert document.status == "queued"
    assert document.processing_error is None
    assert document.processed_at is None


@pytest.mark.asyncio
async def test_worker_claims_and_finalizes_through_service_rpc(monkeypatch):
    calls = []

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, *, headers, json):
            calls.append((url, headers, json))
            return Response(True)

    monkeypatch.setattr(ingestion_queue.settings, "SUPABASE_URL", "https://unit.test")
    monkeypatch.setattr(ingestion_queue.settings, "SUPABASE_SERVICE_ROLE_KEY", "test-only-key")
    monkeypatch.setattr(ingestion_queue.httpx, "AsyncClient", Client)

    await ingestion_queue.finish_job(uuid.UUID(int=1), succeeded=True)

    assert calls[0][0] == "https://unit.test/rest/v1/rpc/complete_document_ingestion_job"
    assert calls[0][1]["Authorization"] == "Bearer test-only-key"
    assert calls[0][2] == {"p_job_id": str(uuid.UUID(int=1))}


def test_queue_migration_keeps_rls_and_bounded_retry_contract():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "supabase" / "migrations" / "20261004120000_durable_document_ingestion_jobs.sql"
    sql = path.read_text(encoding="utf-8")
    assert "document_id UUID NOT NULL UNIQUE" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "CREATE POLICY \"Users can enqueue their documents\"" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "attempt_count < job_row.max_attempts" in sql
    assert "REVOKE ALL ON FUNCTION public.claim_document_ingestion_jobs" in sql
    assert "TO service_role" in sql


@pytest.mark.asyncio
async def test_enqueue_locks_owned_document_without_queue_update_privileges():
    session = FakeSession()
    document = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4())
    await IngestionService.enqueue_document(session, document, uuid.uuid4())
    assert "FOR UPDATE" in str(session.statements[0])
    assert "documents.user_id" in str(session.statements[0])
    assert "documents.workspace_id" in str(session.statements[0])
    assert "FOR UPDATE" not in str(session.statements[1])
    session = FakeSession()
    session.owner_exists = False
    with pytest.raises(ValueError, match="unauthorized"):
        await IngestionService.enqueue_document(session, document, uuid.uuid4())
    assert session.added == []


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_worker_claimed_attempt_runs_pipeline_and_finalizes(monkeypatch, success):
    job_id, document_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    calls = []

    async def load(_job_id):
        return {"claimed_document_id": str(document_id), "claimed_user_id": str(user_id), "claimed_attempt": 1}

    async def process(doc, user, **options):
        assert (doc, user) == (document_id, user_id)
        assert options == {"defer_failure_to_queue": True}
        return success

    async def finish(job, *, succeeded):
        calls.append((job, succeeded))

    monkeypatch.setattr(ingestion_queue, "load_claimed_job", load)
    monkeypatch.setattr(IngestionService, "process_document_background", process)
    monkeypatch.setattr(ingestion_queue, "finish_job", finish)
    assert await ingestion_queue.process_claimed_job(job_id) is success
    assert calls == [(job_id, success)]


@pytest.mark.asyncio
async def test_worker_pipeline_exception_is_failed_and_logs_are_sanitized(monkeypatch, caplog):
    job_id = uuid.uuid4()

    async def load(_job):
        return {"claimed_document_id": str(uuid.uuid4()), "claimed_user_id": str(uuid.uuid4()), "claimed_attempt": 2}

    async def process(*args, **kwargs):
        raise RuntimeError("PRIVATE_DOCUMENT_OR_CREDENTIAL")

    async def finish(_job, *, succeeded):
        assert succeeded is False

    monkeypatch.setattr(ingestion_queue, "load_claimed_job", load)
    monkeypatch.setattr(IngestionService, "process_document_background", process)
    monkeypatch.setattr(ingestion_queue, "finish_job", finish)
    assert await ingestion_queue.process_claimed_job(job_id) is False
    assert "PRIVATE_DOCUMENT_OR_CREDENTIAL" not in caplog.text


@pytest.mark.asyncio
async def test_worker_poll_recovers_from_rpc_failure_and_claims_only_one(monkeypatch):
    stop = asyncio.Event()
    attempts = []

    async def claim(limit):
        attempts.append(limit)
        if len(attempts) == 1:
            raise ConnectionError("private error")
        return [{"job_id": str(uuid.UUID(int=1))}]

    async def process(_job):
        stop.set()
        return True

    # Avoid a wall-clock delay while exercising the real poll loop.
    async def wait(awaitable, timeout):
        assert timeout >= 1.0
        if stop.is_set():
            return await awaitable
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(ingestion_queue, "claim_jobs", claim)
    monkeypatch.setattr(ingestion_queue, "process_claimed_job", process)
    monkeypatch.setattr(ingestion_queue.asyncio, "wait_for", wait)
    await ingestion_queue.run_worker_forever(stop_event=stop)
    assert attempts == [1, 1]


@pytest.mark.asyncio
async def test_stopped_worker_does_not_claim(monkeypatch):
    stop = asyncio.Event()
    stop.set()

    async def claim(**kwargs):
        pytest.fail("Stopped worker should not claim")

    monkeypatch.setattr(ingestion_queue, "claim_jobs", claim)
    await ingestion_queue.run_worker_forever(stop_event=stop)


@pytest.mark.asyncio
async def test_failed_attempt_uses_retry_rpc_without_unbounded_error_detail(monkeypatch):
    calls = []

    async def rpc(function, payload):
        calls.append((function, payload))
        return "queued"

    monkeypatch.setattr(ingestion_queue, "_rpc", rpc)
    await ingestion_queue.finish_job(uuid.UUID(int=1), succeeded=False)
    assert calls == [("fail_document_ingestion_job", {
        "p_job_id": str(uuid.UUID(int=1)), "p_error": "INGESTION_ATTEMPT_FAILED",
    })]


@pytest.mark.asyncio
async def test_worker_failure_status_waits_for_queue_retry_decision(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.document_service import DocumentService

    document_id, user_id = uuid.uuid4(), uuid.uuid4()
    document = SimpleNamespace(status="queued", id=document_id)
    monkeypatch.setattr(DocumentService, "get_document", AsyncMock(return_value=document))
    monkeypatch.setattr(DocumentService, "get_document_file", AsyncMock(side_effect=RuntimeError("parser failed")))
    db = SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock(), rollback=AsyncMock())
    assert await IngestionService.process_document(db, document_id, user_id, defer_failure_to_queue=True) is False
    assert document.status == "processing"
    assert document.processed_at is None
    db.rollback.assert_awaited_once()


def test_new_queue_grants_do_not_weaken_policies_or_service_only_rpc_access():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sql = (root / "supabase/migrations/20261005130000_grant_authenticated_ingestion_job_access.sql").read_text()
    assert "GRANT SELECT, INSERT, DELETE" in sql
    assert "REVOKE UPDATE" in sql
    assert "GRANT ALL" not in sql
    assert "CREATE POLICY" not in sql
    assert "DISABLE ROW LEVEL SECURITY" not in sql
    assert "GRANT EXECUTE" not in sql
