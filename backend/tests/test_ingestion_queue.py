"""Focused tests for durable ingestion job handling without external services."""
import uuid
from types import SimpleNamespace
import pytest

from app.models.ingestion_job import DocumentIngestionJob
from app.services.ingestion_service import IngestionService
from app.workers import ingestion_queue


class FakeSession:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.deleted = []

    async def execute(self, _statement):
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
