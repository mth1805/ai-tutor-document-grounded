"""Local handoff authorization, idempotency and evidence-preserving windows."""
import asyncio
import uuid
import pytest
from fastapi import HTTPException
from app.api.v1 import local_models
from app.core.config import settings, Settings
from app.ml.text_windows import text_windows


def test_windowing_preserves_characters_and_bounds_tokens():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
    original = " tiếng Việt\n" * 200
    windows = text_windows(Tokenizer(), original, 30)
    assert "".join(windows) == original
    assert all(len(window) <= 30 for window in windows)


def test_shared_profile_rejected_outside_development():
    with pytest.raises(ValueError, match="restricted to development"):
        Settings(ENVIRONMENT="test", LOCAL_SHARED_MODELS=True)


def test_internal_token_never_equals_service_key(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "private-service-key")
    assert local_models.worker_token() != "private-service-key"
    with pytest.raises(HTTPException) as error:
        local_models.authorize("Bearer wrong")
    assert error.value.status_code == 401
    local_models.authorize("Bearer " + local_models.worker_token())


@pytest.mark.asyncio
async def test_stale_attempt_rejected_before_document_access(monkeypatch):
    async def load(_job):
        return {"claimed_attempt": 2}
    monkeypatch.setattr(local_models, "load_claimed_job", load)
    with pytest.raises(HTTPException) as error:
        await local_models.embed_attempt(local_models.EmbedAttempt(job_id=uuid.uuid4(), attempt=1))
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_duplicate_handoff_and_disconnect_share_one_task(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    count = 0
    async def process(request):
        nonlocal count
        count += 1
        started.set()
        await release.wait()
        return True
    monkeypatch.setattr(local_models, "embed_attempt", process)
    monkeypatch.setattr(local_models, "authorize", lambda _: None)
    local_models.ready.set()
    request = local_models.EmbedAttempt(job_id=uuid.uuid4(), attempt=1)
    try:
        first = asyncio.create_task(local_models.embed(request, "test"))
        await started.wait()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        second = asyncio.create_task(local_models.embed(request, "test"))
        await asyncio.sleep(0)
        release.set()
        assert (await second).succeeded
        assert count == 1
    finally:
        local_models.ready.clear()


@pytest.mark.asyncio
async def test_worker_local_handoff_failure_does_not_complete_job(monkeypatch):
    from app.workers import ingestion_queue
    from app.services.ingestion_service import IngestionService
    from unittest.mock import AsyncMock
    job_id = uuid.uuid4()
    job = {"claimed_attempt": 1, "claimed_document_id": str(uuid.uuid4()), "claimed_user_id": str(uuid.uuid4())}
    monkeypatch.setattr(settings, "LOCAL_SHARED_MODELS", True)
    monkeypatch.setattr(ingestion_queue, "load_claimed_job", AsyncMock(return_value=job))
    monkeypatch.setattr(IngestionService, "process_document_background", AsyncMock(return_value=True))
    monkeypatch.setattr(ingestion_queue, "request_local_embedding", AsyncMock(return_value=False))
    finish = AsyncMock()
    monkeypatch.setattr(ingestion_queue, "finish_job", finish)
    assert await ingestion_queue.process_claimed_job(job_id) is False
    finish.assert_awaited_once_with(job_id, succeeded=False)


@pytest.mark.asyncio
async def test_stale_embedding_does_not_commit_or_fail_new_attempt(monkeypatch):
    from app.services.embedding_service import EmbeddingService
    from app.services.document_service import DocumentService
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    doc = SimpleNamespace(status="processed", embedding_status="pending")
    monkeypatch.setattr(DocumentService, "get_document", AsyncMock(return_value=doc))
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    assert await EmbeddingService.embed_document(db, uuid.uuid4(), uuid.uuid4(), lease_guard=AsyncMock(return_value=False)) is False
    db.commit.assert_not_awaited()
    assert doc.embedding_status == "pending"


@pytest.mark.asyncio
async def test_embedding_during_warmup_never_loads_models_or_changes_document(monkeypatch):
    from app.services.embedding_service import EmbeddingService
    from app.services.document_service import DocumentService
    from unittest.mock import AsyncMock
    monkeypatch.setattr(settings, "LOCAL_SHARED_MODELS", True)
    local_models.ready.clear()
    fetch = AsyncMock()
    monkeypatch.setattr(DocumentService, "get_document", fetch)
    assert await EmbeddingService.embed_document(None, uuid.uuid4(), uuid.uuid4()) is False
    fetch.assert_not_awaited()
