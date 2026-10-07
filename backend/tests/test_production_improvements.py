"""Local regression coverage; no production storage, data, or models are used."""
import io
import json
import unicodedata
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from docx import Document as WordDocument
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.document_service import DocumentService, _IN_MEMORY_DOCUMENTS
from app.services.storage_service import StorageService, sanitize_filename
from app.services.workspace_service import WorkspaceService
from app.models.ingestion_job import DocumentIngestionJob

NAMES = ["BÀI 5.pdf", "ĐỀ CƯƠNG.docx", "Học máy nâng cao.pdf",
         "Tài liệu NLP tiếng Việt.txt", "Chương 1 – Tổng quan.pdf"]
USER = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
HEADERS = {"Authorization": f"Bearer test-token:{USER}"}


def word_bytes():
    document = WordDocument()
    document.add_paragraph("Tài liệu tiếng Việt về học máy.")
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


def upload(name, content, mime):
    client = TestClient(app)
    workspace = client.post("/api/v1/workspaces", headers=HEADERS, json={"name": "Preview"}).json()["id"]
    result = client.post(f"/api/v1/workspaces/{workspace}/documents", headers=HEADERS,
                         files={"file": (name, content, mime)})
    assert result.status_code == 201
    return client, result.json()


@pytest.mark.parametrize("name", NAMES)
def test_unicode_filename_roundtrip(name):
    from urllib.parse import unquote
    extension = Path(name).suffix
    content = word_bytes() if extension == ".docx" else b"%PDF-1.5 test" if extension == ".pdf" else "Tài liệu".encode()
    mime = {".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pdf": "application/pdf", ".txt": "text/plain"}[extension]
    client, document = upload(name, content, mime)
    assert document["original_filename"] == name
    assert _IN_MEMORY_DOCUMENTS[uuid.UUID(document["id"])].original_filename == name
    assert document["storage_path"] == f'{document["workspace_id"]}/{document["id"]}/source{extension}'
    response = client.get(f'/api/v1/documents/{document["id"]}/download', headers=HEADERS)
    assert response.content == content
    assert unquote(response.headers["content-disposition"].split("filename*=UTF-8''")[1]) == name
    assert sanitize_filename(unicodedata.normalize("NFD", name)) == name


def test_word_preview_success_and_tenant_isolation(monkeypatch):
    from app.services import document_preview
    convert = lambda *args: b"%PDF-1.5 converted preview"
    monkeypatch.setattr(document_preview, "convert_word_to_pdf", convert)
    client, document = upload("ĐỀ CƯƠNG.docx", word_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    route = f'/api/v1/documents/{document["id"]}/preview'
    assert client.get(route).status_code == 401
    assert client.get(route, headers={"Authorization": f"Bearer test-token:{uuid.uuid4()}"}).status_code == 404
    response = client.get(route, headers=HEADERS)
    assert response.status_code == 200
    assert response.content == convert()
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["cache-control"] == "private, no-store"


def test_word_preview_text_fallback(monkeypatch):
    from app.services import document_preview
    def failure(*args):
        raise RuntimeError("private backend details")
    monkeypatch.setattr(document_preview, "convert_word_to_pdf", failure)
    client, document = upload("ĐỀ CƯƠNG.docx", word_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    response = client.get(f'/api/v1/documents/{document["id"]}/preview', headers=HEADERS)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "Tài liệu tiếng Việt" in response.text
    assert "private backend details" not in response.text


@pytest.mark.parametrize("name,content,mime", [
    ("BÀI 5.pdf", b"%PDF-1.5 test", "application/pdf"),
    ("image.png", b"\x89PNG\r\n\x1a\n" + bytes(30), "image/png"),
])
def test_pdf_and_image_preview_regression(name, content, mime):
    client, document = upload(name, content, mime)
    result = client.get(f'/api/v1/documents/{document["id"]}/preview', headers=HEADERS)
    assert result.status_code == 200
    assert result.content == content
    assert result.headers["content-type"] == mime


def test_libreoffice_conversion_isolated_profile_and_cleanup(monkeypatch):
    from app.services import document_preview
    roots = []
    monkeypatch.setattr(document_preview.shutil, "which", lambda _: "soffice")
    def run(command, **kwargs):
        root = Path(command[-1]).parent
        roots.append(root)
        assert command[-1].endswith("source.docx")
        assert any(arg.startswith("-env:UserInstallation=file:") for arg in command)
        assert kwargs["timeout"] == settings.LIBREOFFICE_TIMEOUT_SECONDS
        (root / "source.pdf").write_bytes(b"%PDF-1.5 preview")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(document_preview.subprocess, "run", run)
    assert document_preview.convert_word_to_pdf(word_bytes(), "ĐỀ CƯƠNG.docx").startswith(b"%PDF-")
    assert not roots[0].exists()


@pytest.mark.asyncio
async def test_storage_failure_never_creates_metadata_or_job(monkeypatch):
    monkeypatch.setattr(WorkspaceService, "get_workspace", AsyncMock(return_value=object()))
    monkeypatch.setattr(StorageService, "upload_file", AsyncMock(side_effect=HTTPException(502, "Storage unavailable")))
    db = SimpleNamespace(add=lambda _: pytest.fail("Metadata queued before storage succeeded"), commit=AsyncMock())
    with pytest.raises(HTTPException):
        await DocumentService.upload_document(db, uuid.uuid4(), USER, "BÀI 5.pdf", b"%PDF-1.5", "application/pdf")
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_and_database_commit_compensation(monkeypatch):
    monkeypatch.setattr(WorkspaceService, "get_workspace", AsyncMock(return_value=object()))
    monkeypatch.setattr(DocumentService, "get_document", AsyncMock(return_value=None))
    write = AsyncMock()
    delete = AsyncMock(return_value=True)
    monkeypatch.setattr(StorageService, "upload_file", write)
    monkeypatch.setattr(StorageService, "delete_file", delete)
    added = []
    db = SimpleNamespace(add=added.append, commit=AsyncMock(side_effect=RuntimeError()), rollback=AsyncMock(), refresh=AsyncMock())
    with pytest.raises(HTTPException):
        await DocumentService.upload_document(db, uuid.uuid4(), USER, "BÀI 5.pdf", b"%PDF-1.5", "application/pdf")
    assert len(added) == 2 and isinstance(added[1], DocumentIngestionJob)
    delete.assert_awaited_once_with(write.call_args.kwargs["storage_path"])
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_uncertain_commit_never_deletes_a_referenced_object(monkeypatch):
    monkeypatch.setattr(WorkspaceService, "get_workspace", AsyncMock(return_value=object()))
    monkeypatch.setattr(StorageService, "upload_file", AsyncMock())
    delete = AsyncMock()
    monkeypatch.setattr(StorageService, "delete_file", delete)
    lookup = AsyncMock(return_value=object())
    monkeypatch.setattr(DocumentService, "get_document", lookup)
    db = SimpleNamespace(add=lambda _: None, commit=AsyncMock(side_effect=ConnectionError()), rollback=AsyncMock())
    for unavailable in (False, True):
        if unavailable:
            lookup.side_effect = ConnectionError()
        with pytest.raises(HTTPException):
            await DocumentService.upload_document(db, uuid.uuid4(), USER, "test.pdf", b"%PDF-1.5", "application/pdf")
        delete.assert_not_awaited()


def test_real_local_pdf_docx_unicode_smoke(monkeypatch):
    import shutil
    import fitz
    executable = shutil.which("soffice")
    if not executable:
        pytest.skip("Local LibreOffice executable unavailable")
    # Use the discovered local tool without changing the user's environment file.
    monkeypatch.setattr(settings, "LIBREOFFICE_CMD", executable)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Document grounded learning. " * 8)
    pdf = document.tobytes()
    document.close()
    client, record = upload("Học máy nâng cao.pdf", pdf, "application/pdf")
    assert client.get(f'/api/v1/documents/{record["id"]}/preview', headers=HEADERS).content == pdf
    assert client.get(f'/api/v1/documents/{record["id"]}/download', headers=HEADERS).content == pdf
    original = word_bytes()
    client, record = upload("ĐỀ CƯƠNG.docx", original, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    preview = client.get(f'/api/v1/documents/{record["id"]}/preview', headers=HEADERS)
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "application/pdf"
    assert preview.content.startswith(b"%PDF-")
    assert client.get(f'/api/v1/documents/{record["id"]}/download', headers=HEADERS).content == original
    stored = _IN_MEMORY_DOCUMENTS[uuid.UUID(record["id"])]
    assert stored.status == "processed" and stored.embedding_status == "completed"


@pytest.mark.asyncio
async def test_successful_commit_and_postcommit_refresh_failure_keep_object(monkeypatch):
    monkeypatch.setattr(WorkspaceService, "get_workspace", AsyncMock(return_value=object()))
    write = AsyncMock()
    delete = AsyncMock()
    monkeypatch.setattr(StorageService, "upload_file", write)
    monkeypatch.setattr(StorageService, "delete_file", delete)
    added = []
    async def commit():
        write.assert_awaited_once()
        assert len(added) == 2
    db = SimpleNamespace(add=added.append, commit=commit, rollback=AsyncMock(), refresh=AsyncMock())
    document = await DocumentService.upload_document(db, uuid.uuid4(), USER, "BÀI 5.pdf", b"%PDF-1.5", "application/pdf")
    assert document.status == "queued"
    assert document.storage_path == write.call_args.kwargs["storage_path"]
    write.reset_mock()
    added.clear()
    db.refresh.side_effect = RuntimeError("refresh failed")
    with pytest.raises(RuntimeError):
        await DocumentService.upload_document(db, uuid.uuid4(), USER, "BÀI 5.pdf", b"%PDF-1.5", "application/pdf")
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_storage_encoding_missing_object_and_compensation(monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "private-test-key")
    calls = []
    path = f"{uuid.uuid4()}/{uuid.uuid4()}/BÀI #5%.pdf"
    def handle(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer private-test-key"
        if request.method == "POST":
            return httpx.Response(201, json={"Key": path})
        if request.method == "GET":
            return httpx.Response(200, content=b"%PDF-1.5")
        assert json.loads(request.content) == {"prefixes": [path]}
        return httpx.Response(200, json=[])
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(handle)))
    assert await StorageService.upload_file(path, b"%PDF-1.5", "application/pdf") == path
    assert await StorageService.get_file(path) == b"%PDF-1.5"
    assert await StorageService.delete_file(path)
    assert "authenticated/documents/" in str(calls[1].url)
    assert "B%C3%80I%20%235%25.pdf" in str(calls[1].url)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(lambda _: httpx.Response(400))))
    assert await StorageService.get_file(path) is None


@pytest.mark.asyncio
async def test_production_storage_does_not_use_memory_fallback(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    with pytest.raises(HTTPException) as error:
        await StorageService.upload_file("path", b"%PDF-1.5", "application/pdf")
    assert error.value.status_code == 503


def test_device_selection_and_cpu_fallback_logging(monkeypatch, caplog):
    import torch
    from app.ml.bge_provider import BGEEmbeddingProvider
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert BGEEmbeddingProvider._resolve_device("auto") == "cuda"
    assert BGEEmbeddingProvider._resolve_device("cuda") == "cuda"
    assert BGEEmbeddingProvider._resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert BGEEmbeddingProvider._resolve_device("auto") == "cpu"
    assert BGEEmbeddingProvider._resolve_device("cuda") == "cpu"
    assert "embedding_cpu_fallback" in caplog.text
    assert "falling back to CPU" in caplog.text


def test_modal_gpu_resource_and_cpu_function_configuration(monkeypatch):
    pytest.importorskip("modal")
    import deploy.modal_app as deployment
    assert deployment.INGESTION_GPU == "T4"
    assert deployment.process_ingestion_job._spec_.gpus == "T4"
    assert deployment.fastapi_app._spec_.gpus is None
    assert deployment.poll_ingestion_queue._spec_.gpus is None
    assert repr(deployment.process_ingestion_job._spec_.volumes["/models"]) == repr(deployment.MODEL_CACHE_VOLUME)


def test_bge_cuda_cache_and_warm_singleton_reuse(monkeypatch, tmp_path):
    import sys
    import torch
    from app.ml import loader
    calls = []
    class Model:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))
        def eval(self):
            pass
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "EMBEDDING_DEVICE", "cuda")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "LOCAL_SHARED_MODELS", False)
    monkeypatch.setattr(loader, "sys", SimpleNamespace(modules={}))
    loader.set_embedding_provider(None)
    first = loader.get_embedding_provider()
    assert first is loader.get_embedding_provider()
    assert first.device == "cuda"
    assert calls == [(settings.EMBEDDING_MODEL_NAME, {"cache_folder": str(tmp_path), "device": "cuda"})]


@pytest.mark.asyncio
async def test_correlated_worker_stages_and_exact_storage_path(monkeypatch, caplog):
    from app.workers import ingestion_queue
    from app.services.ingestion.telemetry import context
    client, document = upload("Tài liệu NLP tiếng Việt.txt", ("Học máy nâng cao. " * 100).encode(), "text/plain")
    job_id = uuid.uuid4()
    monkeypatch.setattr(ingestion_queue, "load_claimed_job", AsyncMock(return_value={
        "claimed_document_id": document["id"], "claimed_user_id": str(USER), "claimed_attempt": 1,
    }))
    finish = AsyncMock()
    monkeypatch.setattr(ingestion_queue, "finish_job", finish)
    get_file = AsyncMock(wraps=StorageService.get_file)
    monkeypatch.setattr(StorageService, "get_file", get_file)
    caplog.clear()
    caplog.set_level("INFO")
    assert await ingestion_queue.process_claimed_job(job_id)
    get_file.assert_awaited_once_with(document["storage_path"])
    entries = [json.loads(record.message) for record in caplog.records if record.message.startswith('{"event": "ingestion_stage"')]
    assert {"claimed", "storage_download", "parsing", "chunking", "model_load", "embedding", "ready"} <= {entry["stage"] for entry in entries}
    assert all(entry["job_id"] == str(job_id) and entry["document_id"] == document["id"] and entry["attempt"] == 1 for entry in entries)
    final = entries[-1]
    assert final["chunk_count"] > 0
    assert final["embedding_count"] == final["chunk_count"]
    assert final["embedding_batch_size"] == settings.EMBEDDING_BATCH_SIZE
    assert final["total_ingestion_ms"] > 0
    assert "gpu_device_name" in final
    assert context.get() is None


def test_corrupt_word_preview_fails_cleanly(monkeypatch):
    from app.services import document_preview
    def fail(*args):
        raise RuntimeError("internal provider details")
    monkeypatch.setattr(document_preview, "convert_word_to_pdf", fail)
    client, document = upload("corrupt.docx", b"PK\x03\x04broken", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    response = client.get(f'/api/v1/documents/{document["id"]}/preview', headers=HEADERS)
    assert response.status_code == 422
    assert "internal provider details" not in response.text


def test_existing_legacy_doc_preview_uses_persisted_text_on_conversion_failure(monkeypatch):
    from app.services import document_preview
    from app.services.ingestion_service import IngestionService, _IN_MEMORY_CHUNKS
    def fail(*args):
        raise RuntimeError("Converter unavailable")
    monkeypatch.setattr(document_preview, "convert_word_to_pdf", fail)
    monkeypatch.setattr(IngestionService, "process_document_background", AsyncMock(return_value=True))
    client, record = upload("ĐỀ CƯƠNG.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(32), "application/msword")
    _IN_MEMORY_CHUNKS[uuid.UUID(record["id"])] = [SimpleNamespace(content="Văn bản đã xử lý", chunk_index=0)]
    response = client.get(f'/api/v1/documents/{record["id"]}/preview', headers=HEADERS)
    assert response.status_code == 200
    assert response.text == "Văn bản đã xử lý"
    assert response.headers["content-type"].startswith("text/plain")


@pytest.mark.asyncio
async def test_missing_storage_object_fails_worker_cleanly(monkeypatch, caplog):
    from app.workers import ingestion_queue
    client, record = upload("BÀI 5.pdf", b"%PDF-1.5 original", "application/pdf")
    monkeypatch.setattr(StorageService, "get_file", AsyncMock(return_value=None))
    monkeypatch.setattr(ingestion_queue, "load_claimed_job", AsyncMock(return_value={
        "claimed_document_id": record["id"], "claimed_user_id": str(USER), "claimed_attempt": 1,
    }))
    finish = AsyncMock()
    monkeypatch.setattr(ingestion_queue, "finish_job", finish)
    caplog.set_level("INFO")
    job_id = uuid.uuid4()
    assert await ingestion_queue.process_claimed_job(job_id) is False
    finish.assert_awaited_once_with(job_id, succeeded=False)
    assert "STORAGE_OBJECT_UNAVAILABLE" in _IN_MEMORY_DOCUMENTS[uuid.UUID(record["id"])].processing_error
    assert "DOCUMENT_PROCESSING_FAILED" in caplog.text


@pytest.mark.asyncio
async def test_production_failure_metadata_never_exposes_provider_details(monkeypatch):
    from app.services.ingestion_service import IngestionService
    from app.services.embedding_service import EmbeddingService
    from app.ml.mock_provider import MockEmbeddingProvider
    client, record = upload("Tài liệu NLP tiếng Việt.txt", ("Tài liệu học máy. " * 100).encode(), "text/plain")
    document_id = uuid.UUID(record["id"])
    doc = _IN_MEMORY_DOCUMENTS[document_id]
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    private_detail = "PRIVATE_PROVIDER_SECRET_AND_INTERNAL_PATH"
    monkeypatch.setattr(DocumentService, "get_document_file", AsyncMock(side_effect=RuntimeError(private_detail)))
    assert await IngestionService.process_document(None, document_id, USER) is False
    assert private_detail not in doc.processing_error
    doc.status = "processed"
    provider = MockEmbeddingProvider()
    def fail(*args, **kwargs):
        raise RuntimeError(private_detail)
    monkeypatch.setattr(provider, "encode_batch", fail)
    assert await EmbeddingService.embed_document(None, document_id, USER, provider=provider) is False
    assert private_detail not in doc.embedding_error
