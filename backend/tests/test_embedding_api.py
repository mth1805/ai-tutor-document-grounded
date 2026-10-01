"""API tests for Phase 6 embedding endpoints, status checks, and pipeline integration."""
import uuid
import pytest
from fastapi.testclient import TestClient
import fitz

from app.main import app
from app.services.ingestion_service import IngestionService, _IN_MEMORY_CHUNKS
from app.services.document_service import _IN_MEMORY_DOCUMENTS
from app.services.embedding_service import EmbeddingService
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.ocr.mock import MockOCRProvider
from app.ml.loader import set_embedding_provider
from app.ml.mock_provider import MockEmbeddingProvider

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}
HEADERS_B = {"Authorization": f"Bearer test-token:{USER_B_ID}"}


@pytest.fixture(autouse=True)
def setup_test_providers():
    """Ensure mock embedding provider is used during tests."""
    mock_p = MockEmbeddingProvider()
    set_embedding_provider(mock_p)
    yield
    set_embedding_provider(None)


def _make_sample_pdf() -> bytes:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Dense embedding transforms text into 1024-dimensional semantic space.")
    b = doc.tobytes()
    doc.close()
    return b


@pytest.mark.asyncio
async def test_embed_api_success_and_status():
    """Verify POST /embed schedules embedding and GET /embedding-status returns progress."""
    # 1. User A creates a workspace
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Embedding API WS"})
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # 2. Upload PDF
    pdf_bytes = _make_sample_pdf()
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("vector_guide.pdf", pdf_bytes, "application/pdf")},
    )
    assert up_res.status_code == 201
    doc_id = up_res.json()["id"]

    # 3. Process document chunks (Phase 5)
    pipeline = IngestionPipeline(ocr_provider=MockOCRProvider())
    processed = await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
        pipeline=pipeline,
    )
    assert processed is True

    # 4. Trigger embedding via API (Phase 6)
    embed_res = client.post(f"/api/v1/documents/{doc_id}/embed", headers=HEADERS_A)
    assert embed_res.status_code == 202
    data = embed_res.json()
    assert data["document_id"] == doc_id
    assert data["embedding_status"] in ("processing", "completed")

    # 5. Check embedding status endpoint
    status_res = client.get(f"/api/v1/documents/{doc_id}/embedding-status", headers=HEADERS_A)
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["document_id"] == doc_id
    assert status_data["total_chunks"] >= 1
    assert status_data["embedding_model"] == "BAAI/bge-m3"

    # 6. Retrieve chunks and verify has_embedding is true
    chunks_res = client.get(f"/api/v1/documents/{doc_id}/chunks", headers=HEADERS_A)
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) >= 1
    assert chunks[0]["has_embedding"] is True
    assert chunks[0]["embedding_model"] == "BAAI/bge-m3"
    assert chunks[0]["embedded_at"] is not None


@pytest.mark.asyncio
async def test_embed_api_unprocessed_document_returns_400():
    """Verify that attempting to embed an unprocessed document returns 400 Bad Request."""
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Unprocessed WS"})
    ws_id = ws_res.json()["id"]

    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("raw.pdf", _make_sample_pdf(), "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    # Force status to 'uploaded'
    doc = _IN_MEMORY_DOCUMENTS.get(uuid.UUID(doc_id))
    if doc:
        doc.status = "uploaded"

    res = client.post(f"/api/v1/documents/{doc_id}/embed", headers=HEADERS_A)
    assert res.status_code == 400
    assert "must be in 'processed' state" in res.json()["detail"]


@pytest.mark.asyncio
async def test_embed_api_cross_user_denied():
    """Verify User B cannot embed or check status of User A's document."""
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Security WS"})
    ws_id = ws_res.json()["id"]

    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("secret.pdf", _make_sample_pdf(), "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    # User B attempts to embed User A's document
    res = client.post(f"/api/v1/documents/{doc_id}/embed", headers=HEADERS_B)
    assert res.status_code == 404

    # User B attempts to get embedding status
    res_status = client.get(f"/api/v1/documents/{doc_id}/embedding-status", headers=HEADERS_B)
    assert res_status.status_code == 404


@pytest.mark.asyncio
async def test_pipeline_auto_embed_after_ingestion():
    """Verify that IngestionService automatically embeds document when AUTO_EMBED_AFTER_INGESTION is True."""
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Auto Embed WS"})
    ws_id = ws_res.json()["id"]

    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("auto.pdf", _make_sample_pdf(), "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    # Execute ingestion (with default AUTO_EMBED_AFTER_INGESTION = True)
    pipeline = IngestionPipeline(ocr_provider=MockOCRProvider())
    processed = await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
        pipeline=pipeline,
    )
    assert processed is True

    # Verify document metadata has completed embedding status
    doc_res = client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS_A)
    assert doc_res.status_code == 200
    doc_data = doc_res.json()
    assert doc_data["status"] == "processed"
    assert doc_data["embedding_status"] == "completed"
    assert doc_data["embedded_at"] is not None
