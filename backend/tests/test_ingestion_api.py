"""API and integration tests for Phase 5 document ingestion, status tracking, and chunk endpoints."""
import io
import uuid
import pytest
from fastapi.testclient import TestClient
import fitz

from app.main import app
from app.services.ingestion_service import IngestionService, _IN_MEMORY_CHUNKS
from app.services.document_service import _IN_MEMORY_DOCUMENTS
from app.services.storage_service import _IN_MEMORY_STORAGE
from app.services.ingestion.ocr.mock import MockOCRProvider
from app.services.ingestion.pipeline import IngestionPipeline

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}
HEADERS_B = {"Authorization": f"Bearer test-token:{USER_B_ID}"}


def _make_sample_pdf() -> bytes:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Chapter 1: Grounded Retrieval.\n\nIn document-grounded AI tutors, documents are truth.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Chapter 2: Vector Search.\n\nEmbedding vectors map semantic content into high-dimensional space.")
    b = doc.tobytes()
    doc.close()
    return b


@pytest.mark.asyncio
async def test_document_ingestion_process_and_chunk_retrieval():
    """Verify upload -> process document -> chunk generation -> get chunks API."""
    # 1. User A creates a workspace
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "AI Systems"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # 2. Upload valid PDF
    pdf_bytes = _make_sample_pdf()
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("lecture.pdf", pdf_bytes, "application/pdf")},
    )
    assert up_res.status_code == 201
    doc_id = up_res.json()["id"]
    assert up_res.json()["status"] in ("uploaded", "processing", "processed")

    # 3. Asynchronously execute ingestion using test mock pipeline
    pipeline = IngestionPipeline(ocr_provider=MockOCRProvider())
    success = await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
        pipeline=pipeline,
    )
    assert success is True

    # 4. Check document metadata shows processed
    get_res = client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS_A)
    assert get_res.status_code == 200
    doc_data = get_res.json()
    assert doc_data["status"] == "processed"
    assert doc_data["processing_error"] is None
    assert doc_data["processed_at"] is not None

    # 5. Retrieve chunks via API
    chunks_res = client.get(f"/api/v1/documents/{doc_id}/chunks", headers=HEADERS_A)
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()
    assert len(chunks) >= 1

    # Verify chunk structure
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["document_id"] == doc_id
    assert chunks[0]["workspace_id"] == ws_id
    assert chunks[0]["user_id"] == USER_A_ID
    assert "Chapter 1" in chunks[0]["content"]
    assert chunks[0]["page_number_start"] == 1
    assert chunks[0]["page_number_end"] == 2



@pytest.mark.asyncio
async def test_idempotent_reprocessing_does_not_duplicate_chunks():
    """Verify that re-processing a document completely replaces old chunks."""
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Idempotency Test"},
    )
    ws_id = ws_res.json()["id"]

    pdf_bytes = _make_sample_pdf()
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("notes.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    pipeline = IngestionPipeline(ocr_provider=MockOCRProvider())

    # Run processing iteration 1
    await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
        pipeline=pipeline,
    )
    chunks_run1 = client.get(f"/api/v1/documents/{doc_id}/chunks", headers=HEADERS_A).json()
    count_run1 = len(chunks_run1)

    # Run processing iteration 2 (re-processing)
    await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
        pipeline=pipeline,
    )
    chunks_run2 = client.get(f"/api/v1/documents/{doc_id}/chunks", headers=HEADERS_A).json()
    count_run2 = len(chunks_run2)

    # Chunk count must be identical, not doubled!
    assert count_run1 == count_run2
    assert [c["chunk_index"] for c in chunks_run2] == list(range(count_run2))


def test_process_endpoint_returns_202_accepted():
    """Verify POST /documents/{id}/process schedules task and returns 202."""
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Async Task WS"},
    )
    ws_id = ws_res.json()["id"]

    pdf_bytes = _make_sample_pdf()
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("async.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    proc_res = client.post(f"/api/v1/documents/{doc_id}/process", headers=HEADERS_A)
    assert proc_res.status_code == 202
    data = proc_res.json()
    assert data["document_id"] == doc_id
    assert data["status"] == "processing"


def test_cross_user_isolation_for_chunks_and_processing():
    """Verify User B cannot access User A's chunks or trigger processing."""
    # User A creates workspace and document
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "User A Private"},
    )
    ws_id = ws_res.json()["id"]

    pdf_bytes = _make_sample_pdf()
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("secret.pdf", pdf_bytes, "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    # User B tries to view chunks
    b_chunks = client.get(f"/api/v1/documents/{doc_id}/chunks", headers=HEADERS_B)
    assert b_chunks.status_code == 404

    # User B tries to trigger re-processing
    b_proc = client.post(f"/api/v1/documents/{doc_id}/process", headers=HEADERS_B)
    assert b_proc.status_code == 404


@pytest.mark.asyncio
async def test_processing_failure_preserves_original_file():
    """Verify that if parsing fails, status=failed is recorded but original file remains in storage."""
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Failure WS"},
    )
    ws_id = ws_res.json()["id"]

    # Upload corrupted text file
    corrupt_content = b"%PDF-1.5 but malformed and truncated content that fails parsing"
    up_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("bad.pdf", corrupt_content, "application/pdf")},
    )
    doc_id = up_res.json()["id"]

    # Run processing with OCR unavailable
    pipeline = IngestionPipeline(ocr_provider=MockOCRProvider(available=False))
    success = await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
        pipeline=pipeline,
    )
    assert success is False

    # Check document status is failed with useful error
    doc_res = client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS_A)
    doc_data = doc_res.json()
    assert doc_data["status"] == "failed"
    assert doc_data["processing_error"] is not None
    assert len(doc_data["processing_error"]) > 0

    # Original file must still be downloadable!
    dl_res = client.get(f"/api/v1/documents/{doc_id}/download", headers=HEADERS_A)
    assert dl_res.status_code == 200
    assert dl_res.content == corrupt_content


@pytest.mark.asyncio
async def test_legacy_doc_without_libreoffice_is_marked_failed(monkeypatch):
    """A missing optional converter produces a failed status and retains the uploaded source."""
    monkeypatch.setattr("app.services.ingestion.parsers.doc.shutil.which", lambda _: None)
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Legacy DOC"})
    ws_id = ws_res.json()["id"]
    original = bytes.fromhex("D0CF11E0A1B11AE1") + b"legacy-content"
    upload = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("lesson.doc", original, "application/msword")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    success = await IngestionService.process_document(
        db=None,
        document_id=uuid.UUID(doc_id),
        user_id=uuid.UUID(USER_A_ID),
    )
    assert not success
    doc_data = client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS_A).json()
    assert doc_data["status"] == "failed"
    assert "requires LibreOffice" in doc_data["processing_error"]
    assert client.get(f"/api/v1/documents/{doc_id}/download", headers=HEADERS_A).content == original
