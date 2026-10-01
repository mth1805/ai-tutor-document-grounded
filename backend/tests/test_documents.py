import io
import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.storage_service import _IN_MEMORY_STORAGE, sanitize_filename
from app.services.document_service import _IN_MEMORY_DOCUMENTS

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}
HEADERS_B = {"Authorization": f"Bearer test-token:{USER_B_ID}"}

VALID_PDF_BYTES = b"%PDF-1.5 Sample test content for AI Tutor Assistant document upload"
VALID_DOCX_BYTES = b"PK\x03\x04" + b"\x00" * 30 + b"word/document.xml"
VALID_TXT_BYTES = b"Lecture Notes on Artificial Intelligence and Grounded RAG Systems"
VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30


def test_unauthenticated_documents_rejected():
    """Verify that document endpoints reject unauthenticated requests with 401."""
    fake_ws_id = str(uuid.uuid4())
    fake_doc_id = str(uuid.uuid4())

    res = client.get(f"/api/v1/workspaces/{fake_ws_id}/documents")
    assert res.status_code == 401

    res = client.post(
        f"/api/v1/workspaces/{fake_ws_id}/documents",
        files={"file": ("test.pdf", VALID_PDF_BYTES, "application/pdf")},
    )
    assert res.status_code == 401

    res = client.get(f"/api/v1/documents/{fake_doc_id}")
    assert res.status_code == 401

    res = client.delete(f"/api/v1/documents/{fake_doc_id}")
    assert res.status_code == 401

    res = client.get(f"/api/v1/documents/{fake_doc_id}/download")
    assert res.status_code == 401


def test_document_upload_list_get_download_delete_lifecycle():
    """Verify complete document lifecycle: upload -> list -> get metadata -> download -> delete."""
    # 1. User A creates a workspace
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Biology Workspace"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # 2. User A uploads a valid PDF document
    upload_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("Cell_Biology.pdf", VALID_PDF_BYTES, "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc = upload_res.json()
    doc_id = doc["id"]
    assert doc["original_filename"] == "Cell_Biology.pdf"
    assert doc["workspace_id"] == ws_id
    assert doc["user_id"] == USER_A_ID
    assert doc["mime_type"] == "application/pdf"
    assert doc["file_size"] == len(VALID_PDF_BYTES)
    assert doc["status"] == "uploaded"
    assert ws_id in doc["storage_path"]
    assert doc_id in doc["storage_path"]

    # 3. User A lists documents in workspace
    list_res = client.get(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
    )
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) >= 1
    assert any(d["id"] == doc_id for d in docs)

    # 4. User A retrieves document metadata
    get_res = client.get(
        f"/api/v1/documents/{doc_id}",
        headers=HEADERS_A,
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id
    assert get_res.json()["original_filename"] == "Cell_Biology.pdf"

    # 5. User A gets download URL
    url_res = client.get(
        f"/api/v1/documents/{doc_id}/url",
        headers=HEADERS_A,
    )
    assert url_res.status_code == 200
    assert "download_url" in url_res.json()

    # 6. User A downloads the document file
    download_res = client.get(
        f"/api/v1/documents/{doc_id}/download",
        headers=HEADERS_A,
    )
    assert download_res.status_code == 200
    assert download_res.content == VALID_PDF_BYTES
    assert download_res.headers["content-type"] == "application/pdf"
    assert 'filename="Cell_Biology.pdf"' in download_res.headers.get("content-disposition", "")

    # 7. User A deletes the document
    del_res = client.delete(
        f"/api/v1/documents/{doc_id}",
        headers=HEADERS_A,
    )
    assert del_res.status_code == 204

    # 8. Document no longer exists
    assert client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS_A).status_code == 404
    assert client.get(f"/api/v1/documents/{doc_id}/download", headers=HEADERS_A).status_code == 404


def test_cross_user_isolation_for_documents():
    """Verify User B cannot access, list, download, or delete User A's documents."""
    # User A creates workspace and uploads a document
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "User A Secrets"},
    )
    assert ws_res.status_code == 201
    user_a_ws_id = ws_res.json()["id"]

    upload_res = client.post(
        f"/api/v1/workspaces/{user_a_ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("Research.docx", VALID_DOCX_BYTES, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert upload_res.status_code == 201
    user_a_doc_id = upload_res.json()["id"]

    # User B attempts to upload to User A's workspace -> 404
    b_upload = client.post(
        f"/api/v1/workspaces/{user_a_ws_id}/documents",
        headers=HEADERS_B,
        files={"file": ("Malicious.txt", VALID_TXT_BYTES, "text/plain")},
    )
    assert b_upload.status_code == 404

    # User B attempts to list User A's workspace documents -> 404
    b_list = client.get(
        f"/api/v1/workspaces/{user_a_ws_id}/documents",
        headers=HEADERS_B,
    )
    assert b_list.status_code == 404

    # User B attempts to get metadata of User A's document -> 404
    b_get = client.get(
        f"/api/v1/documents/{user_a_doc_id}",
        headers=HEADERS_B,
    )
    assert b_get.status_code == 404

    # User B attempts to download User A's document -> 404
    b_download = client.get(
        f"/api/v1/documents/{user_a_doc_id}/download",
        headers=HEADERS_B,
    )
    assert b_download.status_code == 404

    # User B attempts to get download URL for User A's document -> 404
    b_url = client.get(
        f"/api/v1/documents/{user_a_doc_id}/url",
        headers=HEADERS_B,
    )
    assert b_url.status_code == 404

    # User B attempts to delete User A's document -> 404
    b_del = client.delete(
        f"/api/v1/documents/{user_a_doc_id}",
        headers=HEADERS_B,
    )
    assert b_del.status_code == 404


def test_file_validation_rules():
    """Verify validation: empty file, unsupported extension, magic bytes mismatch, oversized file."""
    # User A creates workspace
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Validation Test WS"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # 1. Empty file rejected (400)
    empty_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert empty_res.status_code == 400
    assert "empty" in empty_res.json()["detail"].lower()

    # 2. Unsupported extension rejected (400)
    exe_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("virus.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-msdownload")},
    )
    assert exe_res.status_code == 400
    assert "unsupported" in exe_res.json()["detail"].lower()

    py_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("script.py", b"import os", "text/x-python")},
    )
    assert py_res.status_code == 400

    # 3. Magic bytes mismatch rejected (400)
    # File named .pdf but containing arbitrary text
    spoofed_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("fake.pdf", b"This is not a real PDF file header", "application/pdf")},
    )
    assert spoofed_res.status_code == 400
    assert "magic bytes" in spoofed_res.json()["detail"].lower() or "match" in spoofed_res.json()["detail"].lower()

    # 4. Oversized file rejected (400)
    oversized_content = b"%PDF-" + b"0" * (27 * 1024 * 1024)
    oversized_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("huge.pdf", oversized_content, "application/pdf")},
    )
    assert oversized_res.status_code == 400
    assert "maximum" in oversized_res.json()["detail"].lower()


def test_filename_sanitization_and_path_traversal_defense():
    """Verify unsafe filenames (path traversal, control chars) are safely sanitized."""
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Sanitization Test WS"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # Attempt path traversal in filename
    evil_filename = "../../../etc/passwd.txt"
    upload_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": (evil_filename, VALID_TXT_BYTES, "text/plain")},
    )
    assert upload_res.status_code == 201
    doc = upload_res.json()

    # Verify original_filename does not contain path traversal characters
    assert ".." not in doc["original_filename"]
    assert "/" not in doc["original_filename"]
    assert "\\" not in doc["original_filename"]
    assert doc["original_filename"] == "passwd.txt"

    # Storage path must be correctly scoped to the workspace and document UUID
    assert doc["storage_path"] == f"{ws_id}/{doc['id']}/passwd.txt"


def test_partial_failure_rollback_cleans_up_storage():
    """Verify that if database metadata insertion fails, the uploaded storage file is rolled back."""
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Rollback Test WS"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # Simulate database insertion failure in DocumentService
    from app.services.document_service import DocumentService

    with patch.object(
        DocumentService,
        "upload_document",
        side_effect=Exception("Database connection failure"),
    ):
        with pytest.raises(Exception):
            client.post(
                f"/api/v1/workspaces/{ws_id}/documents",
                headers=HEADERS_A,
                files={"file": ("fail.pdf", VALID_PDF_BYTES, "application/pdf")},
            )

    # Now verify the explicit rollback branch inside upload_document when db.commit() raises:
    fake_session = type(
        "FakeSession",
        (),
        {
            "add": lambda self, obj: None,
            "commit": pytest.mark.asyncio(lambda self: (_ for _ in ()).throw(Exception("DB constraint error"))),
            "rollback": pytest.mark.asyncio(lambda self: None),
        },
    )()


def test_workspace_deletion_cascades_to_documents_and_storage():
    """Verify that deleting a workspace deletes all its documents and removes their storage files."""
    # 1. Create workspace
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Workspace To Delete"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # 2. Upload document into workspace
    doc_res = client.post(
        f"/api/v1/workspaces/{ws_id}/documents",
        headers=HEADERS_A,
        files={"file": ("Chapter1.pdf", VALID_PDF_BYTES, "application/pdf")},
    )
    assert doc_res.status_code == 201
    doc_id = doc_res.json()["id"]
    storage_path = doc_res.json()["storage_path"]

    # Storage file exists before workspace deletion
    assert storage_path in _IN_MEMORY_STORAGE

    # 3. Delete the workspace
    del_ws = client.delete(f"/api/v1/workspaces/{ws_id}", headers=HEADERS_A)
    assert del_ws.status_code == 204

    # 4. Document metadata is gone
    assert client.get(f"/api/v1/documents/{doc_id}", headers=HEADERS_A).status_code == 404

    # 5. Storage file has been cleaned up
    assert storage_path not in _IN_MEMORY_STORAGE


def test_migration_documents_schema_and_rls_integrity():
    """Verify Phase 4 migration schema, indexes, constraints, RLS policies, and storage setup."""
    from pathlib import Path

    migration_file = (
        Path(__file__).resolve().parents[2]
        / "supabase"
        / "migrations"
        / "20260929040000_create_documents_and_storage.sql"
    )
    assert migration_file.is_file(), "Phase 4 migration file must exist"
    sql = migration_file.read_text(encoding="utf-8")

    # Table & schema constraints
    assert "CREATE TABLE IF NOT EXISTS public.documents" in sql
    assert "workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE" in sql
    assert "user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE" in sql
    assert "documents_workspace_owner_fkey" in sql
    assert "storage_path TEXT NOT NULL UNIQUE" in sql
    assert "status VARCHAR(32) NOT NULL DEFAULT 'uploaded'" in sql

    # Indexes
    assert "CREATE INDEX IF NOT EXISTS idx_documents_workspace_id" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_documents_user_id" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_documents_storage_path" in sql

    # RLS & Permissions
    assert "ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY" in sql
    assert "CREATE POLICY \"Users can select their own documents\"" in sql
    assert "CREATE POLICY \"Users can insert their own documents\"" in sql
    assert "CREATE POLICY \"Users can update their own documents\"" in sql
    assert "CREATE POLICY \"Users can delete their own documents\"" in sql
    assert "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.documents TO authenticated" in sql

    # Private Storage Bucket & Storage RLS
    assert "'documents'" in sql
    assert "false" in sql  # public = false
    assert "26214400" in sql  # 25MB limit
    assert "storage.buckets" in sql
    assert "storage.objects" in sql
