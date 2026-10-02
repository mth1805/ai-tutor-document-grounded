"""Integration tests for Phase 8 Grounded RAG Chat API and Streaming."""
import uuid
import json
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from app.core.config import settings

from app.main import app
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.document_service import _IN_MEMORY_DOCUMENTS
from app.services.workspace_service import _IN_MEMORY_WORKSPACES
from app.services.conversation_service import _IN_MEMORY_CONVERSATIONS, _IN_MEMORY_MESSAGES
from app.ml.loader import set_embedding_provider, set_reranker_provider
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
from app.llm import set_llm_provider, reset_llm_provider, MockLLMProvider, LLMProviderError

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}
HEADERS_B = {"Authorization": f"Bearer test-token:{USER_B_ID}"}


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure clean test doubles and in-memory stores for every chat test."""
    set_embedding_provider(MockEmbeddingProvider())
    set_reranker_provider(MockRerankerProvider())
    set_llm_provider(MockLLMProvider(
        default_response="Photosynthesis creates glucose and oxygen [Source 1].",
        tokens=["Photosynthesis ", "creates ", "glucose ", "and ", "oxygen ", "[Source 1]."],
    ))
    _IN_MEMORY_CHUNKS.clear()
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_WORKSPACES.clear()
    _IN_MEMORY_CONVERSATIONS.clear()
    _IN_MEMORY_MESSAGES.clear()
    yield
    set_embedding_provider(None)
    set_reranker_provider(None)
    reset_llm_provider()
    _IN_MEMORY_CHUNKS.clear()
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_WORKSPACES.clear()
    _IN_MEMORY_CONVERSATIONS.clear()
    _IN_MEMORY_MESSAGES.clear()


def create_user_workspace_and_conversation(user_headers=HEADERS_A):
    """Helper to setup a workspace and conversation thread."""
    ws_res = client.post("/api/v1/workspaces", headers=user_headers, json={"name": "Biology 101"})
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    conv_res = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers=user_headers,
        json={"title": "Photosynthesis Study"},
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]
    return ws_id, conv_id


def test_unauthenticated_chat_rejected():
    fake_conv = str(uuid.uuid4())
    res = client.post(f"/api/v1/conversations/{fake_conv}/chat", json={"content": "Hello"})
    assert res.status_code == 401


def test_chat_conversation_not_found_or_forbidden():
    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)

    # User B attempts to access User A's conversation
    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_B,
        json={"content": "What is photosynthesis?"},
    )
    assert res.status_code == 404


def test_chat_invalid_chat_mode():
    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Question", "chat_mode": "Invalid Mode Name"},
    )
    assert res.status_code == 422
    assert "Invalid chat mode" in res.text


def test_chat_mismatched_workspace():
    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)
    random_ws = str(uuid.uuid4())

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Question", "workspace_id": random_ws},
    )
    assert res.status_code == 400
    assert "workspace does not match" in res.text


def test_chat_insufficient_evidence_deterministic_fallback(monkeypatch):
    """Verify when workspace has no documents, Gemini is NOT called and fallback is returned."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", False)
    mock_llm = MockLLMProvider()
    set_llm_provider(mock_llm)

    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)

    # Ask question in empty workspace
    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Explain the Calvin cycle in detail."},
    )
    assert res.status_code == 200
    data = res.json()

    # Evidence flag is False
    assert data["has_sufficient_evidence"] is False
    assert "couldn't find enough information" in data["content"]
    assert data["citations"] == []

    # LLM was NOT called!
    assert mock_llm.call_count == 0

    # Assistant fallback message was persisted
    msg_res = client.get(f"/api/v1/conversations/{conv_id}/messages", headers=HEADERS_A)
    assert msg_res.status_code == 200
    messages = msg_res.json()
    assert len(messages) == 2  # user question + assistant fallback
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "couldn't find enough information" in messages[1]["content"]


def test_chat_sync_grounded_answer_and_citations():
    """Verify complete sync generation with document chunks, citations, and message persistence."""
    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)
    ws_uuid = uuid.UUID(ws_id)
    u_uuid = uuid.UUID(USER_A_ID)

    # 1. Register a document and chunk in memory
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        original_filename="Botany_Chapter_3.pdf",
        storage_path=f"{ws_id}/{doc_id}/Botany_Chapter_3.pdf",
        mime_type="application/pdf",
        file_size=1024,
        status="processed",
        embedding_status="completed",
        created_at=datetime.now(timezone.utc),
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    content_text = "Photosynthesis occurs in chloroplasts. Chlorophyll captures sunlight to produce glucose."
    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch([content_text])[0]

    chunk_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        chunk_index=0,
        content=content_text,
        page_number_start=7,
        page_number_end=8,
        token_count=20,
        embedding=vec,
        created_at=datetime.now(timezone.utc),
    )
    _IN_MEMORY_CHUNKS[doc_id] = [chunk]

    mock_llm = MockLLMProvider(
        default_response="Photosynthesis occurs in chloroplasts and generates glucose [Source 1].",
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={
            "content": "Photosynthesis occurs in chloroplasts sunlight glucose",
            "chat_mode": "Detailed Guidance",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is True
    # Verify [Source 1] was converted to [Doc: Botany_Chapter_3.pdf, pp. 7-8]
    assert "[Doc: Botany_Chapter_3.pdf, pp. 7-8]" in data["content"]
    assert len(data["citations"]) == 1
    citation = data["citations"][0]
    assert citation["document_name"] == "Botany_Chapter_3.pdf"
    assert citation["chunk_id"] == str(chunk_id)
    assert citation["page_start"] == 7
    assert citation["page_end"] == 8

    # Verify LLM was called with Detailed Guidance
    assert mock_llm.call_count == 1
    assert "DETAILED GUIDANCE" in mock_llm.last_system_instruction

    # Verify messages persisted in DB
    msg_res = client.get(f"/api/v1/conversations/{conv_id}/messages", headers=HEADERS_A)
    messages = msg_res.json()
    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["citations"]) == 1


def test_chat_streaming_sse():
    """Verify SSE streaming yields status, token, and done events properly."""
    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)
    ws_uuid = uuid.UUID(ws_id)
    u_uuid = uuid.UUID(USER_A_ID)

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        original_filename="Genetics.pdf",
        storage_path=f"{ws_id}/{doc_id}/Genetics.pdf",
        mime_type="application/pdf",
        file_size=2048,
        status="processed",
        embedding_status="completed",
        created_at=datetime.now(timezone.utc),
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    dna_content = "DNA is a double helix composed of adenine, thymine, cytosine, and guanine."
    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch([dna_content])[0]

    chunk_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        chunk_index=0,
        content=dna_content,
        page_number_start=1,
        page_number_end=1,
        token_count=18,
        embedding=vec,
        created_at=datetime.now(timezone.utc),
    )
    _IN_MEMORY_CHUNKS[doc_id] = [chunk]

    mock_llm = MockLLMProvider(
        tokens=["DNA ", "is ", "a ", "double ", "helix ", "[Source 1]."],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "DNA double helix adenine thymine", "chat_mode": "Full Solution"},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    body = res.text
    assert "event: status" in body
    assert "event: token" in body
    assert "event: done" in body

    # Parse done event payload
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_json = json.loads(body[data_idx + 6:data_end])

    assert done_json["has_sufficient_evidence"] is True
    assert "[Doc: Genetics.pdf, p. 1]" in done_json["content"]
    assert len(done_json["citations"]) == 1
    assert done_json["citations"][0]["document_name"] == "Genetics.pdf"


def test_chat_llm_provider_error_handling():
    """Verify provider error during streaming yields event: error gracefully."""
    ws_id, conv_id = create_user_workspace_and_conversation(HEADERS_A)
    ws_uuid = uuid.UUID(ws_id)
    u_uuid = uuid.UUID(USER_A_ID)

    doc_id = uuid.uuid4()
    _IN_MEMORY_DOCUMENTS[doc_id] = Document(
        id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        original_filename="sample.pdf",
        storage_path="path",
        mime_type="application/pdf",
        file_size=10,
        status="processed",
        embedding_status="completed",
        created_at=datetime.now(timezone.utc),
    )

    content_text = "Quantum computing algorithms for factorizing numbers."
    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch([content_text])[0]

    _IN_MEMORY_CHUNKS[doc_id] = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_uuid,
            user_id=u_uuid,
            chunk_index=0,
            content=content_text,
            page_number_start=1,
            page_number_end=1,
            token_count=8,
            embedding=vec,
            created_at=datetime.now(timezone.utc),
        )
    ]

    # Configure mock LLM to throw an LLMProviderError
    set_llm_provider(MockLLMProvider(simulate_error=LLMProviderError("Upstream service unavailable.")))

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "quantum computing algorithms"},
    )
    assert res.status_code == 200
    body = res.text
    assert "event: error" in body
    assert "Upstream service unavailable" in body
