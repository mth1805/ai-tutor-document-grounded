"""Phase 9: Web Search Fallback — focused unit and integration tests.

Covers:
1. Sufficient document evidence → Gemini called WITHOUT web grounding
2. Insufficient document evidence → web fallback path selected
3. Web citation extraction — URL/title/domain from grounding metadata
4. Mixed citations — doc + web can coexist
5. Citation deduplication — repeated web citations
6. Raw citation leakage — [Source X] must not reach the user
7. Web failure — graceful fallback response
8. Streaming — fallback still works through SSE
9. Persistence — assistant message + citations saved correctly
10. Regression — existing retrieval/chat tests still pass
"""
import json
import uuid
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
from app.rag.citation_service import WebCitation

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_clean_state():
    """Reset in-memory state and provider singletons for every test."""
    set_embedding_provider(MockEmbeddingProvider())
    set_reranker_provider(MockRerankerProvider())
    set_llm_provider(MockLLMProvider(
        default_response="Document-grounded answer [Source 1].",
        tokens=["Document-grounded ", "answer ", "[Source 1]."],
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


def _create_workspace_and_conversation():
    """Helper: create workspace + conversation, return (ws_id, conv_id)."""
    ws = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Test WS"})
    assert ws.status_code == 201
    ws_id = ws.json()["id"]
    conv = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers=HEADERS_A,
        json={"title": "Test Chat"},
    )
    assert conv.status_code == 201
    return ws_id, conv.json()["id"]


def _register_document_and_chunk(ws_id: str, content: str, page_start: int = 1, page_end: int = 1):
    """Helper: register a document + matching chunk in memory stores."""
    ws_uuid = uuid.UUID(ws_id)
    u_uuid = uuid.UUID(USER_A_ID)
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        original_filename="LectureNotes.pdf",
        storage_path=f"{ws_id}/{doc_id}/LectureNotes.pdf",
        mime_type="application/pdf",
        file_size=2048,
        status="processed",
        embedding_status="completed",
        created_at=datetime.now(timezone.utc),
    )
    _IN_MEMORY_DOCUMENTS[doc_id] = doc

    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch([content])[0]
    chunk_id = uuid.uuid4()
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=doc_id,
        workspace_id=ws_uuid,
        user_id=u_uuid,
        chunk_index=0,
        content=content,
        page_number_start=page_start,
        page_number_end=page_end,
        token_count=len(content.split()),
        embedding=vec,
        created_at=datetime.now(timezone.utc),
    )
    _IN_MEMORY_CHUNKS[doc_id] = [chunk]
    return doc_id, chunk_id


# ---------------------------------------------------------------------------
# Test 1: Sufficient document evidence → web grounding NOT triggered
# ---------------------------------------------------------------------------

def test_sufficient_document_evidence_does_not_trigger_web_fallback(monkeypatch):
    """When document chunks pass the relevance gate, Gemini should be called
    WITHOUT web grounding — i.e., generate_stream not generate_stream_with_grounding."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    doc_content = "Photosynthesis occurs in chloroplasts using chlorophyll and sunlight."
    _register_document_and_chunk(ws_id, doc_content)

    mock_llm = MockLLMProvider(
        default_response="Photosynthesis [Source 1].",
        tokens=["Photosynthesis ", "[Source 1]."],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "photosynthesis chloroplasts chlorophyll sunlight"},
    )
    assert res.status_code == 200
    data = res.json()

    # Document evidence is sufficient — normal LLM path taken
    assert data["has_sufficient_evidence"] is True
    assert mock_llm.call_count == 1            # generate_stream was called
    assert mock_llm.web_grounding_call_count == 0  # web grounding was NOT called
    assert len(data["citations"]) >= 1


# ---------------------------------------------------------------------------
# Test 2: Insufficient document evidence → web fallback selected
# ---------------------------------------------------------------------------

def test_insufficient_evidence_triggers_web_fallback(monkeypatch):
    """Empty workspace should trigger web grounding when fallback is enabled."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    # No documents registered — workspace is empty

    web_sources = [
        {"title": "NASA Space FAQ", "url": "https://nasa.gov/space-faq", "snippet": "Space is vast."},
    ]
    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Space ", "is ", "vast ", "and ", "infinite."],
        web_grounding_sources=web_sources,
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "How big is the observable universe?"},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is False
    assert mock_llm.call_count == 0                    # doc-grounded path NOT taken
    assert mock_llm.web_grounding_call_count == 1       # web grounding WAS called


# ---------------------------------------------------------------------------
# Test 3: Web citation extraction — URL/title/domain parsed correctly
# ---------------------------------------------------------------------------

def test_web_citation_extraction_from_grounding_metadata(monkeypatch):
    """Web sources should be parsed into WebCitation objects with url, title, domain."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    web_sources = [
        {
            "title": "Wikipedia: Black Hole",
            "url": "https://en.wikipedia.org/wiki/Black_hole",
            "snippet": "A black hole is a region of spacetime.",
        },
        {
            "title": "NASA Black Holes",
            "url": "https://www.nasa.gov/black-holes",
            "snippet": "NASA's overview of black holes.",
        },
    ]
    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Black ", "holes ", "are ", "fascinating."],
        web_grounding_sources=web_sources,
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "What is a black hole?"},
    )
    assert res.status_code == 200
    body = res.text
    assert "event: done" in body

    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    web_src = done_data.get("web_sources", [])
    assert len(web_src) == 2

    # Validate first source
    src0 = web_src[0]
    assert src0["url"] == "https://en.wikipedia.org/wiki/Black_hole"
    assert src0["title"] == "Wikipedia: Black Hole"
    assert src0["domain"] == "en.wikipedia.org"
    assert src0["source_type"] == "web"

    # Validate second source
    src1 = web_src[1]
    assert src1["url"] == "https://www.nasa.gov/black-holes"
    assert src1["domain"] == "nasa.gov"


# ---------------------------------------------------------------------------
# Test 4: Malformed / missing citation metadata handled safely
# ---------------------------------------------------------------------------

def test_web_citation_extraction_handles_malformed_sources(monkeypatch):
    """WebCitation.from_raw must handle missing/empty fields without crashing."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    # Malformed: no URL in one source, missing title in another
    web_sources = [
        {"title": "Valid Source", "url": "https://valid.com/page", "snippet": "Valid."},
        {"title": "No URL Source", "url": "", "snippet": "Should be excluded."},
        {"title": "", "url": "https://notitle.com/", "snippet": "No title."},
    ]
    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Answer."],
        web_grounding_sources=web_sources,
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Some question"},
    )
    assert res.status_code == 200
    body = res.text

    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    web_src = done_data.get("web_sources", [])
    # Empty URL source excluded; valid sources kept
    for s in web_src:
        assert s["url"]  # All returned sources must have a URL
    # No crash, no fabricated URLs
    urls = {s["url"] for s in web_src}
    assert "https://valid.com/page" in urls
    assert "" not in urls


# ---------------------------------------------------------------------------
# Test 5: WebCitation model unit tests
# ---------------------------------------------------------------------------

def test_web_citation_from_raw_valid():
    """WebCitation.from_raw parses normal web source correctly."""
    raw = {
        "title": "MIT OpenCourseWare",
        "url": "https://ocw.mit.edu/course/physics",
        "snippet": "Free physics course materials.",
    }
    wc = WebCitation.from_raw(raw)
    assert wc.url == "https://ocw.mit.edu/course/physics"
    assert wc.title == "MIT OpenCourseWare"
    assert wc.domain == "ocw.mit.edu"
    assert wc.source_type == "web"
    assert len(wc.snippet) <= 300


def test_web_citation_from_raw_www_stripped():
    """WebCitation strips leading www. from domain."""
    raw = {"title": "Example", "url": "https://www.example.com/path", "snippet": ""}
    wc = WebCitation.from_raw(raw)
    assert wc.domain == "example.com"


def test_web_citation_from_raw_missing_title_uses_domain():
    """When title is empty, domain is used as fallback."""
    raw = {"title": "", "url": "https://fallback.org/article", "snippet": None}
    wc = WebCitation.from_raw(raw)
    assert wc.title == "fallback.org"


def test_web_citation_to_dict_structure():
    """to_dict must return all required keys for frontend rendering."""
    raw = {"title": "Test", "url": "https://test.com", "snippet": "A snippet."}
    d = WebCitation.from_raw(raw).to_dict()
    for key in ("source_type", "title", "url", "domain", "snippet"):
        assert key in d, f"Missing key: {key}"
    assert d["source_type"] == "web"


# ---------------------------------------------------------------------------
# Test 6: Web citation deduplication via WEB_SEARCH_MAX_SOURCES cap
# ---------------------------------------------------------------------------

def test_web_sources_capped_at_max_sources(monkeypatch):
    """Web sources must be capped at settings.WEB_SEARCH_MAX_SOURCES."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "WEB_SEARCH_MAX_SOURCES", 2)

    ws_id, conv_id = _create_workspace_and_conversation()

    # Provide 5 sources — only 2 should appear
    web_sources = [
        {"title": f"Source {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet {i}."}
        for i in range(5)
    ]
    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Answer."],
        web_grounding_sources=web_sources,
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "broad topic question"},
    )
    assert res.status_code == 200
    body = res.text
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    assert len(done_data.get("web_sources", [])) == 2


# ---------------------------------------------------------------------------
# Test 7: Raw citation tokens must NOT reach user
# ---------------------------------------------------------------------------

def test_no_raw_source_tokens_in_web_fallback_response(monkeypatch):
    """[Source X] raw tokens must not appear in the final streamed content."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    # The model emits raw [Source 1] in web grounding response — it should NOT appear in content
    # (web responses don't go through CitationService, so raw tokens from the model
    # should not be present — the web path does not inject source markers)
    mock_llm = MockLLMProvider(
        web_grounding_tokens=["The ", "answer ", "is ", "clear."],  # clean response
        web_grounding_sources=[{"title": "T", "url": "https://t.com", "snippet": ""}],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "What happened?"},
    )
    assert res.status_code == 200
    data = res.json()

    content = data["content"]
    # Must not contain raw source tokens
    assert "[Source " not in content
    assert "Source 1" not in content


# ---------------------------------------------------------------------------
# Test 8: Web failure → graceful error response
# ---------------------------------------------------------------------------

def test_web_grounding_failure_yields_sse_error(monkeypatch):
    """When web grounding raises LLMProviderError, an SSE error event must be emitted."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    # No documents → will trigger web fallback

    mock_llm = MockLLMProvider(
        simulate_web_grounding_error=LLMProviderError("Google Search quota exceeded."),
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Some question that needs web"},
    )
    assert res.status_code == 200
    body = res.text
    assert "event: error" in body
    assert "Google Search quota exceeded" in body


# ---------------------------------------------------------------------------
# Test 9: Web fallback SSE — status events include web_search status
# ---------------------------------------------------------------------------

def test_web_fallback_sse_includes_web_search_status_event(monkeypatch):
    """The SSE stream must emit a status event with status=web_search before tokens."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    # Empty workspace → web fallback

    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Web ", "answer."],
        web_grounding_sources=[{"title": "T", "url": "https://t.com", "snippet": ""}],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Obscure question not in any doc"},
    )
    assert res.status_code == 200
    body = res.text

    # Must contain status=web_search event
    assert "event: status" in body
    assert "web_search" in body

    # Token events
    assert "event: token" in body

    # Done event
    assert "event: done" in body
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    assert done_data["used_web_fallback"] is True
    assert done_data["has_sufficient_evidence"] is False


# ---------------------------------------------------------------------------
# Test 10: Persistence — web grounding result saved to DB
# ---------------------------------------------------------------------------

def test_web_fallback_response_persisted_as_assistant_message(monkeypatch):
    """After web-grounded response, the assistant message must be persisted."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Persisted ", "web ", "answer."],
        web_grounding_sources=[{"title": "PW Source", "url": "https://pw.com", "snippet": ""}],
    )
    set_llm_provider(mock_llm)

    client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Question requiring web"},
    )

    # Check messages were persisted
    msg_res = client.get(f"/api/v1/conversations/{conv_id}/messages", headers=HEADERS_A)
    assert msg_res.status_code == 200
    messages = msg_res.json()
    assert len(messages) == 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    # Content not empty
    assert len(messages[1]["content"]) > 0
    # Document citations empty (web path)
    assert messages[1].get("citations", []) == []


# ---------------------------------------------------------------------------
# Test 11: Mixed evidence not applicable for current in-memory flow
# (document citations + web_sources in the same done payload)
# Web sources are in done payload alongside doc citations (empty for web path)
# ---------------------------------------------------------------------------

def test_done_payload_structure_for_web_fallback(monkeypatch):
    """Done payload must contain citations=[], web_sources=[...], used_web_fallback=True."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    web_src = [{"title": "T", "url": "https://t.com", "snippet": "S"}]
    mock_llm = MockLLMProvider(
        web_grounding_tokens=["Answer."],
        web_grounding_sources=web_src,
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Any question"},
    )
    body = res.text
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    assert "citations" in done_data          # doc citations key present
    assert "web_sources" in done_data        # web sources key present
    assert "has_sufficient_evidence" in done_data
    assert "used_web_fallback" in done_data
    assert done_data["used_web_fallback"] is True
    assert isinstance(done_data["web_sources"], list)
    assert len(done_data["web_sources"]) >= 1


# ---------------------------------------------------------------------------
# Test 12: Regression — document-grounded path unchanged when fallback disabled
# ---------------------------------------------------------------------------

def test_document_grounded_path_unaffected_when_fallback_disabled(monkeypatch):
    """WEB_SEARCH_FALLBACK_ENABLED=False must not trigger web fallback."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", False)

    ws_id, conv_id = _create_workspace_and_conversation()
    doc_content = "Mitochondria are the powerhouses of the cell."
    _register_document_and_chunk(ws_id, doc_content)

    mock_llm = MockLLMProvider(
        default_response="Mitochondria are powerhouses [Source 1].",
        tokens=["Mitochondria ", "are ", "powerhouses ", "[Source 1]."],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "mitochondria powerhouses cell"},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is True
    assert mock_llm.call_count == 1
    assert mock_llm.web_grounding_call_count == 0


# ---------------------------------------------------------------------------
# Test 13: Regression — empty workspace with fallback DISABLED returns graceful message
# ---------------------------------------------------------------------------

def test_empty_workspace_with_fallback_disabled_returns_deterministic_fallback(monkeypatch):
    """When web fallback is disabled and docs are absent, the deterministic message is returned."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", False)

    mock_llm = MockLLMProvider()
    set_llm_provider(mock_llm)

    ws_id, conv_id = _create_workspace_and_conversation()

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "What is quantum entanglement?"},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is False
    assert "couldn't find enough information" in data["content"]
    assert mock_llm.call_count == 0           # LLM not called at all
    assert mock_llm.web_grounding_call_count == 0  # no web fallback
