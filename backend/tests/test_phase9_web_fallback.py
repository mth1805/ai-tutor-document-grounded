"""Phase 9: Web Search Fallback — focused unit and integration tests.

Architecture change: Web fallback now uses Tavily (via WebSearchProvider abstraction)
instead of Gemini native Google Search grounding.

Test mapping to requirements:
 1. sufficient document evidence  → Tavily NOT called
 2. insufficient document evidence → Tavily called exactly once
 3. Tavily result parsing → title/url/content/domain preserved in WebCitation
 4. Gemini receives bounded web context (prompt contains [Web Source X])
 5. web citations persisted correctly in done payload
 6. duplicate web URLs deduplicated in WebCitation list
 7. raw [Source X] / [Web Source X] must never reach final visible answer
 8. web-search SSE status event emitted because Tavily path entered
 9. Tavily failure produces SSE error event
10. empty Tavily results produce graceful error event
11. existing document-only path unchanged
12. existing evidence-gate behavior unchanged
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
from app.web_search import set_web_search_provider, reset_web_search_provider
from app.web_search.mock_provider import MockWebSearchProvider
from app.web_search.exceptions import WebSearchAuthError, WebSearchTimeoutError, WebSearchRateLimitError

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
    # Default: no-op web search provider (won't be hit when docs are sufficient)
    set_web_search_provider(MockWebSearchProvider())
    _IN_MEMORY_CHUNKS.clear()
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_WORKSPACES.clear()
    _IN_MEMORY_CONVERSATIONS.clear()
    _IN_MEMORY_MESSAGES.clear()
    yield
    set_embedding_provider(None)
    set_reranker_provider(None)
    reset_llm_provider()
    reset_web_search_provider()
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
# Test 1: Sufficient document evidence → Tavily NOT called
# ---------------------------------------------------------------------------

def test_sufficient_document_evidence_does_not_trigger_web_fallback(monkeypatch):
    """When document chunks pass the relevance gate, Tavily must NOT be called."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    doc_content = "Photosynthesis occurs in chloroplasts using chlorophyll and sunlight."
    _register_document_and_chunk(ws_id, doc_content)

    mock_llm = MockLLMProvider(
        default_response="Photosynthesis [Source 1].",
        tokens=["Photosynthesis ", "[Source 1]."],
    )
    set_llm_provider(mock_llm)

    mock_web = MockWebSearchProvider()
    set_web_search_provider(mock_web)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "photosynthesis chloroplasts chlorophyll sunlight"},
    )
    assert res.status_code == 200
    data = res.json()

    # Document evidence is sufficient — Gemini document path taken
    assert data["has_sufficient_evidence"] is True
    assert mock_llm.call_count == 1        # generate_stream was called (doc path)
    assert mock_web.call_count == 0        # Tavily was NOT called
    assert len(data["citations"]) >= 1


# ---------------------------------------------------------------------------
# Test 2: Insufficient document evidence → Tavily called exactly once
# ---------------------------------------------------------------------------

def test_insufficient_evidence_triggers_tavily_web_fallback(monkeypatch):
    """Empty workspace should trigger Tavily when fallback is enabled."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    # No documents registered — workspace is empty

    mock_web = MockWebSearchProvider(results=[
        {"title": "NASA Space FAQ", "url": "https://nasa.gov/space-faq",
         "content": "Space is vast.", "domain": "nasa.gov"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(
        tokens=["Space ", "is ", "vast ", "and ", "infinite."],
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
    assert mock_web.call_count == 1         # Tavily WAS called exactly once
    assert mock_llm.call_count == 1         # generate_stream (doc-free, Tavily-grounded)


# ---------------------------------------------------------------------------
# Test 3: Tavily result parsing — title/url/content/domain preserved
# ---------------------------------------------------------------------------

def test_tavily_result_parsing_preserved_in_web_citations(monkeypatch):
    """Web sources from Tavily should be parsed into WebCitation objects with correct fields."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    mock_web = MockWebSearchProvider(results=[
        {
            "title": "Wikipedia: Black Hole",
            "url": "https://en.wikipedia.org/wiki/Black_hole",
            "content": "A black hole is a region of spacetime.",
            "domain": "en.wikipedia.org",
        },
        {
            "title": "NASA Black Holes",
            "url": "https://www.nasa.gov/black-holes",
            "content": "NASA's overview of black holes.",
            "domain": "nasa.gov",
        },
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Black ", "holes ", "are ", "fascinating."])
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
# Test 4: Gemini receives bounded web context with [Web Source X] labels
# ---------------------------------------------------------------------------

def test_gemini_receives_bounded_web_context(monkeypatch):
    """The prompt sent to Gemini must contain [Web Source X] context sections from Tavily results."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    mock_web = MockWebSearchProvider(results=[
        {"title": "Source Alpha", "url": "https://alpha.com/page",
         "content": "Alpha content.", "domain": "alpha.com"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Alpha ", "answer."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Tell me about Alpha."},
    )
    assert res.status_code == 200

    # The LLM's last prompt must contain the web context label
    assert mock_llm.last_prompt is not None
    assert "[Web Source 1]" in mock_llm.last_prompt
    assert "https://alpha.com/page" in mock_llm.last_prompt
    assert "Alpha content." in mock_llm.last_prompt
    # Must use WEB_EVIDENCE tag (not DOCUMENT_EVIDENCE)
    assert "<WEB_EVIDENCE>" in mock_llm.last_prompt


# ---------------------------------------------------------------------------
# Test 5: Web citations persisted correctly in done payload
# ---------------------------------------------------------------------------

def test_web_citations_persisted_correctly(monkeypatch):
    """After Tavily-grounded response, done payload must carry correct web_sources structure."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    web_src_raw = [{"title": "PW Source", "url": "https://pw.com", "content": "Info.", "domain": "pw.com"}]
    mock_web = MockWebSearchProvider(results=web_src_raw)
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Persisted ", "web ", "answer."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Question requiring web"},
    )
    assert res.status_code == 200
    body = res.text
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    assert done_data["used_web_fallback"] is True
    assert done_data["has_sufficient_evidence"] is False
    assert "citations" in done_data       # doc citations key present (empty)
    assert done_data["citations"] == []
    assert len(done_data["web_sources"]) == 1

    ws = done_data["web_sources"][0]
    assert ws["source_type"] == "web"
    assert ws["url"] == "https://pw.com"
    assert ws["title"] == "PW Source"
    assert ws["domain"] == "pw.com"


# ---------------------------------------------------------------------------
# Test 6: Duplicate web URLs deduplicated in WebCitation list
# ---------------------------------------------------------------------------

def test_duplicate_web_urls_deduplicated(monkeypatch):
    """Duplicate URLs from Tavily must appear only once in web_sources."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "WEB_SEARCH_MAX_SOURCES", 10)

    ws_id, conv_id = _create_workspace_and_conversation()

    # Tavily mock returns two entries with the same URL
    mock_web = MockWebSearchProvider(results=[
        {"title": "Dup Source", "url": "https://dup.com/page", "content": "Dup.", "domain": "dup.com"},
        {"title": "Dup Source 2", "url": "https://dup.com/page", "content": "Dup again.", "domain": "dup.com"},
        {"title": "Unique", "url": "https://unique.com", "content": "Unique.", "domain": "unique.com"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Answer."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Broad question"},
    )
    assert res.status_code == 200
    body = res.text
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    web_src = done_data.get("web_sources", [])
    urls = [s["url"] for s in web_src]
    # Duplicate URL appears only once
    assert urls.count("https://dup.com/page") == 1
    assert "https://unique.com" in urls


# ---------------------------------------------------------------------------
# Test 7: Raw [Source X] must NOT reach final answer text
# ---------------------------------------------------------------------------

def test_no_raw_source_tokens_in_web_fallback_response(monkeypatch):
    """[Source X] raw tokens must not appear in the final streamed content (web path)."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    mock_web = MockWebSearchProvider(results=[
        {"title": "T", "url": "https://t.com", "content": "Content.", "domain": "t.com"},
    ])
    set_web_search_provider(mock_web)

    # LLM returns clean prose — no Source markers expected
    mock_llm = MockLLMProvider(tokens=["The ", "answer ", "is ", "clear."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "What happened?"},
    )
    assert res.status_code == 200
    data = res.json()

    content = data["content"]
    assert "[Source " not in content
    assert "Source 1" not in content


# ---------------------------------------------------------------------------
# Test 8: web-search SSE status event emitted when Tavily path is entered
# ---------------------------------------------------------------------------

def test_web_search_sse_status_emitted_on_tavily_path(monkeypatch):
    """The SSE stream must emit status=web_search because the backend actually entered Tavily."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    # Empty workspace → web fallback

    mock_web = MockWebSearchProvider(results=[
        {"title": "T", "url": "https://t.com", "content": "Info.", "domain": "t.com"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Web ", "answer."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Obscure question not in any doc"},
    )
    assert res.status_code == 200
    body = res.text

    # Must contain status=web_search (emitted because Tavily was actually entered)
    assert "event: status" in body
    assert "web_search" in body
    assert mock_web.call_count == 1       # confirms the real Tavily path was taken

    # Token events
    assert "event: token" in body

    # Done event with correct flags
    assert "event: done" in body
    done_idx = body.find("event: done")
    data_idx = body.find("data: ", done_idx)
    data_end = body.find("\n\n", data_idx)
    done_data = json.loads(body[data_idx + 6:data_end])

    assert done_data["used_web_fallback"] is True
    assert done_data["has_sufficient_evidence"] is False


# ---------------------------------------------------------------------------
# Test 9: Tavily failure produces SSE error event
# ---------------------------------------------------------------------------

def test_tavily_failure_yields_sse_error(monkeypatch):
    """When Tavily raises WebSearchError, an SSE error event must be emitted."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()
    # No documents → will trigger Tavily fallback

    mock_web = MockWebSearchProvider(
        simulate_error=WebSearchAuthError("Tavily API key is invalid.")
    )
    set_web_search_provider(mock_web)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Some question that needs web"},
    )
    assert res.status_code == 200
    body = res.text
    assert "event: error" in body
    assert "Tavily API key is invalid" in body


# ---------------------------------------------------------------------------
# Test 10: Empty Tavily results produce graceful error event
# ---------------------------------------------------------------------------

def test_empty_tavily_results_produce_graceful_error(monkeypatch):
    """When Tavily returns zero results, an SSE error (not exception) must be emitted."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    # Provider returns empty list (valid call, no results)
    mock_web = MockWebSearchProvider(results=[])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider()
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat",
        headers=HEADERS_A,
        json={"content": "Absolutely obscure query with no web results"},
    )
    assert res.status_code == 200
    body = res.text
    # Must emit an error event — not generate an uncited Gemini answer
    assert "event: error" in body
    # LLM generate_stream must NOT be called (no content to ground it on)
    assert mock_llm.call_count == 0


# ---------------------------------------------------------------------------
# Test 11: Existing document-only path remains unchanged
# ---------------------------------------------------------------------------

def test_document_only_path_unchanged_when_fallback_disabled(monkeypatch):
    """WEB_SEARCH_FALLBACK_ENABLED=False must not trigger Tavily."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", False)

    ws_id, conv_id = _create_workspace_and_conversation()
    doc_content = "Mitochondria are the powerhouses of the cell."
    _register_document_and_chunk(ws_id, doc_content)

    mock_web = MockWebSearchProvider()
    set_web_search_provider(mock_web)

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
    assert mock_llm.call_count == 1      # doc-grounded path taken
    assert mock_web.call_count == 0      # Tavily NOT called


# ---------------------------------------------------------------------------
# Test 12: Existing evidence-gate behavior unchanged
# ---------------------------------------------------------------------------

def test_empty_workspace_with_fallback_disabled_returns_deterministic_fallback(monkeypatch):
    """When web fallback is disabled and docs are absent, deterministic message is returned."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", False)

    mock_llm = MockLLMProvider()
    set_llm_provider(mock_llm)
    mock_web = MockWebSearchProvider()
    set_web_search_provider(mock_web)

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
    assert mock_llm.call_count == 0        # LLM not called at all
    assert mock_web.call_count == 0        # Tavily not called


# ---------------------------------------------------------------------------
# WebCitation unit tests (these test the existing citation_service, unchanged)
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


def test_web_sources_capped_at_max_sources(monkeypatch):
    """Web sources must be capped at settings.WEB_SEARCH_MAX_SOURCES."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "WEB_SEARCH_MAX_SOURCES", 2)

    ws_id, conv_id = _create_workspace_and_conversation()

    # Provide 5 sources — only 2 should appear
    mock_web = MockWebSearchProvider(results=[
        {"title": f"Source {i}", "url": f"https://example.com/{i}",
         "content": f"Snippet {i}.", "domain": "example.com"}
        for i in range(5)
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Answer."])
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


def test_web_fallback_response_persisted_as_assistant_message(monkeypatch):
    """After Tavily-grounded response, the assistant message must be persisted."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)

    ws_id, conv_id = _create_workspace_and_conversation()

    mock_web = MockWebSearchProvider(results=[
        {"title": "PW Source", "url": "https://pw.com", "content": "Info.", "domain": "pw.com"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Persisted ", "web ", "answer."])
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
# Two-Tier Evidence Gate Regression Tests (MIN_ANSWERABLE_RERANK_SCORE)
# These tests verify that the routing decision uses BOTH tiers, and that
# Tavily is invoked on the right paths. Pattern identical to previous tests
# but using MockWebSearchProvider instead of generate_stream_with_grounding.
# ---------------------------------------------------------------------------


class FixedScoreRerankerProvider:
    """Test-only reranker that returns a constant score for all pairs."""

    def __init__(self, score: float):
        self._score = score

    def predict(self, pairs, batch_size=None):
        return [round(self._score, 4)] * len(pairs)


def test_tier1_pass_tier2_fail_triggers_tavily(monkeypatch):
    """Chunks passing Tier 1 (≥0.35) but failing Tier 2 (<0.55) must trigger Tavily."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MIN_ANSWERABLE_RERANK_SCORE", 0.55)
    monkeypatch.setattr(settings, "RETRIEVAL_ROUTING_MODE", "always_quality")
    set_reranker_provider(FixedScoreRerankerProvider(score=0.45))

    ws_id, conv_id = _create_workspace_and_conversation()
    _register_document_and_chunk(ws_id, "This document covers general API concepts.")

    mock_web = MockWebSearchProvider(results=[
        {"title": "Gemini News", "url": "https://news.example.com/gemini",
         "content": "Latest.", "domain": "news.example.com"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Latest ", "Gemini ", "API ", "news."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "What is the latest Gemini API news?"},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is False
    assert mock_web.call_count == 1     # Tavily WAS called (web fallback)
    assert mock_llm.call_count == 1     # generate_stream called (Tavily-grounded)


def test_tier1_pass_tier2_pass_uses_document_path(monkeypatch):
    """Chunks passing both tiers (≥0.55) must use the doc path — Tavily NOT called."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MIN_ANSWERABLE_RERANK_SCORE", 0.55)
    monkeypatch.setattr(settings, "RETRIEVAL_ROUTING_MODE", "always_quality")
    set_reranker_provider(FixedScoreRerankerProvider(score=0.80))

    ws_id, conv_id = _create_workspace_and_conversation()
    _register_document_and_chunk(ws_id, "Photosynthesis occurs in chloroplasts using chlorophyll and sunlight.")

    mock_web = MockWebSearchProvider()
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(
        default_response="Photosynthesis [Source 1].",
        tokens=["Photosynthesis ", "[Source 1]."],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Explain photosynthesis and chloroplasts."},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is True
    assert mock_llm.call_count == 1      # doc-grounded LLM called
    assert mock_web.call_count == 0      # Tavily NOT called


def test_weak_evidence_at_boundary_triggers_tavily(monkeypatch):
    """Score exactly at Tier 1 (0.35) but below Tier 2 (0.55) → Tavily fallback."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MIN_ANSWERABLE_RERANK_SCORE", 0.55)
    monkeypatch.setattr(settings, "RETRIEVAL_ROUTING_MODE", "always_quality")
    set_reranker_provider(FixedScoreRerankerProvider(score=0.35))

    ws_id, conv_id = _create_workspace_and_conversation()
    _register_document_and_chunk(ws_id, "Some tangentially related course content.")

    mock_web = MockWebSearchProvider(results=[
        {"title": "Web Source", "url": "https://web.example.com",
         "content": "Info.", "domain": "web.example.com"},
    ])
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(tokens=["Web ", "answer."])
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Explain the latest research on quantum computing."},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["has_sufficient_evidence"] is False
    assert mock_web.call_count == 1     # Tavily called


def test_min_answerable_threshold_is_the_routing_control(monkeypatch):
    """Lowering MIN_ANSWERABLE_RERANK_SCORE to 0.40 makes a 0.45-scoring chunk sufficient."""
    monkeypatch.setattr(settings, "WEB_SEARCH_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "MIN_ANSWERABLE_RERANK_SCORE", 0.40)
    monkeypatch.setattr(settings, "RETRIEVAL_ROUTING_MODE", "always_quality")
    set_reranker_provider(FixedScoreRerankerProvider(score=0.45))

    ws_id, conv_id = _create_workspace_and_conversation()
    _register_document_and_chunk(ws_id, "Lecture notes on algorithms and complexity.")

    mock_web = MockWebSearchProvider()
    set_web_search_provider(mock_web)

    mock_llm = MockLLMProvider(
        default_response="Algorithm answer [Source 1].",
        tokens=["Algorithm ", "answer ", "[Source 1]."],
    )
    set_llm_provider(mock_llm)

    res = client.post(
        f"/api/v1/conversations/{conv_id}/chat/sync",
        headers=HEADERS_A,
        json={"content": "Explain algorithmic complexity."},
    )
    assert res.status_code == 200
    data = res.json()

    # With threshold at 0.40, score 0.45 passes → doc path used
    assert data["has_sufficient_evidence"] is True
    assert mock_llm.call_count == 1
    assert mock_web.call_count == 0   # Tavily NOT called
