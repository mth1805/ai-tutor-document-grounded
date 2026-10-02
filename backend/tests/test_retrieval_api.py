"""API tests for Phase 7 Hybrid Retrieval and Cross-Encoder search endpoint."""
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.document_service import _IN_MEMORY_DOCUMENTS
from app.services.workspace_service import _IN_MEMORY_WORKSPACES
from app.models.chunk import DocumentChunk
from app.ml.loader import set_embedding_provider, set_reranker_provider
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
from datetime import datetime, timezone

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}
HEADERS_B = {"Authorization": f"Bearer test-token:{USER_B_ID}"}


@pytest.fixture(autouse=True)
def setup_test_providers_and_state():
    """Ensure mock ML providers and clean memory stores for every API test."""
    set_embedding_provider(MockEmbeddingProvider())
    set_reranker_provider(MockRerankerProvider())
    _IN_MEMORY_CHUNKS.clear()
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_WORKSPACES.clear()
    yield
    set_embedding_provider(None)
    set_reranker_provider(None)
    _IN_MEMORY_CHUNKS.clear()
    _IN_MEMORY_DOCUMENTS.clear()
    _IN_MEMORY_WORKSPACES.clear()


def test_retrieval_search_endpoint_unauthorized():
    """Verify endpoint rejects requests without valid bearer authorization."""
    random_ws = str(uuid.uuid4())
    res = client.post(
        f"/api/v1/workspaces/{random_ws}/retrieval/search",
        json={"query": "test query"},
    )
    assert res.status_code == 401


def test_retrieval_search_workspace_not_found_or_forbidden():
    """Verify 404 is returned when workspace does not exist or belongs to another user."""
    # User A creates a workspace
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "User A Workspace"})
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # User B attempts to search User A's workspace
    res = client.post(
        f"/api/v1/workspaces/{ws_id}/retrieval/search",
        headers=HEADERS_B,
        json={"query": "machine learning"},
    )
    assert res.status_code == 404
    assert "Workspace not found" in res.json()["detail"]


def test_retrieval_search_empty_query_rejected():
    """Verify validation rejects empty or whitespace-only search queries."""
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Workspace Test"})
    ws_id = ws_res.json()["id"]

    res = client.post(
        f"/api/v1/workspaces/{ws_id}/retrieval/search",
        headers=HEADERS_A,
        json={"query": "   "},
    )
    assert res.status_code in (400, 422)


def test_retrieval_search_success_with_instrumentation_and_provenance():
    """Verify end-to-end retrieval API returns ranked chunks, timing metrics, and provenance."""
    # 1. User A creates a workspace
    ws_res = client.post("/api/v1/workspaces", headers=HEADERS_A, json={"name": "Physics Workspace"})
    assert ws_res.status_code == 201
    ws_id = uuid.UUID(ws_res.json()["id"])
    user_id = uuid.UUID(USER_A_ID)
    doc_id = uuid.uuid4()

    # 2. Seed chunk with embedding in memory
    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch(["Newton's laws of classical motion"])[0]
    now = datetime.now(timezone.utc)

    chunk = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        chunk_index=0,
        content="Newton's second law states that force equals mass times acceleration (F=ma).",
        page_number_start=3,
        page_number_end=3,
        token_count=15,
        embedding=vec,
        created_at=now,
        updated_at=now,
    )
    _IN_MEMORY_CHUNKS[doc_id] = [chunk]

    # 3. Perform retrieval search
    res = client.post(
        f"/api/v1/workspaces/{ws_id}/retrieval/search",
        headers=HEADERS_A,
        json={
            "query": "Newton's laws of motion acceleration",
            "dense_top_k": 10,
            "lexical_top_k": 10,
            "rrf_k": 60,
            "candidate_pool_size": 10,
            "rerank_top_k": 5,
            "relevance_threshold": 0.35,
        },
    )

    assert res.status_code == 200
    data = res.json()

    assert data["workspace_id"] == str(ws_id)
    assert data["query"] == "Newton's laws of motion acceleration"
    assert data["total_results"] == 1
    assert data["has_sufficient_evidence"] is True
    assert data["relevance_threshold"] == 0.35

    # Check timing metrics
    timings = data["timings"]
    assert "query_embedding_ms" in timings
    assert "dense_retrieval_ms" in timings
    assert "lexical_retrieval_ms" in timings
    assert "rrf_ms" in timings
    assert "rerank_ms" in timings
    assert "total_retrieval_ms" in timings
    assert timings["total_retrieval_ms"] >= 0.0

    # Check chunk payload and provenance
    chunk_res = data["results"][0]
    assert chunk_res["chunk_id"] == str(chunk.id)
    assert chunk_res["document_id"] == str(doc_id)
    assert chunk_res["page_number_start"] == 3
    assert chunk_res["page_number_end"] == 3
    assert chunk_res["chunk_index"] == 0
    assert chunk_res["final_rank"] == 1
    assert chunk_res["passed_relevance_gate"] is True
    assert chunk_res["rerank_score"] is not None
    assert "dense" in chunk_res["retrieval_sources"]

    # Verify NO raw embeddings or sensitive internals leaked
    assert "embedding" not in chunk_res
    assert "raw_vector" not in chunk_res
    assert "storage_path" not in chunk_res
