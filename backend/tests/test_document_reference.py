"""Document references through real HTTP/SSE, retrieval, citations and persistence; no paid APIs."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
from app.services.document_service import _IN_MEMORY_DOCUMENTS
from app.services.document_reference import DocumentReferenceResolver
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService
from app.tools.math_solver import MathSolver
from app.llm import MockLLMProvider, set_llm_provider
from tests.test_chat_api import (
    client, HEADERS_A, HEADERS_B, USER_A_ID, USER_B_ID,
    create_user_workspace_and_conversation, setup_test_environment,
)
from tests.test_math_chat import events


def document(workspace, name="Bài 6. Tranformer Ver2.pdf", *, user=USER_A_ID,
             status="processed", embedding_status="completed", content="Transformer attention combines token information."):
    now = datetime.now(timezone.utc)
    doc = Document(id=uuid.uuid4(), workspace_id=uuid.UUID(workspace), user_id=uuid.UUID(user),
                   original_filename=name, storage_path=str(uuid.uuid4()), mime_type="application/pdf",
                   file_size=100, status=status, embedding_status=embedding_status, created_at=now, updated_at=now)
    _IN_MEMORY_DOCUMENTS[doc.id] = doc
    if content is not None:
        _IN_MEMORY_CHUNKS[doc.id] = [DocumentChunk(
            id=uuid.uuid4(), document_id=doc.id, workspace_id=doc.workspace_id, user_id=doc.user_id,
            content=content, chunk_index=0, token_count=10, page_number_start=3, page_number_end=3,
            embedding=MockEmbeddingProvider().encode_batch([content])[0], created_at=now)]
    return doc


@pytest.fixture
def pipeline(monkeypatch):
    # Exercise the full retrieval pipeline with a deterministic high-confidence model double.
    monkeypatch.setattr(MockRerankerProvider, "predict", lambda self, pairs, **kw: [0.95] * len(pairs))
    original = RetrievalService.retrieve
    requests = []
    async def capture(**kwargs):
        requests.append(kwargs["request"])
        return await original(**kwargs)
    monkeypatch.setattr(RetrievalService, "retrieve", capture)
    monkeypatch.setattr("app.services.rag_service.get_web_search_provider",
                        lambda: pytest.fail("Resolved/ambiguous requests must not search web"))
    return requests


def chat(cid, content, **extra):
    response = client.post(f"/api/v1/conversations/{cid}/chat/sync", headers=HEADERS_A,
                           json={"content": content, **extra})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("query", ["tóm tắt file này", "tóm tắt tài liệu này", "file này nói về gì?",
    "summarize this document", "explain the content of this PDF", "summarize the uploaded file"])
def test_single_embedded_document(query, pipeline):
    ws, cid = create_user_workspace_and_conversation()
    doc = document(ws)
    result = chat(cid, query)
    assert pipeline[0].document_id == doc.id
    assert doc.original_filename in pipeline[0].query
    assert result["has_sufficient_evidence"]
    assert result["citations"][0]["document_id"] == str(doc.id)
    assert result["citations"][0]["page_start"] == 3


def test_explicit_filename_and_persisted_followup(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    first = document(ws, "Transformer.pdf")
    second = document(ws, "Attention.pdf")
    query = "Tóm tắt Transformer.pdf"
    chat(cid, query)
    assert pipeline[-1].query == query
    chat(cid, "Giải thích kỹ hơn phần 2 của tài liệu này")
    assert pipeline[-1].document_id == first.id
    chat(cid, "summarize this document", active_document_id=str(second.id))
    chat(cid, "explain this document")
    assert pipeline[-1].document_id == second.id
    # Another conversation in the same workspace cannot inherit this conversation's reference.
    other = client.post(f"/api/v1/workspaces/{ws}/conversations", headers=HEADERS_A,
                        json={"title": "Other thread"}).json()["id"]
    result = chat(other, "summarize this document")
    assert "Which document" in result["content"]


@pytest.mark.parametrize("field", ["active_document_id", "document_id"])
@pytest.mark.parametrize("multiple", [False, True])
def test_selected_id(pipeline, field, multiple):
    ws, cid = create_user_workspace_and_conversation()
    selected = document(ws, "Selected.pdf")
    if multiple:
        document(ws, "Other.pdf")
    chat(cid, "summarize this document", **{field: str(selected.id)})
    assert pipeline[-1].document_id == selected.id


def test_priority_attachment_viewer_filename_then_history(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    first, second = document(ws, "First.pdf"), document(ws, "Second.pdf")
    chat(cid, "summarize First.pdf", document_id=str(second.id), active_document_id=str(first.id))
    assert pipeline[-1].document_id == second.id
    chat(cid, "summarize Second.pdf", active_document_id=str(first.id))
    assert pipeline[-1].document_id == first.id
    chat(cid, "summarize Second.pdf")
    assert pipeline[-1].document_id == second.id


def test_ambiguous_clarification_streams_and_persists(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    document(ws, "Transformer.pdf")
    document(ws, "Attention.pdf")
    response = client.post(f"/api/v1/conversations/{cid}/chat", headers=HEADERS_A,
                           json={"content": "tóm tắt file này"})
    emitted = events(response.text)
    final = next(data for name, data in emitted if name == "done")
    assert any(name == "token" for name, _ in emitted)
    assert "Transformer.pdf" in final["content"] and "Attention.pdf" in final["content"]
    assert "Bạn muốn" in final["content"] and not pipeline
    history = client.get(f"/api/v1/conversations/{cid}/messages", headers=HEADERS_A).json()
    assert history[-1]["content"] == final["content"]


@pytest.mark.parametrize("endpoint", ["", "/sync"])
@pytest.mark.parametrize("field", ["document_id", "active_document_id"])
@pytest.mark.parametrize("other_user", [False, True])
def test_ids_cannot_cross_workspace_or_owner(endpoint, field, other_user, pipeline):
    ws, cid = create_user_workspace_and_conversation()
    other_ws, _ = create_user_workspace_and_conversation(HEADERS_B if other_user else HEADERS_A)
    foreign = document(other_ws, user=USER_B_ID if other_user else USER_A_ID)
    response = client.post(f"/api/v1/conversations/{cid}/chat{endpoint}", headers=HEADERS_A,
                           json={"content": "summarize this document", field: str(foreign.id)})
    assert response.status_code == 404
    assert not pipeline
    assert client.get(f"/api/v1/conversations/{cid}/messages", headers=HEADERS_A).json() == []


@pytest.mark.parametrize("status,embedding_status,content,expected", [
    ("processing", "pending", None, "processing"),
    ("queued", "pending", None, "queued"),
    ("processed", "processing", "text", "processing"),
    ("failed", "failed", None, "failed"),
    ("processed", "completed", None, "no usable"),
    ("processed", "completed", "  ", "no usable"),
])
def test_unready_state(status, embedding_status, content, expected, pipeline):
    ws, cid = create_user_workspace_and_conversation()
    document(ws, status=status, embedding_status=embedding_status, content=content)
    result = chat(cid, "summarize this document")
    assert expected in result["content"] and not pipeline
    assert "no uploaded" not in result["content"]


def test_unembedded_chunks_are_not_eligible(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    doc = document(ws)
    _IN_MEMORY_CHUNKS[doc.id][0].embedding = None
    assert "no usable" in chat(cid, "summarize this document")["content"]
    assert not pipeline


def test_normal_question_is_not_forced_to_one_document(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    document(ws)
    result = chat(cid, "Explain attention mechanisms")
    assert pipeline[-1].document_id is None
    assert pipeline[-1].query == "Explain attention mechanisms"
    assert result["has_sufficient_evidence"]


def test_unknown_filename_does_not_silently_choose_only_document(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    document(ws)
    assert "couldn't match" in chat(cid, "summarize Missing.pdf")["content"]
    assert not pipeline


def test_removed_history_document_does_not_switch_to_remaining_document(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    previous = document(ws, "Previous.pdf")
    document(ws, "Unrelated.pdf")
    chat(cid, "summarize Previous.pdf")
    del _IN_MEMORY_DOCUMENTS[previous.id]
    result = chat(cid, "explain this document")
    assert "no longer" in result["content"]
    assert len(pipeline) == 1


def test_only_eligible_document_ignores_other_users_and_processing_documents(pipeline):
    ws, cid = create_user_workspace_and_conversation()
    eligible = document(ws)
    document(ws, "Private.pdf", user=USER_B_ID)
    processing = document(ws, "Pending.pdf", status="processing", embedding_status="pending")
    chat(cid, "summarize this document")
    assert pipeline[-1].document_id == eligible.id
    result = chat(cid, "summarize this document", active_document_id=str(processing.id))
    assert "processing" in result["content"]
    assert len(pipeline) == 1


@pytest.mark.asyncio
async def test_sql_eligibility_batches_documents_and_excludes_unusable_content():
    ws, _ = create_user_workspace_and_conversation()
    first, second = document(ws), document(ws, "Other.pdf")
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.scalars.return_value.all.return_value = [first.id]
    result = await DocumentReferenceResolver.usable_ids(db, [first, second], first.workspace_id, first.user_id)
    assert result == {first.id} and db.execute.await_count == 1
    sql = str(db.execute.call_args.args[0])
    assert "document_chunks.workspace_id" in sql and "document_chunks.user_id" in sql
    assert "embedding IS NOT NULL" in sql and "trim(document_chunks.content)" in sql


def test_selected_document_math_and_citations_stream(monkeypatch, pipeline):
    ws, cid = create_user_workspace_and_conversation()
    doc = document(ws, "Exercises.pdf", content="Bài 3: Giải phương trình x^2 - 5*x + 6 = 0")
    async def solve(task, timeout_seconds):
        return MathSolver.solve(task)
    monkeypatch.setattr(MathSolver, "execute", solve)
    set_llm_provider(MockLLMProvider(tokens=["Factor the equation ", "[Source 1]."]))
    response = client.post(f"/api/v1/conversations/{cid}/chat", headers=HEADERS_A, json={
        "content": "giải bài 3 trong file này", "active_document_id": str(doc.id), "chat_mode": "Full Solution"})
    emitted = events(response.text)
    final = next(data for name, data in emitted if name == "done")
    assert final["solver"]["verified"] and final["solver"]["used"]
    assert "x ∈ {2, 3}" in final["content"]
    assert "[Doc: Exercises.pdf, p. 3]" in final["content"]
    assert final["citations"][0]["document_id"] == str(doc.id)
    assert len([1 for name, _ in emitted if name == "token"]) >= 3
    history = client.get(f"/api/v1/conversations/{cid}/messages", headers=HEADERS_A).json()
    assert history[-1]["citations"] == final["citations"]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["dense", "lexical"])
async def test_document_filter_applied_before_limit_in_memory(channel):
    ws, _ = create_user_workspace_and_conversation()
    doc = document(ws)
    document(ws, "Other.pdf")
    method = getattr(RetrievalService, f"{channel}_retrieve")
    args = {"query": "attention"} if channel == "lexical" else {
        "query_vector": _IN_MEMORY_CHUNKS[doc.id][0].embedding}
    result = await method(None, doc.workspace_id, doc.user_id, top_k=1, document_id=doc.id, **args)
    assert len(result) == 1 and result[0].document_id == doc.id


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["dense", "lexical"])
async def test_sql_channels_filter_workspace_owner_and_document(channel):
    db = AsyncMock()
    db.execute.return_value = MagicMock()
    db.execute.return_value.fetchall.return_value = []
    args = {"query": "attention"} if channel == "lexical" else {"query_vector": [1.0]}
    ws, user, doc_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await getattr(RetrievalService, f"{channel}_retrieve")(
        db, ws, user, top_k=1, document_id=doc_id, **args)
    for call in db.execute.call_args_list:
        sql, params = call.args
        assert "workspace_id = :workspace_id" in str(sql) and "user_id = :user_id" in str(sql)
        assert "document_id = CAST(:document_id AS uuid)" in str(sql)
        assert params["workspace_id"] == ws and params["user_id"] == user
        assert params["document_id"] == str(doc_id)
