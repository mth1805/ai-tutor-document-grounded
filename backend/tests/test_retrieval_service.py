"""Service-level tests for Phase 7 Hybrid Retrieval + Cross-Encoder Reranking."""
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.chunk import DocumentChunk
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService, RetrievalCandidate
from app.services.rag_service import RAGService
from app.schemas.retrieval import RetrievalRequest
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider


@pytest.fixture(autouse=True)
def clean_in_memory_chunks():
    _IN_MEMORY_CHUNKS.clear()
    yield
    _IN_MEMORY_CHUNKS.clear()


@pytest.mark.asyncio
async def test_dense_retrieval_tenant_filtering_and_top_k():
    """Verify dense retrieval respects workspace and user boundaries and limits top_k."""
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    emb_provider = MockEmbeddingProvider()
    vec_match = emb_provider.encode_batch(["machine learning algorithms"])[0]
    vec_other = emb_provider.encode_batch(["cooking recipes and baking"])[0]

    now = datetime.now(timezone.utc)
    # Workspace A, User A chunks
    chunks_a = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_a,
            workspace_id=ws_a,
            user_id=user_a,
            chunk_index=0,
            content="Machine learning algorithms build mathematical models.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            embedding=vec_match,
            created_at=now,
            updated_at=now,
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_a,
            workspace_id=ws_a,
            user_id=user_a,
            chunk_index=1,
            content="Baking bread requires flour, yeast, and warm water.",
            page_number_start=2,
            page_number_end=2,
            token_count=10,
            embedding=vec_other,
            created_at=now,
            updated_at=now,
        ),
    ]
    _IN_MEMORY_CHUNKS[doc_a] = chunks_a

    # Workspace B, User B chunks (should never be returned for user A)
    chunks_b = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_b,
            workspace_id=ws_b,
            user_id=user_b,
            chunk_index=0,
            content="Secret user B document on machine learning algorithms.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            embedding=vec_match,
            created_at=now,
            updated_at=now,
        )
    ]
    _IN_MEMORY_CHUNKS[doc_b] = chunks_b

    # Retrieve for Workspace A, User A
    results = await RetrievalService.dense_retrieve(
        db=None,
        workspace_id=ws_a,
        user_id=user_a,
        query_vector=vec_match,
        top_k=5,
    )

    assert len(results) == 2
    # The machine learning chunk should have highest cosine similarity (1.0)
    assert results[0].chunk_id == chunks_a[0].id
    assert results[0].dense_score is not None
    assert results[0].dense_score > results[1].dense_score
    # User B's chunk must never appear
    assert all(r.chunk_id != chunks_b[0].id for r in results)


@pytest.mark.asyncio
async def test_lexical_retrieval_keyword_matching_and_tenant_isolation():
    """Verify lexical retrieval matches search terms and isolates workspaces."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Photosynthesis occurs in the chloroplasts of plant cells.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            created_at=now,
            updated_at=now,
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=1,
            content="Cellular respiration converts glucose into ATP in mitochondria.",
            page_number_start=2,
            page_number_end=2,
            token_count=10,
            created_at=now,
            updated_at=now,
        ),
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    # Query for "chloroplasts photosynthesis"
    results = await RetrievalService.lexical_retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        query="chloroplasts photosynthesis",
        top_k=5,
    )

    assert len(results) >= 1
    assert results[0].chunk_id == chunks[0].id
    assert results[0].lexical_score is not None
    assert results[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_end_to_end_retrieval_pipeline():
    """Verify complete retrieval pipeline: dense + lexical -> RRF -> Cross-Encoder -> gate."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch(["quantum entanglement mechanics"])[0]

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Quantum entanglement is a phenomenon where particles remain connected.",
            page_number_start=5,
            page_number_end=5,
            token_count=12,
            embedding=vec,
            created_at=now,
            updated_at=now,
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=1,
            content="Classical mechanics describes the motion of macroscopic objects.",
            page_number_start=12,
            page_number_end=12,
            token_count=10,
            embedding=emb_provider.encode_batch(["classical mechanics physics"])[0],
            created_at=now,
            updated_at=now,
        ),
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    request = RetrievalRequest(
        query="quantum entanglement",
        dense_top_k=10,
        lexical_top_k=10,
        rrf_k=60,
        candidate_pool_size=10,
        rerank_top_k=5,
        relevance_threshold=0.35,
    )

    response = await RetrievalService.retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        request=request,
        embedding_provider=emb_provider,
        reranker_provider=MockRerankerProvider(),
    )

    assert response.workspace_id == ws_id
    assert response.query == "quantum entanglement"
    assert response.total_results == 2
    assert response.has_sufficient_evidence is True
    assert response.timings.total_retrieval_ms >= 0.0
    assert response.timings.query_embedding_ms >= 0.0

    # Top chunk should be quantum entanglement
    top_chunk = response.results[0]
    assert top_chunk.chunk_id == chunks[0].id
    assert top_chunk.final_rank == 1
    assert top_chunk.passed_relevance_gate is True
    assert top_chunk.rerank_score is not None
    assert top_chunk.page_number_start == 5
    assert top_chunk.page_number_end == 5


@pytest.mark.asyncio
async def test_graceful_degradation_when_dense_fails():
    """Verify that when dense retrieval encounters an error, lexical retrieval still returns results."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Special relativity was proposed by Albert Einstein.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            created_at=now,
            updated_at=now,
        )
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    request = RetrievalRequest(query="relativity Einstein")

    # Simulate dense retrieval failure
    with patch.object(
        RetrievalService,
        "dense_retrieve",
        side_effect=RuntimeError("pgvector connection dropped"),
    ):
        response = await RetrievalService.retrieve(
            db=None,
            workspace_id=ws_id,
            user_id=user_id,
            request=request,
            embedding_provider=MockEmbeddingProvider(),
            reranker_provider=MockRerankerProvider(),
        )

        assert response.total_results >= 1
        assert response.diagnostics is not None
        assert "dense_error" in response.diagnostics
        assert "pgvector connection dropped" in response.diagnostics["dense_error"]


@pytest.mark.asyncio
async def test_graceful_degradation_when_lexical_fails():
    """Verify that when lexical retrieval fails, dense retrieval continues."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch(["thermodynamics heat transfer"])[0]

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Thermodynamics deals with heat, work, and temperature.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            embedding=vec,
            created_at=now,
            updated_at=now,
        )
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    request = RetrievalRequest(query="thermodynamics heat")

    # Simulate lexical retrieval failure
    with patch.object(
        RetrievalService,
        "lexical_retrieve",
        side_effect=RuntimeError("tsvector syntax error"),
    ):
        response = await RetrievalService.retrieve(
            db=None,
            workspace_id=ws_id,
            user_id=user_id,
            request=request,
            embedding_provider=emb_provider,
            reranker_provider=MockRerankerProvider(),
        )

        assert response.total_results >= 1
        assert response.diagnostics is not None
        assert "lexical_error" in response.diagnostics
        assert "tsvector syntax error" in response.diagnostics["lexical_error"]


@pytest.mark.asyncio
async def test_dense_retrieval_sql_vector_parameter_binding():
    """Verify dense retrieval SQL uses CAST(:vec_literal AS vector) and binds vec_literal."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    query_vector = [0.1, 0.2, 0.3, 0.4]

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_db.execute.return_value = mock_result

    results = await RetrievalService.dense_retrieve(
        db=mock_db,
        workspace_id=ws_id,
        user_id=user_id,
        query_vector=query_vector,
        top_k=5,
    )

    assert results == []
    mock_db.execute.assert_awaited_once()

    stmt, params = mock_db.execute.await_args[0]
    # Verify the compiled SQL binds vec_literal cleanly without parse errors
    compiled_params = stmt.compile().params
    assert "vec_literal" in compiled_params, "vec_literal must be recognized as a bind parameter by SQLAlchemy"
    assert "workspace_id" in compiled_params
    assert "user_id" in compiled_params
    assert "top_k" in compiled_params

    # Verify that the passed params dictionary contains vec_literal
    assert "vec_literal" in params
    assert params["vec_literal"].startswith("[0.1,")
    assert params["workspace_id"] == ws_id
    assert params["user_id"] == user_id
    assert params["top_k"] == 5


@pytest.mark.asyncio
async def test_dense_retrieval_failure_rolls_back_and_does_not_poison_session():
    """Verify that an error in dense retrieval triggers db.rollback() and leaves the session clean."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    query_vector = [0.1, 0.2, 0.3, 0.4]

    mock_db = AsyncMock()
    mock_db.execute.side_effect = RuntimeError("syntax error at or near ':'")

    with pytest.raises(RuntimeError):
        await RetrievalService.dense_retrieve(
            db=mock_db,
            workspace_id=ws_id,
            user_id=user_id,
            query_vector=query_vector,
            top_k=5,
        )

    # Rollback must have been called on failure
    mock_db.rollback.assert_awaited()

    # Subsequent query on the session must now work
    mock_db.execute.side_effect = None
    mock_row = MagicMock()
    mock_row.id = doc_id
    mock_row.original_filename = "test_document.pdf"
    mock_res = MagicMock()
    mock_res.all.return_value = [mock_row]
    mock_db.execute.return_value = mock_res

    doc_names = await RAGService.resolve_document_names(mock_db, {doc_id}, user_id)
    assert doc_names[doc_id] == "test_document.pdf"


@pytest.mark.asyncio
async def test_resolve_document_names_handles_aborted_transaction_and_retries():
    """Verify that if the session has an aborted transaction, resolve_document_names rolls back and retries."""
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_db = AsyncMock()
    mock_row = MagicMock()
    mock_row.id = doc_id
    mock_row.original_filename = "document_recovered.pdf"
    mock_res = MagicMock()
    mock_res.all.return_value = [mock_row]

    # First execute fails with transaction aborted error, second execute succeeds after rollback
    mock_db.execute.side_effect = [
        RuntimeError("current transaction is aborted, commands ignored until end of transaction block"),
        mock_res,
    ]

    doc_names = await RAGService.resolve_document_names(mock_db, {doc_id}, user_id)
    assert doc_names[doc_id] == "document_recovered.pdf"
    mock_db.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_hybrid_retrieval_degrades_gracefully_with_db_dense_failure():
    """Verify hybrid retrieval handles dense failure against db session without unhandled exception."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            chunk_index=0,
            content="Neural networks and transformers learn deep representations.",
            page_number_start=1,
            page_number_end=1,
            token_count=10,
            created_at=now,
            updated_at=now,
        )
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks

    mock_db = AsyncMock()
    request = RetrievalRequest(query="neural networks")

    lex_cands = [
        RetrievalCandidate(
            chunk_id=chunks[0].id,
            document_id=doc_id,
            workspace_id=ws_id,
            user_id=user_id,
            content=chunks[0].content,
            page_number_start=1,
            page_number_end=1,
            chunk_index=0,
            lexical_score=0.85,
            retrieval_sources={"lexical"},
        )
    ]

    with patch.object(
        RetrievalService,
        "dense_retrieve",
        side_effect=RuntimeError("syntax error at or near ':'"),
    ), patch.object(
        RetrievalService,
        "lexical_retrieve",
        return_value=lex_cands,
    ):
        response = await RetrievalService.retrieve(
            db=mock_db,
            workspace_id=ws_id,
            user_id=user_id,
            request=request,
            embedding_provider=MockEmbeddingProvider(),
            reranker_provider=MockRerankerProvider(),
        )

        assert response.total_results >= 1
        assert response.diagnostics is not None
        assert "dense_error" in response.diagnostics
        assert "syntax error" in response.diagnostics["dense_error"]
        # Session rollback must have been called
        mock_db.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_normal_dense_and_lexical_retrieval_both_succeed():
    """Verify normal dense and lexical retrieval channels both contribute candidates."""
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    emb_provider = MockEmbeddingProvider()
    vec = emb_provider.encode_batch(["quantum mechanics physics"])[0]

    chunk_dense = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        chunk_index=0,
        content="Quantum mechanics physics explains subatomic particle behavior.",
        page_number_start=1,
        page_number_end=1,
        token_count=10,
        embedding=vec,
        created_at=now,
        updated_at=now,
    )
    chunk_lexical = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        chunk_index=1,
        content="Wave-particle duality is a fundamental principle of physics.",
        page_number_start=2,
        page_number_end=2,
        token_count=10,
        created_at=now,
        updated_at=now,
    )
    _IN_MEMORY_CHUNKS[doc_id] = [chunk_dense, chunk_lexical]

    request = RetrievalRequest(query="quantum mechanics physics")
    response = await RetrievalService.retrieve(
        db=None,
        workspace_id=ws_id,
        user_id=user_id,
        request=request,
        embedding_provider=emb_provider,
        reranker_provider=MockRerankerProvider(),
    )

    assert response.total_results >= 1
    assert response.has_sufficient_evidence is True
    # At least one result must have passed the relevance gate
    assert any(c.passed_relevance_gate for c in response.results)

