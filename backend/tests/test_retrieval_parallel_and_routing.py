"""Tests for Parallel Dense + Lexical Retrieval and Fast/Quality Path Routing."""
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.chunk import DocumentChunk
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.services.retrieval_service import RetrievalService
from app.schemas.retrieval import RetrievalRequest, RetrievalRoutingMode
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
from app.rag.router import RetrievalRouter, RoutingDecision


@pytest.fixture(autouse=True)
def clean_in_memory_chunks():
    _IN_MEMORY_CHUNKS.clear()
    yield
    _IN_MEMORY_CHUNKS.clear()


@pytest.fixture
def sample_corpus():
    workspace_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    emb_provider = MockEmbeddingProvider()
    texts = [
        "Machine learning algorithms optimize numerical objective functions.",
        "Deep neural networks require backpropagation and gradient descent.",
        "Photosynthesis converts light energy into chemical energy in plant cells.",
    ]
    vectors = emb_provider.encode_batch(texts)

    chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            workspace_id=workspace_id,
            user_id=user_id,
            chunk_index=i,
            content=text,
            page_number_start=1,
            page_number_end=1,
            token_count=12,
            embedding=vec,
            created_at=now,
            updated_at=now,
        )
        for i, (text, vec) in enumerate(zip(texts, vectors))
    ]
    _IN_MEMORY_CHUNKS[doc_id] = chunks
    return workspace_id, user_id, doc_id, chunks, emb_provider


@pytest.mark.asyncio
async def test_parallel_retrieval_both_channels_succeed(sample_corpus):
    """Verify both dense and lexical channels execute concurrently and return combined RRF results."""
    workspace_id, user_id, doc_id, chunks, emb_provider = sample_corpus
    rerank_mock = MagicMock(spec=MockRerankerProvider())
    rerank_mock.predict.return_value = [0.85, 0.75]


    req = RetrievalRequest(
        query="machine learning optimization",
        dense_top_k=5,
        lexical_top_k=5,
        candidate_pool_size=10,
        rerank_top_k=3,
        routing_mode="always_quality",
    )

    resp = await RetrievalService.retrieve(
        db=None,
        workspace_id=workspace_id,
        user_id=user_id,
        request=req,
        embedding_provider=emb_provider,
        reranker_provider=rerank_mock,
    )

    assert len(resp.results) > 0
    assert resp.timings.dense_retrieval_ms >= 0.0
    assert resp.timings.lexical_retrieval_ms >= 0.0
    assert resp.timings.total_retrieval_ms > 0.0
    assert resp.routing_path == "QUALITY"
    assert rerank_mock.predict.called


@pytest.mark.asyncio
async def test_parallel_retrieval_channel_failure_degradation(sample_corpus):
    """Verify that if one retrieval channel raises an error, the pipeline degrades gracefully."""
    workspace_id, user_id, doc_id, chunks, emb_provider = sample_corpus
    rerank_provider = MockRerankerProvider()

    req = RetrievalRequest(
        query="photosynthesis light",
        dense_top_k=5,
        lexical_top_k=5,
        candidate_pool_size=10,
        rerank_top_k=3,
        routing_mode="always_fast",
    )

    # Patch dense_retrieve to throw an exception
    with patch.object(
        RetrievalService, "dense_retrieve", side_effect=RuntimeError("Dense DB connection timeout")
    ):
        resp = await RetrievalService.retrieve(
            db=None,
            workspace_id=workspace_id,
            user_id=user_id,
            request=req,
            embedding_provider=emb_provider,
            reranker_provider=rerank_provider,
        )

        # Lexical should still succeed and find the photosynthesis chunk
        assert len(resp.results) > 0
        assert resp.results[0].chunk_id == chunks[2].id
        assert resp.routing_path == "FAST"


@pytest.mark.asyncio
async def test_always_fast_routing_skips_cross_encoder(sample_corpus):
    """Verify always_fast routing completely skips Cross-Encoder reranking."""
    workspace_id, user_id, doc_id, chunks, emb_provider = sample_corpus
    rerank_mock = MagicMock(spec=MockRerankerProvider())

    req = RetrievalRequest(
        query="machine learning",
        routing_mode="always_fast",
    )

    resp = await RetrievalService.retrieve(
        db=None,
        workspace_id=workspace_id,
        user_id=user_id,
        request=req,
        embedding_provider=emb_provider,
        reranker_provider=rerank_mock,
    )

    # Reranker should NEVER be called in always_fast mode
    rerank_mock.predict.assert_not_called()
    assert resp.routing_path == "FAST"
    assert resp.timings.rerank_ms == 0.0
    assert resp.diagnostics["routing_mode"] == "always_fast"
    assert resp.diagnostics["routing_path"] == "FAST"


@pytest.mark.asyncio
async def test_always_quality_routing_invokes_cross_encoder(sample_corpus):
    """Verify always_quality routing always invokes Cross-Encoder reranking."""
    workspace_id, user_id, doc_id, chunks, emb_provider = sample_corpus
    rerank_mock = MagicMock(spec=MockRerankerProvider())
    rerank_mock.predict.return_value = [0.92, 0.40, 0.10]

    req = RetrievalRequest(
        query="machine learning",
        routing_mode="always_quality",
    )

    resp = await RetrievalService.retrieve(
        db=None,
        workspace_id=workspace_id,
        user_id=user_id,
        request=req,
        embedding_provider=emb_provider,
        reranker_provider=rerank_mock,
    )

    assert rerank_mock.predict.called
    assert resp.routing_path == "QUALITY"
    assert resp.diagnostics["routing_mode"] == "always_quality"
    assert resp.diagnostics["routing_path"] == "QUALITY"



@pytest.mark.asyncio
async def test_adaptive_routing_logic():
    """Unit test for RetrievalRouter.decide logic on high-confidence vs low-confidence scenarios."""
    from app.schemas.retrieval import RetrievedChunk

    def make_res(score: float, sources=None):
        return RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="test chunk",
            page_number_start=1,
            page_number_end=1,
            chunk_index=0,
            rrf_score=score,
            final_rank=1,
            passed_relevance_gate=True,
            retrieval_sources=sources or ["dense", "lexical"],
        )

    # Scenario 1: Empty candidates -> FAST (no candidates to rerank)
    d1 = RetrievalRouter.decide([], mode=RetrievalRoutingMode.ADAPTIVE)
    assert d1.path == "FAST"

    # Scenario 2: Single candidate (strong -> FAST, weak -> QUALITY)
    d2_strong = RetrievalRouter.decide([make_res(0.0328)], mode=RetrievalRoutingMode.ADAPTIVE)
    assert d2_strong.path == "FAST"
    d2_weak = RetrievalRouter.decide([make_res(0.01)], mode=RetrievalRoutingMode.ADAPTIVE)
    assert d2_weak.path == "QUALITY"


    # Scenario 3: High score, dual consensus, and clear gap -> FAST
    d3 = RetrievalRouter.decide(
        [make_res(0.032, ["dense", "lexical"]), make_res(0.016, ["dense"]), make_res(0.015, ["lexical"])],
        mode=RetrievalRoutingMode.ADAPTIVE,
        rrf_score_threshold=0.025,
        score_gap_threshold=0.003,
    )
    assert d3.path == "FAST"
    assert d3.top_rrf_score == 0.032
    assert d3.score_gap == 0.016

    # Scenario 4: Ambiguous candidates with small gap -> QUALITY
    d4 = RetrievalRouter.decide(
        [make_res(0.026), make_res(0.025), make_res(0.024)],
        mode=RetrievalRoutingMode.ADAPTIVE,
        rrf_score_threshold=0.025,
        score_gap_threshold=0.003,
    )
    assert d4.path == "QUALITY"

    # Scenario 5: Low top score -> QUALITY
    d5 = RetrievalRouter.decide(
        [make_res(0.018), make_res(0.010)],
        mode=RetrievalRoutingMode.ADAPTIVE,
        rrf_score_threshold=0.025,
        score_gap_threshold=0.003,
    )
    assert d5.path == "QUALITY"

    # Scenario 6: Calibrated settings defaults
    d6 = RetrievalRouter.decide(
        [make_res(0.0328, ["dense", "lexical"]), make_res(0.0320, ["dense", "lexical"])],
        mode=RetrievalRoutingMode.ADAPTIVE,
    )
    assert d6.path == "FAST"


