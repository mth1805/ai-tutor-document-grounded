"""Deterministic unit tests for Reciprocal Rank Fusion (RRF)."""
import uuid
import pytest
from app.services.retrieval_service import RetrievalService, RetrievalCandidate


def _create_candidate(chunk_id: uuid.UUID, content: str, chunk_index: int) -> RetrievalCandidate:
    ws_id = uuid.uuid4()
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=doc_id,
        workspace_id=ws_id,
        user_id=user_id,
        content=content,
        page_number_start=1,
        page_number_end=1,
        chunk_index=chunk_index,
    )


def test_rrf_deterministic_scoring_and_ranking():
    """Verify RRF computes exact expected scores: RRF(d) = sum(1 / (k + rank))."""
    id_1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
    id_2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
    id_3 = uuid.UUID("33333333-3333-3333-3333-333333333333")
    id_4 = uuid.UUID("44444444-4444-4444-4444-444444444444")

    # Dense channel: id_1 (rank 1), id_2 (rank 2), id_4 (rank 3)
    c1_dense = _create_candidate(id_1, "Content 1", chunk_index=0)
    c1_dense.dense_score = 0.95
    c2_dense = _create_candidate(id_2, "Content 2", chunk_index=1)
    c2_dense.dense_score = 0.85
    c4_dense = _create_candidate(id_4, "Content 4", chunk_index=3)
    c4_dense.dense_score = 0.75
    dense_list = [c1_dense, c2_dense, c4_dense]

    # Lexical channel: id_1 (rank 1), id_4 (rank 2), id_3 (rank 3)
    c1_lex = _create_candidate(id_1, "Content 1", chunk_index=0)
    c1_lex.lexical_score = 0.90
    c4_lex = _create_candidate(id_4, "Content 4", chunk_index=3)
    c4_lex.lexical_score = 0.80
    c3_lex = _create_candidate(id_3, "Content 3", chunk_index=2)
    c3_lex.lexical_score = 0.70
    lexical_list = [c1_lex, c4_lex, c3_lex]

    k = 60
    fused = RetrievalService.reciprocal_rank_fusion(
        dense_candidates=dense_list,
        lexical_candidates=lexical_list,
        rrf_k=k,
        pool_size=10,
    )

    # All 4 unique chunks should be present
    assert len(fused) == 4
    fused_ids = [c.chunk_id for c in fused]

    # Expected calculations with k=60:
    # id_1: rank_dense=1, rank_lex=1 -> 1/61 + 1/61 = 2/61 ≈ 0.032787
    # id_4: rank_dense=3, rank_lex=2 -> 1/63 + 1/62 ≈ 0.015873 + 0.016129 = 0.032002
    # id_2: rank_dense=2, rank_lex=None -> 1/62 ≈ 0.016129
    # id_3: rank_dense=None, rank_lex=3 -> 1/63 ≈ 0.015873

    assert fused_ids[0] == id_1
    assert fused_ids[1] == id_4
    assert fused_ids[2] == id_2
    assert fused_ids[3] == id_3

    # Check scores match within 1e-5
    assert pytest.approx(fused[0].rrf_score, abs=1e-5) == 2.0 / 61.0
    assert pytest.approx(fused[1].rrf_score, abs=1e-5) == (1.0 / 63.0 + 1.0 / 62.0)
    assert pytest.approx(fused[2].rrf_score, abs=1e-5) == 1.0 / 62.0
    assert pytest.approx(fused[3].rrf_score, abs=1e-5) == 1.0 / 63.0


def test_rrf_deduplication_and_sources():
    """Verify that chunks appearing in both channels are deduplicated and sources preserved."""
    cid = uuid.uuid4()
    c_dense = _create_candidate(cid, "Shared content", chunk_index=0)
    c_dense.dense_score = 0.88
    c_lex = _create_candidate(cid, "Shared content", chunk_index=0)
    c_lex.lexical_score = 0.77

    fused = RetrievalService.reciprocal_rank_fusion(
        dense_candidates=[c_dense],
        lexical_candidates=[c_lex],
        rrf_k=60,
        pool_size=10,
    )

    assert len(fused) == 1
    chunk = fused[0]
    assert chunk.chunk_id == cid
    assert chunk.dense_score == 0.88
    assert chunk.lexical_score == 0.77
    assert chunk.retrieval_sources == {"dense", "lexical"}
    assert chunk.dense_rank == 1
    assert chunk.lexical_rank == 1


def test_rrf_candidate_pool_truncation():
    """Verify candidate_pool_size truncates candidates beyond the limit."""
    dense_items = [
        _create_candidate(uuid.uuid4(), f"Doc {i}", chunk_index=i)
        for i in range(10)
    ]
    lexical_items = [
        _create_candidate(uuid.uuid4(), f"Lex {i}", chunk_index=i + 10)
        for i in range(10)
    ]

    fused = RetrievalService.reciprocal_rank_fusion(
        dense_candidates=dense_items,
        lexical_candidates=lexical_items,
        rrf_k=60,
        pool_size=5,
    )

    assert len(fused) == 5


def test_rrf_single_channel_candidates():
    """Verify RRF handles cases where one channel returns zero candidates."""
    dense_items = [_create_candidate(uuid.uuid4(), "Dense only", chunk_index=0)]
    dense_items[0].dense_score = 0.9

    fused = RetrievalService.reciprocal_rank_fusion(
        dense_candidates=dense_items,
        lexical_candidates=[],
        rrf_k=60,
        pool_size=10,
    )

    assert len(fused) == 1
    assert fused[0].retrieval_sources == {"dense"}
    assert fused[0].lexical_score is None
    assert pytest.approx(fused[0].rrf_score, abs=1e-5) == 1.0 / 61.0
