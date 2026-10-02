"""Unit tests for Relevance Gate after Cross-Encoder reranking."""
import uuid
from app.services.retrieval_service import RetrievalService, RetrievalCandidate


def _create_scored_candidate(rerank_score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content="Test content",
        page_number_start=1,
        page_number_end=1,
        chunk_index=0,
        rerank_score=rerank_score,
    )


def test_relevance_gate_passes_when_at_least_one_candidate_exceeds_threshold():
    """Verify that gate confirms sufficient evidence when candidates meet the threshold."""
    candidates = [
        _create_scored_candidate(0.85),
        _create_scored_candidate(0.40),
        _create_scored_candidate(0.20),
    ]
    threshold = 0.35
    gated, has_evidence = RetrievalService.apply_relevance_gate(candidates, threshold=threshold)

    assert has_evidence is True
    assert gated[0].passed_relevance_gate is True
    assert gated[1].passed_relevance_gate is True
    assert gated[2].passed_relevance_gate is False


def test_relevance_gate_rejects_when_all_candidates_below_threshold():
    """Verify that gate flags insufficient evidence when no candidate meets the threshold."""
    candidates = [
        _create_scored_candidate(0.30),
        _create_scored_candidate(0.25),
        _create_scored_candidate(0.12),
    ]
    threshold = 0.35
    gated, has_evidence = RetrievalService.apply_relevance_gate(candidates, threshold=threshold)

    assert has_evidence is False
    assert all(c.passed_relevance_gate is False for c in gated)


def test_relevance_gate_exact_boundary():
    """Verify that score exactly matching the threshold passes."""
    candidates = [_create_scored_candidate(0.35)]
    gated, has_evidence = RetrievalService.apply_relevance_gate(candidates, threshold=0.35)

    assert has_evidence is True
    assert gated[0].passed_relevance_gate is True


def test_relevance_gate_empty_candidates():
    """Verify that gate on empty candidates returns False without errors."""
    gated, has_evidence = RetrievalService.apply_relevance_gate([], threshold=0.35)
    assert has_evidence is False
    assert gated == []
