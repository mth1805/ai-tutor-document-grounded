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


# ---------------------------------------------------------------------------
# Two-Tier Gate Unit Tests
# The RAGService applies MIN_ANSWERABLE_RERANK_SCORE on top of the existing
# has_sufficient_evidence flag. These tests verify the score extraction logic
# and edge cases at the boundary between the two thresholds.
# ---------------------------------------------------------------------------

def test_tier1_zone_has_evidence_true_but_top_score_below_answerable():
    """Candidates scoring in the Tier-1 zone [0.35, 0.55) have has_evidence=True
    but top_rerank_score < MIN_ANSWERABLE_RERANK_SCORE.

    The apply_relevance_gate result (Tier 1) is unchanged; the RAGService
    must then apply Tier 2 separately. This test verifies the gate output
    that the RAGService receives when chunks are topically adjacent only.
    """
    # Score 0.45: passes 0.35 gate (Tier 1), falls below 0.55 answerable bar (Tier 2)
    candidates = [
        _create_scored_candidate(0.45),
        _create_scored_candidate(0.38),
    ]
    gated, has_evidence = RetrievalService.apply_relevance_gate(candidates, threshold=0.35)

    assert has_evidence is True                        # Tier 1 sees evidence
    assert gated[0].passed_relevance_gate is True
    assert gated[1].passed_relevance_gate is True

    # Tier 2 check (as done in rag_service.py)
    top_rerank_score = max(
        (c.rerank_score for c in gated if c.rerank_score is not None), default=0.0
    )
    min_answerable = 0.55
    evidence_is_answerable = has_evidence and top_rerank_score >= min_answerable

    assert top_rerank_score == 0.45
    assert evidence_is_answerable is False             # Tier 2 rejects → web fallback correct


def test_tier2_passes_when_top_score_meets_answerable_threshold():
    """Candidates with top score ≥ 0.55 pass both tiers → document path correct."""
    candidates = [
        _create_scored_candidate(0.82),
        _create_scored_candidate(0.41),
    ]
    gated, has_evidence = RetrievalService.apply_relevance_gate(candidates, threshold=0.35)

    top_rerank_score = max(
        (c.rerank_score for c in gated if c.rerank_score is not None), default=0.0
    )
    min_answerable = 0.55
    evidence_is_answerable = has_evidence and top_rerank_score >= min_answerable

    assert has_evidence is True
    assert top_rerank_score == 0.82
    assert evidence_is_answerable is True              # Both tiers pass → doc path correct


def test_top_rerank_score_uses_best_candidate_not_first():
    """top_rerank_score must be the maximum across all candidates, not the first."""
    # Candidates in non-sorted order to verify max() is used
    candidates = [
        _create_scored_candidate(0.40),
        _create_scored_candidate(0.72),  # highest, but not first
        _create_scored_candidate(0.50),
    ]
    gated, has_evidence = RetrievalService.apply_relevance_gate(candidates, threshold=0.35)

    top_rerank_score = max(
        (c.rerank_score for c in gated if c.rerank_score is not None), default=0.0
    )
    assert top_rerank_score == 0.72                    # must pick the highest, not the first
    assert has_evidence is True

