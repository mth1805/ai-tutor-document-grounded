"""Tests for Cross-Encoder reranker provider, singleton loader, and lifecycle."""
import os
import pytest
from app.ml.reranker_base import BaseRerankerProvider
from app.ml.reranker_mock import MockRerankerProvider
from app.ml.loader import (
    get_reranker_provider,
    set_reranker_provider,
    warmup_reranker_model,
)


def test_mock_reranker_provider_interface():
    """Verify MockRerankerProvider satisfies BaseRerankerProvider interface."""
    provider = MockRerankerProvider()
    assert isinstance(provider, BaseRerankerProvider)
    assert provider.model_name == "mock-reranker-bge-reranker-v2-m3"
    assert provider.device == "cpu"
    assert provider.version == "mock-rerank-1.0.0"


def test_mock_reranker_predict_scoring():
    """Verify MockRerankerProvider computes higher scores for matching query terms."""
    provider = MockRerankerProvider()
    query = "photosynthesis in plants"
    relevant_doc = "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients."
    irrelevant_doc = "Quantum computing uses qubits to perform complex matrix calculations."

    scores = provider.predict(
        pairs=[(query, relevant_doc), (query, irrelevant_doc)],
        batch_size=2,
    )
    assert len(scores) == 2
    # Relevant document should score high (> 0.50)
    assert scores[0] >= 0.50
    # Irrelevant document should score low (< 0.35)
    assert scores[1] < 0.35
    assert scores[0] > scores[1]


def test_mock_reranker_empty_pairs():
    """Verify predicting on empty pairs returns an empty list without error."""
    provider = MockRerankerProvider()
    assert provider.predict([]) == []


def test_reranker_singleton_loader_reuse():
    """Verify get_reranker_provider returns the identical singleton instance across calls."""
    p1 = get_reranker_provider()
    p2 = get_reranker_provider()
    assert p1 is p2
    assert isinstance(p1, BaseRerankerProvider)


def test_set_reranker_provider_override():
    """Verify set_reranker_provider allows swapping the provider instance."""
    original = get_reranker_provider()
    custom_mock = MockRerankerProvider(model_name="custom-test-reranker")
    try:
        set_reranker_provider(custom_mock)
        assert get_reranker_provider() is custom_mock
        assert get_reranker_provider().model_name == "custom-test-reranker"
    finally:
        set_reranker_provider(original)


def test_warmup_reranker_model_runs():
    """Verify warmup hook executes cleanly without exceptions."""
    warmup_reranker_model()


@pytest.mark.skipif(
    not os.getenv("RUN_REAL_RERANKER_TEST"),
    reason="Optional smoke test: set RUN_REAL_RERANKER_TEST=1 to download and test real CrossEncoder model.",
)
def test_real_cross_encoder_smoke():
    """Optional smoke test downloading and testing real BAAI/bge-reranker-v2-m3 model.

    Verifies:
    - Model downloads / loads from local persistent cache
    - Query-document scoring works across both English and Vietnamese
    - Output score list shape and range [0.0, 1.0] are correct
    - Relevant passages score significantly higher than irrelevant passages
    """
    from app.ml.reranker_provider import CrossEncoderRerankerProvider

    target_model = os.getenv("TEST_REAL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    provider = CrossEncoderRerankerProvider(model_name=target_model)
    assert provider.model_name == target_model

    pairs = [
        # English: Relevant vs Irrelevant
        ("What is machine learning?", "Machine learning is a subfield of artificial intelligence focusing on algorithms."),
        ("What is machine learning?", "Photosynthesis converts light energy into chemical energy in green plants."),
        # Vietnamese: Relevant vs Irrelevant
        ("Trí tuệ nhân tạo là gì?", "Trí tuệ nhân tạo là ngành khoa học máy tính nghiên cứu về các hệ thống thông minh."),
        ("Trí tuệ nhân tạo là gì?", "Định lý Pythagoras phát biểu về bình phương cạnh huyền trong tam giác vuông."),
    ]

    scores = provider.predict(pairs)
    assert len(scores) == 4
    for score in scores:
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # Relevant documents must score significantly higher than irrelevant documents
    assert scores[0] > scores[1], f"English relevant {scores[0]} should exceed irrelevant {scores[1]}"
    if "TinyBERT" not in target_model:
        assert scores[2] > scores[3], f"Vietnamese relevant {scores[2]} should exceed irrelevant {scores[3]}"
