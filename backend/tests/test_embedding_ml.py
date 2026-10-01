"""Unit tests for ML embedding providers, model caching, and singleton lifecycle."""
import os
import math
import pytest

from app.ml.base import BaseEmbeddingProvider
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.bge_provider import BGEEmbeddingProvider
from app.ml.loader import (
    get_embedding_provider,
    set_embedding_provider,
    warmup_embedding_model,
)
from app.core.config import settings


def test_mock_embedding_provider_initialization():
    """Verify MockEmbeddingProvider properties and configuration."""
    provider = MockEmbeddingProvider()
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.dimension == 1024
    assert provider.device == "cpu"
    assert provider.version == "mock-1.0.0"


def test_mock_embedding_provider_deterministic_output():
    """Verify that same text yields identical vectors and different text yields distinct vectors."""
    provider = MockEmbeddingProvider()
    vec1 = provider.encode_batch(["Document-grounded AI Tutor"])[0]
    vec2 = provider.encode_batch(["Document-grounded AI Tutor"])[0]
    vec3 = provider.encode_batch(["Different text about vector indexing"])[0]

    assert len(vec1) == 1024
    assert vec1 == vec2
    assert vec1 != vec3


def test_mock_embedding_provider_normalization():
    """Verify that L2 normalization produces vectors with unit norm."""
    provider = MockEmbeddingProvider()
    vectors = provider.encode_batch(["Hello world", "Artificial Intelligence"], normalize=True)
    assert len(vectors) == 2

    for vec in vectors:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-5


def test_mock_embedding_provider_batching_and_empty():
    """Verify batching behavior and empty input handling."""
    provider = MockEmbeddingProvider()
    assert provider.encode_batch([]) == []

    texts = [f"Chunk number {i}" for i in range(25)]
    vectors = provider.encode_batch(texts, batch_size=8)
    assert len(vectors) == 25
    assert all(len(v) == 1024 for v in vectors)


def test_device_resolution_and_cpu_fallback(monkeypatch):
    """Verify that requesting CUDA falls back safely to CPU when CUDA is unavailable."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert BGEEmbeddingProvider._resolve_device("auto") == "cpu"
    assert BGEEmbeddingProvider._resolve_device("cuda") == "cpu"
    assert BGEEmbeddingProvider._resolve_device("cpu") == "cpu"


def test_device_resolution_when_cuda_available(monkeypatch):
    """Verify that requesting CUDA or auto selects CUDA when available."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert BGEEmbeddingProvider._resolve_device("auto") == "cuda"
    assert BGEEmbeddingProvider._resolve_device("cuda") == "cuda"
    assert BGEEmbeddingProvider._resolve_device("cpu") == "cpu"


def test_singleton_loader_reuses_instance():
    """Verify that get_embedding_provider returns the exact same singleton instance."""
    set_embedding_provider(None)
    p1 = get_embedding_provider()
    p2 = get_embedding_provider()
    assert p1 is p2


def test_custom_provider_injection():
    """Verify that set_embedding_provider allows injecting test mock providers."""
    custom = MockEmbeddingProvider(dimension=512, version="custom-test")
    set_embedding_provider(custom)
    active = get_embedding_provider()
    assert active is custom
    assert active.dimension == 512
    set_embedding_provider(None)


def test_warmup_hook_executes_safely():
    """Verify warmup_embedding_model runs without throwing."""
    set_embedding_provider(MockEmbeddingProvider())
    try:
        warmup_embedding_model()
    finally:
        set_embedding_provider(None)


@pytest.mark.skipif(
    os.getenv("RUN_REAL_BGE_M3_TEST") != "1",
    reason="Optional smoke test: Set RUN_REAL_BGE_M3_TEST=1 to run real BGE-M3 weight loading.",
)
def test_real_bge_m3_smoke():
    """Explicitly optional smoke test for real BGE-M3 model weights download/cache."""
    provider = BGEEmbeddingProvider()
    assert provider.dimension == 1024
    vec = provider.encode_batch(["Smoke test sentence for BGE-M3"])[0]
    assert len(vec) == 1024
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4
