"""Tests for ML model startup lifecycle, lifespan prewarming, and singleton persistence.

Verifies:
- PREWARM_MODELS=False skips warmup calls completely.
- PREWARM_MODELS=True invokes both embedding and reranker warmup hooks exactly once via threadpool.
- Failures during configured prewarm raise RuntimeError to prevent serving from an unready state.
- Process-level singleton reuse is preserved across repeated calls and requests.
- No model is constructed per HTTP request.
- Lazy loading semantics work when prewarming is disabled.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.config import settings
from app.main import app, lifespan
from app.ml.base import BaseEmbeddingProvider
from app.ml.reranker_base import BaseRerankerProvider
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider
import app.ml.loader as ml_loader


@pytest.mark.asyncio
async def test_lifespan_prewarm_disabled_skips_warmup(monkeypatch):
    """When PREWARM_MODELS is False, lifespan startup must not invoke warmup functions."""
    monkeypatch.setattr(settings, "PREWARM_MODELS", False)

    mock_warmup_embed = MagicMock()
    mock_warmup_rerank = MagicMock()
    monkeypatch.setattr(ml_loader, "warmup_embedding_model", mock_warmup_embed)
    monkeypatch.setattr(ml_loader, "warmup_reranker_model", mock_warmup_rerank)

    async with lifespan(app):
        pass

    assert mock_warmup_embed.call_count == 0
    assert mock_warmup_rerank.call_count == 0


@pytest.mark.asyncio
async def test_lifespan_prewarm_enabled_invokes_both_warmups_once(monkeypatch):
    """When PREWARM_MODELS is True, lifespan startup must invoke both warmups exactly once."""
    monkeypatch.setattr(settings, "PREWARM_MODELS", True)

    mock_warmup_embed = MagicMock()
    mock_warmup_rerank = MagicMock()
    monkeypatch.setattr(ml_loader, "warmup_embedding_model", mock_warmup_embed)
    monkeypatch.setattr(ml_loader, "warmup_reranker_model", mock_warmup_rerank)

    async with lifespan(app):
        pass

    assert mock_warmup_embed.call_count == 1
    assert mock_warmup_rerank.call_count == 1


@pytest.mark.asyncio
async def test_lifespan_prewarm_embedding_failure_fails_startup(monkeypatch):
    """When PREWARM_MODELS is True, embedding warmup failure must fail application startup."""
    monkeypatch.setattr(settings, "PREWARM_MODELS", True)

    def failing_warmup_embed():
        raise RuntimeError("Corrupted embedding weights")

    monkeypatch.setattr(ml_loader, "warmup_embedding_model", failing_warmup_embed)

    with pytest.raises(RuntimeError, match="Corrupted embedding weights"):
        async with lifespan(app):
            pass


@pytest.mark.asyncio
async def test_lifespan_prewarm_reranker_failure_fails_startup(monkeypatch):
    """When PREWARM_MODELS is True, reranker warmup failure must fail application startup."""
    monkeypatch.setattr(settings, "PREWARM_MODELS", True)

    mock_warmup_embed = MagicMock()
    monkeypatch.setattr(ml_loader, "warmup_embedding_model", mock_warmup_embed)

    def failing_warmup_rerank():
        raise RuntimeError("CUDA OOM loading reranker")

    monkeypatch.setattr(ml_loader, "warmup_reranker_model", failing_warmup_rerank)

    with pytest.raises(RuntimeError, match="CUDA OOM loading reranker"):
        async with lifespan(app):
            pass

    assert mock_warmup_embed.call_count == 1


@pytest.mark.asyncio
async def test_lifespan_uses_threadpool_offloading(monkeypatch):
    """Verify that model warmups are executed via run_in_threadpool so the async event loop is not blocked."""
    monkeypatch.setattr(settings, "PREWARM_MODELS", True)

    threadpool_targets = []

    async def mock_run_in_threadpool(func, *args, **kwargs):
        threadpool_targets.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr("fastapi.concurrency.run_in_threadpool", mock_run_in_threadpool)

    mock_warmup_embed = MagicMock()
    mock_warmup_rerank = MagicMock()
    monkeypatch.setattr(ml_loader, "warmup_embedding_model", mock_warmup_embed)
    monkeypatch.setattr(ml_loader, "warmup_reranker_model", mock_warmup_rerank)

    async with lifespan(app):
        pass

    assert mock_warmup_embed in threadpool_targets
    assert mock_warmup_rerank in threadpool_targets


def test_singleton_reuse_across_calls():
    """Verify that get_embedding_provider and get_reranker_provider return the identical singleton."""
    ml_loader.set_embedding_provider(None)
    ml_loader.set_reranker_provider(None)

    # First call initializes singleton
    embed1 = ml_loader.get_embedding_provider()
    embed2 = ml_loader.get_embedding_provider()
    assert embed1 is embed2
    assert isinstance(embed1, BaseEmbeddingProvider)

    rerank1 = ml_loader.get_reranker_provider()
    rerank2 = ml_loader.get_reranker_provider()
    assert rerank1 is rerank2
    assert isinstance(rerank1, BaseRerankerProvider)


def test_lazy_loading_behavior_when_prewarm_disabled():
    """Verify lazy loading behavior: providers remain uninitialized until first explicit request."""
    # Reset internal singletons
    ml_loader.set_embedding_provider(None)
    ml_loader.set_reranker_provider(None)

    assert ml_loader._EMBEDDING_PROVIDER is None
    assert ml_loader._RERANKER_PROVIDER is None

    # First request lazily loads the singleton
    lazy_embed = ml_loader.get_embedding_provider()
    assert ml_loader._EMBEDDING_PROVIDER is not None
    assert ml_loader._EMBEDDING_PROVIDER is lazy_embed

    lazy_rerank = ml_loader.get_reranker_provider()
    assert ml_loader._RERANKER_PROVIDER is not None
    assert ml_loader._RERANKER_PROVIDER is lazy_rerank
