import sys
import threading
import logging
from typing import Optional

from app.core.config import settings
from app.ml.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_EMBEDDING_PROVIDER: Optional[BaseEmbeddingProvider] = None
_LOCK = threading.Lock()

_RERANKER_PROVIDER = None
_RERANKER_LOCK = threading.Lock()


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Retrieves or lazily initializes the process-wide embedding provider singleton.

    Never creates multiple model instances across HTTP requests.
    In test environments, defaults to MockEmbeddingProvider to avoid downloading multi-GB weights.
    """
    global _EMBEDDING_PROVIDER

    if _EMBEDDING_PROVIDER is not None:
        return _EMBEDDING_PROVIDER

    with _LOCK:
        if _EMBEDDING_PROVIDER is not None:
            return _EMBEDDING_PROVIDER

        if (
            getattr(settings, "USE_MOCK_EMBEDDING", False)
            or settings.ENVIRONMENT == "test"
            or "pytest" in sys.modules
        ):
            from app.ml.mock_provider import MockEmbeddingProvider

            logger.info("Initializing MockEmbeddingProvider for test environment.")
            _EMBEDDING_PROVIDER = MockEmbeddingProvider()
        else:
            from app.ml.bge_provider import BGEEmbeddingProvider

            logger.info("Initializing BGEEmbeddingProvider singleton...")
            _EMBEDDING_PROVIDER = BGEEmbeddingProvider()

        return _EMBEDDING_PROVIDER


def set_embedding_provider(provider: Optional[BaseEmbeddingProvider]) -> None:
    """Explicitly sets or resets the singleton embedding provider (primarily for tests)."""
    global _EMBEDDING_PROVIDER
    with _LOCK:
        _EMBEDDING_PROVIDER = provider


def warmup_embedding_model() -> None:
    """Lifespan warmup hook to ensure model weights are pre-loaded during application startup."""
    try:
        logger.info("Pre-warming embedding model on process startup...")
        provider = get_embedding_provider()
        # Warmup pass with tiny text
        provider.encode_batch(["warmup text"], normalize=True, batch_size=1)
        logger.info(
            "Embedding model (%s) warmed up successfully on device '%s'.",
            provider.model_name,
            provider.device,
        )
    except Exception as e:
        logger.error("Failed to pre-warm embedding model: %s", e, exc_info=True)
        raise RuntimeError(f"Startup prewarm failed for embedding model: {e}") from e


def get_reranker_provider():
    """Retrieves or lazily initializes the process-wide Cross-Encoder reranker singleton.

    Never creates multiple model instances across HTTP requests.
    In test environments, defaults to MockRerankerProvider to avoid downloading large weights.
    """
    global _RERANKER_PROVIDER

    if _RERANKER_PROVIDER is not None:
        return _RERANKER_PROVIDER

    with _RERANKER_LOCK:
        if _RERANKER_PROVIDER is not None:
            return _RERANKER_PROVIDER

        if (
            getattr(settings, "USE_MOCK_RERANKER", False)
            or settings.ENVIRONMENT == "test"
            or "pytest" in sys.modules
        ):
            from app.ml.reranker_mock import MockRerankerProvider

            logger.info("Initializing MockRerankerProvider for test environment.")
            _RERANKER_PROVIDER = MockRerankerProvider()
        else:
            from app.ml.reranker_provider import CrossEncoderRerankerProvider

            logger.info("Initializing CrossEncoderRerankerProvider singleton...")
            _RERANKER_PROVIDER = CrossEncoderRerankerProvider()

        return _RERANKER_PROVIDER


def set_reranker_provider(provider) -> None:
    """Explicitly sets or resets the singleton reranker provider (primarily for tests)."""
    global _RERANKER_PROVIDER
    with _RERANKER_LOCK:
        _RERANKER_PROVIDER = provider


def warmup_reranker_model() -> None:
    """Lifespan warmup hook to ensure Cross-Encoder weights are pre-loaded during application startup."""
    try:
        logger.info("Pre-warming Cross-Encoder reranker model on process startup...")
        provider = get_reranker_provider()
        provider.predict([("warmup query", "warmup text")], batch_size=1)
        logger.info(
            "Reranker model (%s) warmed up successfully on device '%s'.",
            provider.model_name,
            provider.device,
        )
    except Exception as e:
        logger.error("Failed to pre-warm reranker model: %s", e, exc_info=True)
        raise RuntimeError(f"Startup prewarm failed for reranker model: {e}") from e
