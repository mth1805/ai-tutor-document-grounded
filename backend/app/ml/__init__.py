"""ML models, embedding, and reranker provider package."""
from app.ml.base import BaseEmbeddingProvider
from app.ml.reranker_base import BaseRerankerProvider
from app.ml.loader import (
    get_embedding_provider,
    set_embedding_provider,
    warmup_embedding_model,
    get_reranker_provider,
    set_reranker_provider,
    warmup_reranker_model,
)
from app.ml.mock_provider import MockEmbeddingProvider
from app.ml.reranker_mock import MockRerankerProvider

__all__ = [
    "BaseEmbeddingProvider",
    "BaseRerankerProvider",
    "get_embedding_provider",
    "set_embedding_provider",
    "warmup_embedding_model",
    "get_reranker_provider",
    "set_reranker_provider",
    "warmup_reranker_model",
    "MockEmbeddingProvider",
    "MockRerankerProvider",
]
