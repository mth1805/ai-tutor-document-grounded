"""Base abstraction for vector embedding providers."""
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense embedding models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality produced by this provider."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Model or provider version identifier for reproducibility."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Active computing device (e.g., 'cpu', 'cuda')."""
        pass

    @abstractmethod
    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Encodes a list of text strings into normalized dense embedding vectors.

        Args:
            texts: List of text strings to embed.
            normalize: Whether to L2-normalize vectors for cosine similarity.
            batch_size: Optional batch size override; defaults to configured setting.

        Returns:
            List of float lists with dimension matching self.dimension.
        """
        pass
