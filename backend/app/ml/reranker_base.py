"""Base abstraction for Cross-Encoder reranker providers."""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseRerankerProvider(ABC):
    """Abstract interface for Cross-Encoder reranker models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the reranker model."""
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
    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Computes relevance scores for (query, document_text) pairs.

        Args:
            pairs: List of (query, document_text) string tuples.
            batch_size: Optional batch size override; defaults to configured setting.

        Returns:
            List of float relevance scores matching the length and order of pairs.
        """
        pass
