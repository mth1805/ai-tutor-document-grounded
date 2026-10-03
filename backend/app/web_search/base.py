"""Abstract base for web search providers.

All concrete web search integrations must implement WebSearchProvider.
WebSearchResult is the canonical output type used by rag_service.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WebSearchResult:
    """Structured result returned by any WebSearchProvider implementation."""

    title: str
    url: str
    content: str          # Main body / snippet from the page
    domain: str           # Derived from URL netloc (www. stripped)
    score: Optional[float] = None   # Relevance score when the provider supplies one

    def to_dict(self) -> dict:
        """Converts to a dict compatible with WebCitation.from_raw()."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.content,
            "domain": self.domain,
        }


class WebSearchProvider(ABC):
    """Abstract interface for pluggable web search backends.

    Concrete implementations must be thread-safe and must NOT log API keys.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[WebSearchResult]:
        """Execute a web search and return structured results.

        Args:
            query: The user query string to search for.
            max_results: Maximum number of results to return.

        Returns:
            List of WebSearchResult, ordered by relevance descending.
            Returns an empty list when no results are found — never raises
            on empty results, only on hard failures (auth, network, timeout).

        Raises:
            WebSearchAuthError: Invalid or missing API key.
            WebSearchTimeoutError: Request timed out.
            WebSearchRateLimitError: Provider rate limit hit.
            WebSearchError: Any other provider-level error.
        """
        ...
