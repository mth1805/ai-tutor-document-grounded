"""Provider-specific exception types for web search failures.

Distinct exception classes allow rag_service.py to handle different failure
modes (auth vs. timeout vs. rate limit) with appropriate logging and user
messages without importing provider-specific libraries.
"""


class WebSearchError(Exception):
    """Base class for all web search provider failures."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class WebSearchAuthError(WebSearchError):
    """Raised when the provider rejects the API key (401/403)."""


class WebSearchTimeoutError(WebSearchError):
    """Raised when the provider request exceeds the configured timeout."""


class WebSearchRateLimitError(WebSearchError):
    """Raised when the provider returns a rate-limit / quota response (429)."""
