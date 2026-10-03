"""Singleton factory and injection utilities for the web search provider.

Pattern mirrors app/llm/factory.py for consistency.
"""
import logging
from typing import Optional

from app.core.config import settings
from app.web_search.base import WebSearchProvider

logger = logging.getLogger(__name__)

_WEB_SEARCH_PROVIDER_INSTANCE: Optional[WebSearchProvider] = None


def get_web_search_provider() -> WebSearchProvider:
    """Returns the cached web search provider singleton, creating it on first call."""
    global _WEB_SEARCH_PROVIDER_INSTANCE
    if _WEB_SEARCH_PROVIDER_INSTANCE is not None:
        return _WEB_SEARCH_PROVIDER_INSTANCE

    if not settings.TAVILY_API_KEY:
        from app.web_search.exceptions import WebSearchAuthError
        raise WebSearchAuthError(
            "Tavily API key is not configured. Set TAVILY_API_KEY in backend/.env."
        )

    from app.web_search.tavily_provider import TavilyWebSearchProvider
    logger.info("Initializing TavilyWebSearchProvider (max_results=%d)", settings.TAVILY_MAX_RESULTS)
    _WEB_SEARCH_PROVIDER_INSTANCE = TavilyWebSearchProvider()
    return _WEB_SEARCH_PROVIDER_INSTANCE


def set_web_search_provider(provider: Optional[WebSearchProvider]) -> None:
    """Explicitly inject a web search provider singleton (useful for test fixtures)."""
    global _WEB_SEARCH_PROVIDER_INSTANCE
    _WEB_SEARCH_PROVIDER_INSTANCE = provider


def reset_web_search_provider() -> None:
    """Resets the cached singleton (useful for test teardown)."""
    global _WEB_SEARCH_PROVIDER_INSTANCE
    _WEB_SEARCH_PROVIDER_INSTANCE = None
