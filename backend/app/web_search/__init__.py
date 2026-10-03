"""Web search provider abstraction package.

Exposes a thin WebSearchProvider interface and Tavily concrete implementation.
The singleton factory keeps provider construction out of request handlers.
"""
from app.web_search.base import WebSearchProvider, WebSearchResult
from app.web_search.tavily_provider import TavilyWebSearchProvider
from app.web_search.factory import get_web_search_provider, set_web_search_provider, reset_web_search_provider

__all__ = [
    "WebSearchProvider",
    "WebSearchResult",
    "TavilyWebSearchProvider",
    "get_web_search_provider",
    "set_web_search_provider",
    "reset_web_search_provider",
]
