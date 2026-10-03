"""Tavily web search provider implementation.

Uses the official tavily-python SDK (tavily.TavilyClient).
The API key is loaded exclusively from settings — it is NEVER logged.
"""
import asyncio
import logging
from typing import List
from urllib.parse import urlparse

from app.core.config import settings
from app.web_search.base import WebSearchProvider, WebSearchResult
from app.web_search.exceptions import (
    WebSearchAuthError,
    WebSearchError,
    WebSearchRateLimitError,
    WebSearchTimeoutError,
)

logger = logging.getLogger(__name__)


def _derive_domain(url: str) -> str:
    """Extracts clean domain (no www.) from a URL string."""
    try:
        netloc = urlparse(url).netloc
        return netloc.removeprefix("www.") or url[:40]
    except Exception:
        return url[:40]


class TavilyWebSearchProvider(WebSearchProvider):
    """Concrete web search backend powered by Tavily Search API.

    Wraps the synchronous TavilyClient in asyncio.to_thread() so the
    async event loop is never blocked by the underlying HTTP call.
    """

    def __init__(self, api_key: str | None = None, timeout: float | None = None):
        resolved_key = api_key or settings.TAVILY_API_KEY
        if not resolved_key or not resolved_key.strip():
            raise WebSearchAuthError(
                "Tavily API key is not configured. Set TAVILY_API_KEY in backend/.env."
            )
        self._api_key = resolved_key.strip()
        self._timeout = timeout if timeout is not None else settings.WEB_SEARCH_TIMEOUT_SECONDS
        # Lazy-import to avoid hard import failure when the package is absent in tests
        self._client = self._build_client()

    def _build_client(self):
        try:
            from tavily import TavilyClient  # type: ignore[import]
            return TavilyClient(api_key=self._api_key)
        except ImportError as exc:
            raise WebSearchError(
                "tavily-python is not installed. Run: pip install tavily-python>=0.3.0"
            ) from exc

    def _sync_search(self, query: str, max_results: int) -> List[dict]:
        """Blocking Tavily search — must be called inside a thread."""
        return self._client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        ).get("results", [])

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[WebSearchResult]:
        """Calls Tavily in a threadpool and normalises results to WebSearchResult."""
        logger.info("[Tavily] Searching: query_len=%d, max_results=%d", len(query), max_results)

        try:
            raw_results: List[dict] = await asyncio.wait_for(
                asyncio.to_thread(self._sync_search, query, max_results),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.error("[Tavily] Request timed out after %.1fs", self._timeout)
            raise WebSearchTimeoutError(
                f"Tavily search timed out after {self._timeout:.0f}s."
            )
        except Exception as exc:
            # Map HTTP status codes surfaced through the SDK to typed exceptions
            exc_str = str(exc).lower()
            if "401" in exc_str or "403" in exc_str or "invalid api key" in exc_str or "unauthorized" in exc_str:
                logger.error("[Tavily] Authentication failed (key rejected).")
                raise WebSearchAuthError("Tavily API key is invalid or has been revoked.") from exc
            if "429" in exc_str or "rate limit" in exc_str or "quota" in exc_str:
                logger.error("[Tavily] Rate limit exceeded.")
                raise WebSearchRateLimitError("Tavily rate limit or quota exceeded.") from exc
            logger.error("[Tavily] Unexpected error: %s", type(exc).__name__)
            raise WebSearchError(f"Tavily search failed: {type(exc).__name__}") from exc

        results: List[WebSearchResult] = []
        for item in raw_results:
            url = (item.get("url") or "").strip()
            if not url or not url.startswith(("http://", "https://")):
                continue  # Skip malformed / non-HTTP results
            title = (item.get("title") or "").strip()
            content = (item.get("content") or item.get("snippet") or "").strip()
            domain = _derive_domain(url)
            if not title:
                title = domain
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    content=content[:1000],   # bounded per result
                    domain=domain,
                    score=item.get("score"),
                )
            )

        logger.info("[Tavily] Returned %d valid results.", len(results))
        return results
