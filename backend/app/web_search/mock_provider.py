"""Mock web search provider for unit and integration tests.

Never makes real HTTP calls. Returns deterministic, configurable results.
"""
from typing import List, Optional
from app.web_search.base import WebSearchProvider, WebSearchResult


class MockWebSearchProvider(WebSearchProvider):
    """Test double for WebSearchProvider — deterministic and zero-network."""

    def __init__(
        self,
        results: Optional[List[dict]] = None,
        simulate_error: Optional[Exception] = None,
    ):
        """
        Args:
            results: List of raw result dicts (title, url, content, domain).
                     Defaults to a single placeholder result.
            simulate_error: If set, search() raises this exception immediately.
        """
        self.simulate_error = simulate_error
        self.call_count: int = 0
        self.last_query: Optional[str] = None
        self.last_max_results: Optional[int] = None

        if results is not None:
            self._raw_results = results
        else:
            self._raw_results = [
                {
                    "title": "Mock Web Result",
                    "url": "https://mock.example.com/result",
                    "content": "Mock snippet from the web.",
                    "domain": "mock.example.com",
                }
            ]

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[WebSearchResult]:
        self.call_count += 1
        self.last_query = query
        self.last_max_results = max_results

        if self.simulate_error is not None:
            raise self.simulate_error

        results = []
        for item in self._raw_results[:max_results]:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            from urllib.parse import urlparse
            try:
                domain = item.get("domain") or urlparse(url).netloc.removeprefix("www.") or url[:40]
            except Exception:
                domain = url[:40]
            title = (item.get("title") or domain).strip()
            content = (item.get("content") or item.get("snippet") or "").strip()
            results.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    content=content,
                    domain=domain,
                )
            )
        return results
