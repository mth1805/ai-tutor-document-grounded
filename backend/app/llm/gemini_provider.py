"""Google Gemini provider implementation using google-genai SDK."""
import logging
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncIterator, Optional
from google import genai
from google.genai import types
from google.genai import errors

from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMQuotaError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


def _retry_delay_seconds(err: Exception) -> float | None:
    """Read explicit retry guidance from Google error details or Retry-After."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    raw_details = getattr(err, "details", None)
    if raw_details is not None:
        def find(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    normalized = str(key).lower().replace("_", "").replace("-", "")
                    if normalized in {"retrydelay", "retryafter", "retrydelayseconds", "retryafterseconds"}:
                        if isinstance(item, (int, float)):
                            return max(0.0, float(item))
                        if isinstance(item, str):
                            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds?)?\s*", item, re.I)
                            if match:
                                value = float(match.group(1))
                                return value / 1000 if (match.group(2) or "s").lower() == "ms" else value
                    result = find(item)
                    if result is not None:
                        return result
            elif isinstance(value, (list, tuple)):
                for item in value:
                    result = find(item)
                    if result is not None:
                        return result
            return None
        delay = find(raw_details)
        if delay is not None:
            return delay
    message = str(getattr(err, "message", None) or "")
    match = re.search(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds?)?", message, re.I)
    if match:
        value = float(match.group(1))
        return value / 1000 if (match.group(2) or "s").lower() == "ms" else value
    return None


def _is_daily_quota(err: Exception) -> bool:
    message = str(getattr(err, "message", None) or "").lower()
    try:
        detail_text = json.dumps(getattr(err, "details", None), ensure_ascii=True, default=str).lower()
    except (TypeError, ValueError):
        detail_text = ""
    text = f"{message} {detail_text}"
    markers = ("daily quota", "per day", "per-day", "per_day", "per 24 hours", "per24hours",
               "requestsperday", "tokensperday", "daily_limit", "daily limit")
    return any(marker in text for marker in markers)


class GeminiProvider(BaseLLMProvider):
    """Production LLM provider for Google Gemini models via official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ):
        resolved_key = api_key or settings.GEMINI_API_KEY
        if not resolved_key or not resolved_key.strip():
            raise LLMConfigurationError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in your environment or backend/.env."
            )
        self.api_key = resolved_key.strip()
        self.model_name = model_name or settings.GEMINI_MODEL
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.LLM_MAX_OUTPUT_TOKENS
        )
        self.thinking_level = settings.GEMINI_THINKING_LEVEL

        # Reuse single client across requests (avoids recreating heavyweight client per request)
        self._client = genai.Client(api_key=self.api_key)

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> types.GenerateContentConfig:
        """Assembles types.GenerateContentConfig from instance settings and overrides."""
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)

        return types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            system_instruction=system_instruction,
        )

    def _diagnostic(self, err: Exception) -> dict:
        """Return safe SDK diagnostics without including client credentials."""
        code = getattr(err, "code", None)
        response = getattr(err, "response", None)
        if not isinstance(code, int):
            code = getattr(response, "status_code", None)
        http_status = code if isinstance(code, int) and 100 <= code <= 599 else None
        provider_status = getattr(err, "status", None)
        provider_message = getattr(err, "message", None) or str(err)
        if self.api_key:
            provider_message = str(provider_message).replace(self.api_key, "[REDACTED]")
        normalized_status = str(provider_status or "").upper()
        error_class = type(err).__name__
        daily_quota = _is_daily_quota(err)
        if daily_quota:
            classification, retryable, quota_scope = "daily_quota", False, "daily"
        elif http_status == 429 or normalized_status in {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"}:
            classification, retryable = "quota", True
            quota_scope = "unknown"
        elif http_status in {408, 504} or isinstance(err, TimeoutError) \
                or normalized_status == "DEADLINE_EXCEEDED":
            classification, retryable = "timeout", True
            quota_scope = None
        elif ((http_status is not None and http_status >= 500) or error_class == "ServerError"
              or normalized_status in {"INTERNAL", "UNAVAILABLE"}):
            classification, retryable = "server_error", True
            quota_scope = None
        elif http_status in {401, 403}:
            classification, retryable = "authentication", False
            quota_scope = None
        elif http_status is not None and 400 <= http_status < 500:
            classification, retryable = "client_error", False
            quota_scope = None
        else:
            classification, retryable = "unknown", False
            quota_scope = None
        return {"exception_class": error_class, "http_status": http_status,
                "provider_status": provider_status, "provider_message": str(provider_message),
                "classification": classification, "retryable": retryable,
                "retry_delay_seconds": _retry_delay_seconds(err), "quota_scope": quota_scope}

    def _log_failure(self, operation: str, err: Exception) -> None:
        diagnostic = self._diagnostic(err)
        logger.error(
            "Gemini %s failed: exception=%s http_status=%s provider_status=%s classification=%s retryable=%s message=%s",
            operation, diagnostic["exception_class"], diagnostic["http_status"],
            diagnostic["provider_status"], diagnostic["classification"],
            diagnostic["retryable"], diagnostic["provider_message"],
        )

    def _map_error(self, err: Exception) -> Exception:
        """Map SDK errors while retaining safe, actionable provider diagnostics."""
        diagnostic = self._diagnostic(err)
        details = {"provider": "google-genai", **diagnostic}
        if isinstance(err, errors.APIError):
            code = diagnostic["http_status"]
            msg = diagnostic["provider_message"]
            if code in (401, 403):
                return LLMAuthenticationError(
                    f"Gemini authentication failed: {msg}", details=details
                )
            if diagnostic["classification"] == "quota":
                return LLMQuotaError(
                    f"Gemini rate limit or quota exceeded: {msg}", details=details
                )
            return LLMProviderError(
                f"Gemini API error ({code} {diagnostic['provider_status']}): {msg}", details=details
            )
        if isinstance(err, TimeoutError):
            return LLMTimeoutError(f"Gemini request timed out: {diagnostic['provider_message']}", details=details)
        return LLMProviderError(f"Gemini generation error: {diagnostic['provider_message']}", details=details)

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Synchronously or asynchronously produces non-streamed text completion."""
        config = self._build_config(system_instruction=system_instruction, **kwargs)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            self._log_failure("non-streaming generation", e)
            raise self._map_error(e) from e

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Asynchronously streams chunks of text tokens directly from Gemini API."""
        config = self._build_config(system_instruction=system_instruction, **kwargs)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            async for chunk in stream:
                text_chunk = chunk.text
                if text_chunk:
                    yield text_chunk
        except Exception as e:
            self._log_failure("streaming generation", e)
            raise self._map_error(e) from e

    async def generate_with_grounding(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, list[dict]]:
        """Non-streaming generation with Google Search grounding enabled.

        Returns:
            Tuple of (full_text, web_sources_list).
            web_sources_list is a list of dicts with keys: title, url, snippet.
        """
        config = self._build_config(system_instruction=system_instruction, **kwargs)
        # Inject google_search tool for grounding
        config = config.model_copy(
            update={"tools": [types.Tool(google_search=types.GoogleSearch())]}
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            full_text = response.text or ""
            web_sources = self._extract_grounding_sources(response)
            return full_text, web_sources
        except Exception as e:
            self._log_failure("grounded generation", e)
            raise self._map_error(e) from e

    async def generate_stream_with_grounding(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, list[dict]]]:
        """Streams text tokens with Google Search grounding enabled.

        Yields tuples of (token_chunk, web_sources).
        web_sources is populated on the final chunk once grounding metadata is available;
        intermediate chunks yield an empty list.
        """
        config = self._build_config(
            system_instruction=system_instruction,
            **kwargs,
        ).model_copy(
            update={"tools": [types.Tool(google_search=types.GoogleSearch())]}
        )
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            last_chunk = None
            async for chunk in stream:
                last_chunk = chunk
                text_chunk = chunk.text
                if text_chunk:
                    yield text_chunk, []
            # After streaming completes, extract grounding from the final chunk
            if last_chunk is not None:
                web_sources = self._extract_grounding_sources(last_chunk)
                if web_sources:
                    # Signal end-of-stream with sources via empty token + sources
                    yield "", web_sources
        except Exception as e:
            self._log_failure("grounded streaming generation", e)
            raise self._map_error(e) from e

    @staticmethod
    def _extract_grounding_sources(response_or_chunk) -> list[dict]:
        """Extracts web source metadata from Gemini grounding metadata.

        Supports both full GenerateContentResponse and streaming GenerateContentResponse chunks.
        Returns a deduplicated list of {title, url, snippet} dicts.
        """
        sources: list[dict] = []
        seen_urls: set[str] = set()

        try:
            candidates = getattr(response_or_chunk, "candidates", None) or []
            for candidate in candidates:
                grounding_meta = getattr(candidate, "grounding_metadata", None)
                if grounding_meta is None:
                    continue

                # grounding_chunks contains web search results
                grounding_chunks = getattr(grounding_meta, "grounding_chunks", None) or []
                for gc in grounding_chunks:
                    web = getattr(gc, "web", None)
                    if web is None:
                        continue
                    url = getattr(web, "uri", None) or getattr(web, "url", None) or ""
                    title = getattr(web, "title", "") or ""
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    sources.append({"title": title, "url": url, "snippet": ""})

                # grounding_supports may have richer snippet text per support
                supports = getattr(grounding_meta, "grounding_supports", None) or []
                for support in supports:
                    text_seg = getattr(support, "segment", None)
                    snippet_text = getattr(text_seg, "text", "") if text_seg else ""
                    chunk_indices = getattr(support, "grounding_chunk_indices", []) or []
                    for ci in chunk_indices:
                        if 0 <= ci < len(sources):
                            if not sources[ci]["snippet"]:
                                sources[ci]["snippet"] = (snippet_text or "")[:300]
        except Exception as exc:
            logger.debug("Could not extract grounding sources: %s", exc)

        return sources
