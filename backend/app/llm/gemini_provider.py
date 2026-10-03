"""Google Gemini provider implementation using google-genai SDK."""
import logging
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

    def _map_error(self, err: Exception) -> Exception:
        """Translates SDK-specific errors into standardized application exceptions."""
        if isinstance(err, errors.APIError):
            code = getattr(err, "code", None)
            msg = getattr(err, "message", str(err))
            if code in (401, 403):
                return LLMAuthenticationError(
                    f"Gemini authentication failed: {msg}", details={"code": code}
                )
            if code == 429:
                return LLMQuotaError(
                    f"Gemini rate limit or quota exceeded: {msg}", details={"code": code}
                )
            return LLMProviderError(
                f"Gemini API error ({code}): {msg}", details={"code": code}
            )
        if isinstance(err, TimeoutError):
            return LLMTimeoutError(f"Gemini request timed out: {err}")
        return LLMProviderError(f"Gemini generation error: {err}")

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
            logger.error("Gemini non-streaming generation failed: %s", type(e).__name__)
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
            logger.error("Gemini streaming generation failed: %s", type(e).__name__)
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
            logger.error("Gemini grounded generation failed: %s", type(e).__name__)
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
            logger.error("Gemini streaming grounded generation failed: %s", type(e).__name__)
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
