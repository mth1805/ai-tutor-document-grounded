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
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        resolved_key = api_key or settings.GEMINI_API_KEY
        if not resolved_key or not resolved_key.strip():
            raise LLMConfigurationError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in your environment or backend/.env."
            )
        self.api_key = resolved_key.strip()
        self.model_name = model_name or settings.GEMINI_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.LLM_MAX_OUTPUT_TOKENS
        )
        self.top_p = top_p if top_p is not None else settings.LLM_TOP_P

        # Reuse single client across requests (avoids recreating heavyweight client per request)
        self._client = genai.Client(api_key=self.api_key)

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> types.GenerateContentConfig:
        """Assembles types.GenerateContentConfig from instance settings and overrides."""
        temperature = kwargs.get("temperature", self.temperature)
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)
        top_p = kwargs.get("top_p", self.top_p)

        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
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
