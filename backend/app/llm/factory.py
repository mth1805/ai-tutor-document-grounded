"""Factory and singleton manager for LLM generation providers."""
import logging
from typing import Optional
from app.core.config import settings
from app.llm.base import BaseLLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.mock_provider import MockLLMProvider
from app.llm.exceptions import LLMConfigurationError

logger = logging.getLogger(__name__)

_LLM_PROVIDER_INSTANCE: Optional[BaseLLMProvider] = None


def get_llm_provider() -> BaseLLMProvider:
    """Returns the cached LLM provider singleton, creating it if needed."""
    global _LLM_PROVIDER_INSTANCE
    if _LLM_PROVIDER_INSTANCE is not None:
        return _LLM_PROVIDER_INSTANCE

    provider_name = (settings.LLM_PROVIDER or "gemini").lower().strip()

    if provider_name == "mock":
        logger.info("Initializing MockLLMProvider (LLM_PROVIDER=mock)")
        _LLM_PROVIDER_INSTANCE = MockLLMProvider()
        return _LLM_PROVIDER_INSTANCE

    if provider_name == "gemini":
        if not settings.GEMINI_API_KEY:
            # In testing or development without a key, log warning and use MockLLMProvider
            if settings.ENVIRONMENT in ("development", "test"):
                logger.warning(
                    "GEMINI_API_KEY is not configured in %s environment. Falling back to MockLLMProvider.",
                    settings.ENVIRONMENT,
                )
                _LLM_PROVIDER_INSTANCE = MockLLMProvider()
                return _LLM_PROVIDER_INSTANCE
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not configured. Please supply a valid Gemini API key in backend/.env."
            )
        logger.info("Initializing GeminiProvider (model=%s)", settings.GEMINI_MODEL)
        _LLM_PROVIDER_INSTANCE = GeminiProvider()
        return _LLM_PROVIDER_INSTANCE

    raise LLMConfigurationError(f"Unsupported LLM_PROVIDER: '{provider_name}'. Must be 'gemini' or 'mock'.")


def set_llm_provider(provider: Optional[BaseLLMProvider]) -> None:
    """Explicitly inject an LLM provider singleton (useful for test fixtures)."""
    global _LLM_PROVIDER_INSTANCE
    _LLM_PROVIDER_INSTANCE = provider


def reset_llm_provider() -> None:
    """Resets the cached singleton instance."""
    global _LLM_PROVIDER_INSTANCE
    _LLM_PROVIDER_INSTANCE = None
