"""Unit tests for Phase 8 LLM provider abstractions."""
import pytest
from unittest.mock import MagicMock
from google.genai.errors import APIError
from google.genai import types

from app.core.config import Settings, settings
from app.llm import (
    BaseLLMProvider,
    GeminiProvider,
    MockLLMProvider,
    get_llm_provider,
    set_llm_provider,
    reset_llm_provider,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMQuotaError,
    LLMProviderError,
)


@pytest.mark.asyncio
async def test_mock_llm_provider_generate():
    provider = MockLLMProvider(default_response="Mock answer [Source 1]")
    result = await provider.generate("Test question", system_instruction="Be helpful")

    assert result == "Mock answer [Source 1]"
    assert provider.last_prompt == "Test question"
    assert provider.last_system_instruction == "Be helpful"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_mock_llm_provider_stream():
    tokens = ["Hello ", "world", "!"]
    provider = MockLLMProvider(tokens=tokens)

    streamed: list[str] = []
    async for token in provider.generate_stream("Stream prompt"):
        streamed.append(token)

    assert streamed == tokens
    assert "".join(streamed) == "Hello world!"
    assert provider.last_prompt == "Stream prompt"


@pytest.mark.asyncio
async def test_mock_llm_provider_simulated_error():
    provider = MockLLMProvider(simulate_error=LLMProviderError("Simulated failure"))

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate("Should fail")
    assert "Simulated failure" in str(exc_info.value)

    with pytest.raises(LLMProviderError):
        async for _ in provider.generate_stream("Should fail"):
            pass


def test_gemini_provider_missing_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiProvider(api_key=None)
    assert "Gemini API key is not configured" in str(exc_info.value)


def test_gemini_provider_error_mapping():
    provider = GeminiProvider(api_key="test-dummy-key")

    # 401 Unauthorized
    err_401 = APIError(401, {"error": {"message": "Invalid API key"}})
    mapped_401 = provider._map_error(err_401)
    assert isinstance(mapped_401, LLMAuthenticationError)
    assert "authentication failed" in str(mapped_401)

    # 429 Rate Limit
    err_429 = APIError(429, {"error": {"message": "Quota exceeded"}})
    mapped_429 = provider._map_error(err_429)
    assert isinstance(mapped_429, LLMQuotaError)
    assert "rate limit" in str(mapped_429)

    # 500 Generic Error
    err_500 = APIError(500, {"error": {"message": "Internal Google server error"}})
    mapped_500 = provider._map_error(err_500)
    assert isinstance(mapped_500, LLMProviderError)
    assert "500" in str(mapped_500)


def test_gemini_provider_uses_gemini_3_thinking_config(monkeypatch):
    assert Settings.model_fields["GEMINI_MODEL"].default == "gemini-3.5-flash-lite"
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setattr(settings, "GEMINI_THINKING_LEVEL", "medium")
    monkeypatch.setattr(settings, "LLM_MAX_OUTPUT_TOKENS", 2048)
    provider = GeminiProvider(api_key="test-dummy-key")

    standard_config = provider._build_config(system_instruction="Be helpful")

    assert provider.model_name == "gemini-3.5-flash-lite"
    assert standard_config.max_output_tokens == 2048
    assert standard_config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert standard_config.system_instruction == "Be helpful"
    assert "temperature" not in standard_config.model_fields_set
    assert "top_p" not in standard_config.model_fields_set


def test_llm_factory_and_injection(monkeypatch):
    reset_llm_provider()

    # In test/development environment without key, falls back gracefully to MockLLMProvider
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(settings, "ENVIRONMENT", "test")

    provider = get_llm_provider()
    assert isinstance(provider, MockLLMProvider)

    # Inject custom mock provider
    custom_mock = MockLLMProvider(default_response="Custom injection")
    set_llm_provider(custom_mock)
    assert get_llm_provider() is custom_mock

    reset_llm_provider()
