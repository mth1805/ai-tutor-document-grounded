"""Unit tests for Phase 8 LLM provider abstractions."""
import asyncio
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace
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
    LLMTimeoutError,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(("finish_reason", "expected_complete"), [
    (types.FinishReason.STOP, True),
    (types.FinishReason.MAX_TOKENS, False),
    (types.FinishReason.SAFETY, False),
    (None, False),
])
async def test_gemini_provider_preserves_candidate_completion_metadata(
    monkeypatch, finish_reason, expected_complete,
):
    provider = GeminiProvider(api_key="test-dummy-key")

    async def fake_generate_response(*args, **kwargs):
        return SimpleNamespace(
            text="generated response",
            candidates=[SimpleNamespace(finish_reason=finish_reason)],
            usage_metadata=SimpleNamespace(
                prompt_token_count=120, candidates_token_count=45, total_token_count=165,
            ),
        )

    monkeypatch.setattr(provider, "_generate_response", fake_generate_response)
    result = await provider.generate_with_metadata("summary prompt")

    assert result.text == "generated response"
    assert result.completed is expected_complete
    assert result.finish_reason == (finish_reason.name if finish_reason is not None else "MISSING_FINISH_REASON")
    assert result.candidate_count == 1
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (120, 45, 165)


@pytest.mark.asyncio
async def test_gemini_provider_reports_missing_candidate_and_missing_finish_reason(monkeypatch):
    provider = GeminiProvider(api_key="test-dummy-key")
    responses = iter([
        SimpleNamespace(text="", candidates=[], usage_metadata=None),
        SimpleNamespace(text="", candidates=[SimpleNamespace(finish_reason=None)], usage_metadata=None),
    ])

    async def fake_generate_response(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(provider, "_generate_response", fake_generate_response)
    missing_candidate = await provider.generate_with_metadata("private prompt")
    missing_finish_reason = await provider.generate_with_metadata("private prompt")

    assert missing_candidate.finish_reason == "MISSING_CANDIDATE"
    assert missing_candidate.candidate_count == 0
    assert not missing_candidate.completed
    assert missing_finish_reason.finish_reason == "MISSING_FINISH_REASON"
    assert not missing_finish_reason.completed


@pytest.mark.asyncio
@pytest.mark.parametrize(("finish", "status"), [
    (types.FinishReason.STOP, "completed"),
    (types.FinishReason.MAX_TOKENS, "incomplete"),
    (types.FinishReason.SAFETY, "blocked"),
    (types.FinishReason.RECITATION, "blocked"),
])
async def test_gemini_stream_classifies_finish_reason(finish, status):
    provider = GeminiProvider(api_key="test-dummy-key")

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                yield SimpleNamespace(
                    text="partial answer",
                    candidates=[SimpleNamespace(finish_reason=finish)],
                )
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await provider.generate_stream_with_metadata("private prompt")

    assert result.text == "partial answer"
    assert result.generation_status == status
    assert result.completed is (status == "completed")


@pytest.mark.asyncio
async def test_gemini_stream_uses_metadata_from_final_chunk_and_keeps_no_usage_safe():
    provider = GeminiProvider(api_key="test-dummy-key")

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                yield SimpleNamespace(text="Hello ", candidates=[], usage_metadata=None)
                yield SimpleNamespace(
                    text="world", candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=20, candidates_token_count=3,
                        thoughts_token_count=7, total_token_count=30,
                    ), model_version="model-v", response_id="response-123",
                )
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await provider.generate_stream_with_metadata("private prompt")
    assert result.text == "Hello world"
    assert (result.input_tokens, result.output_tokens, result.thoughts_tokens, result.total_tokens) == (20, 3, 7, 30)
    assert (result.model_version, result.response_id) == ("model-v", "response-123")
    assert result.generation_status == "completed"

    class NoUsageModels:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                yield SimpleNamespace(text="answer", candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)])
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=NoUsageModels()))
    no_usage = await provider.generate_stream_with_metadata("private prompt")
    assert no_usage.text == "answer"
    assert no_usage.input_tokens is None and no_usage.total_tokens is None


@pytest.mark.asyncio
async def test_gemini_stream_without_candidate_or_finish_is_unverified():
    provider = GeminiProvider(api_key="test-dummy-key")

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                yield SimpleNamespace(text="text but no candidate", candidates=[])
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await provider.generate_stream_with_metadata("prompt")
    assert result.text == "text but no candidate"
    assert result.generation_status == "unverified"

    class MissingFinishModels:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                yield SimpleNamespace(text="candidate without finish", candidates=[SimpleNamespace(finish_reason=None)])
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=MissingFinishModels()))
    no_finish = await provider.generate_stream_with_metadata("prompt")
    assert no_finish.text == "candidate without finish"
    assert no_finish.candidate_count == 1
    assert no_finish.finish_reason is None
    assert no_finish.generation_status == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError("late"), ConnectionResetError("lost")])
async def test_gemini_stream_preserves_partial_text_on_timeout_or_network_error(exception):
    provider = GeminiProvider(api_key="test-dummy-key")

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                yield SimpleNamespace(text="partial ", candidates=[])
                raise exception
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await provider.generate_stream_with_metadata("private prompt")
    assert result.text == "partial "
    assert result.generation_status == "interrupted"
    assert result.error_type


@pytest.mark.asyncio
async def test_gemini_stream_timeout_before_first_chunk_is_interrupted(monkeypatch):
    provider = GeminiProvider(api_key="test-dummy-key")
    monkeypatch.setattr(settings, "LLM_STREAMING_TIMEOUT_SECONDS", 0.01)

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                await asyncio.sleep(0.1)
                yield SimpleNamespace(text="late", candidates=[])
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await provider.generate_stream_with_metadata("prompt")
    assert result.text == ""
    assert result.generation_status == "interrupted"


@pytest.mark.asyncio
async def test_gemini_stream_quota_error_is_failed_not_max_tokens():
    provider = GeminiProvider(api_key="test-dummy-key")

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                raise APIError(429, {"error": {"message": "Quota exceeded"}})
                yield  # make this an async generator
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    result = await provider.generate_stream_with_metadata("private prompt")
    assert result.generation_status == "failed"
    assert result.finish_reason is None
    assert result.error_classification == "quota"
    assert result.http_status == 429


@pytest.mark.asyncio
async def test_legacy_stream_provider_is_unverified_and_preserves_accumulated_text():
    class LegacyProvider(BaseLLMProvider):
        async def generate(self, prompt, system_instruction=None, **kwargs):
            return "legacy"

        async def generate_stream(self, prompt, system_instruction=None, **kwargs):
            yield "legacy "
            yield "stream"

    result = await LegacyProvider().generate_stream_with_metadata("prompt")
    assert result.text == "legacy stream"
    assert result.generation_status == "unverified"
    assert result.completed is False


@pytest.mark.asyncio
async def test_gemini_generate_keeps_legacy_string_return_contract(monkeypatch):
    provider = GeminiProvider(api_key="test-dummy-key")

    async def fake_generate_response(*args, **kwargs):
        return SimpleNamespace(text="legacy response", candidates=[])

    monkeypatch.setattr(provider, "_generate_response", fake_generate_response)
    assert await provider.generate("normal RAG prompt") == "legacy response"


def test_generation_budget_and_timeout_defaults_are_separate():
    config = Settings(_env_file=None, ENVIRONMENT="test")

    assert config.CHAT_MAX_OUTPUT_TOKENS == 8192
    assert config.LLM_MAX_OUTPUT_TOKENS == 2048
    assert config.LLM_STREAMING_TIMEOUT_SECONDS == 90.0


@pytest.mark.parametrize("field,value", [
    ("CHAT_MAX_OUTPUT_TOKENS", 127),
    ("CHAT_MAX_OUTPUT_TOKENS", 8193),
    ("LLM_STREAMING_TIMEOUT_SECONDS", 0),
    ("LLM_STREAMING_TIMEOUT_SECONDS", -1),
])
def test_invalid_chat_budget_and_streaming_timeout_are_rejected(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, ENVIRONMENT="test", **{field: value})


@pytest.mark.asyncio
async def test_gemini_provider_applies_configured_timeout(monkeypatch):
    provider = GeminiProvider(api_key="test-dummy-key")
    monkeypatch.setattr(settings, "LLM_STREAMING_TIMEOUT_SECONDS", 0.01)

    class SlowModels:
        async def generate_content(self, **kwargs):
            await asyncio.sleep(0.1)

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SlowModels()))

    with pytest.raises(LLMTimeoutError):
        await provider._generate_response("prompt")

    class SlowStreamModels:
        async def generate_content_stream(self, **kwargs):
            async def chunks():
                await asyncio.sleep(0.1)
                yield SimpleNamespace(text="late chunk")
            return chunks()

    provider._client = SimpleNamespace(aio=SimpleNamespace(models=SlowStreamModels()))
    with pytest.raises(LLMTimeoutError):
        async for _ in provider.generate_stream("prompt"):
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
    chat_config = provider._build_config(
        system_instruction="Be helpful",
        max_output_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
    )

    assert provider.model_name == "gemini-3.5-flash-lite"
    assert standard_config.max_output_tokens == 2048
    assert chat_config.max_output_tokens == 8192
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
