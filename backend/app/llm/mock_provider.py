"""Mock LLM provider for unit and integration testing without external API calls."""
import asyncio
from typing import AsyncIterator, List, Optional
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import LLMProviderError


class MockLLMProvider(BaseLLMProvider):
    """Test-double LLM provider providing deterministic generation without network calls."""

    def __init__(
        self,
        default_response: str = "This is a grounded mock response based on the document evidence. [Source 1]",
        tokens: Optional[List[str]] = None,
        simulate_error: Optional[Exception] = None,
        token_delay: float = 0.0,
        # Phase 9: web grounding support
        web_grounding_response: str = "Web-grounded answer about the topic.",
        web_grounding_tokens: Optional[List[str]] = None,
        web_grounding_sources: Optional[List[dict]] = None,
        simulate_web_grounding_error: Optional[Exception] = None,
    ):
        self.default_response = default_response
        self.tokens = (
            tokens
            if tokens is not None
            else [
                "This ",
                "is ",
                "a ",
                "grounded ",
                "mock ",
                "response ",
                "based ",
                "on ",
                "the ",
                "document ",
                "evidence. ",
                "[Source 1]",
            ]
        )
        self.simulate_error = simulate_error
        self.token_delay = token_delay

        # Phase 9: web grounding defaults
        self.web_grounding_response = web_grounding_response
        self.web_grounding_tokens = web_grounding_tokens or [
            "Web-grounded ",
            "answer ",
            "about ",
            "the ",
            "topic.",
        ]
        self.web_grounding_sources = web_grounding_sources or [
            {
                "title": "Example Web Source",
                "url": "https://example.com/article",
                "snippet": "Relevant snippet about the topic.",
            }
        ]
        self.simulate_web_grounding_error = simulate_web_grounding_error

        # Spy attributes for test inspection
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None
        self.call_count: int = 0
        self.web_grounding_call_count: int = 0

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        self.call_count += 1

        if self.simulate_error:
            raise self.simulate_error

        return self.default_response

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        self.call_count += 1

        if self.simulate_error:
            raise self.simulate_error

        for token in self.tokens:
            if self.token_delay > 0:
                await asyncio.sleep(self.token_delay)
            yield token

    async def generate_stream_with_grounding(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[tuple[str, list[dict]]]:
        """Phase 9: Simulates streaming generation with Google Search grounding.

        Yields (token_chunk, web_sources) tuples:
        - Intermediate yields: (token, [])
        - Final yield: ("", web_sources)
        """
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        self.web_grounding_call_count += 1

        if self.simulate_web_grounding_error:
            raise self.simulate_web_grounding_error

        for token in self.web_grounding_tokens:
            if self.token_delay > 0:
                await asyncio.sleep(self.token_delay)
            yield token, []

        # Signal end-of-stream with sources
        if self.web_grounding_sources:
            yield "", self.web_grounding_sources
