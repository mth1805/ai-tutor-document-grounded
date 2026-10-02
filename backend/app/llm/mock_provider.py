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

        # Spy attributes for test inspection
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None
        self.call_count: int = 0

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
