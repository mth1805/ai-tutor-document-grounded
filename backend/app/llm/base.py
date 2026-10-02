"""Base abstract class for LLM providers."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class BaseLLMProvider(ABC):
    """Abstract interface defining required methods for LLM answer generation."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Non-streaming text generation for testing or internal synthesis."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Asynchronously stream generated text tokens for real-time chat consumption."""
        pass
