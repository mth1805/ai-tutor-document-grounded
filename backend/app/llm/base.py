"""Base abstract class for LLM providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass(frozen=True)
class GenerationResult:
    """Text plus completion status for callers that must reject partial output."""

    text: str
    completed: bool = True
    finish_reason: str | None = None


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

    async def generate_with_metadata(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """Backward-compatible completion metadata; legacy providers are assumed complete."""
        text = await self.generate(prompt, system_instruction=system_instruction, **kwargs)
        return GenerationResult(text=text or "", completed=True)

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Asynchronously stream generated text tokens for real-time chat consumption."""
        pass
