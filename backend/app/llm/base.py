"""Base abstract class for LLM providers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import time


@dataclass(frozen=True)
class GenerationResult:
    """Generated text and safe operational metadata for a single model call."""

    text: str
    completed: bool = False
    finish_reason: str | None = None
    candidate_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thoughts_tokens: int | None = None
    model_version: str | None = None
    response_id: str | None = None
    generation_duration_ms: float | None = None
    time_to_first_token_ms: float | None = None
    generation_status: str | None = None
    error_type: str | None = None
    error_classification: str | None = None
    http_status: int | None = None

    def __post_init__(self) -> None:
        status = self.generation_status
        if status is None:
            reason = (self.finish_reason or "").upper()
            if reason == "STOP" or (self.completed and not reason):
                status = "completed"
            elif reason == "MAX_TOKENS":
                status = "incomplete"
            elif reason in {"SAFETY", "RECITATION"}:
                status = "blocked"
            else:
                status = "unverified"
            object.__setattr__(self, "generation_status", status)
        object.__setattr__(self, "completed", status == "completed")


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
        """Legacy non-streaming providers cannot verify a normal finish reason."""
        text = await self.generate(prompt, system_instruction=system_instruction, **kwargs)
        return GenerationResult(text=text or "", generation_status="unverified")

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Asynchronously stream generated text tokens for real-time chat consumption."""
        pass

    async def generate_stream_with_metadata(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """Buffer a legacy provider stream and conservatively report its outcome.

        Existing provider interfaces remain valid. Providers that do not expose a
        finish reason are explicitly unverified even when their iterator exhausts.
        """
        started = time.perf_counter()
        chunks: list[str] = []
        first_token_ms = None
        try:
            async for chunk in self.generate_stream(
                prompt, system_instruction=system_instruction, **kwargs
            ):
                if chunk:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    chunks.append(chunk)
            return GenerationResult(
                text="".join(chunks),
                generation_status="unverified",
                generation_duration_ms=(time.perf_counter() - started) * 1000,
                time_to_first_token_ms=first_token_ms,
            )
        except Exception as exc:
            name = type(exc).__name__
            class_names = {cls.__name__ for cls in type(exc).__mro__}
            interrupted = bool(class_names & {
                "LLMTimeoutError", "TimeoutError", "TimeoutException", "ReadTimeout", "ConnectTimeout",
                "ConnectionError", "ConnectionResetError", "NetworkError", "RequestError", "ReadError",
                "WriteError", "ConnectError", "RemoteProtocolError", "OSError",
            })
            details = getattr(exc, "details", None)
            http_status = details.get("http_status") if isinstance(details, dict) else None
            error_classification = details.get("classification") if isinstance(details, dict) else None
            return GenerationResult(
                text="".join(chunks),
                generation_status="interrupted" if interrupted else "failed",
                generation_duration_ms=(time.perf_counter() - started) * 1000,
                time_to_first_token_ms=first_token_ms,
                error_type=name,
                error_classification=error_classification if isinstance(error_classification, str) else None,
                http_status=http_status if isinstance(http_status, int) else None,
            )
