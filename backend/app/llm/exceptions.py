"""Application-level exceptions for LLM providers."""


class LLMError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMConfigurationError(LLMError):
    """Raised when provider configuration or API keys are missing or invalid."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when provider authentication fails (e.g. invalid API key)."""
    pass


class LLMQuotaError(LLMError):
    """Raised when provider rate limit or quota is exceeded."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when generation times out."""
    pass


class LLMProviderError(LLMError):
    """Raised when provider returns an unhandled runtime error."""
    pass
