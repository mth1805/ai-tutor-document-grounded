"""Schemas for Phase 8 Grounded RAG Chat API."""
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Payload for initiating grounded chat generation with streaming answer."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User prompt or question grounded in uploaded workspace documents.",
    )
    chat_mode: str = Field(
        default="Detailed Guidance",
        description="Guidance mode: 'Light Guidance', 'Detailed Guidance', or 'Full Solution'.",
    )
    workspace_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional workspace ID verification against parent conversation.",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message content cannot be empty or contain only whitespace.")
        return stripped


class ChatResponse(BaseModel):
    """Payload returned for synchronous (non-streamed) chat generation."""

    message_id: Optional[str] = None
    conversation_id: uuid.UUID
    content: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    has_sufficient_evidence: bool = True
