import uuid
from datetime import datetime
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CitationItem(BaseModel):
    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    page_start: int
    page_end: int
    snippet: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Author role"
    )
    content: str = Field(
        ..., min_length=1, description="Message text content"
    )
    citations: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Structured document citations"
    )


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    citations: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Structured document citations"
    )
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
