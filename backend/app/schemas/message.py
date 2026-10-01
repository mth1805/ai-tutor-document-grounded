import uuid
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Author role"
    )
    content: str = Field(
        ..., min_length=1, description="Message text content"
    )


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
