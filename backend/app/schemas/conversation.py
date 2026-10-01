import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ConversationCreate(BaseModel):
    title: str = Field(
        default="New Conversation",
        min_length=1,
        max_length=255,
        description="Conversation thread title",
    )


class ConversationUpdate(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Updated conversation title",
    )


class ConversationResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
