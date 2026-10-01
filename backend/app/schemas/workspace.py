import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class WorkspaceBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Workspace display name")


class WorkspaceCreate(WorkspaceBase):
    pass


class WorkspaceUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Updated workspace name")


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
