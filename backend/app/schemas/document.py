import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, computed_field


class DocumentResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    original_filename: str
    storage_path: str
    mime_type: str
    file_size: int
    status: str
    created_at: datetime
    updated_at: datetime
    processing_started_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    download_url: Optional[str] = None

    # Phase 6: Embedding lifecycle and status
    embedding_status: Optional[str] = "pending"
    embedding_started_at: Optional[datetime] = None
    embedded_at: Optional[datetime] = None
    embedding_error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if hasattr(obj, "embedding_status") and getattr(obj, "embedding_status", None) is None:
            try:
                setattr(obj, "embedding_status", "pending")
            except Exception:
                pass
        return super().model_validate(obj, *args, **kwargs)


class DocumentDownloadResponse(BaseModel):
    download_url: str
    expires_in: int


class DocumentChunkResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    chunk_index: int
    content: str
    page_number_start: int
    page_number_end: int
    token_count: int
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None
    embedded_at: Optional[datetime] = None
    created_at: datetime

    @computed_field
    def has_embedding(self) -> bool:
        return self.embedded_at is not None

    model_config = ConfigDict(from_attributes=True)


class DocumentProcessResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    message: str


class DocumentEmbedResponse(BaseModel):
    document_id: uuid.UUID
    embedding_status: str
    message: str


class DocumentEmbeddingStatusResponse(BaseModel):
    document_id: uuid.UUID
    embedding_status: str
    total_chunks: int
    embedded_chunks: int
    embedding_model: Optional[str] = None
    embedding_started_at: Optional[datetime] = None
    embedded_at: Optional[datetime] = None
    embedding_error: Optional[str] = None
