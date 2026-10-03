import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional, List, Any
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func, UniqueConstraint, Computed
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.models.workspace import Base

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.workspace import Workspace


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Phase 6: Vector embeddings and reproducibility metadata
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedding_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Phase 7: Multilingual full-text search tsvector (Vietnamese + English via simple + english)
    tsv: Mapped[Optional[Any]] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', coalesce(content, '')) || to_tsvector('english', coalesce(content, ''))"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="document_chunks_doc_chunk_uniq"),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="chunks"
    )
    workspace: Mapped["Workspace"] = relationship(
        "Workspace"
    )
