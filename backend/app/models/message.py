import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Any
from sqlalchemy import String, Text, DateTime, ForeignKey, ForeignKeyConstraint, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.workspace import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    __table_args__ = (
        UniqueConstraint("id", "conversation_id", "role", name="messages_id_conversation_role_key"),
        UniqueConstraint("id", "conversation_id", "generation_id", name="messages_id_conversation_generation_key"),
        ForeignKeyConstraint(
            ["generation_id", "conversation_id"],
            ["chat_generation_runs.generation_id", "chat_generation_runs.conversation_id"],
            ondelete="CASCADE",
            name="messages_generation_run_fkey",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[Any]] = mapped_column(
        JSONB, default=list, nullable=True
    )
    generation_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    generation_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    generation_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
