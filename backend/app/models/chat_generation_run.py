"""Durable chat generation, private checkpoint, and subscriber lease models."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, CHAR, CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.workspace import Base


class ChatGenerationRun(Base):
    __tablename__ = "chat_generation_runs"

    generation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_message_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_message_role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    request_fingerprint: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    chat_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    workspace_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    active_document_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    state: Mapped[str] = mapped_column(String(24), nullable=False, default="queued")
    termination_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cancel_reason: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    assistant_message_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    dispatch_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dispatch_token: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    dispatch_lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    next_dispatch_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_dispatch_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    modal_call_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    worker_token: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    worker_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    worker_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    disconnect_grace_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    generation_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("state IN ('queued','dispatching','running','cancel_requested','completed','incomplete','interrupted','blocked','failed','unverified')", name="chat_generation_runs_state_check"),
        CheckConstraint("(state IN ('queued','dispatching','running','cancel_requested') AND assistant_message_id IS NULL AND finished_at IS NULL AND termination_reason IS NULL) OR (state IN ('completed','incomplete','interrupted','blocked','failed','unverified') AND assistant_message_id IS NOT NULL AND finished_at IS NOT NULL AND termination_reason IS NOT NULL)", name="chat_generation_runs_terminal_shape_check"),
        UniqueConstraint("generation_id", "conversation_id", name="chat_generation_runs_generation_conversation_key"),
        ForeignKeyConstraint(["conversation_id", "user_id"], ["conversations.id", "conversations.user_id"], ondelete="CASCADE", name="chat_generation_runs_conversation_owner_fkey"),
        ForeignKeyConstraint(["user_message_id", "conversation_id", "user_message_role"], ["messages.id", "messages.conversation_id", "messages.role"], ondelete="CASCADE", name="chat_generation_runs_user_message_fkey"),
        ForeignKeyConstraint(["assistant_message_id", "conversation_id", "generation_id"], ["messages.id", "messages.conversation_id", "messages.generation_id"], ondelete="NO ACTION", deferrable=True, initially="DEFERRED", name="chat_generation_runs_assistant_message_fkey"),
        Index("chat_generation_one_active_per_conversation_idx", "conversation_id", unique=True,
              postgresql_where=text("state IN ('queued','dispatching','running','cancel_requested')")),
    )


class ChatGenerationCheckpoint(Base):
    __tablename__ = "chat_generation_private_checkpoints"

    generation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_generation_runs.generation_id", ondelete="CASCADE"), primary_key=True)
    checkpoint_delta: Mapped[str] = mapped_column(Text, nullable=False, default="")
    checkpoint_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    partial_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    safe_partial_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    worker_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ChatGenerationSubscriber(Base):
    __tablename__ = "chat_generation_subscribers"

    generation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_generation_runs.generation_id", ondelete="CASCADE"), primary_key=True)
    subscriber_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_checkpoint_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
