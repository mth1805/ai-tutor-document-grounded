from sqlalchemy import DateTime, SmallInteger

from app.models.chat_generation_run import ChatGenerationContextSnapshot, ChatGenerationRun
from app.models.conversation import Conversation


def test_chat_generation_run_metadata_maps_nullable_timezone_aware_finished_at():
    column = ChatGenerationRun.__table__.c.finished_at

    assert isinstance(column.type, DateTime)
    assert column.type.timezone is True
    assert column.nullable is True


def test_chat_generation_run_metadata_maps_all_migration_columns():
    expected_columns = {
        "generation_id",
        "conversation_id",
        "user_id",
        "user_message_id",
        "user_message_role",
        "request_fingerprint",
        "chat_mode",
        "workspace_id",
        "document_id",
        "active_document_id",
        "state",
        "termination_reason",
        "cancel_reason",
        "assistant_message_id",
        "dispatch_attempts",
        "dispatch_token",
        "dispatch_lease_expires_at",
        "next_dispatch_at",
        "last_dispatch_error",
        "modal_call_id",
        "worker_token",
        "worker_epoch",
        "worker_heartbeat_at",
        "worker_lease_expires_at",
        "disconnect_grace_started_at",
        "generation_metadata",
        "created_at",
        "updated_at",
        "finished_at",
    }

    assert set(ChatGenerationRun.__table__.columns.keys()) == expected_columns


def test_chat_generation_context_snapshot_maps_private_migration_columns():
    expected = {
        "generation_id", "snapshot_version", "snapshot_payload", "canonical_snapshot",
        "context_hash", "canonical_byte_count", "jsonb_rendered_byte_count", "created_at",
    }
    assert set(ChatGenerationContextSnapshot.__table__.columns.keys()) == expected
    assert isinstance(ChatGenerationContextSnapshot.__table__.c.snapshot_version.type, SmallInteger)
    assert Conversation.__table__.c.conversation_version.type.python_type is int
