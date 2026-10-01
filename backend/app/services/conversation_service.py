import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.workspace import Workspace
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.conversation import ConversationCreate, ConversationUpdate
from app.schemas.message import MessageCreate
from app.services.workspace_service import WorkspaceService, _IN_MEMORY_WORKSPACES

# In-memory storage fallback for local development/testing without live PostgreSQL
_IN_MEMORY_CONVERSATIONS: dict[uuid.UUID, Conversation] = {}
_IN_MEMORY_MESSAGES: dict[uuid.UUID, Message] = {}


class ConversationService:
    @staticmethod
    async def list_conversations(
        db: Optional[AsyncSession], workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[List[Conversation]]:
        """Lists all conversations for a workspace after verifying user ownership."""
        # Check workspace ownership first (IDOR defense)
        workspace = await WorkspaceService.get_workspace(db, workspace_id, user_id)
        if not workspace:
            return None

        if db is not None:
            stmt = (
                select(Conversation)
                .where(
                    Conversation.workspace_id == workspace_id,
                    Conversation.user_id == user_id,
                )
                .order_by(Conversation.created_at.desc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

        return [
            c
            for c in _IN_MEMORY_CONVERSATIONS.values()
            if c.workspace_id == workspace_id and c.user_id == user_id
        ]

    @staticmethod
    async def get_conversation(
        db: Optional[AsyncSession], conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[Conversation]:
        """Gets a conversation ensuring user ownership."""
        if db is not None:
            stmt = select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

        c = _IN_MEMORY_CONVERSATIONS.get(conversation_id)
        if c and c.user_id == user_id:
            return c
        return None

    @staticmethod
    async def create_conversation(
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        data: ConversationCreate,
    ) -> Optional[Conversation]:
        """Creates a new conversation inside the specified workspace."""
        # Verify workspace exists and belongs to user
        workspace = await WorkspaceService.get_workspace(db, workspace_id, user_id)
        if not workspace:
            return None

        now = datetime.now(timezone.utc)
        conversation = Conversation(
            id=uuid.uuid4(),
            workspace_id=workspace_id,
            user_id=user_id,
            title=data.title.strip() if data.title else "New Conversation",
            created_at=now,
            updated_at=now,
        )

        if db is not None:
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            return conversation

        _IN_MEMORY_CONVERSATIONS[conversation.id] = conversation
        return conversation

    @staticmethod
    async def update_conversation(
        db: Optional[AsyncSession],
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        data: ConversationUpdate,
    ) -> Optional[Conversation]:
        """Renames a conversation after verifying user ownership."""
        now = datetime.now(timezone.utc)

        if db is not None:
            stmt = (
                update(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
                .values(title=data.title.strip(), updated_at=now)
                .returning(Conversation)
            )
            result = await db.execute(stmt)
            updated = result.scalar_one_or_none()
            if updated:
                await db.commit()
            return updated

        c = _IN_MEMORY_CONVERSATIONS.get(conversation_id)
        if c and c.user_id == user_id:
            c.title = data.title.strip()
            c.updated_at = now
            return c
        return None

    @staticmethod
    async def delete_conversation(
        db: Optional[AsyncSession], conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        """Deletes a conversation after verifying user ownership (cascades to messages)."""
        if db is not None:
            stmt = delete(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            result = await db.execute(stmt)
            if result.rowcount > 0:
                await db.commit()
                return True
            return False

        c = _IN_MEMORY_CONVERSATIONS.get(conversation_id)
        if c and c.user_id == user_id:
            del _IN_MEMORY_CONVERSATIONS[conversation_id]
            # Cascade delete in-memory messages
            to_delete = [
                m_id
                for m_id, m in _IN_MEMORY_MESSAGES.items()
                if m.conversation_id == conversation_id
            ]
            for m_id in to_delete:
                del _IN_MEMORY_MESSAGES[m_id]
            return True
        return False

    @staticmethod
    async def list_messages(
        db: Optional[AsyncSession], conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[List[Message]]:
        """Lists messages for a conversation, verifying conversation ownership."""
        conversation = await ConversationService.get_conversation(
            db, conversation_id, user_id
        )
        if not conversation:
            return None

        if db is not None:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

        return sorted(
            [
                m
                for m in _IN_MEMORY_MESSAGES.values()
                if m.conversation_id == conversation_id
            ],
            key=lambda x: x.created_at,
        )

    @staticmethod
    async def create_message(
        db: Optional[AsyncSession],
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        data: MessageCreate,
    ) -> Optional[Message]:
        """Appends a new message to a conversation after verifying ownership."""
        conversation = await ConversationService.get_conversation(
            db, conversation_id, user_id
        )
        if not conversation:
            return None

        now = datetime.now(timezone.utc)
        message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role=data.role,
            content=data.content.strip(),
            created_at=now,
        )

        if db is not None:
            db.add(message)
            # Touch conversation updated_at
            conversation.updated_at = now
            await db.commit()
            await db.refresh(message)
            return message

        _IN_MEMORY_MESSAGES[message.id] = message
        conversation.updated_at = now
        return message
