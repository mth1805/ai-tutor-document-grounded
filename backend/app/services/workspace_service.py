import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.workspace import Workspace
from app.schemas.workspace import WorkspaceCreate, WorkspaceUpdate

# In-memory storage fallback for local development/testing without live PostgreSQL
_IN_MEMORY_WORKSPACES: dict[uuid.UUID, Workspace] = {}


class WorkspaceService:
    @staticmethod
    async def list_workspaces(
        db: Optional[AsyncSession], user_id: uuid.UUID
    ) -> List[Workspace]:
        """Lists all workspaces belonging to the authenticated user."""
        if db is not None:
            stmt = (
                select(Workspace)
                .where(Workspace.user_id == user_id)
                .order_by(Workspace.created_at.desc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

        # In-memory fallback
        return [
            ws
            for ws in _IN_MEMORY_WORKSPACES.values()
            if ws.user_id == user_id
        ]

    @staticmethod
    async def get_workspace(
        db: Optional[AsyncSession], workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> Optional[Workspace]:
        """Gets a single workspace, ensuring user ownership."""
        if db is not None:
            stmt = select(Workspace).where(
                Workspace.id == workspace_id, Workspace.user_id == user_id
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

        ws = _IN_MEMORY_WORKSPACES.get(workspace_id)
        if ws and ws.user_id == user_id:
            return ws
        return None

    @staticmethod
    async def create_workspace(
        db: Optional[AsyncSession], user_id: uuid.UUID, data: WorkspaceCreate
    ) -> Workspace:
        """Creates a new workspace assigned to the authenticated user."""
        now = datetime.now(timezone.utc)
        workspace = Workspace(
            id=uuid.uuid4(),
            user_id=user_id,
            name=data.name.strip(),
            created_at=now,
            updated_at=now,
        )

        if db is not None:
            db.add(workspace)
            await db.commit()
            await db.refresh(workspace)
            return workspace

        _IN_MEMORY_WORKSPACES[workspace.id] = workspace
        return workspace

    @staticmethod
    async def update_workspace(
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        data: WorkspaceUpdate,
    ) -> Optional[Workspace]:
        """Renames a workspace after verifying user ownership."""
        now = datetime.now(timezone.utc)

        if db is not None:
            stmt = (
                update(Workspace)
                .where(Workspace.id == workspace_id, Workspace.user_id == user_id)
                .values(name=data.name.strip(), updated_at=now)
                .returning(Workspace)
            )
            result = await db.execute(stmt)
            updated = result.scalar_one_or_none()
            if updated:
                await db.commit()
            return updated

        ws = _IN_MEMORY_WORKSPACES.get(workspace_id)
        if ws and ws.user_id == user_id:
            ws.name = data.name.strip()
            ws.updated_at = now
            return ws
        return None

    @staticmethod
    async def delete_workspace(
        db: Optional[AsyncSession], workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        """Deletes a workspace after verifying user ownership."""
        if db is not None:
            # Query documents in workspace to clean up their storage files
            from app.models.document import Document
            from app.services.storage_service import StorageService

            doc_stmt = select(Document.storage_path).where(
                Document.workspace_id == workspace_id,
                Document.user_id == user_id,
            )
            doc_res = await db.execute(doc_stmt)
            storage_paths = list(doc_res.scalars().all())

            stmt = delete(Workspace).where(
                Workspace.id == workspace_id, Workspace.user_id == user_id
            )
            result = await db.execute(stmt)
            if result.rowcount > 0:
                await db.commit()
                for path in storage_paths:
                    await StorageService.delete_file(path)
                return True
            return False

        ws = _IN_MEMORY_WORKSPACES.get(workspace_id)
        if ws and ws.user_id == user_id:
            del _IN_MEMORY_WORKSPACES[workspace_id]
            # Match the database ON DELETE CASCADE behavior in local mode.
            from app.services.conversation_service import (
                _IN_MEMORY_CONVERSATIONS,
                _IN_MEMORY_MESSAGES,
            )
            from app.services.document_service import _IN_MEMORY_DOCUMENTS
            from app.services.storage_service import StorageService

            doc_ids = [
                d_id
                for d_id, d in _IN_MEMORY_DOCUMENTS.items()
                if d.workspace_id == workspace_id
            ]
            for d_id in doc_ids:
                doc = _IN_MEMORY_DOCUMENTS.pop(d_id, None)
                if doc:
                    await StorageService.delete_file(doc.storage_path)

            conversation_ids = {
                conversation_id
                for conversation_id, conversation in _IN_MEMORY_CONVERSATIONS.items()
                if conversation.workspace_id == workspace_id
            }
            for conversation_id in conversation_ids:
                del _IN_MEMORY_CONVERSATIONS[conversation_id]
            for message_id, message in list(_IN_MEMORY_MESSAGES.items()):
                if message.conversation_id in conversation_ids:
                    del _IN_MEMORY_MESSAGES[message_id]
            return True
        return False
