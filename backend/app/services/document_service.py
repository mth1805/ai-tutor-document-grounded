import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from app.models.document import Document
from app.models.ingestion_job import DocumentIngestionJob
from app.services.workspace_service import WorkspaceService
from app.services.ingestion.telemetry import event
from app.services.storage_service import (
    StorageService,
    validate_upload_file,
)

logger = logging.getLogger(__name__)

# In-memory storage fallback for local development/testing without live PostgreSQL
_IN_MEMORY_DOCUMENTS: dict[uuid.UUID, Document] = {}


class DocumentService:
    """Business logic for document metadata persistence, storage coordination, and authorization."""

    @classmethod
    async def upload_document(
        cls,
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str,
        content: bytes,
        mime_type: Optional[str] = None,
    ) -> Optional[Document]:
        """Validates file, uploads to private storage, and persists metadata.

        Returns:
            Created Document model, or None if workspace not found or unauthorized.
        """
        # 1. Authorize: verify workspace exists and belongs to the user
        workspace = await WorkspaceService.get_workspace(db, workspace_id, user_id)
        if not workspace:
            return None

        # 2. Validate file size, extension, and content magic bytes (raises 400 on error)
        clean_name, validated_mime, file_size = validate_upload_file(
            filename=filename, content=content, mime_type=mime_type
        )

        # 3. Generate deterministic IDs and workspace-scoped storage path
        document_id = uuid.uuid4()
        storage_path = StorageService.generate_storage_path(
            workspace_id=workspace_id,
            document_id=document_id,
            filename=clean_name,
        )

        # 4. Upload original file to private Storage
        await StorageService.upload_file(
            storage_path=storage_path,
            content=content,
            mime_type=validated_mime,
        )

        now = datetime.now(timezone.utc)
        doc = Document(
            id=document_id,
            workspace_id=workspace_id,
            user_id=user_id,
            original_filename=clean_name,
            storage_path=storage_path,
            mime_type=validated_mime,
            file_size=file_size,
            status="uploaded",
            embedding_status="pending",
            created_at=now,
            updated_at=now,
        )

        # 5. Persist metadata in PostgreSQL with transactional rollback on failure
        if db is not None:
            try:
                doc.status = "queued"
                db.add(doc)
                job = DocumentIngestionJob(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    workspace_id=doc.workspace_id,
                    user_id=doc.user_id,
                    status="queued",
                    attempt_count=0,
                    max_attempts=3,
                )
                db.add(job)
                await db.commit()
            except Exception as e:
                logger.error(
                    "upload_metadata_failed document_id=%s category=%s",
                    document_id, type(e).__name__,
                )
                # A lost commit acknowledgement is ambiguous. Check for a
                # committed row before compensation; if the DB is unavailable,
                # retain the object for reconciliation rather than breaking a job.
                try:
                    await db.rollback()
                    committed = await cls.get_document(db, document_id, user_id)
                    if committed is None and not await StorageService.delete_file(storage_path):
                        logger.error("upload_compensation_failed document_id=%s", document_id)
                except Exception as cleanup_error:
                    logger.error("upload_compensation_failed document_id=%s category=%s", document_id, type(cleanup_error).__name__)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to record document metadata in database",
                )
            # The metadata/job transaction has committed. A refresh failure must
            # never delete the object now referenced by that committed transaction.
            await db.refresh(doc)
            event("queued", document_id=str(document_id), job_id=str(job.id), attempt=0)
            return doc

        # In-memory fallback
        _IN_MEMORY_DOCUMENTS[doc.id] = doc
        return doc

    @classmethod
    async def list_documents(
        cls,
        db: Optional[AsyncSession],
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[List[Document]]:
        """Lists all documents for a workspace after verifying user ownership."""
        workspace = await WorkspaceService.get_workspace(db, workspace_id, user_id)
        if not workspace:
            return None

        if db is not None:
            stmt = (
                select(Document)
                .where(
                    Document.workspace_id == workspace_id,
                    Document.user_id == user_id,
                )
                .order_by(Document.created_at.desc())
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())

        return sorted(
            [
                d
                for d in _IN_MEMORY_DOCUMENTS.values()
                if d.workspace_id == workspace_id and d.user_id == user_id
            ],
            key=lambda x: x.created_at,
            reverse=True,
        )

    @classmethod
    async def get_document(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[Document]:
        """Gets a document by ID ensuring user ownership."""
        if db is not None:
            stmt = select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
            result = await db.execute(stmt)
            return result.scalar_one_or_none()

        doc = _IN_MEMORY_DOCUMENTS.get(document_id)
        if doc and doc.user_id == user_id:
            return doc
        return None

    @classmethod
    async def delete_document(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """Deletes a document metadata record and removes its original file from storage."""
        doc = await cls.get_document(db, document_id, user_id)
        if not doc:
            return False

        storage_path = doc.storage_path

        if db is not None:
            stmt = delete(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
            result = await db.execute(stmt)
            if result.rowcount > 0:
                await db.commit()
                # Remove file from storage
                await StorageService.delete_file(storage_path)
                return True
            return False

        del _IN_MEMORY_DOCUMENTS[document_id]
        await StorageService.delete_file(storage_path)
        return True

    @classmethod
    async def get_document_file(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> tuple[Optional[Document], Optional[bytes]]:
        """Retrieves document metadata and raw file bytes after verifying authorization."""
        doc = await cls.get_document(db, document_id, user_id)
        if not doc:
            return None, None

        file_bytes = await StorageService.get_file(doc.storage_path)
        return doc, file_bytes
