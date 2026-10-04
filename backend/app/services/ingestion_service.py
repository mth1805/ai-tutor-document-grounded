"""Ingestion Service for managing document parsing, chunk persistence, and idempotent reprocessing."""
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool

from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.models.ingestion_job import DocumentIngestionJob
from app.services.document_service import DocumentService
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.ingestion.models import ProcessedChunk
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

# In-memory storage for test/development environments when live database is disabled
_IN_MEMORY_CHUNKS: dict[uuid.UUID, List[DocumentChunk]] = {}


class IngestionService:
    """Coordinates parsing, OCR, cleaning, chunking, and database persistence."""

    @classmethod
    async def enqueue_document(
        cls,
        db: AsyncSession,
        document: Document,
        user_id: uuid.UUID,
    ) -> bool:
        """Persist/re-arm one owner-scoped job; a unique document key prevents duplicates."""
        result = await db.execute(
            select(DocumentIngestionJob)
            .where(
                DocumentIngestionJob.document_id == document.id,
                DocumentIngestionJob.user_id == user_id,
            )
            .with_for_update()
        )
        existing = result.scalar_one_or_none()
        if existing and existing.status in {"queued", "processing"}:
            return False
        if existing:
            await db.delete(existing)
            # Release the per-document unique key within this transaction before
            # inserting the replacement terminal-job retry row.
            await db.flush()

        document.status = "queued"
        document.processing_error = None
        document.processed_at = None
        db.add(DocumentIngestionJob(
            document_id=document.id,
            workspace_id=document.workspace_id,
            user_id=user_id,
            status="queued",
            attempt_count=0,
            max_attempts=3,
        ))
        return True

    @classmethod
    async def process_document(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        pipeline: Optional[IngestionPipeline] = None,
    ) -> bool:
        """Processes an uploaded document, parses text, produces chunks, and commits them atomically.

        Guarantees:
        - Atomic replacement of old chunks (idempotent re-processing).
        - Document status transitions: 'processing' -> 'processed' (or 'failed').
        - Original file in storage is never deleted upon processing failure.
        - Processing runs in threadpool to keep asyncio event loop responsive.

        Returns:
            True if processing and chunk persistence succeeded, False otherwise.
        """
        # 1. Fetch document and verify authorization
        doc = await DocumentService.get_document(db, document_id, user_id)
        if not doc:
            logger.warning("Document %s not found or unauthorized for user %s", document_id, user_id)
            return False

        # 2. Transition status to 'processing'
        now = datetime.now(timezone.utc)
        doc.status = "processing"
        doc.processing_started_at = now
        doc.processing_error = None

        if db is not None:
            try:
                await db.commit()
                await db.refresh(doc)
            except Exception as e:
                logger.error("Failed to update document status to 'processing': %s", e)
                await db.rollback()

        # 3. Retrieve raw file bytes from private storage
        try:
            _, file_bytes = await DocumentService.get_document_file(db, document_id, user_id)
            if not file_bytes:
                raise ValueError(f"Original file for document {document_id} was not found in storage.")

            # 4. Run CPU-bound parsing, cleaning, and chunking in threadpool
            active_pipeline = pipeline or IngestionPipeline()
            chunks: List[ProcessedChunk] = await run_in_threadpool(
                active_pipeline.process,
                file_bytes,
                doc.original_filename,
                doc.mime_type,
            )

            # 5. Atomic persistence: delete previous chunks & insert new chunks in a single transaction
            processed_time = datetime.now(timezone.utc)

            if db is not None:
                # Delete existing chunks for idempotency
                await db.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                )

                # Insert new chunks
                for c in chunks:
                    chunk_model = DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        workspace_id=doc.workspace_id,
                        user_id=doc.user_id,
                        chunk_index=c.chunk_index,
                        content=c.content,
                        page_number_start=c.page_number_start,
                        page_number_end=c.page_number_end,
                        token_count=c.token_count,
                        created_at=processed_time,
                        updated_at=processed_time,
                    )
                    db.add(chunk_model)

                doc.status = "processed"
                doc.processed_at = processed_time
                doc.processing_error = None

                await db.commit()
                await db.refresh(doc)
            else:
                # In-memory persistence
                _IN_MEMORY_CHUNKS[doc.id] = [
                    DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        workspace_id=doc.workspace_id,
                        user_id=doc.user_id,
                        chunk_index=c.chunk_index,
                        content=c.content,
                        page_number_start=c.page_number_start,
                        page_number_end=c.page_number_end,
                        token_count=c.token_count,
                        created_at=processed_time,
                        updated_at=processed_time,
                    )
                    for c in chunks
                ]
                doc.status = "processed"
                doc.processed_at = processed_time
                doc.processing_error = None

            logger.info(
                "Document %s (%s) successfully processed with %d chunks.",
                doc.id,
                doc.original_filename,
                len(chunks),
            )

            # Phase 6 Integration: Automatically trigger embedding for processed document
            from app.core.config import settings
            if settings.AUTO_EMBED_AFTER_INGESTION:
                from app.services.embedding_service import EmbeddingService
                logger.info("Auto-triggering Phase 6 embedding for processed document %s", doc.id)
                if not await EmbeddingService.embed_document(db, doc.id, user_id):
                    raise RuntimeError("Embedding stage failed")

            return True

        except Exception as err:
            logger.error("Processing failed for document %s: %s", document_id, err, exc_info=True)
            failed_time = datetime.now(timezone.utc)
            error_msg = str(err)

            if db is not None:
                try:
                    await db.rollback()
                    # Re-fetch document and mark as failed
                    doc = await DocumentService.get_document(db, document_id, user_id)
                    if doc:
                        doc.status = "failed"
                        doc.processing_error = error_msg
                        doc.processed_at = failed_time
                        await db.commit()
                except Exception as db_err:
                    logger.critical("Failed to record error state for document %s: %s", document_id, db_err)
            else:
                doc.status = "failed"
                doc.processing_error = error_msg
                doc.processed_at = failed_time

            return False

    @classmethod
    async def process_document_background(
        cls,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        pipeline: Optional[IngestionPipeline] = None,
    ) -> bool:
        """Asynchronous background worker execution for document ingestion."""
        if AsyncSessionLocal is None:
            return await cls.process_document(None, document_id, user_id, pipeline=pipeline)

        try:
            from app.db.session import get_db
            from app.main import app
            if get_db in app.dependency_overrides:
                return await cls.process_document(None, document_id, user_id, pipeline=pipeline)
        except Exception:
            pass

        async with AsyncSessionLocal() as session:
            session.info["rls_user_id"] = user_id
            return await cls.process_document(session, document_id, user_id, pipeline=pipeline)

    @classmethod
    async def list_document_chunks(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[List[DocumentChunk]]:
        """Lists all chunks for an authorized document in sequential order."""
        # 1. Verify document ownership
        doc = await DocumentService.get_document(db, document_id, user_id)
        if not doc:
            return None

        # 2. Retrieve chunks
        if db is not None:
            stmt = (
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.user_id == user_id,
                )
                .order_by(DocumentChunk.chunk_index.asc())
            )
            res = await db.execute(stmt)
            return list(res.scalars().all())

        return sorted(
            _IN_MEMORY_CHUNKS.get(document_id, []),
            key=lambda c: c.chunk_index,
        )
