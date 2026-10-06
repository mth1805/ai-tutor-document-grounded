"""Embedding Service for batch vector generation, pgvector persistence, and embedding lifecycle."""
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable, Awaitable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.services.document_service import DocumentService, _IN_MEMORY_DOCUMENTS
from app.services.ingestion_service import _IN_MEMORY_CHUNKS
from app.ml.base import BaseEmbeddingProvider
from app.ml.loader import get_embedding_provider
from app.db.session import AsyncSessionLocal
from app.ml.local_runtime import infer

logger = logging.getLogger(__name__)


class StaleEmbeddingAttempt(RuntimeError):
    """An expired queue attempt must not overwrite a newer attempt's state."""


class EmbeddingService:
    """Manages batch embedding computation, atomic vector persistence, and lifecycle transitions."""

    @classmethod
    async def embed_document(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        provider: Optional[BaseEmbeddingProvider] = None,
        batch_size: Optional[int] = None,
        lease_guard: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> bool:
        """Computes embeddings for all chunks of a processed document and commits them atomically.

        Guarantees:
        - Only documents in 'processed' state can be embedded.
        - Heavy model inference is offloaded via run_in_threadpool.
        - Vectors are written directly to existing canonical chunk records (idempotent; no duplicate chunks).
        - Document lifecycle transitions: 'pending' -> 'processing' -> 'completed' (or 'failed').
        - On failure, transaction is rolled back so partial embeddings are not committed.
        """
        if settings.LOCAL_SHARED_MODELS:
            from app.ml.local_runtime import ready
            if not ready.is_set():
                logger.warning("local_embedding_unavailable models_ready=false")
                return False
        # 1. Fetch document and enforce tenant isolation
        doc = await DocumentService.get_document(db, document_id, user_id)
        if not doc:
            logger.warning("Document %s not found or unauthorized for user %s", document_id, user_id)
            return False
        if lease_guard is not None and not await lease_guard():
            return False

        # 2. Enforce Phase 5 prerequisite: Document MUST be 'processed'
        if doc.status != "processed":
            err_msg = (
                f"Document {document_id} cannot be embedded because its status is '{doc.status}'. "
                "Document must be in 'processed' state after Phase 5 chunking."
            )
            logger.warning(err_msg)
            raise ValueError(err_msg)

        # 3. Transition embedding status to 'processing'
        now = datetime.now(timezone.utc)
        doc.embedding_status = "processing"
        doc.embedding_started_at = now
        doc.embedding_error = None

        if db is not None:
            try:
                await db.commit()
                await db.refresh(doc)
            except Exception as e:
                logger.error("embedding_status_update_failed document_id=%s category=%s", document_id, type(e).__name__)
                await db.rollback()

        try:
            logger.info("embedding_started document_id=%s", document_id)
            # 4. Fetch canonical chunks
            chunks: List[DocumentChunk] = []
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
                chunks = list(res.scalars().all())
            else:
                chunks = [
                    c for c in _IN_MEMORY_CHUNKS.get(document_id, [])
                    if c.user_id == user_id
                ]
                chunks.sort(key=lambda c: c.chunk_index)

            if not chunks:
                if lease_guard is not None and not await lease_guard():
                    raise StaleEmbeddingAttempt()
                logger.info("Document %s has 0 chunks to embed. Marking embedding as completed.", document_id)
                completed_time = datetime.now(timezone.utc)
                doc.embedding_status = "completed"
                doc.embedded_at = completed_time
                doc.embedding_error = None
                if db is not None:
                    await db.commit()
                    await db.refresh(doc)
                logger.info("embedding_completed document_id=%s count=0", document_id)
                return True

            # 5. Extract texts and prepare embedding provider
            active_provider = provider or get_embedding_provider()
            effective_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

            texts = [c.content for c in chunks]

            # 6. Execute batch embedding in threadpool to keep asyncio event loop responsive
            logger.info(
                "Embedding %d chunks for document %s (batch_size=%d, provider=%s, device=%s)...",
                len(texts),
                document_id,
                effective_batch_size,
                active_provider.model_name,
                active_provider.device,
            )

            vectors: List[List[float]] = []
            if settings.LOCAL_SHARED_MODELS:
                for content in texts:
                    vectors.extend(await infer(active_provider.encode_batch, [content],
                        normalize=settings.EMBEDDING_NORMALIZE, batch_size=1, background=True))
            else:
                vectors = await run_in_threadpool(active_provider.encode_batch, texts,
                    normalize=settings.EMBEDDING_NORMALIZE, batch_size=effective_batch_size)

            if len(vectors) != len(chunks):
                raise ValueError(
                    f"Generated vector count ({len(vectors)}) does not match chunk count ({len(chunks)})"
                )

            # 7. Atomic persistence: Update existing canonical chunks with embedding vectors
            if lease_guard is not None and not await lease_guard():
                raise StaleEmbeddingAttempt()
            completed_time = datetime.now(timezone.utc)

            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
                chunk.embedding_model = active_provider.model_name
                chunk.embedding_version = active_provider.version
                chunk.embedded_at = completed_time
                chunk.updated_at = completed_time

            doc.embedding_status = "completed"
            doc.embedded_at = completed_time
            doc.embedding_error = None

            if db is not None:
                await db.commit()
                await db.refresh(doc)
            else:
                # Update in-memory document state
                doc_record = _IN_MEMORY_DOCUMENTS.get(document_id)
                if doc_record:
                    doc_record.embedding_status = "completed"
                    doc_record.embedded_at = completed_time
                    doc_record.embedding_error = None

            logger.info(
                "embedding_completed document_id=%s count=%d dimension=%d model=%s",
                document_id,
                len(chunks),
                active_provider.dimension,
                active_provider.model_name,
            )
            return True

        except Exception as err:
            logger.error("embedding_failed document_id=%s category=%s", document_id, type(err).__name__)
            failed_time = datetime.now(timezone.utc)
            error_msg = str(err)

            if isinstance(err, StaleEmbeddingAttempt):
                if db is not None:
                    await db.rollback()
                return False

            if db is not None:
                try:
                    await db.rollback()
                    # Re-fetch document and persist failed embedding status
                    doc = await DocumentService.get_document(db, document_id, user_id)
                    if doc:
                        doc.embedding_status = "failed"
                        doc.embedding_error = error_msg
                        await db.commit()
                except Exception as db_err:
                    logger.critical(
                        "embedding_failure_record_failed document_id=%s category=%s",
                        document_id,
                        type(db_err).__name__,
                    )
            else:
                doc.embedding_status = "failed"
                doc.embedding_error = error_msg
                doc_record = _IN_MEMORY_DOCUMENTS.get(document_id)
                if doc_record:
                    doc_record.embedding_status = "failed"
                    doc_record.embedding_error = error_msg

            return False

    @classmethod
    async def embed_document_background(
        cls,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        provider: Optional[BaseEmbeddingProvider] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Asynchronous background worker task for document embedding."""
        if AsyncSessionLocal is None:
            await cls.embed_document(None, document_id, user_id, provider=provider, batch_size=batch_size)
            return

        try:
            from app.db.session import get_db
            from app.main import app
            if get_db in app.dependency_overrides:
                await cls.embed_document(None, document_id, user_id, provider=provider, batch_size=batch_size)
                return
        except Exception:
            pass

        async with AsyncSessionLocal() as session:
            session.info["rls_user_id"] = user_id
            await cls.embed_document(session, document_id, user_id, provider=provider, batch_size=batch_size)

    @classmethod
    async def get_embedding_status(
        cls,
        db: Optional[AsyncSession],
        document_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves embedding progress, chunk statistics, and timestamps for an authorized document."""
        doc = await DocumentService.get_document(db, document_id, user_id)
        if not doc:
            return None

        chunks: List[DocumentChunk] = []
        if db is not None:
            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.user_id == user_id,
            )
            res = await db.execute(stmt)
            chunks = list(res.scalars().all())
        else:
            chunks = [
                c for c in _IN_MEMORY_CHUNKS.get(document_id, [])
                if c.user_id == user_id
            ]

        total_chunks = len(chunks)
        embedded_chunks = sum(1 for c in chunks if getattr(c, "embedded_at", None) is not None)
        model_name = next((c.embedding_model for c in chunks if getattr(c, "embedding_model", None)), None)

        return {
            "document_id": doc.id,
            "embedding_status": getattr(doc, "embedding_status", "pending"),
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "embedding_model": model_name or settings.EMBEDDING_MODEL_NAME,
            "embedding_started_at": getattr(doc, "embedding_started_at", None),
            "embedded_at": getattr(doc, "embedded_at", None),
            "embedding_error": getattr(doc, "embedding_error", None),
        }
