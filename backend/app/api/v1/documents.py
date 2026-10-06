import uuid
from typing import List
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Response,
    BackgroundTasks,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.security import get_current_user, AuthenticatedUser
from app.schemas.document import (
    DocumentResponse,
    DocumentDownloadResponse,
    DocumentChunkResponse,
    DocumentProcessResponse,
    DocumentEmbedResponse,
    DocumentEmbeddingStatusResponse,
)
from app.services.document_service import DocumentService
from app.services.storage_service import StorageService
from app.services.ingestion_service import IngestionService
from app.services.embedding_service import EmbeddingService
from app.core.config import settings

router = APIRouter(tags=["documents"])


@router.post(
    "/workspaces/{workspace_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Workspace Document",
)
async def upload_document(
    workspace_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, TXT, PNG, JPG, WEBP)"),
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Uploads a document to private storage, records its metadata, and schedules background ingestion."""
    # Read at most one byte beyond the configured maximum so oversized uploads
    # are rejected without buffering an unbounded body in application memory.
    content = await file.read(settings.MAX_FILE_SIZE_BYTES + 1)
    if len(content) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the configured size limit.",
        )
    filename = file.filename or "document.bin"
    mime_type = file.content_type

    doc = await DocumentService.upload_document(
        db=db,
        workspace_id=workspace_id,
        user_id=current_user.id,
        filename=filename,
        content=content,
        mime_type=mime_type,
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found or unauthorized",
        )

    # Database-backed deployments rely on the durable job row committed with
    # the document; only the in-memory development adapter uses BackgroundTasks.
    if db is None:
        background_tasks.add_task(
            IngestionService.process_document_background,
            doc.id,
            current_user.id,
        )

    return DocumentResponse.model_validate(doc)



@router.get(
    "/workspaces/{workspace_id}/documents",
    response_model=List[DocumentResponse],
    summary="List Workspace Documents",
)
async def list_documents(
    workspace_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[DocumentResponse]:
    """Lists all documents belonging to a workspace in reverse chronological order."""
    docs = await DocumentService.list_documents(db, workspace_id, current_user.id)
    if docs is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found or unauthorized",
        )
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document Metadata",
)
async def get_document(
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Retrieves document metadata by ID, ensuring user ownership."""
    doc = await DocumentService.get_document(db, document_id, current_user.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Deletes a document and its stored file, ensuring user ownership."""
    deleted = await DocumentService.delete_document(
        db, document_id, current_user.id
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )


@router.get(
    "/documents/{document_id}/download",
    summary="Download Document File",
)
async def download_document(
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Streams original file contents for authenticated and authorized users."""
    doc, content = await DocumentService.get_document_file(
        db, document_id, current_user.id
    )
    if not doc or content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found or unauthorized",
        )

    return Response(
        content=content,
        media_type=doc.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{doc.original_filename}"',
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get(
    "/documents/{document_id}/url",
    response_model=DocumentDownloadResponse,
    summary="Get Secure Document Access URL",
)
async def get_document_url(
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentDownloadResponse:
    """Returns a secure signed URL or access endpoint for private document retrieval."""
    doc = await DocumentService.get_document(db, document_id, current_user.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )

    signed_url = await StorageService.create_signed_url(doc.storage_path, expires_in=3600)
    if not signed_url or signed_url.startswith("/api/v1"):
        # Local direct fallback
        signed_url = f"/api/v1/documents/{document_id}/download"

    return DocumentDownloadResponse(download_url=signed_url, expires_in=3600)


@router.post(
    "/documents/{document_id}/process",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start or Restart Document Ingestion",
)
async def process_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentProcessResponse:
    """Schedules background parsing, OCR, cleaning, and chunking for an authorized document."""
    doc = await DocumentService.get_document(db, document_id, current_user.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )

    if db is not None:
        try:
            enqueued = await IngestionService.enqueue_document(db, doc, current_user.id)
            if enqueued:
                await db.commit()
                await db.refresh(doc)
        except Exception:
            await db.rollback()
            raise
        response_status = doc.status
    else:
        doc.status = "processing"
        response_status = "processing"
        background_tasks.add_task(
            IngestionService.process_document_background,
            document_id,
            current_user.id,
        )

    return DocumentProcessResponse(
        document_id=document_id,
        status=response_status,
        message="Document ingestion queued." if response_status == "queued" else "Document ingestion is already running.",
    )


@router.get(
    "/documents/{document_id}/chunks",
    response_model=List[DocumentChunkResponse],
    summary="List Document Chunks",
)
async def list_document_chunks(
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[DocumentChunkResponse]:
    """Retrieves all parsed and structure-aware chunks for an authorized document."""
    chunks = await IngestionService.list_document_chunks(
        db, document_id, current_user.id
    )
    if chunks is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )

    return [DocumentChunkResponse.model_validate(c) for c in chunks]


@router.post(
    "/documents/{document_id}/embed",
    response_model=DocumentEmbedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start or Restart Document Embedding",
)
async def embed_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentEmbedResponse:
    """Schedules background BGE-M3 embedding generation and pgvector persistence for an authorized processed document."""
    doc = await DocumentService.get_document(db, document_id, current_user.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )

    if doc.status != "processed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document must be in 'processed' state before generating embeddings (current status: '{doc.status}').",
        )

    # Immediately mark embedding_status as processing
    from app.core.config import settings
    if settings.LOCAL_SHARED_MODELS:
        from app.ml.local_runtime import ready
        if not ready.is_set():
            raise HTTPException(503, "Local models are warming up; retry shortly")
    doc.embedding_status = "processing"
    if db is not None:
        try:
            await db.commit()
            await db.refresh(doc)
        except Exception:
            await db.rollback()

    # Enqueue background task
    background_tasks.add_task(
        EmbeddingService.embed_document_background,
        document_id,
        current_user.id,
    )

    return DocumentEmbedResponse(
        document_id=document_id,
        embedding_status="processing",
        message="Document embedding scheduled successfully.",
    )


@router.get(
    "/documents/{document_id}/embedding-status",
    response_model=DocumentEmbeddingStatusResponse,
    summary="Get Document Embedding Status",
)
async def get_document_embedding_status(
    document_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentEmbeddingStatusResponse:
    """Retrieves embedding progress, chunk statistics, and timestamps for an authorized document."""
    status_info = await EmbeddingService.get_embedding_status(
        db, document_id, current_user.id
    )
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized",
        )

    return DocumentEmbeddingStatusResponse.model_validate(status_info)
