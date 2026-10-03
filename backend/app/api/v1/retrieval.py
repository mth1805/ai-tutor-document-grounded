"""Retrieval API router for Phase 7 Hybrid Search and Reranking."""
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthenticatedUser, get_current_user
from app.db.session import get_db
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.services.workspace_service import WorkspaceService
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces/{workspace_id}/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Execute Hybrid Retrieval and Cross-Encoder Reranking",
    description=(
        "Retrieves and reranks document chunks within an authorized workspace using "
        "pgvector dense semantic search, PostgreSQL FTS lexical search, RRF fusion, "
        "and Cross-Encoder reranking."
    ),
)
async def search_workspace_chunks(
    workspace_id: uuid.UUID,
    payload: RetrievalRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: Optional[AsyncSession] = Depends(get_db),
) -> RetrievalResponse:
    """Executes the Phase 7 retrieval pipeline for the authenticated workspace owner."""
    # 1. Enforce strict tenant isolation: verify that workspace exists and belongs to current user
    workspace = await WorkspaceService.get_workspace(db, workspace_id, current_user.id)
    if not workspace:
        logger.warning(
            "Unauthorized or missing workspace %s accessed by user %s",
            workspace_id,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found or unauthorized.",
        )

    # 2. Execute retrieval pipeline
    try:
        response = await RetrievalService.retrieve(
            db=db,
            workspace_id=workspace_id,
            user_id=current_user.id,
            request=payload,
        )
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(
            "Retrieval pipeline failed for workspace %s: %s",
            workspace_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during retrieval and reranking.",
        )
