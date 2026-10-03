"""Chat API router for Phase 8 Grounded RAG Generation and Streaming."""
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import AuthenticatedUser, get_current_user
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGService
from app.rag.prompt_builder import ChatMode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations/{conversation_id}/chat", tags=["chat"])


@router.post(
    "",
    summary="Stream Grounded AI Tutor Response",
    description=(
        "Retrieves workspace document evidence using Phase 7 hybrid search, "
        "enforces evidence gating, and streams grounded response tokens via Server-Sent Events (SSE)."
    ),
)
async def stream_chat_message(
    conversation_id: uuid.UUID,
    payload: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: Optional[AsyncSession] = Depends(get_db),
) -> StreamingResponse:
    """Streams grounded response tokens for a user query."""
    # 1. Authorize: verify conversation exists and belongs to current user
    conversation = await ConversationService.get_conversation(db, conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized.",
        )

    # 2. Validate workspace consistency
    if payload.workspace_id and payload.workspace_id != conversation.workspace_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Specified workspace does not match conversation.",
        )

    # 3. Validate chat mode server-side
    try:
        ChatMode.from_str(payload.chat_mode)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )

    stream_generator = RAGService.stream_chat(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        query=payload.content,
        chat_mode_str=payload.chat_mode,
        workspace_id=payload.workspace_id,
    )

    return StreamingResponse(
        stream_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/sync",
    response_model=ChatResponse,
    summary="Synchronous Grounded AI Tutor Response",
    description="Executes grounded RAG generation non-streamed (for testing and benchmarks).",
)
async def sync_chat_message(
    conversation_id: uuid.UUID,
    payload: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: Optional[AsyncSession] = Depends(get_db),
) -> ChatResponse:
    """Non-streaming generation returning the complete assembled assistant response."""
    conversation = await ConversationService.get_conversation(db, conversation_id, current_user.id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized.",
        )

    if payload.workspace_id and payload.workspace_id != conversation.workspace_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Specified workspace does not match conversation.",
        )

    try:
        ChatMode.from_str(payload.chat_mode)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )

    result = await RAGService.generate_chat(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id,
        query=payload.content,
        chat_mode_str=payload.chat_mode,
        workspace_id=payload.workspace_id,
    )

    return ChatResponse(
        message_id=result.get("message_id"),
        conversation_id=conversation_id,
        content=result.get("content", ""),
        citations=result.get("citations", []),
        has_sufficient_evidence=result.get("has_sufficient_evidence", True),
    )
