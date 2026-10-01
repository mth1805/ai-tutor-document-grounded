import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.security import get_current_user, AuthenticatedUser
from app.schemas.message import MessageCreate, MessageResponse
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations/{conversation_id}/messages", tags=["messages"])


@router.get(
    "",
    response_model=List[MessageResponse],
    summary="List Conversation Messages",
)
async def list_messages(
    conversation_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[MessageResponse]:
    """Retrieves all messages for a conversation in chronological order."""
    messages = await ConversationService.list_messages(
        db, conversation_id, current_user.id
    )
    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized",
        )
    return [MessageResponse.model_validate(m) for m in messages]


@router.post(
    "",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Message",
)
async def create_message(
    conversation_id: uuid.UUID,
    data: MessageCreate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Persists a new message in the conversation thread (Phase 3 persistence only)."""
    message = await ConversationService.create_message(
        db, conversation_id, current_user.id, data
    )
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized",
        )
    return MessageResponse.model_validate(message)
