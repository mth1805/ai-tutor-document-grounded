import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.security import get_current_user, AuthenticatedUser
from app.schemas.conversation import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
)
from app.services.conversation_service import ConversationService

router = APIRouter(tags=["conversations"])


@router.get(
    "/workspaces/{workspace_id}/conversations",
    response_model=List[ConversationResponse],
    summary="List Workspace Conversations",
)
async def list_workspace_conversations(
    workspace_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[ConversationResponse]:
    """Retrieves all conversations belonging to the specified workspace."""
    conversations = await ConversationService.list_conversations(
        db, workspace_id, current_user.id
    )
    if conversations is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found or unauthorized",
        )
    return [ConversationResponse.model_validate(c) for c in conversations]


@router.post(
    "/workspaces/{workspace_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Conversation",
)
async def create_workspace_conversation(
    workspace_id: uuid.UUID,
    data: ConversationCreate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Creates a new conversation thread inside the specified workspace."""
    conversation = await ConversationService.create_conversation(
        db, workspace_id, current_user.id, data
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found or unauthorized",
        )
    return ConversationResponse.model_validate(conversation)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get Conversation Details",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Retrieves details of a specific conversation."""
    conversation = await ConversationService.get_conversation(
        db, conversation_id, current_user.id
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized",
        )
    return ConversationResponse.model_validate(conversation)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Rename Conversation",
)
async def update_conversation(
    conversation_id: uuid.UUID,
    data: ConversationUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Renames an existing conversation after verifying ownership."""
    conversation = await ConversationService.update_conversation(
        db, conversation_id, current_user.id, data
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized",
        )
    return ConversationResponse.model_validate(conversation)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Conversation",
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Deletes a conversation after verifying ownership."""
    deleted = await ConversationService.delete_conversation(
        db, conversation_id, current_user.id
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or unauthorized",
        )
