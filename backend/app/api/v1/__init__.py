"""API v1 package."""
from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.workspaces import router as workspaces_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.messages import router as messages_router
from app.api.v1.documents import router as documents_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router, prefix="", tags=["health"])
api_v1_router.include_router(workspaces_router, prefix="", tags=["workspaces"])
api_v1_router.include_router(conversations_router, prefix="", tags=["conversations"])
api_v1_router.include_router(messages_router, prefix="", tags=["messages"])
api_v1_router.include_router(documents_router, prefix="", tags=["documents"])
