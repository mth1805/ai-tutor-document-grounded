"""Private local worker handoff; never mounted in production or Modal."""
import asyncio
import hashlib
import hmac
import uuid
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.ml.local_runtime import ready
from app.services.embedding_service import EmbeddingService
from app.services.document_service import DocumentService
from app.workers.ingestion_queue import load_claimed_job

router = APIRouter()
_active: dict[tuple[uuid.UUID, int], asyncio.Task] = {}


def worker_token() -> str:
    key = settings.SUPABASE_SERVICE_ROLE_KEY
    if not key:
        raise RuntimeError("Local worker authentication is not configured")
    return hmac.new(key.encode(), b"ai-tutor-local-model-handoff-v1", hashlib.sha256).hexdigest()


def authorize(authorization: str | None):
    if not authorization or not hmac.compare_digest(authorization.encode(), ("Bearer " + worker_token()).encode()):
        raise HTTPException(401, "Invalid local worker credentials")


class EmbedAttempt(BaseModel):
    job_id: uuid.UUID
    attempt: int = Field(ge=1, le=10)


class ModelState(BaseModel):
    ready: bool


class EmbedResult(BaseModel):
    succeeded: bool


@router.get("/internal/local-models", response_model=ModelState)
async def model_state(authorization: str | None = Header(default=None)):
    authorize(authorization)
    return ModelState(ready=ready.is_set())


async def embed_attempt(request: EmbedAttempt) -> bool:
    job = await load_claimed_job(request.job_id)
    if not job or int(job["claimed_attempt"]) != request.attempt:
        raise HTTPException(409, "Stale or unavailable ingestion attempt")
    document_id = uuid.UUID(job["claimed_document_id"])
    user_id = uuid.UUID(job["claimed_user_id"])
    if AsyncSessionLocal is None:
        raise HTTPException(503, "Database unavailable")
    async with AsyncSessionLocal() as session:
        session.info["rls_user_id"] = user_id
        doc = await DocumentService.get_document(session, document_id, user_id)
        if doc is None:
            raise HTTPException(404, "Document unavailable")
        if doc.embedding_status == "completed":
            return True
        async def current_attempt():
            current = await load_claimed_job(request.job_id)
            return bool(current and int(current["claimed_attempt"]) == request.attempt)
        return await EmbeddingService.embed_document(session, document_id, user_id, lease_guard=current_attempt)


@router.post("/internal/local-models/embed", response_model=EmbedResult)
async def embed(request: EmbedAttempt, authorization: str | None = Header(default=None)):
    authorize(authorization)
    if not ready.is_set():
        raise HTTPException(503, "Local models unavailable")
    key = (request.job_id, request.attempt)
    task = _active.get(key)
    if task is None:
        task = asyncio.create_task(embed_attempt(request))
        _active[key] = task
        def finished(completed):
            _active.pop(key, None)
            # Consume exceptions when the caller disconnected; retries remain idempotent.
            if not completed.cancelled():
                completed.exception()
        task.add_done_callback(finished)
    return EmbedResult(succeeded=await asyncio.shield(task))
