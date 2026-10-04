"""Durable Supabase Postgres ingestion queue worker helpers.

Only queue-control RPCs use the backend-only service key. Actual document reads
and writes run through the existing user-scoped authenticated SQLAlchemy session.
"""
import asyncio
import logging
import os
import uuid
from typing import Any

import httpx

from app.core.config import settings
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


class QueueConfigurationError(RuntimeError):
    """Raised when a worker is started without its private queue credentials."""


def _rpc_config() -> tuple[str, str]:
    base_url = settings.SUPABASE_URL
    service_key = settings.SUPABASE_SERVICE_ROLE_KEY
    if not base_url or not service_key:
        raise QueueConfigurationError("Supabase queue worker is not configured")
    return base_url.rstrip("/"), service_key


async def _rpc(function: str, payload: dict[str, Any]) -> Any:
    base_url, service_key = _rpc_config()
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{base_url}/rest/v1/rpc/{function}", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()


async def claim_jobs(limit: int = 5) -> list[dict[str, Any]]:
    """Atomically claim due jobs; expired leases are recovered by the database RPC."""
    result = await _rpc(
        "claim_document_ingestion_jobs",
        {"p_limit": max(1, min(limit, 10)), "p_lease_seconds": 3600},
    )
    return result if isinstance(result, list) else []


async def load_claimed_job(job_id: uuid.UUID) -> dict[str, Any] | None:
    rows = await _rpc("get_claimed_document_ingestion_job", {"p_job_id": str(job_id)})
    return rows[0] if isinstance(rows, list) and rows else None


async def finish_job(job_id: uuid.UUID, *, succeeded: bool) -> None:
    if succeeded:
        result = await _rpc("complete_document_ingestion_job", {"p_job_id": str(job_id)})
        if result is not True:
            raise RuntimeError("Queue job was no longer in a processing state")
        return
    await _rpc(
        "fail_document_ingestion_job",
        {"p_job_id": str(job_id), "p_error": "INGESTION_ATTEMPT_FAILED"},
    )


async def process_claimed_job(job_id: uuid.UUID) -> bool:
    """Run one claimed job; parser/provider detail is kept out of queue logs."""
    job = await load_claimed_job(job_id)
    if not job:
        logger.warning("ingestion job unavailable job_id=%s", job_id)
        return False
    document_id = uuid.UUID(job["claimed_document_id"])
    user_id = uuid.UUID(job["claimed_user_id"])
    attempt = int(job["claimed_attempt"])
    logger.info(
        "ingestion job started job_id=%s document_id=%s attempt=%d",
        job_id, document_id, attempt,
    )
    try:
        succeeded = await IngestionService.process_document_background(document_id, user_id)
    except Exception as exc:
        logger.error(
            "ingestion job raised job_id=%s category=%s",
            job_id, type(exc).__name__, exc_info=True,
        )
        succeeded = False
    await finish_job(job_id, succeeded=succeeded)
    logger.info(
        "ingestion job finished job_id=%s document_id=%s succeeded=%s",
        job_id, document_id, succeeded,
    )
    return succeeded


async def run_worker_forever(poll_seconds: float = 5.0) -> None:
    """Local/dev worker process; production uses the scheduled Modal poller."""
    while True:
        jobs = await claim_jobs()
        if not jobs:
            await asyncio.sleep(poll_seconds)
            continue
        for job in jobs:
            await process_claimed_job(uuid.UUID(job["job_id"]))


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker_forever())
