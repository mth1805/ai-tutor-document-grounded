"""Durable Supabase Postgres ingestion queue worker helpers.

Only queue-control RPCs use the backend-only service key. Actual document reads
and writes run through the existing user-scoped authenticated SQLAlchemy session.
"""
import asyncio
import logging
import os
import signal
import uuid
from time import perf_counter
from typing import Any

import httpx

from app.core.config import settings
from app.services.ingestion_service import IngestionService
from app.services.ingestion.telemetry import context, event

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
    jobs = result if isinstance(result, list) else []
    for job in jobs:
        event("claimed", job_id=job.get("job_id"), document_id=job.get("claimed_document_id"), attempt=job.get("claimed_attempt"))
    return jobs


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
    started = perf_counter()
    token = context.set({"job_id": str(job_id), "document_id": None, "attempt": None, "metrics": {
        "OCR_ms": 0, "embedding_count": 0, "chunk_count": 0,
        "storage_download_ms": 0, "parsing_ms": 0, "chunking_ms": 0,
        "model_load_ms": 0, "embedding_ms": 0, "vector_store_ms": 0,
        "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
        "model_device": None,
    }})
    succeeded = False
    try:
        succeeded = await _process_claimed_job(job_id)
        return succeeded
    finally:
        state = context.get()
        event("ready" if succeeded else "failed", **state["metrics"],
              total_ingestion_ms=round((perf_counter() - started) * 1000, 2),
              final_ingestion_status="ready" if succeeded else "failed")
        context.reset(token)


async def _process_claimed_job(job_id: uuid.UUID) -> bool:
    """Run one claimed job; parser/provider detail is kept out of queue logs."""
    job = await load_claimed_job(job_id)
    if not job:
        logger.warning("ingestion_job_unavailable job_id=%s", job_id)
        return False
    document_id = uuid.UUID(job["claimed_document_id"])
    user_id = uuid.UUID(job["claimed_user_id"])
    attempt = int(job["claimed_attempt"])
    context.get().update(document_id=str(document_id), attempt=attempt)
    import torch
    available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if available else None
    context.get()["metrics"].update(cuda_available=available, gpu_device_name=device_name)
    event("claimed", cuda_available=available, gpu_device_name=device_name)
    logger.info(
        "ingestion_job_claimed job_id=%s document_id=%s attempt=%d",
        job_id, document_id, attempt,
    )
    try:
        succeeded = await IngestionService.process_document_background(
            document_id, user_id, defer_failure_to_queue=True,
        )
        if succeeded and settings.LOCAL_SHARED_MODELS:
            succeeded = await request_local_embedding(job_id, attempt)
    except Exception as exc:
        logger.error(
            "ingestion_job_failed job_id=%s category=%s",
            job_id, type(exc).__name__,
        )
        succeeded = False
    if settings.LOCAL_SHARED_MODELS:
        current = await load_claimed_job(job_id)
        if not current or int(current["claimed_attempt"]) != attempt:
            logger.warning("ingestion_attempt_stale job_id=%s attempt=%d", job_id, attempt)
            return False
    await finish_job(job_id, succeeded=succeeded)
    logger.info(
        "%s job_id=%s document_id=%s attempt=%d",
        "ingestion_job_completed" if succeeded else "ingestion_job_failed",
        job_id, document_id, attempt,
    )
    return succeeded


async def run_worker_forever(
    poll_seconds: float = 5.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Local/dev worker process; production uses the scheduled Modal poller."""
    stop = stop_event or asyncio.Event()
    # Claim only the job being worked on: a sequential CPU worker must not lease
    # a batch whose later jobs might expire before their parsing even starts.
    while not stop.is_set():
        try:
            if settings.LOCAL_SHARED_MODELS:
                await wait_local_models(stop)
                if stop.is_set():
                    break
            jobs = await claim_jobs(limit=1)
            for job in jobs:
                # Drain a claimed attempt on SIGTERM; don't leave an idle lease.
                await process_claimed_job(uuid.UUID(job["job_id"]))
        except Exception as exc:
            # The queue keeps the lease/retry record if finalization fails.
            # Transient RPC/network failures must not terminate the only worker.
            logger.error("ingestion_worker_poll_failed category=%s", type(exc).__name__)
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(1.0, poll_seconds))
        except asyncio.TimeoutError:
            pass


def local_headers() -> dict[str, str]:
    from app.api.v1.local_models import worker_token
    return {"Authorization": "Bearer " + worker_token()}


async def wait_local_models(stop: asyncio.Event) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        while not stop.is_set():
            try:
                response = await client.get(settings.LOCAL_MODEL_API_URL + "/internal/local-models", headers=local_headers())
                response.raise_for_status()
                if response.json().get("ready") is True:
                    return
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass


async def request_local_embedding(job_id: uuid.UUID, attempt: int) -> bool:
    # A retry attaches to the same backend task. Leave ample time within the lease.
    async with httpx.AsyncClient(timeout=1500) as client:
        for retry in range(2):
            try:
                response = await client.post(
                    settings.LOCAL_MODEL_API_URL + "/internal/local-models/embed",
                    headers=local_headers(), json={"job_id": str(job_id), "attempt": attempt},
                )
                response.raise_for_status()
                return response.json().get("succeeded") is True
            except httpx.TransportError:
                if retry:
                    raise
        return False


async def main() -> None:
    """CPU Compose entrypoint with signal-aware shutdown and pool cleanup."""
    from app.db.session import AsyncSessionLocal, engine

    _rpc_config()
    if AsyncSessionLocal is None:
        raise QueueConfigurationError("Local ingestion worker requires a configured PostgreSQL connection")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Native Windows development.
            signal.signal(sig, lambda *_args: loop.call_soon_threadsafe(stop.set))
    logger.info("ingestion_worker_started concurrency=1 poll_seconds=5")
    try:
        await run_worker_forever(stop_event=stop)
    finally:
        if engine is not None:
            await engine.dispose()
        logger.info("ingestion_worker_stopped")


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(main())
