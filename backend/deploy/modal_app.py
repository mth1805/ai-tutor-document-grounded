"""Modal ASGI deployment wrapper for the existing FastAPI application.

Run from the repository root with `modal deploy backend/deploy/modal_app.py`.
No Modal resources are created until that explicit command is run.
"""
from pathlib import Path
import os
import asyncio
import uuid

import modal

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
MAX_CONCURRENT_INPUTS = int(os.getenv("MODAL_MAX_CONCURRENT_INPUTS", "10"))
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    os.getenv("MODAL_MODEL_CACHE_VOLUME", "ai-tutor-model-cache"), create_if_missing=True
)
INGESTION_GPU = os.getenv("MODAL_INGESTION_GPU", "").strip() or None

app = modal.App("ai-tutor-api")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(BACKEND / "requirements.lock.txt"))
    .add_local_dir(str(BACKEND / "app"), remote_path="/root/backend/app")
    .env({"PYTHONPATH": "/root/backend", "HF_HOME": "/models/huggingface"})
)

# The persistent volume prevents model downloads on container restarts. It is
# mounted read/write because first-use cache population may occur in a worker.
secrets = [modal.Secret.from_name(os.getenv("MODAL_SECRET_NAME", "ai-tutor-production"))]


@app.function(
    image=image,
    secrets=secrets,
    cpu=float(os.getenv("MODAL_API_CPU", "4")),
    memory=int(os.getenv("MODAL_API_MEMORY_MB", "8192")),
    volumes={"/models": MODEL_CACHE_VOLUME},
    timeout=150,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.asgi_app()
def fastapi_app():
    """Expose the current application object and preserve its REST/SSE routes."""
    from app.main import app as api

    return api


@app.function(
    image=image,
    secrets=secrets,
    timeout=60,
    schedule=modal.Period(seconds=15),
)
def poll_ingestion_queue():
    """Claim durable jobs on CPU and dispatch only actual jobs to the worker."""
    from app.workers.ingestion_queue import claim_jobs

    jobs = asyncio.run(claim_jobs(limit=10))
    for job in jobs:
        process_ingestion_job.spawn(job["job_id"])


@app.function(
    image=image,
    secrets=secrets,
    cpu=float(os.getenv("MODAL_INGESTION_CPU", "4")),
    memory=int(os.getenv("MODAL_INGESTION_MEMORY_MB", "12288")),
    gpu=INGESTION_GPU,
    volumes={"/models": MODEL_CACHE_VOLUME},
    timeout=1800,
    scaledown_window=300,
)
def process_ingestion_job(job_id: str):
    """Process a single durable job; GPU is allocated only to this worker function."""
    from app.workers.ingestion_queue import process_claimed_job

    return asyncio.run(process_claimed_job(uuid.UUID(job_id)))
