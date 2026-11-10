"""Modal ASGI deployment wrapper for the existing FastAPI application.

Run from the repository root with `modal deploy backend/deploy/modal_app.py`.
No Modal resources are created until that explicit command is run.

Path-safety note: ROOT / parents[2] cannot be used unconditionally because Modal
uploads this file to /root/modal_app.py (only two ancestors), which raises an
IndexError at import time.  BACKEND is resolved as parents[1] of this file,
which is the ``backend/`` directory in the local repo.  All BACKEND references
are used exclusively at image build time (pip-requirements paths and add_local_dir);
they are never evaluated inside a running container.
"""
from pathlib import Path
import os
import asyncio
import uuid
import logging

import modal
from fastapi import HTTPException, Request

# ``backend/deploy/modal_app.py``  ->  parents[0]=deploy/  parents[1]=backend/
# On Modal the file is /root/modal_app.py; parents[2] would raise IndexError.
# BACKEND is only consumed at image build time (requirements paths, add_local_dir),
# never inside a running container, so resolving from __file__ is safe.
BACKEND = Path(__file__).resolve().parent.parent
MAX_CONCURRENT_INPUTS = int(os.getenv("MODAL_MAX_CONCURRENT_INPUTS", "10"))
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    os.getenv("MODAL_MODEL_CACHE_VOLUME", "ai-tutor-model-cache"), create_if_missing=True
)
READ_ONLY_MODEL_CACHE_VOLUME = MODEL_CACHE_VOLUME.with_mount_options(read_only=True)
_configured_inference_gpu = os.getenv("MODAL_INFERENCE_GPU", "").strip()
_legacy_ingestion_gpu = os.getenv("MODAL_INGESTION_GPU")
if _configured_inference_gpu:
    INFERENCE_GPU = _configured_inference_gpu
elif _legacy_ingestion_gpu is not None:
    logging.getLogger(__name__).warning(
        "MODAL_INGESTION_GPU is deprecated; use MODAL_INFERENCE_GPU. It never allocates a GPU to ingestion."
    )
    INFERENCE_GPU = _legacy_ingestion_gpu.strip() or "T4"
else:
    INFERENCE_GPU = "T4"
GPU_WORKER_TIMEOUT = int(os.getenv("GPU_WORKER_TIMEOUT_SECONDS", "20"))
CPU_REQUIREMENTS = BACKEND / "requirements.lock.txt"
GPU_REQUIREMENTS = BACKEND / "requirements-gpu.lock.txt"

app = modal.App("ai-tutor-api")
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Image uploads support scanned PDFs/images and legacy .doc files, so install
    # their native tools alongside the Python-only PDF/DOCX dependencies.
    .apt_install(
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "tesseract-ocr-vie",
        "libreoffice-writer",
    )
)
# The API and durable ingestion worker stay CPU-only. A separate inference
# endpoint owns CUDA so GPU scheduling failures cannot strand claimed jobs.
image = (
    base_image
    .pip_install_from_requirements(str(CPU_REQUIREMENTS))
    .env({"PYTHONPATH": "/root/backend", "HF_HOME": "/models/huggingface"})
    .add_local_dir(str(BACKEND / "app"), remote_path="/root/backend/app")
)
# Keep CUDA dependencies and the inference image small and exclusive to the
# bounded inference service; OCR utilities are only needed by CPU ingestion.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(GPU_REQUIREMENTS))
    .env({"PYTHONPATH": "/root/backend", "HF_HOME": "/models/huggingface"})
    .add_local_dir(str(BACKEND / "app"), remote_path="/root/backend/app")
)
ingestion_image = image
inference_image = gpu_image

# Runtime functions mount the cache read-only. Run prepare_model_cache once
# after deployment and again after changing either model checkpoint.
secrets = [modal.Secret.from_name(os.getenv("MODAL_SECRET_NAME", "ai-tutor-production"))]


@app.function(
    image=image,
    secrets=secrets,
    cpu=float(os.getenv("MODAL_API_CPU", "4")),
    memory=int(os.getenv("MODAL_API_MEMORY_MB", "8192")),
    volumes={"/models": READ_ONLY_MODEL_CACHE_VOLUME},
    timeout=150,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.asgi_app()
def fastapi_app():
    """Expose the current application object and preserve its REST/SSE routes."""
    os.environ["EMBEDDING_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["RERANKER_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
    image=ingestion_image,
    secrets=secrets,
    cpu=float(os.getenv("MODAL_INGESTION_CPU", "4")),
    memory=int(os.getenv("MODAL_INGESTION_MEMORY_MB", "12288")),
    gpu=None,
    volumes={"/models": READ_ONLY_MODEL_CACHE_VOLUME},
    timeout=1800,
    scaledown_window=300,
)
def process_ingestion_job(job_id: str):
    """Process parsing and persistence on CPU; embeddings use bounded GPU inference."""
    # Set worker-only options before importing Settings; the inference endpoint
    # is the only CUDA resource and the worker can fall back to local CPU.
    os.environ["DEVICE_MODE"] = "auto"
    os.environ["EMBEDDING_DEVICE"] = "auto"
    os.environ["RERANKER_DEVICE"] = "auto"
    # Match HF_HOME's normal hub subdirectory so CPU/GPU workers share weights.
    os.environ["EMBEDDING_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["RERANKER_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["AUTO_EMBED_AFTER_INGESTION"] = "true"
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    from app.workers.ingestion_queue import process_claimed_job

    # Reuse the event loop as well as the model singleton in warm containers:
    # SQLAlchemy's pooled async connections belong to their original loop.
    global _worker_runner
    if "_worker_runner" not in globals():
        _worker_runner = asyncio.Runner()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return _worker_runner.run(process_claimed_job(uuid.UUID(job_id)))


@app.function(
    image=inference_image,
    secrets=secrets,
    cpu=float(os.getenv("MODAL_INFERENCE_CPU", "4")),
    memory=int(os.getenv("MODAL_INFERENCE_MEMORY_MB", "12288")),
    gpu=None,
    volumes={"/models": MODEL_CACHE_VOLUME},
    timeout=3600,
    max_containers=1,
    scaledown_window=60,
)
def prepare_model_cache():
    """Preload both public model checkpoints before read-only runtime mounts."""
    os.environ["DEVICE_MODE"] = "cpu"
    os.environ["GPU_FALLBACK_ENABLED"] = "false"
    os.environ["EMBEDDING_DEVICE"] = "cpu"
    os.environ["RERANKER_DEVICE"] = "cpu"
    os.environ["EMBEDDING_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["RERANKER_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from app.ml.loader import get_embedding_provider, get_reranker_provider

    embedding = get_embedding_provider()
    embedding.encode_batch(["model cache warmup"], normalize=True, batch_size=1)
    reranker = get_reranker_provider()
    reranker.predict([("model cache warmup", "model cache warmup")], batch_size=1)
    MODEL_CACHE_VOLUME.commit()
    logging.getLogger(__name__).info("model_cache_prepared embedding=%s reranker=%s",
                                     embedding.model_name, reranker.model_name)
    return True


@app.function(
    image=inference_image,
    secrets=secrets,
    cpu=float(os.getenv("MODAL_INFERENCE_CPU", "4")),
    memory=int(os.getenv("MODAL_INFERENCE_MEMORY_MB", "12288")),
    gpu=INFERENCE_GPU,
    volumes={"/models": READ_ONLY_MODEL_CACHE_VOLUME},
    timeout=max(60, GPU_WORKER_TIMEOUT),
    scaledown_window=300,
    max_containers=int(os.getenv("MODAL_INFERENCE_MAX_CONTAINERS", "2")),
)
@modal.concurrent(max_inputs=1)
@modal.fastapi_endpoint(method="POST")
async def gpu_inference(request: Request):
    """Private GPU inference endpoint. Inputs are data only; it never writes jobs."""
    import os
    from app.ml.device_runtime import is_valid_bearer_token
    expected = os.getenv("MODAL_INFERENCE_TOKEN", "")
    supplied = request.headers.get("authorization", "")
    if not is_valid_bearer_token(expected, supplied):
        raise HTTPException(status_code=401, detail="Unauthorized")
    payload = await request.json()
    operation = payload.get("operation") if isinstance(payload, dict) else None
    values = payload.get("values") if isinstance(payload, dict) else None
    batch_size = payload.get("batch_size", 16) if isinstance(payload, dict) else 16
    if (not isinstance(batch_size, int) or batch_size < 1 or batch_size > 128
            or not isinstance(values, list) or len(values) > 128):
        raise HTTPException(status_code=422, detail="Inference batch exceeds limits")
    if operation == "embed" and any(not isinstance(v, str) or len(v) > 50000 for v in values):
        raise HTTPException(status_code=422, detail="Invalid inference input")
    if operation == "rerank" and any(
        not isinstance(pair, (list, tuple)) or len(pair) != 2
        or any(not isinstance(part, str) or len(part) > 50000 for part in pair)
        for pair in values
    ):
        raise HTTPException(status_code=422, detail="Invalid inference input")
    os.environ["DEVICE_MODE"] = "cuda"
    # Fail the remote request if CUDA cannot run; the CPU-only caller owns the
    # bounded fallback, so the GPU worker never silently computes on CPU.
    os.environ["GPU_FALLBACK_ENABLED"] = "false"
    os.environ["EMBEDDING_DEVICE"] = "cuda"
    os.environ["RERANKER_DEVICE"] = "cuda"
    os.environ["EMBEDDING_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["RERANKER_MODEL_CACHE_DIR"] = "/models/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        if operation == "embed" and isinstance(values, list) and all(isinstance(v, str) for v in values):
            from app.ml.loader import get_embedding_provider
            provider = get_embedding_provider()
            import torch
            outputs = provider.encode_batch(values, normalize=bool(payload.get("normalize", True)), batch_size=int(batch_size))
        elif operation == "rerank" and isinstance(values, list):
            from app.ml.loader import get_reranker_provider
            if not all(isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, str) for x in v) for v in values):
                raise HTTPException(status_code=422, detail="Invalid inference input")
            provider = get_reranker_provider()
            import torch
            outputs = provider.predict(values, batch_size=int(batch_size))
        else:
            raise HTTPException(status_code=422, detail="Invalid inference operation or input")
        device = provider.device
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if device != "cuda" or not isinstance(gpu, str) or not gpu.strip():
            raise RuntimeError("GPU inference did not confirm CUDA execution")
        return {"ok": True, "outputs": outputs, "device": device, "gpu": gpu}
    except HTTPException:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error("gpu_inference_failed operation=%s category=%s", operation, type(exc).__name__)
        raise HTTPException(status_code=503, detail="GPU inference failed") from exc
