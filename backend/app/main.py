from contextlib import asynccontextmanager
import asyncio
import logging
import time
from typing import AsyncGenerator
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from app.core.config import settings
from app.core.cors import setup_cors
from app.schemas.health import HealthResponse
from app.api.v1 import api_v1_router
from app.api.v1.health import readiness_check
from app.db.session import engine
from app.core.observability import RequestIdFilter, request_id_context

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(RequestIdFilter())
logger = logging.getLogger("app.http")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Pre-warm embedding and reranker models if configured
    model_task = None
    if settings.LOCAL_SHARED_MODELS:
        from app.ml.local_runtime import serve_models
        model_task = asyncio.create_task(serve_models())
    elif settings.PREWARM_MODELS:
        from fastapi.concurrency import run_in_threadpool
        from app.ml.loader import warmup_embedding_model, warmup_reranker_model

        await run_in_threadpool(warmup_embedding_model)
        await run_in_threadpool(warmup_reranker_model)
    try:
        yield
    finally:
        if model_task is not None:
            model_task.cancel()
            await asyncio.gather(model_task, return_exceptions=True)
        if engine is not None:
            await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID")
    try:
        request_id = str(uuid.UUID(request_id)) if request_id else str(uuid.uuid4())
    except (ValueError, TypeError, AttributeError):
        request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request_context_token = request_id_context.set(request_id)
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > settings.MAX_REQUEST_BODY_BYTES:
        request_id_context.reset(request_context_token)
        return JSONResponse(
            status_code=413,
            content={"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body exceeds the configured size limit", "details": {}}},
            headers={"X-Request-ID": request_id},
        )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception("request failed request_id=%s route=%s latency_ms=%.2f", request_id, request.url.path, elapsed_ms)
        request_id_context.reset(request_context_token)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request response started request_id=%s method=%s route=%s status=%s latency_ms=%.2f",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
    )

    previous_background = response.background

    async def log_response_completion():
        if previous_background is not None:
            await previous_background()
        total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request response finished request_id=%s method=%s route=%s status=%s total_request_latency_ms=%.2f",
            request_id, request.method, request.url.path, response.status_code, total_ms,
        )

    response.background = BackgroundTask(log_response_completion)
    request_id_context.reset(request_context_token)
    return response


@app.exception_handler(Exception)
async def sanitized_unhandled_exception(request: Request, exc: Exception):
    """Log details internally and return a safe error envelope to clients."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.exception("unhandled error request_id=%s category=%s", request_id, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An internal error occurred", "details": {"request_id": request_id}}},
        headers={"X-Request-ID": request_id},
    )

# Configure CORS
setup_cors(app)

# Root-level health check endpoint for container/orchestrator probes
@app.get("/health", response_model=HealthResponse, summary="Root Health Check", tags=["health"])
async def root_health() -> HealthResponse:
    """Returns basic health status for the API service."""
    return HealthResponse(status="ok")


app.add_api_route("/ready", readiness_check, methods=["GET"], response_model=HealthResponse, tags=["health"])


# Include Versioned API Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)
if settings.LOCAL_SHARED_MODELS:
    from app.api.v1.local_models import router as local_models_router
    app.include_router(local_models_router, include_in_schema=False)
