from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from app.core.config import settings
from app.core.cors import setup_cors
from app.schemas.health import HealthResponse
from app.api.v1 import api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Pre-warm embedding model if configured
    if settings.PREWARM_MODELS:
        from app.ml.loader import warmup_embedding_model
        warmup_embedding_model()
    yield
    # Shutdown: Clean up connections / resources


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Configure CORS
setup_cors(app)

# Root-level health check endpoint for container/orchestrator probes
@app.get("/health", response_model=HealthResponse, summary="Root Health Check", tags=["health"])
async def root_health() -> HealthResponse:
    """Returns basic health status for the API service."""
    return HealthResponse(status="ok")


# Include Versioned API Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)
