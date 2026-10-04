from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text
from app.core.config import settings
from app.db.session import engine
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """Returns basic health status for the API service."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse, summary="Readiness Check")
async def readiness_check() -> HealthResponse:
    """Check configuration and database connectivity without loading ML models."""
    production = settings.ENVIRONMENT.lower() in {"production", "prod"}
    missing = []
    if production:
        if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
            missing.append("auth_configuration")
        if engine is None:
            missing.append("database_configuration")
        else:
            try:
                async with engine.connect() as connection:
                    await connection.execute(text("SELECT 1"))
            except Exception:
                missing.append("database_unavailable")
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "dependencies": missing},
        )
    return HealthResponse(status="ok")
