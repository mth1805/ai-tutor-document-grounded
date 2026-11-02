"""Focused production configuration and operational endpoint checks."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.main import app


def test_production_configuration_accepts_required_deployment_values():
    configured = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        SUPABASE_URL="https://project.supabase.co",
        SUPABASE_ANON_KEY="public-test-key",
        SUPABASE_DB_URL="postgresql+asyncpg://db.example/postgres",
        GEMINI_API_KEY="test-gemini-key",
        TAVILY_API_KEY="test-tavily-key",
        BACKEND_CORS_ORIGINS=["https://app.example.com"],
    )
    assert configured.resolved_database_url == "postgresql+asyncpg://db.example/postgres"


@pytest.mark.parametrize("origins", [["*"], ["http://localhost:3000"]])
def test_production_configuration_rejects_untrusted_cors_origins(origins):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            SUPABASE_URL="https://project.supabase.co",
            SUPABASE_ANON_KEY="public-test-key",
            SUPABASE_DB_URL="postgresql+asyncpg://db.example/postgres",
            GEMINI_API_KEY="test-gemini-key",
            TAVILY_API_KEY="test-tavily-key",
            BACKEND_CORS_ORIGINS=origins,
        )


def test_production_configuration_requires_database_and_provider_secrets():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ENVIRONMENT="production",
            SUPABASE_URL="https://project.supabase.co",
            SUPABASE_ANON_KEY="public-test-key",
            BACKEND_CORS_ORIGINS=["https://app.example.com"],
        )


def test_health_and_readiness_are_lightweight(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok"}
    assert client.get("/ready").headers.get("x-request-id")


def test_production_readiness_fails_when_database_is_unavailable(monkeypatch):
    import app.api.v1.health as health

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(health, "engine", None)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"


def test_request_body_limit_and_error_response_are_sanitized():
    client = TestClient(app)
    too_large = client.post("/health", headers={"content-length": "27000001"})
    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "REQUEST_TOO_LARGE"


@pytest.mark.asyncio
async def test_unhandled_exception_handler_does_not_return_exception_text():
    from starlette.requests import Request
    from app.main import sanitized_unhandled_exception

    scope = {
        "type": "http", "method": "GET", "path": "/internal", "headers": [],
        "query_string": b"", "server": ("test", 80), "client": ("test", 1),
        "scheme": "http",
    }
    response = await sanitized_unhandled_exception(Request(scope), RuntimeError("private traceback path"))
    assert response.status_code == 500
    assert b"private traceback path" not in response.body
    assert b"INTERNAL_SERVER_ERROR" in response.body


def test_cors_middleware_uses_configured_origins_only():
    allowed = [middleware.kwargs.get("allow_origins") for middleware in app.user_middleware
               if middleware.cls.__name__ == "CORSMiddleware"]
    assert allowed
    assert settings.BACKEND_CORS_ORIGINS in allowed
    assert "*" not in allowed[0]


def test_chat_stream_rejects_missing_auth_without_creating_stream():
    import uuid

    response = TestClient(app).post(
        f"/api/v1/conversations/{uuid.uuid4()}/chat",
        json={"content": "test"},
    )
    assert response.status_code == 401


def test_modal_app_image_construction_order():
    pytest.importorskip("modal")
    import deploy.modal_app as modal_app

    # Ensure add_local_dir is at the end of the chain (represented as Image(local files))
    # and all build steps (pip installs, env setup) occur BEFORE add_local_dir.
    assert repr(modal_app.image) == "Image(local files)"
    assert repr(modal_app.gpu_image) == "Image(local files)"
    assert modal_app.app.name == "ai-tutor-api"
    assert len(modal_app.secrets) == 1
    assert modal_app.MODEL_CACHE_VOLUME is not None
    assert repr(modal_app.fastapi_app) == "Function(fastapi_app)"
    assert repr(modal_app.poll_ingestion_queue) == "Function(poll_ingestion_queue)"
    assert repr(modal_app.process_ingestion_job) == "Function(process_ingestion_job)"

