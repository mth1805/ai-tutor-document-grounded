import sys
from pathlib import Path

import pytest

# Support the documented repository-root invocation: python -m pytest backend/tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.db.session import get_db
from app.main import app


@pytest.fixture(autouse=True)
def use_development_environment_for_tests(monkeypatch):
    """Enable explicitly local-only auth and storage behavior in test runs."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", None)


@pytest.fixture(autouse=True)
def isolate_api_tests_from_configured_database():
    """Keep CRUD API tests deterministic; the RLS test owns its real DB engine."""
    async def use_in_memory_database():
        yield None

    app.dependency_overrides[get_db] = use_in_memory_database
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def use_mock_embedding_for_tests():
    """Ensure tests always use MockEmbeddingProvider to prevent downloading heavy model weights."""
    from app.ml.loader import set_embedding_provider
    from app.ml.mock_provider import MockEmbeddingProvider

    set_embedding_provider(MockEmbeddingProvider())
    try:
        yield
    finally:
        set_embedding_provider(None)
