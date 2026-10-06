import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
import app.core.security as security

client = TestClient(app)

USER_1_ID = "11111111-1111-1111-1111-111111111111"
USER_2_ID = "22222222-2222-2222-2222-222222222222"

USER_1_HEADERS = {"Authorization": f"Bearer test-token:{USER_1_ID}"}
USER_2_HEADERS = {"Authorization": f"Bearer test-token:{USER_2_ID}"}


def test_unauthenticated_request_rejected():
    """Verify that requests without Authorization header return 401."""
    response = client.get("/api/v1/workspaces")
    assert response.status_code == 401
    assert "detail" in response.json()


def test_development_test_token_is_rejected_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    response = client.get("/api/v1/workspaces", headers=USER_1_HEADERS)
    assert response.status_code == 401


def test_unsigned_jwt_is_rejected():
    unsigned_token = (
        "eyJhbGciOiJub25lIn0."
        "eyJzdWIiOiIxMTExMTExMS0xMTExLTExMTEtMTExMS0xMTExMTExMTExMTEifQ."
    )
    response = client.get(
        "/api/v1/workspaces",
        headers={"Authorization": f"Bearer {unsigned_token}"},
    )
    assert response.status_code == 401


def test_workspace_creation_uses_verified_supabase_auth_identity(monkeypatch):
    """The bearer token's verified Supabase user ID must reach the workspace row."""
    auth_user_id = "9b0bcf8a-f3e4-4d87-9a35-d59e11675a9e"
    access_token = "verified-supabase-access-token"

    class AuthResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"id": auth_user_id, "email": "student@example.test"}

    class AuthClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url, headers):
            assert url == "https://auth.example.test/auth/v1/user"
            assert headers["Authorization"] == f"Bearer {access_token}"
            return AuthResponse()

    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "SUPABASE_JWT_SECRET", None)
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://auth.example.test")
    monkeypatch.setattr(settings, "SUPABASE_ANON_KEY", "test-anon-key")
    monkeypatch.setattr(security.httpx, "AsyncClient", AuthClient)

    response = client.post(
        "/api/v1/workspaces",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"name": "Auth identity regression"},
    )

    assert response.status_code == 201
    assert response.json()["user_id"] == auth_user_id
    assert response.json()["user_id"] != USER_1_ID


def test_create_and_list_workspace():
    """Verify authenticated user can create and list their workspaces."""
    create_res = client.post(
        "/api/v1/workspaces",
        headers=USER_1_HEADERS,
        json={"name": "Machine Learning 101"},
    )
    assert create_res.status_code == 201
    created_ws = create_res.json()
    assert created_ws["name"] == "Machine Learning 101"
    assert created_ws["user_id"] == USER_1_ID
    ws_id = created_ws["id"]

    # List workspaces for user 1
    list_res = client.get("/api/v1/workspaces", headers=USER_1_HEADERS)
    assert list_res.status_code == 200
    workspaces = list_res.json()
    assert any(w["id"] == ws_id for w in workspaces)


def test_update_workspace():
    """Verify authenticated user can rename their workspace."""
    # Create workspace
    create_res = client.post(
        "/api/v1/workspaces",
        headers=USER_1_HEADERS,
        json={"name": "Deep Learning Draft"},
    )
    assert create_res.status_code == 201
    ws_id = create_res.json()["id"]

    # Rename
    update_res = client.patch(
        f"/api/v1/workspaces/{ws_id}",
        headers=USER_1_HEADERS,
        json={"name": "Deep Learning Final"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Deep Learning Final"


def test_delete_workspace():
    """Verify authenticated user can delete their workspace."""
    create_res = client.post(
        "/api/v1/workspaces",
        headers=USER_1_HEADERS,
        json={"name": "Workspace To Delete"},
    )
    assert create_res.status_code == 201
    ws_id = create_res.json()["id"]

    # Delete
    del_res = client.delete(f"/api/v1/workspaces/{ws_id}", headers=USER_1_HEADERS)
    assert del_res.status_code == 204

    # Confirm deletion from list
    list_res = client.get("/api/v1/workspaces", headers=USER_1_HEADERS)
    assert not any(w["id"] == ws_id for w in list_res.json())


def test_cross_user_isolation_prevent_idor():
    """Verify User 2 cannot access, update, or delete User 1's workspace."""
    # User 1 creates a private workspace
    create_res = client.post(
        "/api/v1/workspaces",
        headers=USER_1_HEADERS,
        json={"name": "User 1 Secret Notes"},
    )
    assert create_res.status_code == 201
    user_1_ws_id = create_res.json()["id"]

    # User 2 tries to list - should NOT see User 1's workspace
    user_2_list = client.get("/api/v1/workspaces", headers=USER_2_HEADERS)
    assert user_2_list.status_code == 200
    assert not any(w["id"] == user_1_ws_id for w in user_2_list.json())

    # User 2 tries to rename User 1's workspace - should be rejected with 404
    user_2_update = client.patch(
        f"/api/v1/workspaces/{user_1_ws_id}",
        headers=USER_2_HEADERS,
        json={"name": "Hacked Name"},
    )
    assert user_2_update.status_code == 404

    # User 2 tries to delete User 1's workspace - should be rejected with 404
    user_2_delete = client.delete(
        f"/api/v1/workspaces/{user_1_ws_id}",
        headers=USER_2_HEADERS,
    )
    assert user_2_delete.status_code == 404
