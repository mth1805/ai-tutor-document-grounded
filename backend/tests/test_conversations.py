import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

HEADERS_A = {"Authorization": f"Bearer test-token:{USER_A_ID}"}
HEADERS_B = {"Authorization": f"Bearer test-token:{USER_B_ID}"}


def test_unauthenticated_conversations_rejected():
    """Verify unauthenticated requests return 401."""
    fake_ws_id = str(uuid.uuid4())
    res = client.get(f"/api/v1/workspaces/{fake_ws_id}/conversations")
    assert res.status_code == 401

    fake_conv_id = str(uuid.uuid4())
    res = client.get(f"/api/v1/conversations/{fake_conv_id}/messages")
    assert res.status_code == 401


def test_conversation_crud_and_messages_lifecycle():
    """Verify full lifecycle: workspace -> conversation -> messages -> cascade delete."""
    # 1. User A creates a workspace
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Physics 101"},
    )
    assert ws_res.status_code == 201
    ws_id = ws_res.json()["id"]

    # 2. User A creates a conversation in own workspace
    conv_res = client.post(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers=HEADERS_A,
        json={"title": "Quantum Mechanics Discussion"},
    )
    assert conv_res.status_code == 201
    conv = conv_res.json()
    assert conv["title"] == "Quantum Mechanics Discussion"
    assert conv["workspace_id"] == ws_id
    assert conv["user_id"] == USER_A_ID
    conv_id = conv["id"]

    # 3. User A lists conversations in workspace
    list_res = client.get(
        f"/api/v1/workspaces/{ws_id}/conversations",
        headers=HEADERS_A,
    )
    assert list_res.status_code == 200
    conversations = list_res.json()
    assert len(conversations) >= 1
    assert any(c["id"] == conv_id for c in conversations)

    # 4. User A renames conversation
    update_res = client.patch(
        f"/api/v1/conversations/{conv_id}",
        headers=HEADERS_A,
        json={"title": "Advanced Quantum Theory"},
    )
    assert update_res.status_code == 200
    assert update_res.json()["title"] == "Advanced Quantum Theory"

    # 5. User A creates messages
    msg1_res = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        headers=HEADERS_A,
        json={"role": "user", "content": "What is the Schrödinger equation?"},
    )
    assert msg1_res.status_code == 201
    msg1 = msg1_res.json()
    assert msg1["role"] == "user"
    assert msg1["content"] == "What is the Schrödinger equation?"
    assert msg1["conversation_id"] == conv_id

    msg2_res = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        headers=HEADERS_A,
        json={
            "role": "assistant",
            "content": "It describes the wave function of a quantum-mechanical system.",
        },
    )
    assert msg2_res.status_code == 201

    # 6. User A lists messages (chronological order)
    messages_res = client.get(
        f"/api/v1/conversations/{conv_id}/messages",
        headers=HEADERS_A,
    )
    assert messages_res.status_code == 200
    messages = messages_res.json()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # 7. User A deletes conversation
    del_res = client.delete(
        f"/api/v1/conversations/{conv_id}",
        headers=HEADERS_A,
    )
    assert del_res.status_code == 204

    # 8. Deleting conversation removes its messages (cascade)
    verify_msgs = client.get(
        f"/api/v1/conversations/{conv_id}/messages",
        headers=HEADERS_A,
    )
    assert verify_msgs.status_code == 404


def test_cross_user_isolation_for_conversations_and_messages():
    """Verify User B cannot access, list, or post messages in User A's workspace/conversations."""
    # User A creates workspace and conversation
    ws_res = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "User A Private Study"},
    )
    assert ws_res.status_code == 201
    user_a_ws_id = ws_res.json()["id"]

    conv_res = client.post(
        f"/api/v1/workspaces/{user_a_ws_id}/conversations",
        headers=HEADERS_A,
        json={"title": "Confidential Research"},
    )
    assert conv_res.status_code == 201
    user_a_conv_id = conv_res.json()["id"]

    # User B attempts to create conversation in User A's workspace -> 404
    b_create_conv = client.post(
        f"/api/v1/workspaces/{user_a_ws_id}/conversations",
        headers=HEADERS_B,
        json={"title": "Intruder Thread"},
    )
    assert b_create_conv.status_code == 404

    # User B attempts to list conversations in User A's workspace -> 404
    b_list_conv = client.get(
        f"/api/v1/workspaces/{user_a_ws_id}/conversations",
        headers=HEADERS_B,
    )
    assert b_list_conv.status_code == 404

    # User B attempts to get details of User A's conversation -> 404
    b_get_conv = client.get(
        f"/api/v1/conversations/{user_a_conv_id}",
        headers=HEADERS_B,
    )
    assert b_get_conv.status_code == 404

    # User B attempts to rename User A's conversation -> 404
    b_rename_conv = client.patch(
        f"/api/v1/conversations/{user_a_conv_id}",
        headers=HEADERS_B,
        json={"title": "Hacked Title"},
    )
    assert b_rename_conv.status_code == 404

    # User B attempts to post message to User A's conversation -> 404
    b_post_msg = client.post(
        f"/api/v1/conversations/{user_a_conv_id}/messages",
        headers=HEADERS_B,
        json={"role": "user", "content": "Hacked message"},
    )
    assert b_post_msg.status_code == 404

    # User B attempts to list messages of User A's conversation -> 404
    b_list_msgs = client.get(
        f"/api/v1/conversations/{user_a_conv_id}/messages",
        headers=HEADERS_B,
    )
    assert b_list_msgs.status_code == 404

    # User B attempts to delete User A's conversation -> 404
    b_del_conv = client.delete(
        f"/api/v1/conversations/{user_a_conv_id}",
        headers=HEADERS_B,
    )
    assert b_del_conv.status_code == 404


def test_workspace_delete_cascades_through_conversations_and_messages():
    workspace = client.post(
        "/api/v1/workspaces",
        headers=HEADERS_A,
        json={"name": "Cascade test"},
    ).json()
    conversation = client.post(
        f"/api/v1/workspaces/{workspace['id']}/conversations",
        headers=HEADERS_A,
        json={"title": "Cascade child"},
    ).json()
    message_response = client.post(
        f"/api/v1/conversations/{conversation['id']}/messages",
        headers=HEADERS_A,
        json={"role": "user", "content": "Remove with parent"},
    )
    assert message_response.status_code == 201

    deleted = client.delete(
        f"/api/v1/workspaces/{workspace['id']}", headers=HEADERS_A
    )
    assert deleted.status_code == 204
    assert client.get(
        f"/api/v1/conversations/{conversation['id']}", headers=HEADERS_A
    ).status_code == 404
    assert client.get(
        f"/api/v1/conversations/{conversation['id']}/messages", headers=HEADERS_A
    ).status_code == 404
