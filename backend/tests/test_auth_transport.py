import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from app.core import security


@pytest.mark.asyncio
@pytest.mark.parametrize('outcomes, expected_status, expected_calls', [
    ([httpx.ReadTimeout('PRIVATE_SESSION_TOKEN'), 200], None, 2),
    ([401], 401, 1),
    ([httpx.ReadTimeout('PRIVATE_SESSION_TOKEN'), httpx.ReadTimeout('PRIVATE_SESSION_TOKEN')], 503, 2),
    ([500], 503, 1),
])
async def test_auth_transport_retry_is_bounded_and_never_accepts_invalid_sessions(
    monkeypatch, caplog, outcomes, expected_status, expected_calls,
):
    user_id = uuid.uuid4()
    calls = []
    monkeypatch.setattr(security.settings, 'SUPABASE_JWT_SECRET', None)
    monkeypatch.setattr(security.settings, 'SUPABASE_URL', 'https://dev.example.invalid')
    monkeypatch.setattr(security.settings, 'SUPABASE_ANON_KEY', 'test-anon-key')
    monkeypatch.setattr(security.settings, 'SUPABASE_AUTH_TIMEOUT_SECONDS', 15)
    monkeypatch.setattr(security.asyncio, 'sleep', AsyncMock())

    class Client:
        def __init__(self, *, timeout):
            assert timeout == 15

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, *, headers):
            calls.append(url)
            outcome = outcomes[len(calls)-1]
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(status_code=outcome, json=lambda: {'id': str(user_id)})

    monkeypatch.setattr(security.httpx, 'AsyncClient', Client)
    if expected_status:
        with pytest.raises(HTTPException) as caught:
            await security.verify_supabase_token('PRIVATE_SESSION_TOKEN')
        assert caught.value.status_code == expected_status
    else:
        assert (await security.verify_supabase_token('PRIVATE_SESSION_TOKEN')).id == user_id
    assert len(calls) == expected_calls
    assert 'PRIVATE_SESSION_TOKEN' not in caplog.text
